from typing import List, Tuple, Callable, Optional, Dict, Literal
import numpy as np
import scipy.sparse as sp

class HighPerformanceHypergraph:
    """
    Core data structure for Hypergraph-Evolved Pipelines (HEP).
    Optimized for fast feature generation using sparse matrices.
    
    Attributes:
        num_vertices (int): Number of initial features (nodes).
        hyperedges (List[dict]): List of hyperedge metadata [{'components': [idxs], 'func': 'sum'}].
        incidence_matrix (sp.dok_matrix): Incidence matrix H (Vertices x Edges) for mutation.
        _cached_csr (Optional[sp.csr_matrix]): Cached CSR matrix for fast multiplication.
    """
    
    def __init__(self, num_vertices: int):
        self.num_vertices = num_vertices
        self.hyperedges: List[Dict] = []
        # Matrix H: Rows = Features, Cols = Hyperedges
        # H[i, j] = 1 if feature i is in hyperedge j
        self.incidence_matrix = sp.dok_matrix((num_vertices, 0), dtype=np.float32)
        self._cached_csr: Optional[sp.csr_matrix] = None
        self._cached_groups: Optional[Dict[str, sp.csr_matrix]] = None

    def add_hyperedge(self, node_indices: List[int], function: str = 'sum'):
        """
        Adds a new hyperedge connecting specified nodes.
        """
        # Resize matrix if needed (DOK is not good for resizing, but we can stack)
        # Actually, for frequent component-wise growth, list of columns might be better, 
        # then assemble CSR. But let's stick to concept: local DOK/LIL updates.
        
        # Simple implementation: append column to internal list/structures 
        # and rebuild sparse matrix lazily or on transform.
        
        edge_id = len(self.hyperedges)
        self.hyperedges.append({'components': node_indices, 'func': function})
        
        # Invalidate cache
        self._cached_csr = None
        self._cached_groups = None

    def _build_matrices(self):
        """
        Groups edges by function and builds sparse matrices for vectorization.
        """
        groups = {}
        # Group edge indices by function
        func_indices = {}
        for i, edge in enumerate(self.hyperedges):
            func = edge['func']
            if func not in func_indices:
                func_indices[func] = []
            func_indices[func].append(i)

        for func, edge_idxs in func_indices.items():
            # Build sub-matrix H_func (Vertices x K_edges)
            # data, row, col for COO construction
            data = []
            rows = []
            cols = []
            
            for new_col_idx, original_edge_idx in enumerate(edge_idxs):
                nodes = self.hyperedges[original_edge_idx]['components']
                for node in nodes:
                    if node < self.num_vertices:
                        rows.append(node)
                        cols.append(new_col_idx)
                        data.append(1.0)
            
            matrix_shape = (self.num_vertices, len(edge_idxs))
            # Create CSC for fast column slicing or CSR for fast multiplication? 
            # X (Samples x Features) @ H (Features x Edges) -> (Samples x Edges)
            # We need standard MatMul, so H should be compatible. 
            # sp.csc_matrix is often better for "vector of columns".
            groups[func] = sp.csc_matrix((data, (rows, cols)), shape=matrix_shape)
            
        self._cached_groups = groups

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Generates new features from input data X.
        X shape: (N_samples, N_features)
        Returns: (N_samples, N_hyperedges)
        """
        if not self.hyperedges:
            return np.zeros((data.shape[0], 0))
            
        if self._cached_groups is None:
            self._build_matrices()
            
        n_samples = data.shape[0]
        n_edges = len(self.hyperedges)
        # Use float64 by default to handle mean/product and avoid int casting errors
        # If data is already float32, maybe preserve it? 
        # For now, let's use float64 for safety or data.dtype if it's floating.
        res_dtype = np.float64
        if np.issubdtype(data.dtype, np.floating):
             res_dtype = data.dtype
             
        result = np.zeros((n_samples, n_edges), dtype=res_dtype)
        
        # Helper to map local group index back to global result index
        # We need to know which edge went where. 
        # Re-iterating is fast enough for metadata.
        
        current_col_indices = {func: 0 for func in self._cached_groups}
        global_indices_map = {} # func -> [global_indices]
        
        for i, edge in enumerate(self.hyperedges):
            func = edge['func']
            if func not in global_indices_map:
                global_indices_map[func] = []
            global_indices_map[func].append(i)

        # Vectorized execution
        X = sp.csr_matrix(data) if sp.issparse(data) else data

        for func, H_sub in self._cached_groups.items():
            global_idxs = global_indices_map[func]
            
            if func == 'sum':
                # Linear combination: X @ H
                # (N x F) @ (F x E_sub) -> (N x E_sub)
                res_sub = X @ H_sub
                if sp.issparse(res_sub):
                    res_sub = res_sub.toarray()
                result[:, global_idxs] = res_sub
                
            elif func == 'product':
                # Non-linear. Cannot use simple matrix mult.
                # But we can assume H_sub is binary.
                # For each col in H_sub, we need product of corresponding columns in X.
                # Optimization: log(prod) = sum(log), but sign issues.
                # Fallback to iteration over columns of H_sub (edges).
                # Since H_sub is CSC, getting column indices is fast.
                
                for local_i, global_i in enumerate(global_idxs):
                    # Get feature indices for this edge
                    # slice CSC column
                    start = H_sub.indptr[local_i]
                    end = H_sub.indptr[local_i+1]
                    feature_idxs = H_sub.indices[start:end]
                    counts = H_sub.data[start:end]
                    
                    # Compute product
                    if len(feature_idxs) > 0:
                        # Start with ones
                        term_prod = np.ones(n_samples, dtype=res_dtype)
                        for idx, count in zip(feature_idxs, counts):
                            if count == 1:
                                term_prod *= data[:, idx]
                            else:
                                term_prod *= (data[:, idx] ** count)
                        result[:, global_i] = term_prod
                        
            elif func == 'mean':
                 # Same as sum, but divide by degree
                 # Degree vector
                 degrees = np.array(H_sub.sum(axis=0)).flatten()
                 # valid degrees only
                 with np.errstate(divide='ignore', invalid='ignore'):
                     res_sub = X @ H_sub
                     if sp.issparse(res_sub):
                         res_sub = res_sub.toarray()
                     res_sub = res_sub / degrees[None, :]
                     # Handle div by zero if any
                     res_sub = np.nan_to_num(res_sub)
                 result[:, global_idxs] = res_sub
            
            else:
                 # Fallback for other funcs
                 for local_i, global_i in enumerate(global_idxs):
                    start = H_sub.indptr[local_i]
                    end = H_sub.indptr[local_i+1]
                    feature_idxs = H_sub.indices[start:end]
                    
                    if len(feature_idxs) > 0:
                        cols = data[:, feature_idxs]
                        if func == 'max':
                            result[:, global_i] = np.max(cols, axis=1)
                        elif func == 'min':
                            result[:, global_i] = np.min(cols, axis=1)
                            
        return result

    def __len__(self):
        return len(self.hyperedges)
