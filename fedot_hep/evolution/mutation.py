import random
from copy import deepcopy
from typing import List, Callable, Any, Optional

from golem.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot_hep.web.operation import HypergraphFeatureConstructor

# Define available aggregation functions
AGGREGATION_FUNCTIONS = ['sum', 'product', 'mean', 'max', 'min']

def get_hep_node(graph: OptGraph) -> Optional[OptNode]:
    """Finds the HypergraphFeatureConstructor node in the graph."""
    for node in graph.nodes:
        # Identify by name (registered operation string)
        node_name = str(node.name)
        if 'hypergraph' in node_name:
            return node
    return None

def hypergraph_mutation(graph: OptGraph, **kwargs) -> Any:
    """
    Applies a random mutation to the hypergraph structure within the pipeline.
    This function is compliant with FEDOT's mutation interface.
    """
    new_graph = deepcopy(graph)
    hep_node = get_hep_node(new_graph)
    if not hep_node:
        return new_graph
    
    params = hep_node.parameters
    if 'hyperedges' not in params or not params['hyperedges']:
        # If empty, always start with an addition
        _add_hyperedge_mutation(hep_node)
        return new_graph

    # Select mutation type
    mutation_type = random.choice([
        _add_hyperedge_mutation,
        _remove_hyperedge_mutation,
        _modify_hyperedge_mutation
    ])
    
    mutation_type(hep_node)
    
    return new_graph

def _add_hyperedge_mutation(node: OptNode):
    """Adds a new random hyperedge."""
    params = node.parameters
    hyperedges = params.get('hyperedges', [])
    
    # We need to know the number of features to generate valid indices.
    # This is tricky: params usually don't know input shape until fit.
    # But we can try to guess or use a metadata field if we stored it.
    # OR we can assume a large enough number and rely on validation/pruning?
    # Better: The node should store `num_vertices` in params after first fit.
    
    num_vertices = params.get('num_vertices')
    if not num_vertices:
        # If we don't know the number of features, we can't safely add an edge 
        # relying on indices. 
        # Fallback: Do nothing or try to preserve existing max index.
        if hyperedges:
            # Estimate from existing edges
            max_idx = 0
            for edge in hyperedges:
                if edge['components']:
                    max_idx = max(max_idx, max(edge['components']))
            num_vertices = max_idx + 1 # At least this many
        else:
            # Empty graph and no metadata. Can't add.
            return

    # Create random edge
    # Random degree between 2 and 5 (heuristic)
    degree = random.randint(2, min(5, num_vertices))
    components = random.sample(range(num_vertices), degree)
    func = random.choice(AGGREGATION_FUNCTIONS)
    
    new_edge = {'components': components, 'func': func}
    hyperedges.append(new_edge)
    
    # Update params
    params.update(hyperedges=hyperedges)

def _remove_hyperedge_mutation(node: OptNode):
    """Removes a random hyperedge."""
    params = node.parameters
    hyperedges = params.get('hyperedges', [])
    
    if not hyperedges:
        return

    # Randomly remove one
    idx = random.randrange(len(hyperedges))
    hyperedges.pop(idx)
    
    params.update(hyperedges=hyperedges)

def _modify_hyperedge_mutation(node: OptNode):
    """Modifies an existing hyperedge (change function or components)."""
    params = node.parameters
    hyperedges = params.get('hyperedges', [])
    
    if not hyperedges:
        return

    edge_idx = random.randrange(len(hyperedges))
    edge = hyperedges[edge_idx]
    
    # Coin flip: change function or components
    if random.random() < 0.5:
        # Change function
        new_func = random.choice([f for f in AGGREGATION_FUNCTIONS if f != edge['func']])
        edge['func'] = new_func
    else:
        # Change components
        num_vertices = params.get('num_vertices')
        if not num_vertices:
             # Try to infer
             max_idx = 0
             for e in hyperedges:
                 if e['components']:
                     max_idx = max(max_idx, max(e['components']))
             num_vertices = max_idx + 1

        action = random.choice(['add', 'remove', 'replace'])
        components = edge['components']
        
        if action == 'add' and len(components) < num_vertices:
            # Add a node not currently in the edge
            available = list(set(range(num_vertices)) - set(components))
            if available:
                components.append(random.choice(available))
                
        elif action == 'remove' and len(components) > 2:
            # Remove a random node, keeping at least 2
            components.pop(random.randrange(len(components)))
            
        elif action == 'replace':
            # Swap one node
            idx_to_replace = random.randrange(len(components))
            available = list(set(range(num_vertices)) - set(components))
            if available:
                components[idx_to_replace] = random.choice(available)
    
    # Parameter update is by reference for lists usually, but let's be explicit
    hyperedges[edge_idx] = edge
    params.update(hyperedges=hyperedges)
