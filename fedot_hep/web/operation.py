from typing import Optional
import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_hep.core.structure import HighPerformanceHypergraph

class HypergraphFeatureConstructor(DataOperationImplementation):
    """
    FEDOT DataOperation implementation for Hypergraph-Evolved Pipelines (HEP).
    Wraps HighPerformanceHypergraph to generate features within the pipeline.
    """
    
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.hypergraph: Optional[HighPerformanceHypergraph] = None
        
    def fit(self, input_data: InputData):
        """
        Initializes the hypergraph structure.
        In the evolutionary loop, structure comes from params.
        If params are empty, we might initialize a default or random structure.
        """
        num_features = input_data.features.shape[1]
        self.hypergraph = HighPerformanceHypergraph(num_vertices=num_features)
        
        # Store num_vertices in params so mutations can know the feature space size
        self.params.update(num_vertices=num_features)
        
        # Hydrate hypergraph from parameters if they exist
        # Params: {'hyperedges': [{'components': [...], 'func': '...'}, ...]}
        hyperedges = self.params.get('hyperedges')
        if hyperedges:
            for edge in hyperedges:
                self.hypergraph.add_hyperedge(edge['components'], edge['func'])
        
        return self.hypergraph

    def transform(self, input_data: InputData) -> OutputData:
        """
        Generates new features and appends them to the input.
        """
        if not self.hypergraph:
            # If transform called without fit (persistence case?), try to restore
            num_features = input_data.features.shape[1]
            self.hypergraph = HighPerformanceHypergraph(num_vertices=num_features)
            hyperedges = self.params.get('hyperedges', [])
            for edge in hyperedges:
                self.hypergraph.add_hyperedge(edge['components'], edge['func'])

        # Generate new features
        # self.hypergraph.transform returns (N, E)
        new_features = self.hypergraph.transform(input_data.features)
        
        # Concatenate with original features
        # [Original | New]
        concatenated_features = np.hstack((input_data.features, new_features))
        
        # Update feature names
        new_names = []
        if input_data.features_names:
            base_names = input_data.features_names
            new_names.extend(base_names)
            for i, edge in enumerate(self.hypergraph.hyperedges):
                func = edge['func']
                # components are indices
                comp_names = [str(base_names[c]) if c < len(base_names) else f"f{c}" 
                              for c in edge['components']]
                name = f"{func}({','.join(comp_names)})"
                new_names.append(name)
        
        output_data = self._convert_to_output(input_data, concatenated_features)
        if new_names:
            output_data.features_names = new_names
            
        return output_data

    def get_params(self) -> OperationParameters:
        """
        Syncs internal hypergraph state back to params before saving/copying.
        """
        if self.hypergraph:
            self.params.update(hyperedges=self.hypergraph.hyperedges)
        return self.params
