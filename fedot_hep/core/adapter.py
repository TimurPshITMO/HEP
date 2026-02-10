from fedot.core.pipelines.adapters import PipelineAdapter
from fedot_hep.web.operation import HypergraphFeatureConstructor

class HypergraphAdapter(PipelineAdapter):
    """
    Adapter for Hypergraph-Evolved Pipelines.
    Ensures that HypergraphFeatureConstructor nodes are correctly handled
    during OptGraph <-> Pipeline conversion.
    
    Since we store hypergraph structure in OperationParameters ('hyperedges'),
    the standard PipelineAdapter logic (which copies 'params') is sufficient 
    for preserving the genome.
    
    This class is reserved for future extensions where we might need 
    to handle complex constraints or non-parameter state validation.
    """
    def __init__(self):
        super().__init__()
