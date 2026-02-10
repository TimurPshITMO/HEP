from golem.core.optimisers.graph import OptGraph
from fedot_hep.evolution.mutation import get_hep_node

def has_no_multiple_hep_nodes(graph: OptGraph):
    """Verifies that the graph contains at most one HEP node."""
    hep_nodes = []
    for node in graph.nodes:
        if 'hypergraph' in str(node.name):
            hep_nodes.append(node)
    
    if len(hep_nodes) > 1:
        raise ValueError("Pipeline has multiple hypergraph nodes, which is forbidden in strict benchmarking mode.")
    return True

def has_hep_node_as_primary(graph: OptGraph):
    """Verifies that the HEP node is a primary node (leaf/data source)."""
    hep_node = get_hep_node(graph)
    if not hep_node:
        return True # Rule doesn't apply
        
    if hep_node.nodes_from:
        raise ValueError("Hypergraph node must be a primary node (operate on raw features).")
    return True

def has_at_least_one_hep_node(graph: OptGraph):
    """Strict variant: ensures the HEP node is present and wasn't pruned."""
    if not get_hep_node(graph):
        raise ValueError("Pipeline must contain at least one hypergraph node (Strict HEP mode).")
    return True
