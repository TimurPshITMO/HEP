from copy import deepcopy
from typing import Any, List, Tuple
import random

from golem.core.optimisers.graph import OptGraph
from fedot_hep.evolution.mutation import get_hep_node

def hypergraph_crossover(graph_first: OptGraph, graph_second: OptGraph, **kwargs) -> Any:
    """
    Performs crossover between two hypergraph-based pipelines.
    Exchanges hyperedges between the HypergraphFeatureConstructor nodes.
    
    Returns:
        List[OptGraph]: A list containing the two new offspring graphs.
    """
    # Create copies for offspring
    child_1 = deepcopy(graph_first)
    child_2 = deepcopy(graph_second)
    
    hep_node_1 = get_hep_node(child_1)
    hep_node_2 = get_hep_node(child_2)
    
    # If either graph lacks a HEP node, crossover is not possible (or just returns copies)
    if not hep_node_1 or not hep_node_2:
        return [child_1, child_2]
        
    edges_1 = hep_node_1.parameters.get('hyperedges', [])
    edges_2 = hep_node_2.parameters.get('hyperedges', [])
    
    # If one of them has no edges, we can still swap (give some to empty)
    if not edges_1 and not edges_2:
        return [child_1, child_2]
        
    # Perform single-point crossover on the lists of edges
    # We treat the list of edges as the genome
    
    min_len = min(len(edges_1), len(edges_2))
    if min_len < 2:
        # If too small for cut point, just swap everything (or do nothing)
        # Let's swap content completely? No, that's just swapping individuals.
        # Let's try to mix if one is empty and other is not.
        if len(edges_1) > 0 and len(edges_2) == 0:
             # Split edges_1 into two
             split_idx = len(edges_1) // 2
             new_edges_1 = edges_1[:split_idx]
             new_edges_2 = edges_1[split_idx:]
        elif len(edges_2) > 0 and len(edges_1) == 0:
             split_idx = len(edges_2) // 2
             new_edges_1 = edges_2[:split_idx]
             new_edges_2 = edges_2[split_idx:]
        else:
             # Both have 0 or 1 edge. Reference implementation:
             # If both have 1, we can swap them? That's just swapping parents.
             # Maybe verify probability?
             # For now, return as is.
             return [child_1, child_2]
    else:
        # Standard crossover
        # Choose cut point. 
        # Note: lists might be different lengths. 
        # Crossover point k1 for parent 1, k2 for parent 2? 
        # Usually single point implies arrays of same length. 
        # For variable length, we can pick a random cut point in each.
        
        cx_point_1 = random.randint(1, len(edges_1) - 1) if len(edges_1) > 1 else 0
        cx_point_2 = random.randint(1, len(edges_2) - 1) if len(edges_2) > 1 else 0
        
        # Offspring 1 gets: P1[:c1] + P2[c2:]
        # Offspring 2 gets: P2[:c2] + P1[c1:]
        
        new_edges_1 = edges_1[:cx_point_1] + edges_2[cx_point_2:]
        new_edges_2 = edges_2[:cx_point_2] + edges_1[cx_point_1:]
        
    # Update parameters
    hep_node_1.parameters.update(hyperedges=new_edges_1)
    hep_node_2.parameters.update(hyperedges=new_edges_2)
    
    return [child_1, child_2]
