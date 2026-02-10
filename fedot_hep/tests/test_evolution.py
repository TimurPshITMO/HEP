import pytest
from golem.core.optimisers.graph import OptGraph, OptNode
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_hep.web.operation import HypergraphFeatureConstructor
from fedot_hep.evolution.mutation import hypergraph_mutation, get_hep_node

@pytest.fixture
def hep_graph():
    # Create a simple graph with one HEP node
    params = OperationParameters(
        hyperedges=[{'components': [0, 1], 'func': 'sum'}],
        num_vertices=10
    )
    hep_node = OptNode(content={'name': 'hypergraph_feature_constructor', 'params': params})
    # Manually attach operation instance to mimic FEDOT node initialization
    # In real FEDOT, content['name'] is used to look up operation, 
    # but here we can inject the instance if needed, or get_hep_node checks instance
    op = HypergraphFeatureConstructor(params)
    hep_node.operation = op
    
    graph = OptGraph(hep_node)
    return graph

def test_get_hep_node(hep_graph):
    node = get_hep_node(hep_graph)
    assert node is not None
    assert isinstance(node.operation, HypergraphFeatureConstructor)

def test_mutation_returns_new_graph(hep_graph):
    old_params = hep_graph.nodes[0].parameters.to_dict()
    new_graph = hypergraph_mutation(hep_graph)
    
    assert new_graph is not hep_graph
    assert new_graph.nodes[0] is not hep_graph.nodes[0]

def test_add_edge_mutation():
    # Setup params manually
    params = OperationParameters(hyperedges=[], num_vertices=5)
    node = OptNode(content={'name': 'hep', 'params': params})
    node.operation = HypergraphFeatureConstructor(params)
    graph = OptGraph(node)
    
    # Force add mutation by mocking random? 
    # Or just run mutation multiple times until change happens?
    # Let's call the internal _add_hyperedge_mutation directly for unit testing specific logic
    from fedot_hep.evolution.mutation import _add_hyperedge_mutation
    
    _add_hyperedge_mutation(node)
    
    edges = node.parameters.get('hyperedges')
    assert len(edges) == 1
    assert 'components' in edges[0]
    assert 'func' in edges[0]
    # Check degree constraints
    assert len(edges[0]['components']) >= 2

def test_remove_edge_mutation():
    params = OperationParameters(
        hyperedges=[{'components': [0, 1], 'func': 'sum'}],
        num_vertices=5
    )
    node = OptNode(content={'name': 'hep', 'params': params})
    node.operation = HypergraphFeatureConstructor(params)
    
    from fedot_hep.evolution.mutation import _remove_hyperedge_mutation
    _remove_hyperedge_mutation(node)
    
    edges = node.parameters.get('hyperedges')
    assert len(edges) == 0

def test_modify_edge_mutation():
    edge = {'components': [0, 1], 'func': 'sum'}
    params = OperationParameters(
        hyperedges=[edge],
        num_vertices=5
    )
    node = OptNode(content={'name': 'hep', 'params': params})
    node.operation = HypergraphFeatureConstructor(params)
    
    from fedot_hep.evolution.mutation import _modify_hyperedge_mutation
    
    # It randomizes change, so let's run it and check if *something* changed
    initial_snapshot = str(edge)
    
    # Try a few times to ensure change (randomness might pick same func or something invisible)
    changed = False
    for _ in range(10):
        _modify_hyperedge_mutation(node)
        current_edge = node.parameters.get('hyperedges')[0]
        if str(current_edge) != initial_snapshot:
            changed = True
            break
            
    assert changed

def test_crossover():
    from fedot_hep.evolution.crossover import hypergraph_crossover
    
    # Parent 1: [sum(0,1), product(2,3)]
    params1 = OperationParameters(
        hyperedges=[
            {'components': [0, 1], 'func': 'sum'},
            {'components': [2, 3], 'func': 'product'}
        ],
        num_vertices=10
    )
    node1 = OptNode(content={'name': 'hep', 'params': params1})
    node1.operation = HypergraphFeatureConstructor(params1)
    graph1 = OptGraph(node1)
    
    # Parent 2: [mean(4,5), max(6,7)]
    params2 = OperationParameters(
        hyperedges=[
            {'components': [4, 5], 'func': 'mean'},
            {'components': [6, 7], 'func': 'max'}
        ],
        num_vertices=10
    )
    node2 = OptNode(content={'name': 'hep', 'params': params2})
    node2.operation = HypergraphFeatureConstructor(params2)
    graph2 = OptGraph(node2)
    
    # Perform crossover
    # FEDOT mutation returns NEW graph with updated params
    # We should ensure get_hep_node works on it
    
    offspring = hypergraph_crossover(graph1, graph2)
    
    assert len(offspring) == 2
    child1, child2 = offspring
    
    edges1 = child1.nodes[0].parameters.get('hyperedges')
    edges2 = child2.nodes[0].parameters.get('hyperedges')
    
    # Check simple swap logic
    total_edges_p1 = 2
    total_edges_p2 = 2
    total_offspring  = len(edges1) + len(edges2)
    
    assert total_offspring == total_edges_p1 + total_edges_p2
    
def test_crossover_empty_parent():
    from fedot_hep.evolution.crossover import hypergraph_crossover
    
    # Parent 1 has 2 edges, Parent 2 has 0
    params1 = OperationParameters(
        hyperedges=[
            {'components': [0, 1], 'func': 'sum'},
            {'components': [2, 3], 'func': 'product'}
        ],
        num_vertices=10
    )
    node1 = OptNode(content={'name': 'hep', 'params': params1})
    node1.operation = HypergraphFeatureConstructor(params1)
    graph1 = OptGraph(node1)
    
    params2 = OperationParameters(hyperedges=[], num_vertices=10)
    node2 = OptNode(content={'name': 'hep', 'params': params2})
    node2.operation = HypergraphFeatureConstructor(params2)
    graph2 = OptGraph(node2)
    
    offspring = hypergraph_crossover(graph1, graph2)
    
    edges1 = offspring[0].nodes[0].parameters.get('hyperedges')
    edges2 = offspring[1].nodes[0].parameters.get('hyperedges')
    
    # Logic is to split Parent1's edges between children if P2 is empty
    assert len(edges1) == 1
    assert len(edges2) == 1
