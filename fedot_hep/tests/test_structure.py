import pytest
import numpy as np
import scipy.sparse as sp
from fedot_hep.core.structure import HighPerformanceHypergraph

@pytest.fixture
def sample_data():
    # 5 samples, 4 features
    return np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 2.0, 2.0, 2.0],
        [0.0, 1.0, 0.0, 1.0],
        [10.0, 5.0, 2.0, 1.0],
        [-1.0, 0.0, 1.0, 0.0]
    ])

def test_initialization():
    graph = HighPerformanceHypergraph(num_vertices=10)
    assert graph.num_vertices == 10
    assert len(graph.hyperedges) == 0

def test_add_hyperedge():
    graph = HighPerformanceHypergraph(num_vertices=5)
    graph.add_hyperedge([0, 1], 'sum')
    assert len(graph.hyperedges) == 1
    assert graph.hyperedges[0]['func'] == 'sum'
    assert graph.hyperedges[0]['components'] == [0, 1]

def test_transform_sum(sample_data):
    graph = HighPerformanceHypergraph(num_vertices=4)
    # Sum of col 0 and 1
    graph.add_hyperedge([0, 1], 'sum')
    
    res = graph.transform(sample_data)
    
    assert res.shape == (5, 1)
    expected = sample_data[:, 0] + sample_data[:, 1]
    assert np.allclose(res[:, 0], expected)

def test_transform_product(sample_data):
    graph = HighPerformanceHypergraph(num_vertices=4)
    # Product of col 2 and 3
    graph.add_hyperedge([2, 3], 'product')
    
    res = graph.transform(sample_data)
    
    expected = sample_data[:, 2] * sample_data[:, 3]
    assert np.allclose(res[:, 0], expected)

def test_transform_mean(sample_data):
    graph = HighPerformanceHypergraph(num_vertices=4)
    # Mean of col 0, 1, 2
    graph.add_hyperedge([0, 1, 2], 'mean')
    
    res = graph.transform(sample_data)
    
    expected = np.mean(sample_data[:, [0, 1, 2]], axis=1)
    assert np.allclose(res[:, 0], expected)

def test_multiple_edges(sample_data):
    graph = HighPerformanceHypergraph(num_vertices=4)
    graph.add_hyperedge([0, 1], 'sum')      # Edge 0
    graph.add_hyperedge([2, 3], 'product')  # Edge 1
    graph.add_hyperedge([0, 3], 'max')      # Edge 2
    
    res = graph.transform(sample_data)
    
    assert res.shape == (5, 3)
    assert np.allclose(res[:, 0], sample_data[:, 0] + sample_data[:, 1])
    assert np.allclose(res[:, 1], sample_data[:, 2] * sample_data[:, 3])
    assert np.allclose(res[:, 2], np.maximum(sample_data[:, 0], sample_data[:, 3]))

def test_sparse_input():
    # Construct a sparse matrix
    data = sp.csr_matrix([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0]
    ])
    
    graph = HighPerformanceHypergraph(num_vertices=3)
    graph.add_hyperedge([0, 1], 'sum')
    
    res = graph.transform(data)
    
    assert isinstance(res, np.ndarray) # Should return dense array according to current implementation
    expected = np.array([1.0 + 0.0, 0.0 + 3.0])
    assert np.allclose(res[:, 0], expected)

def test_invalid_indices():
    graph = HighPerformanceHypergraph(num_vertices=3)
    # 5 is out of bounds
    graph.add_hyperedge([0, 5], 'sum')
    
    # Should probably ignore out of bound index or raise error. 
    # Current implementation ignores it in _build_matrices check.
    
    data = np.ones((2, 3))
    res = graph.transform(data)
    
    # Effectively only col 0 is summed
    assert np.allclose(res[:, 0], data[:, 0])
