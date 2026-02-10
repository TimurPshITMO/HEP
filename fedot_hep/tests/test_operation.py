import pytest
import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot_hep.web.operation import HypergraphFeatureConstructor
from fedot.core.operations.operation_parameters import OperationParameters

@pytest.fixture
def simple_input_data():
    features = np.array([[1, 2], [3, 4], [5, 6]])
    target = np.array([0, 1, 0])
    task = Task(TaskTypesEnum.classification)
    input_data = InputData(
        idx=np.arange(3),
        features=features,
        target=target,
        task=task,
        data_type=DataTypesEnum.table,
        features_names=['f0', 'f1']
    )
    return input_data

def test_operation_fit_transform(simple_input_data):
    # Initialize with params
    params = OperationParameters(hyperedges=[{'components': [0, 1], 'func': 'sum'}])
    operation = HypergraphFeatureConstructor(params)
    
    # Fit
    operation.fit(simple_input_data)
    assert operation.hypergraph is not None
    assert len(operation.hypergraph.hyperedges) == 1
    
    # Transform
    output = operation.transform(simple_input_data)
    
    # Check shape: 2 original + 1 new = 3 cols
    assert output.predict.shape == (3, 3)
    
    # Check value
    expected_new_col = simple_input_data.features[:, 0] + simple_input_data.features[:, 1]
    assert np.allclose(output.predict[:, 2], expected_new_col)

    
    # Check names
    assert len(output.features_names) == 3
    assert output.features_names[2] == 'sum(f0,f1)'

def test_persistence_via_params(simple_input_data):
    # Simulate saving/loading where params are preserved but object is re-instantiated
    params = OperationParameters(hyperedges=[{'components': [0, 0], 'func': 'product'}])
    
    operation = HypergraphFeatureConstructor(params)
    # Note: fit is NOT called, mimicking restore->transform flow in some cases, 
    # or just checking if transform handles initialization
    
    output = operation.transform(simple_input_data)
    
    # Should self-initialize from params
    assert output.predict.shape == (3, 3)
    expected = simple_input_data.features[:, 0] * simple_input_data.features[:, 0]
    assert np.allclose(output.predict[:, 2], expected)

def test_get_params_sync(simple_input_data):
    operation = HypergraphFeatureConstructor()
    operation.fit(simple_input_data)
    
    # Modifying hypergraph manually (simulating mutation)
    operation.hypergraph.add_hyperedge([1, 1], 'mean')
    
    # Check if get_params returns updated structure
    params = operation.get_params()
    assert 'hyperedges' in params._parameters
    edges = params._parameters['hyperedges']
    assert len(edges) == 1
    assert edges[0]['func'] == 'mean'
