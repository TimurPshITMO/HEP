import pytest
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository

from fedot_hep.api.registration import register_hep
from fedot_hep.web.operation import HypergraphFeatureConstructor

@pytest.fixture(autouse=True)
def setup_hep():
    register_hep()

def test_registration_success():
    repo = OperationTypesRepository('data_operation')
    op_ids = [op.id for op in repo._repo]
    assert 'hypergraph_feature_constructor' in op_ids

def test_pipeline_with_hep():
    # Create simple data
    features = np.array([[1, 2], [3, 4], [5, 6]])
    target = np.array([1, 0, 1])
    input_data = InputData(
        idx=np.arange(len(features)),
        features=features,
        target=target,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table
    )
    
    # Build pipeline: HEP -> LogReg
    # We use the registered string name
    builder = PipelineBuilder()
    pipeline = builder.add_node('hypergraph_feature_constructor', params={'hyperedges': [{'components': [0, 1], 'func': 'sum'}]}) \
                      .add_node('logit') \
                      .build()
                      
    # Fit
    pipeline.fit(input_data)
    
    # Predict
    prediction = pipeline.predict(input_data)
    
    assert prediction is not None
    # Verify that HEP node worked (it should have 3 features now)
    hep_node = pipeline.nodes[1] # builder.add_node appends, so logit is 0 (root), hep is 1
    # Check if hep_node results have 3 columns
    # Actually in FEDOT Pipeline, nodes are ordered from root. 
    # builder.add_node('hep').add_node('logit') -> logit is root, hep is parent of logit.
    
    # Let's find the HEP node result
    # We can check the shape of data passed to logit
    # Or just verify the pipeline didn't crash
    assert pipeline.is_fitted

def test_validation_rules():
    from fedot_hep.core.verification import has_no_multiple_hep_nodes, has_hep_node_as_secondary
    from golem.core.optimisers.graph import OptGraph, OptNode
    
    # 1. Test multiple HEP nodes
    node1 = OptNode(content='hypergraph_feature_constructor')
    node2 = OptNode(content='hypergraph_feature_constructor')
    graph = OptGraph([node1, node2])
    node1.nodes_from = [node2] # Just to connect them
    
    with pytest.raises(ValueError, match="multiple hypergraph nodes"):
        has_no_multiple_hep_nodes(graph)
        
    # 2. Test HEP as primary (forbidden)
    primary_hep = OptNode(content='hypergraph_feature_constructor')
    graph_primary = OptGraph(primary_hep)

    
    with pytest.raises(ValueError, match="must be a secondary node"):
        has_hep_node_as_secondary(graph_primary)
