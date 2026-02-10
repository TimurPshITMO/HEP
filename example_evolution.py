import numpy as np
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_hep.api.registration import register_hep, patch_fedot_for_hep

def run_hep_evolution_example():
    # 1. Register and Patch FEDOT
    register_hep()
    patch_fedot_for_hep()
    
    # Integrity Check
    from fedot.api.api_utils.api_params_repository import ApiParamsRepository
    print(f"DEBUG: ApiParamsRepository._get_default_mutations is {ApiParamsRepository._get_default_mutations}")

    # 2. Prepare synthetic data
    # Interaction detection problem
    n_samples = 400
    features = np.random.rand(n_samples, 5)
    # Target heavily depends on interaction: x0*x1 + x2*x0
    target = (features[:, 0] * features[:, 1] + features[:, 2] * features[:, 0] + np.random.normal(0, 0.05, n_samples)).reshape(-1, 1)
    
    input_data = InputData(
        idx=np.arange(n_samples),
        features=features,
        target=target,
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table
    )

    # 3. Build Initial Assumption with HEP node
    initial_pipeline = PipelineBuilder() \
        .add_node('hypergraph_feature_constructor') \
        .add_node('ridge') \
        .build()

    # 4. Initialize Fedot
    # We use a short timeout and include our initial assumption
    model = Fedot(
        problem='regression',
        timeout=2, 
        preset='fast_train',
        initial_assumption=initial_pipeline,
        n_jobs=1 # Single job to avoid potential multiprocessing issues with monkey-patching
    )

    print("Starting HEP evolution (PatchMode)...")
    pipeline = model.fit(features=input_data.features, target=input_data.target)
    
    print("\nFinal Pipeline Structure:")
    print(pipeline.descriptive_id)
    
    # Check if hypergraph node evolved
    hep_node = None
    for node in pipeline.nodes:
        if 'hypergraph' in str(node.name):
            hep_node = node
            break
            
    if hep_node:
        print("\n=== Evolved Hypergraph Details ===")
        edges = hep_node.parameters.get('hyperedges', [])
        print(f"Number of edges: {len(edges)}")
        for i, edge in enumerate(edges[:5]):
             print(f" Edge {i}: {edge}")
    else:
        print("HEP node was pruned.")

    prediction = model.predict(features=features)
    print("\nEvolution complete.")

if __name__ == '__main__':
    run_hep_evolution_example()

