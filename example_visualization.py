import numpy as np
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_hep.api.registration import register_hep, patch_fedot_for_hep
from fedot_hep.visualization.hep_explainer import HEPExplainer

def run_visualization_demo():
    # 1. Register and Patch
    register_hep()
    patch_fedot_for_hep()

    # 2. Prepare complex interaction data
    n_samples = 400
    n_features = 12
    features = np.random.rand(n_samples, n_features)
    # Target depends on multiple interactions
    # y = x0*x1 + x2*x3*x4 + x5/x6 + x9*x10 + x11
    target = (features[:, 0] * features[:, 1] + 
              features[:, 2] * features[:, 3] * features[:, 4] + 
              features[:, 5] / (features[:, 6] + 0.1) + 
              features[:, 9] * features[:, 10] + 
              features[:, 11] +
              np.random.normal(0, 0.05, n_samples)).reshape(-1, 1)
    
    input_data = InputData(
        idx=np.arange(n_samples),
        features=features,
        target=target,
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table
    )

    # 3. Initial Assumption
    # Pre-setting num_vertices to help mutations in the first generation
    initial_pipeline = PipelineBuilder() \
        .add_node('hypergraph_feature_constructor', params={'num_vertices': n_features}) \
        .add_node('rfr') \
        .build()

    # 4. Run shorter evolution with rich logging
    print("Running evolution for high-quality visualization data (approx 10 min)...")
    model = Fedot(
        problem='regression',
        timeout=10, 
        preset='fast_train',
        initial_assumption=initial_pipeline,
        n_jobs=1,
        with_tuning=False 
    )

    model.fit(input_data)
    
    # 5. Extract History and Explainer
    history = model.history
    explainer = HEPExplainer(output_dir='research_results')

    print("\nGenerating Research Visualizations...")
    
    # A. Survival Plot (Persistence)
    explainer.plot_edge_survival(history)
    
    # B. Feature Interaction Importance
    feature_names = [f"Sensor_{i}" for i in range(n_features)]
    explainer.plot_feature_importance(history, feature_names=feature_names)
    
    # C. Best Hypergraph Structure
    best_pipeline = model.current_pipeline
    hep_node = None
    for node in best_pipeline.nodes:
        if 'hypergraph' in str(node.name):
            hep_node = node
            break
            
    if hep_node:
        explainer.plot_hypergraph(hep_node, title="Final Evolved Feature Interactions")
    else:
        print("No HEP node in final pipeline, skipping structure plot.")

    print("\nVisualization demo complete. Check the 'research_results' directory.")

if __name__ == '__main__':
    run_visualization_demo()
