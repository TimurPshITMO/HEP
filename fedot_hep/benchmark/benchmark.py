import time
import pandas as pd
import numpy as np
from typing import Dict, Any

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot_hep.api.registration import register_hep, patch_fedot_for_hep

class ParallelBenchmarkRunner:
    """
    Runner for comparative benchmarks between Standard FEDOT and HEP-FEDOT.
    """
    def __init__(self, timeout_mins: float = 10, presets: str = 'fast_train'):
        self.timeout = timeout_mins
        self.presets = presets
        register_hep()
        patch_fedot_for_hep()

    def run_benchmark(self, train_data: InputData, test_data: InputData) -> pd.DataFrame:
        results = []
        
        # 1. Standard FEDOT
        print("--- Running Standard FEDOT Baseline ---")
        fedot_std = Fedot(problem='regression', timeout=self.timeout, preset=self.presets)
        start_time = time.time()
        fedot_std.fit(train_data)
        std_fit_time = time.time() - start_time
        
        std_pred = fedot_std.predict(test_data)
        std_rmse = self._calc_rmse(test_data.target, std_pred)
        
        results.append({
            'Config': 'Standard FEDOT',
            'RMSE': std_rmse,
            'Fit Time (s)': std_fit_time,
            'Constructed Features': 0
        })

        # 2. HEP-FEDOT
        print("\n--- Running HEP-FEDOT ---")
        # In this implementation, we ensure HEP is present
        # Custom initialization or requirements would go here
        fedot_hep = Fedot(problem='regression', timeout=self.timeout, preset=self.presets)
        
        start_time = time.time()
        fedot_hep.fit(train_data)
        hep_fit_time = time.time() - start_time
        
        hep_pred = fedot_hep.predict(test_data)
        hep_rmse = self._calc_rmse(test_data.target, hep_pred)
        
        # Count hyperedges in the best model
        hep_node_count = 0
        for node in fedot_hep.current_pipeline.nodes:
            if 'hypergraph' in str(node.name):
                # Sum of edges for all hep nodes
                hep_node_count += len(node.parameters.get('hyperedges', []))

        results.append({
            'Config': 'HEP-FEDOT',
            'RMSE': hep_rmse,
            'Fit Time (s)': hep_fit_time,
            'Constructed Features': hep_node_count
        })

        return pd.DataFrame(results)

    def _calc_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
