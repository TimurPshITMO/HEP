"""Benchmark 3: OFAT (one-factor-at-a-time) hyperparameter sensitivity study.

Varies one HEP hyperparameter at a time while keeping the rest at their
defaults. Runs on the synthetic_regression dataset across multiple seeds.

Answers: which hyperparameter has the most impact on HEP's best fitness?
Note that due to fitness caching, doubling pop_size is cheaper than doubling
n_generations — this benchmark makes that trade-off empirically visible.
"""

from __future__ import annotations
import logging
import random as _random
from typing import List, Optional

import numpy as np
import pandas as pd

from benchmarks.config import DEFAULT_HEP_PARAMS, SEEDS
from benchmarks.datasets import DatasetInfo, get_dataset
from benchmarks.hep_wrapper import HEPTransformer
from benchmarks.profiler import ResourceProfiler

logger = logging.getLogger(__name__)

ABLATION_GRID = {
    'pop_size':          [10, 20, 40, 80],
    'n_generations':     [5, 10, 20, 40],
    'mut_rate':          [0.2, 0.4, 0.6, 0.8],
    'elitism_count':     [1, 2, 4],
    'available_functions': [
        ['sum', 'mean'],
        ['sum', 'mean', 'max', 'min'],
        ['sum', 'mean', 'max', 'min', 'std', 'prod'],
    ],
}


def run_one_ablation_trial(
    dataset: DatasetInfo,
    param_name: str,
    param_value,
    seed: int,
    base_params: Optional[dict] = None,
) -> dict:
    # Set seed at trial level for reproducible HEP evolution.
    np.random.seed(seed)
    _random.seed(seed)

    base_params = (base_params or DEFAULT_HEP_PARAMS).copy()
    base_params[param_name] = param_value

    hep = HEPTransformer(
        problem_type=dataset.problem_type,
        random_state=seed,
        **{k: v for k, v in base_params.items() if k != 'cv'},
        cv=base_params.get('cv', 3),
    )

    with ResourceProfiler() as prof:
        hep.fit(dataset.X, dataset.y)

    # Retrieve convergence info from tracker history
    from hep_engine import EvolutionTracker
    import json, os
    history_path = hep.history_path_
    best_per_gen, avg_per_gen = [], []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        for gen in history:
            fitnesses = [ind['fitness'] for ind in gen['population']]
            best_per_gen.append(max(fitnesses))
            avg_per_gen.append(float(np.mean(fitnesses)))

    return {
        'param_name':        param_name,
        'param_value':       str(param_value),
        'seed':              seed,
        'best_fitness':      hep.best_fitness_,
        'avg_fitness':       float(np.mean(avg_per_gen)) if avg_per_gen else float('nan'),
        'n_generations_run': len(best_per_gen),
        'fit_time_s':        prof.elapsed_s,
    }


def run_benchmark_3(
    dataset: Optional[DatasetInfo] = None,
    seeds: Optional[List[int]] = None,
    ablation_grid: Optional[dict] = None,
    base_params: Optional[dict] = None,
) -> pd.DataFrame:
    if dataset is None:
        dataset = get_dataset('synthetic_regression')
    if seeds is None:
        seeds = SEEDS
    if ablation_grid is None:
        ablation_grid = ABLATION_GRID

    rows = []
    total = sum(len(v) for v in ablation_grid.values()) * len(seeds)
    done = 0

    for param_name, values in ablation_grid.items():
        for param_value in values:
            for seed in seeds:
                done += 1
                logger.info(
                    f"[B3 {done}/{total}] {param_name}={param_value} | seed={seed}"
                )
                try:
                    row = run_one_ablation_trial(dataset, param_name, param_value, seed, base_params)
                    rows.append(row)
                except Exception as exc:
                    logger.error(f"  FAILED: {exc}")
                    rows.append({
                        'param_name': param_name,
                        'param_value': str(param_value),
                        'seed': seed,
                        'status': f'error: {exc}',
                    })

    return pd.DataFrame(rows)
