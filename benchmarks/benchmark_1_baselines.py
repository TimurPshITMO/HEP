"""Benchmark 1: Baseline ML models with raw features vs HEP-augmented features.

For each (dataset, model, seed) triple:
  - Fit model on raw X_train → score on X_test
  - Fit HEPTransformer on X_train → augment X_train and X_test → fit model → score
Report CV score, test score, delta, HEP fit time and peak memory.
"""

from __future__ import annotations
import logging
import random
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.config import DEFAULT_HEP_PARAMS, SEEDS
from benchmarks.datasets import DatasetInfo, get_all_datasets
from benchmarks.hep_wrapper import HEPTransformer
from benchmarks.profiler import ResourceProfiler

logger = logging.getLogger(__name__)


def _get_models(problem_type: str) -> dict:
    if problem_type == 'regression':
        return {
            'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearModel':      Ridge(alpha=1.0),
        }
    return {
        'RandomForest':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LinearModel':      LogisticRegression(max_iter=1000, random_state=42),
    }


def _test_score(model, X_test, y_test, problem_type: str) -> float:
    y_pred = model.predict(X_test)
    if problem_type == 'regression':
        return float(r2_score(y_test, y_pred))
    return float(f1_score(y_test, y_pred, average='macro'))


def run_one_trial(
    dataset: DatasetInfo,
    model_name: str,
    model_template,
    hep_params: dict,
    seed: int,
    cv: int = 3,
    test_size: float = 0.2,
) -> dict:
    # Set seed at trial level — ensures HEP evolution is reproducible without
    # polluting global state inside the transformer (see hep_wrapper.py).
    np.random.seed(seed)
    random.seed(seed)

    base = dict(
        dataset=dataset.name,
        model=model_name,
        seed=seed,
        problem_type=dataset.problem_type,
        n_raw_features=dataset.n_features,
    )
    scoring = 'r2' if dataset.problem_type == 'regression' else 'f1_macro'

    X_tr, X_te, y_tr, y_te = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed
    )

    # ---- WITHOUT HEP ----
    pipe_raw = Pipeline([('scaler', StandardScaler()), ('model', clone(model_template))])
    cv_raw = cross_val_score(pipe_raw, X_tr, y_tr, cv=cv, scoring=scoring)
    pipe_raw.fit(X_tr, y_tr)
    test_raw = _test_score(pipe_raw, X_te, y_te, dataset.problem_type)

    # ---- WITH HEP ----
    hep = HEPTransformer(
        problem_type=dataset.problem_type,
        random_state=seed,
        **hep_params,
    )
    # 1. Сначала честная кросс-валидация внутри Pipeline (исключает Target Leakage)
    pipe_hep_cv = Pipeline([('hep', hep), ('scaler', StandardScaler()), ('model', clone(model_template))])
    cv_hep = cross_val_score(pipe_hep_cv, X_tr, y_tr, cv=cv, scoring=scoring)

    # 2. Обучение на всём train для holdout-теста и замеров времени
    pipe_hep_test = Pipeline([('hep', hep), ('scaler', StandardScaler()), ('model', clone(model_template))])
    
    with ResourceProfiler() as prof:
        pipe_hep_test.fit(X_tr, y_tr)
        
    test_hep = _test_score(pipe_hep_test, X_te, y_te, dataset.problem_type)
    fitted_hep = pipe_hep_test.named_steps['hep']

    return {
        **base,
        'n_hep_features':    fitted_hep.n_hep_features_,
        'cv_score_raw':      float(np.mean(cv_raw)),
        'cv_score_hep':      float(np.mean(cv_hep)),
        'test_score_raw':    test_raw,
        'test_score_hep':    test_hep,
        'delta_test_score':  test_hep - test_raw,
        'hep_fit_time_s':    prof.elapsed_s,
        'hep_peak_mem_mb':   prof.peak_mem_mb,
        'best_genome_fitness': fitted_hep.best_fitness_,
    }


def run_benchmark_1(
    datasets: Optional[List[DatasetInfo]] = None,
    hep_params: Optional[dict] = None,
    seeds: Optional[List[int]] = None,
    output_dir: str = 'benchmarks/results',
) -> pd.DataFrame:
    if datasets is None:
        datasets = get_all_datasets()
    if hep_params is None:
        hep_params = DEFAULT_HEP_PARAMS.copy()
    if seeds is None:
        seeds = SEEDS

    rows = []
    total = sum(len(_get_models(d.problem_type)) for d in datasets) * len(seeds)
    done = 0

    for dataset in datasets:
        models = _get_models(dataset.problem_type)
        for model_name, model_tmpl in models.items():
            for seed in seeds:
                done += 1
                logger.info(f"[B1 {done}/{total}] {dataset.name} | {model_name} | seed={seed}")
                try:
                    row = run_one_trial(dataset, model_name, model_tmpl, hep_params, seed)
                    rows.append(row)
                except Exception as exc:
                    logger.error(f"  FAILED: {exc}")
                    rows.append({
                        'dataset': dataset.name, 'model': model_name, 'seed': seed,
                        'status': f'error: {exc}',
                    })

    return pd.DataFrame(rows)
