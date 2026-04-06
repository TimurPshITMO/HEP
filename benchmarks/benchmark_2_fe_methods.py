"""Benchmark 2: HEP vs other feature engineering methods.

Fixed downstream model: RandomForest (regression or classification).
Each FE method is a callable: (X_train, y_train, X_test) -> (X_tr_t, X_te_t, n_features).
Results include test score, CV score, fit time, peak memory per (dataset, method, seed).
"""

from __future__ import annotations
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from benchmarks.config import DEFAULT_HEP_PARAMS, SEEDS
from benchmarks.datasets import DatasetInfo, get_all_datasets
from benchmarks.hep_wrapper import HEPTransformer
from benchmarks.profiler import ResourceProfiler

logger = logging.getLogger(__name__)

# Each method returns (X_train_t, X_test_t, n_features, metadata_dict)
# metadata_dict is optional (may be {}); used to pass extras like history_path.
MethodFn = Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, int, dict]]


def _get_rf(problem_type: str):
    if problem_type == 'regression':
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def _test_score(model, X_te, y_te, problem_type: str) -> float:
    y_pred = model.predict(X_te)
    if problem_type == 'regression':
        return float(r2_score(y_te, y_pred))
    return float(f1_score(y_te, y_pred, average='macro'))


def build_method_registry(dataset: DatasetInfo, hep_params: dict, seed: int) -> Dict[str, MethodFn]:
    """Return ordered dict of FE method callables for a given dataset."""
    n_features = dataset.n_features
    problem_type = dataset.problem_type
    methods: Dict[str, MethodFn] = {}

    # 1. No feature engineering
    def no_fe(X_tr, y_tr, X_te):
        return X_tr, X_te, X_tr.shape[1], {}
    methods['NoFE'] = no_fe

    # 2. PolynomialFeatures degree=2
    def poly2_clean(X_tr, y_tr, X_te):
        pf = PolynomialFeatures(degree=2, include_bias=False)
        Xtr_t = pf.fit_transform(X_tr)
        Xte_t = pf.transform(X_te)
        return Xtr_t, Xte_t, Xtr_t.shape[1], {}
    methods['Poly2'] = poly2_clean

    # 3. PolynomialFeatures degree=2 interaction_only
    def poly2_interact(X_tr, y_tr, X_te):
        pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        Xtr_t = pf.fit_transform(X_tr)
        Xte_t = pf.transform(X_te)
        return Xtr_t, Xte_t, Xtr_t.shape[1], {}
    methods['Poly2Interact'] = poly2_interact

    # 4. HEP — returns history_path in metadata for convergence curve
    def hep_method(X_tr, y_tr, X_te):
        hep = HEPTransformer(
            problem_type=problem_type,
            random_state=seed,
            inner_model=_get_rf(problem_type),  # match downstream RF(100)
            **hep_params,
        )
        hep.fit(X_tr, y_tr)
        Xtr_t = hep.transform(X_tr)
        Xte_t = hep.transform(X_te)
        return Xtr_t, Xte_t, hep.n_hep_features_, {'history_path': hep.history_path_}
    methods['HEP'] = hep_method

    # 6. FeatureTools DFS (optional)
    try:
        import featuretools as ft
        def featuretools_dfs(X_tr, y_tr, X_te):
            import pandas as _pd
            col_names = [f'f{i}' for i in range(X_tr.shape[1])]

            def _build_fm(X):
                df = _pd.DataFrame(X, columns=col_names)
                df['id'] = range(len(df))
                es = ft.EntitySet(id='bench')
                es.add_dataframe(dataframe_name='data', dataframe=df, index='id')
                fm, _ = ft.dfs(entityset=es, target_dataframe_name='data',
                               max_depth=1, verbose=False)
                return fm.values

            Xtr_t = _build_fm(X_tr)
            Xte_t = _build_fm(X_te)
            return Xtr_t, Xte_t, Xtr_t.shape[1], {}
        methods['FeatureTools'] = featuretools_dfs
    except ImportError:
        logger.info("featuretools not installed — FeatureTools DFS skipped")
    except Exception as e:
        logger.warning(f"FeatureTools setup error: {e}")

    # 7. AutoFeat (optional)
    try:
        from autofeat import AutoFeatRegressor, AutoFeatClassifier
        def autofeat_method(X_tr, y_tr, X_te):
            cls = AutoFeatRegressor if problem_type == 'regression' else AutoFeatClassifier
            af = cls(verbose=0)
            Xtr_t = af.fit_transform(X_tr, y_tr)
            Xte_t = af.transform(X_te)
            return Xtr_t, Xte_t, Xtr_t.shape[1], {}
        methods['AutoFeat'] = autofeat_method
    except ImportError:
        logger.info("autofeat not installed — AutoFeat skipped")
    except Exception as e:
        logger.warning(f"AutoFeat setup error: {e}")

    return methods


def run_one_fe_trial(
    dataset: DatasetInfo,
    method_name: str,
    method_fn: MethodFn,
    hep_params: dict,
    seed: int,
    cv: int = 3,
    test_size: float = 0.2,
) -> dict:
    # Set seed at trial level for reproducible HEP evolution.
    np.random.seed(seed)
    random.seed(seed)

    base = dict(
        dataset=dataset.name,
        problem_type=dataset.problem_type,
        method=method_name,
        seed=seed,
        status='ok',
    )
    rf = _get_rf(dataset.problem_type)

    X_tr, X_te, y_tr, y_te = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed
    )

    try:
        from sklearn.model_selection import KFold, StratifiedKFold
        # 1. Честная Кросс-Валидация (без утечки целевой переменной)
        cv_scores = []
        if cv > 0:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed) if dataset.problem_type == 'classification' else KFold(n_splits=cv, shuffle=True, random_state=seed)
            for tr_idx, va_idx in splitter.split(X_tr, y_tr):
                X_f_tr, y_f_tr = X_tr[tr_idx], y_tr[tr_idx]
                X_f_va, y_f_va = X_tr[va_idx], y_tr[va_idx]
                
                # Фитим FE только на фолде (предотвращаем утечку!)
                X_f_tr_t, X_f_va_t, _, _ = method_fn(X_f_tr, y_f_tr, X_f_va)
                
                pipe_cv = Pipeline([('scaler', StandardScaler()), ('model', clone(rf))])
                pipe_cv.fit(X_f_tr_t, y_f_tr)
                cv_scores.append(_test_score(pipe_cv, X_f_va_t, y_f_va, dataset.problem_type))

        # 2. Обучение на всем Train для holdout-теста и профилирования
        with ResourceProfiler() as prof:
            X_tr_t, X_te_t, n_features, meta = method_fn(X_tr, y_tr, X_te)

        # Downstream: Pipeline with StandardScaler + RF
        pipe = Pipeline([('scaler', StandardScaler()), ('model', clone(rf))])
        pipe.fit(X_tr_t, y_tr)
        test_sc = _test_score(pipe, X_te_t, y_te, dataset.problem_type)

        return {
            **base,
            'n_features':   n_features,
            'cv_score':     float(np.mean(cv_scores)) if len(cv_scores) > 0 else float('nan'),
            'test_score':   test_sc,
            'fit_time_s':   prof.elapsed_s,
            'peak_mem_mb':  prof.peak_mem_mb,
            'history_path': meta.get('history_path', ''),
        }
    except Exception as exc:
        logger.error(f"  [{method_name}] FAILED: {exc}")
        return {**base, 'status': f'error: {exc}'}


def run_benchmark_2(
    datasets: Optional[List[DatasetInfo]] = None,
    hep_params: Optional[dict] = None,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    if datasets is None:
        datasets = get_all_datasets()
    if hep_params is None:
        hep_params = DEFAULT_HEP_PARAMS.copy()
    if seeds is None:
        seeds = SEEDS

    rows = []
    for dataset in datasets:
        for seed in seeds:
            methods = build_method_registry(dataset, hep_params, seed)
            for method_name, method_fn in methods.items():
                logger.info(f"[B2] {dataset.name} | {method_name} | seed={seed}")
                row = run_one_fe_trial(dataset, method_name, method_fn, hep_params, seed)
                rows.append(row)

    return pd.DataFrame(rows)
