"""Dataset registry for HEP benchmarks. All datasets are returned raw (unscaled)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class DatasetInfo:
    name: str
    X: np.ndarray
    y: np.ndarray
    problem_type: str          # 'regression' | 'classification'
    description: str
    feature_names: List[str] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0

    def __post_init__(self):
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        if not self.feature_names:
            self.feature_names = [f"x{i}" for i in range(self.n_features)]


def load_synthetic_regression() -> DatasetInfo:
    """1000 samples, 12 features, nonlinear target — exact formula from run_benchmark.py."""
    # Use local RandomState instead of np.random.seed() to avoid poisoning the
    # global numpy RNG for the rest of the benchmark process.
    rng = np.random.RandomState(42)
    X = rng.uniform(-3, 3, (1000, 12))
    y = (
        X[:, 0] * X[:, 1]
        + X[:, 2] ** 2
        + X[:, 3] * X[:, 4] * X[:, 5]
        + np.sin(X[:, 6]) * X[:, 7]
        + rng.normal(0, 0.2, 1000)
    )
    feature_names = [f"f{i}" for i in range(12)]
    return DatasetInfo(
        name='synthetic_regression',
        X=X, y=y,
        problem_type='regression',
        description='Nonlinear synthetic (run_benchmark.py formula)',
        feature_names=feature_names,
    )


def load_california_housing() -> DatasetInfo:
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    return DatasetInfo(
        name='california_housing',
        X=d.data, y=d.target,
        problem_type='regression',
        description='California housing prices (20640 samples)',
        feature_names=list(d.feature_names),
    )


def load_diabetes() -> DatasetInfo:
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return DatasetInfo(
        name='diabetes',
        X=d.data, y=d.target,
        problem_type='regression',
        description='Diabetes progression (442 samples, 10 features)',
        feature_names=list(d.feature_names),
    )


def load_breast_cancer() -> DatasetInfo:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return DatasetInfo(
        name='breast_cancer',
        X=d.data, y=d.target,
        problem_type='classification',
        description='Breast cancer Wisconsin (569 samples, 30 features)',
        feature_names=list(d.feature_names),
    )


def load_wine() -> DatasetInfo:
    from sklearn.datasets import load_wine
    d = load_wine()
    return DatasetInfo(
        name='wine',
        X=d.data, y=d.target,
        problem_type='classification',
        description='Wine recognition (178 samples, 13 features)',
        feature_names=list(d.feature_names),
    )


_REGISTRY = {
    'synthetic_regression': load_synthetic_regression,
    'california_housing':   load_california_housing,
    'diabetes':             load_diabetes,
    'breast_cancer':        load_breast_cancer,
    'wine':                 load_wine,
}


def get_dataset(name: str) -> DatasetInfo:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def get_all_datasets() -> List[DatasetInfo]:
    return [_REGISTRY[name]() for name in _REGISTRY]
