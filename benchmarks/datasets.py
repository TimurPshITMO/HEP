"""Dataset registry for HEP benchmarks. All datasets are returned raw (unscaled).

Mandatory datasets (always available, sklearn-based):
    synthetic_regression, california_housing, diabetes, breast_cancer, wine

Optional real-world datasets (require: poetry install -E real-datasets):
    superconductivity, appliances_energy, qsar_biodegradation, gas_sensor_array
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List
import numpy as np

_logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Optional real-world datasets
# Install: poetry install -E real-datasets  OR  pip install ucimlrepo openml
# ---------------------------------------------------------------------------

def load_superconductivity() -> DatasetInfo:
    """Superconductivity Tc regression, 21 263 × 81, OpenML 44964."""
    try:
        import openml
    except ImportError:
        raise ImportError("pip install ucimlrepo openml  OR  poetry install -E real-datasets")
    ds = openml.datasets.get_dataset(44964)
    X_df, y_s, _, _ = ds.get_data(target=ds.default_target_attribute)
    return DatasetInfo(
        name='superconductivity',
        X=X_df.to_numpy(dtype=np.float64),
        y=np.array(y_s, dtype=np.float64).ravel(),
        problem_type='regression',
        description='Superconductivity critical temperature Tc (21 263 × 81, OpenML 44964)',
        feature_names=list(X_df.columns),
    )


def load_appliances_energy() -> DatasetInfo:
    """Appliances energy prediction, 19 735 × 28, UCI."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError("pip install ucimlrepo openml  OR  poetry install -E real-datasets")
    ds = fetch_ucirepo(name="Appliances energy prediction")
    X = ds.data.features.select_dtypes(include=[np.number])  # drop datetime 'date' column
    y = ds.data.targets.iloc[:, 0]                           # 'Appliances' energy in Wh
    return DatasetInfo(
        name='appliances_energy',
        X=X.to_numpy(dtype=np.float64),
        y=np.array(y, dtype=np.float64).ravel(),
        problem_type='regression',
        description='Appliances energy prediction (19 735 × 28, UCI)',
        feature_names=list(X.columns),
    )


def load_qsar_biodegradation() -> DatasetInfo:
    """QSAR biodegradation binary classification, 1 055 × 41, UCI."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError("pip install ucimlrepo openml  OR  poetry install -E real-datasets")
    from sklearn.preprocessing import LabelEncoder
    ds = fetch_ucirepo(name="QSAR biodegradation")
    X = ds.data.features.to_numpy(dtype=np.float64)
    # Labels may be strings ('RB'/'NRB') or integers (1/2) — LabelEncoder handles both
    y = LabelEncoder().fit_transform(ds.data.targets.iloc[:, 0].astype(str))
    return DatasetInfo(
        name='qsar_biodegradation',
        X=X,
        y=y,
        problem_type='classification',
        description='QSAR Biodegradation binary classification (1 055 × 41, UCI)',
        feature_names=list(ds.data.features.columns),
    )


def load_gas_sensor_array() -> DatasetInfo:
    """Gas sensor array drift, 13 910 × 128, 10-class, UCI."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError("pip install ucimlrepo openml  OR  poetry install -E real-datasets")
    from sklearn.preprocessing import LabelEncoder
    ds = fetch_ucirepo(name="Gas Sensor Array Drift Dataset")
    X = ds.data.features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(ds.data.targets.iloc[:, 0].astype(str))
    return DatasetInfo(
        name='gas_sensor_array',
        X=X,
        y=y,
        problem_type='classification',
        description='Gas Sensor Array Drift 10-class (13 910 × 128, UCI)',
        feature_names=[f"sensor_{i}" for i in range(X.shape[1])],
    )


_OPTIONAL_REGISTRY: dict = {
    'superconductivity':   load_superconductivity,
    'appliances_energy':   load_appliances_energy,
    'qsar_biodegradation': load_qsar_biodegradation,
    'gas_sensor_array':    load_gas_sensor_array,
}


def get_dataset(name: str) -> DatasetInfo:
    if name in _REGISTRY:
        return _REGISTRY[name]()
    if name in _OPTIONAL_REGISTRY:
        return _OPTIONAL_REGISTRY[name]()  # raises ImportError with helpful message if missing
    raise KeyError(
        f"Unknown dataset '{name}'. "
        f"Available: {list(_REGISTRY) + list(_OPTIONAL_REGISTRY)}"
    )


def get_all_datasets() -> List[DatasetInfo]:
    datasets = [_REGISTRY[name]() for name in _REGISTRY]
    for name, loader in _OPTIONAL_REGISTRY.items():
        try:
            datasets.append(loader())
        except Exception as exc:  # ImportError, network errors, API changes
            _logger.warning(f"Skipping optional dataset '{name}': {exc}")
    return datasets
