"""Centralized configuration for the HEP benchmark suite."""

SEEDS = [0, 1, 2, 3, 4]

DEFAULT_HEP_PARAMS = dict(
    pop_size=20,
    n_generations=20,
    mut_rate=0.4,
    cross_rate=0.5,
    elitism_count=2,
    cv=3,
    timeout=300,
    complexity_penalty=0.0,
)

DATASET_NAMES = [
    'synthetic_regression',
    'california_housing',
    'diabetes',
    'breast_cancer',
    'wine',
]
