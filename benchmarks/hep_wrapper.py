"""sklearn-compatible wrapper around HEP's EvolutionaryOptimizer.

HEPTransformer.transform() returns an *augmented* matrix (original features
concatenated with HEP-derived features), because that is exactly what
FitnessEvaluator evaluated during evolution. Using the augmented matrix
downstream ensures that benchmark scores match the fitness optimised for.
"""

from __future__ import annotations
import os
import uuid
import random
import logging
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from hep_engine import FitnessEvaluator, EvolutionaryOptimizer

logger = logging.getLogger(__name__)


class HEPTransformer(BaseEstimator, TransformerMixin):
    """Wraps EvolutionaryOptimizer as a sklearn transformer.

    Parameters
    ----------
    pop_size : int
        Population size for the genetic algorithm.
    n_generations : int
        Number of generations to evolve.
    mut_rate : float
        Mutation probability per individual.
    cross_rate : float
        Crossover probability per pair.
    elitism_count : int
        Number of elite individuals preserved each generation.
    problem_type : str
        'regression' or 'classification'.
    cv : int
        Cross-validation folds used by FitnessEvaluator.
    timeout : float
        Max wall-clock seconds for the evolution loop.
    n_jobs : int or None
        Parallelism passed to FitnessEvaluator (sklearn n_jobs convention).
    complexity_penalty : float
        L0 regularization coefficient λ: fitness = CV_score − λ·|E|.
        0.0 disables regularization. Typical range: 0.001–0.01.
    inner_model : sklearn estimator or None
        Proxy model used by FitnessEvaluator to score candidate hypergraphs
        during evolution via cross-validation. If None, defaults to a
        lightweight RandomForest(n_estimators=50, max_depth=5) for speed.

        For scientifically clean benchmarks, pass the same model used
        downstream: ``inner_model=clone(model_template)``. This ensures HEP
        evolves features optimised for the actual evaluation architecture:

        - Ridge/LogReg: near-zero cost (linear fits take milliseconds)
        - RandomForest(100): ~2× vs default proxy, within timeout budget
        - GradientBoosting(100): ~10× slower; timeout caps the run and
          returns the best genome found so far

        The default RF(50, depth=5) proxy is kept for exploratory use where
        cross-architecture generalisability is the experimental goal.
    random_state : int or None
        Seeds both numpy and Python random for reproducibility.
    history_dir : str or None
        Directory for EvolutionTracker JSON output. If None, a unique
        temporary path is generated per fit() call so that multi-seed runs
        don't overwrite each other's JSON files.
    """

    def __init__(
        self,
        pop_size: int = 20,
        n_generations: int = 20,
        mut_rate: float = 0.4,
        cross_rate: float = 0.5,
        elitism_count: int = 2,
        problem_type: str = 'regression',
        cv: int = 3,
        timeout: float = 300.0,
        n_jobs: Optional[int] = -1,
        complexity_penalty: float = 0.0,
        available_functions: Optional[List[str]] = None,
        inner_model=None,
        random_state: Optional[int] = 42,
        history_dir: Optional[str] = None,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.elitism_count = elitism_count
        self.problem_type = problem_type
        self.cv = cv
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.complexity_penalty = complexity_penalty
        self.available_functions = available_functions
        self.inner_model = inner_model
        self.random_state = random_state
        self.history_dir = history_dir

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HEPTransformer':
        import random

        # Apply random_state to global PRNG for reproducible evolution.
        # isinstance(int) guard: random.seed() cannot accept np.random.RandomState
        # objects and would raise TypeError (sklearn allows all three types).
        # Technical Debt: global seed is an sklearn anti-pattern (global state
        # poisoning). Correct fix: local RNG instance inside GeneticOperators.
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Build default inner model with same random_state for reproducible CV scores.
        # Use local variable — modifying self.inner_model in fit() would violate
        # sklearn's requirement that __init__ params are not mutated in fit().
        inner = self.inner_model
        if inner is None:
            if self.problem_type == 'regression':
                inner = RandomForestRegressor(
                    n_estimators=50, max_depth=5,
                    n_jobs=self.n_jobs, random_state=self.random_state,
                )
            else:
                inner = RandomForestClassifier(
                    n_estimators=50, max_depth=5,
                    n_jobs=self.n_jobs, random_state=self.random_state,
                )

        evaluator = FitnessEvaluator(
            X, y,
            problem_type=self.problem_type,
            cv=self.cv,
            complexity_penalty=self.complexity_penalty,
            n_jobs=self.n_jobs,
            model=inner,
        )

        optimizer = EvolutionaryOptimizer(
            pop_size=self.pop_size,
            mut_rate=self.mut_rate,
            cross_rate=self.cross_rate,
            elitism_count=self.elitism_count,
            available_functions=self.available_functions,
        )
        # NOTE: EvolutionaryOptimizer.__init__ creates EvolutionTracker(output_dir='history'),
        # which immediately calls os.makedirs('history') in CWD. This is a side effect of
        # hep_engine internals and cannot be avoided without modifying hep_engine.
        # Data is correctly routed to run_dir below via output_dir override.

        # Each fit() gets a unique tracker directory to avoid collisions when
        # running multiple seeds in the same process.
        run_dir = self.history_dir or os.path.join(
            '/tmp', 'hep_bench', uuid.uuid4().hex[:12]
        )
        os.makedirs(run_dir, exist_ok=True)
        optimizer.tracker.output_dir = run_dir  # must be set BEFORE run()

        logger.info(
            f"Starting HEP evolution: pop={self.pop_size}, "
            f"gens={self.n_generations}, history={run_dir}"
        )
        population = optimizer.run(
            evaluator,
            n_generations=self.n_generations,
            timeout=self.timeout,
        )

        best = population.best()
        self.best_genome_ = best.genome
        self.best_individual_ = best
        self.n_features_in_ = X.shape[1]
        self.n_hep_features_ = len(best.genome)
        self.best_fitness_ = best.fitness
        self.history_path_ = os.path.join(run_dir, 'full_history.json')

        logger.info(
            f"HEP done: best_fitness={self.best_fitness_:.4f}, "
            f"n_hep_features={self.n_hep_features_}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['best_genome_', 'n_features_in_'])

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but HEPTransformer was fitted "
                f"on {self.n_features_in_} features."
            )

        X_hep = self.best_genome_.transform(X)

        if X_hep.shape[1] == 0:
            # Empty genome — return original unchanged (matches FitnessEvaluator)
            return X

        return np.hstack([X, X_hep])

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        check_is_fitted(self, ['best_genome_', 'n_features_in_'])
        if input_features is not None:
            orig = list(input_features)
        else:
            orig = [f"x{i}" for i in range(self.n_features_in_)]

        hep_names = []
        for edge in self.best_genome_.edges.values():
            nodes_str = '_'.join(map(str, edge.node_indices))
            hep_names.append(f"hep_{edge.function_name}_{nodes_str}")

        return orig + hep_names
