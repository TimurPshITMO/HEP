# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HEP (Hypergraph Evolution for Pipelines)** is an AutoFE (Automated Feature Engineering) framework using genetic algorithms on dynamic hypergraphs. It evolves a population of `Hypergraph` genomes, where each edge represents a feature transformation (e.g., `sum([f0, f1, f2])`), and evaluates fitness via cross-validated sklearn models.

## Commands

```bash
# Install dependencies
poetry install

# Run all tests
poetry run python -m pytest tests/ -v

# Run a single test file
poetry run python -m pytest tests/test_core.py -v

# Run a single test
poetry run python -m pytest tests/test_core.py::test_hyperedge_signature -v

# Run the legacy benchmark
poetry run python run_benchmark.py

# Launch the demo notebook
poetry run jupyter notebook notebooks/hep_showcase.ipynb

# Run the full benchmark suite (all 3 benchmarks)
python benchmarks/run_all.py --benchmark all --max-samples 5000

# Quick smoke test (fast params, 1 dataset)
python benchmarks/run_all.py \
    --benchmark all --datasets synthetic_regression \
    --seeds 0 1 --hep-generations 5 --hep-pop-size 10 --hep-timeout 60

# Run only a specific benchmark (1=baselines, 2=FE methods, 3=ablation)
python benchmarks/run_all.py --benchmark 2 --seeds 0 1 2
```

## Architecture

The framework is in `hep_engine/` and follows a clean separation of concerns:

- **`core.py`** — `Hyperedge` and `Hypergraph` data structures. A `Hyperedge` maps a set of feature indices to an aggregation function (`sum`, `mean`, `max`, `min`, `std`, `prod`). Edges are identified by MD5 signatures for deduplication and caching. `Hypergraph.transform(X)` applies all edges to produce new feature columns.

- **`evolution.py`** — `Individual` (wraps a `Hypergraph` genome with fitness/lineage metadata), `Population` (list of individuals), and `GeneticOperators` (mutation: add/remove/modify edges; crossover: swap edge sets between two parents).

- **`evaluator.py`** — `FitnessEvaluator` concatenates HEP-transformed features with the original features and scores via cross-validation (R² for regression, macro F1 for classification). Caches intermediate `transform` results by edge signature.

- **`optimizer.py`** — `EvolutionaryOptimizer` runs the main loop: tournament selection (k=3), crossover, mutation, elitism, and fitness caching by genome signature. Supports a `timeout` parameter.

- **`tracker.py`** — `EvolutionTracker` writes per-generation JSON files (`gen_XXX.json`) and a `full_history.json` to an output directory for post-hoc analysis.

- **`visualizer.py`** — `HEPVisualizer` renders hypergraphs using HyperNetX (falls back to NetworkX). Can generate per-generation animation frames from `full_history.json`.

## Key Design Decisions

- **Fitness caching** operates at two levels: `FitnessEvaluator` caches `transform` outputs by edge signature; `EvolutionaryOptimizer` caches full fitness scores by genome signature. This avoids redundant computation when identical genomes appear across generations.
- **Genome identity** is determined by MD5 hashes, not object identity. Two independently created `Hyperedge`s with the same nodes and function have the same signature.
- **Feature augmentation**: HEP features are always concatenated with the original features before evaluation — the evolved edges add features, they do not replace them.
- **`test_engine.py` has a known import bug** — it imports from `hep_engine.engine` which doesn't exist (correct modules are `evaluator` and `optimizer`).
- **Logging** — engine modules use `logging.getLogger(__name__)` with a `NullHandler` in `__init__.py`. Per-generation output is at `DEBUG` level; enable with `--verbose` in benchmark scripts or configure a handler in calling code.

## Benchmark Suite (`benchmarks/`)

Three benchmarks, results in `benchmarks/results/`:

| File | Purpose |
|---|---|
| `benchmark_1_baselines.py` | Same model with raw features vs HEP-augmented (multiple models × datasets × seeds) |
| `benchmark_2_fe_methods.py` | HEP vs NoFE / Poly2 / Poly2Interact / PCA_expand / FeatureTools* / AutoFeat* |
| `benchmark_3_ablation.py` | OFAT sensitivity: pop_size, n_generations, mut_rate, elitism_count, available_functions |
| `hep_wrapper.py` | `HEPTransformer(BaseEstimator, TransformerMixin)` — sklearn-compatible wrapper |
| `reporting.py` | CSV save, bar charts with errorbars, convergence curve, Pareto scatter, Friedman test |

`*` optional — gracefully skipped if not installed (`poetry install -E benchmarks` to add them).

**Key design notes for `benchmarks/`:**
- All models are wrapped in `Pipeline([StandardScaler, model])` to prevent data leakage.
- `HEPTransformer.transform()` returns the *augmented* matrix (original + HEP features) — matching what `FitnessEvaluator` optimised for during evolution.
- Each `HEPTransformer.fit()` writes tracker JSON to a unique `/tmp/hep_bench/{uuid}` path — prevents multi-seed tracker file collisions.
- B3 uses the same `hep_params` passed to `run_all.py` as the OFAT base, so `--hep-generations 5` in a test run is respected.
