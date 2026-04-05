"""Entry point for the HEP benchmark suite.

Usage examples:

    # Smoke test — fast, single dataset, 2 seeds, 5 generations
    python benchmarks/run_all.py \\
        --benchmark all \\
        --datasets synthetic_regression \\
        --seeds 0 1 \\
        --hep-generations 5 \\
        --hep-pop-size 10 \\
        --hep-timeout 60 \\
        --verbose

    # Full run
    python benchmarks/run_all.py --benchmark all --max-samples 5000
"""

from __future__ import annotations
import argparse
import logging
import os
import sys

# Ensure project root is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from benchmarks.config import DEFAULT_HEP_PARAMS, DATASET_NAMES, SEEDS
from benchmarks.datasets import get_all_datasets, get_dataset
from benchmarks.reporting import (
    plot_ablation,
    plot_benchmark_1,
    plot_benchmark_2,
    plot_convergence_curve,
    plot_feature_count_vs_score,
    plot_pareto_memory_score,
    print_results_table,
    print_summary_table,
    save_results_csv,
)


def _parse_args():
    p = argparse.ArgumentParser(description='Run HEP benchmark suite')
    p.add_argument('--benchmark', choices=['1', '2', '3', 'all'], default='all')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Subset of dataset names (default: all)')
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='Random seeds (default: config.SEEDS)')
    p.add_argument('--hep-generations',       type=int,   default=DEFAULT_HEP_PARAMS['n_generations'])
    p.add_argument('--hep-pop-size',          type=int,   default=DEFAULT_HEP_PARAMS['pop_size'])
    p.add_argument('--hep-timeout',           type=float, default=DEFAULT_HEP_PARAMS['timeout'])
    p.add_argument('--hep-complexity-penalty', type=float, default=DEFAULT_HEP_PARAMS.get('complexity_penalty', 0.0))
    p.add_argument('--max-samples',     type=int, default=5000,
                   help='Subsample datasets larger than this (default: 5000)')
    p.add_argument('--output-dir',      default='benchmarks/results')
    p.add_argument('--verbose', action='store_true',
                   help='Enable DEBUG logging (shows per-generation output)')
    return p.parse_args()


def _setup_logging(output_dir: str, verbose: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'benchmark.log')),
        ],
    )


def _subsample(datasets, max_samples: int):
    from sklearn.utils import resample
    result = []
    for d in datasets:
        if d.n_samples > max_samples:
            logging.getLogger(__name__).info(
                f"{d.name}: subsampling {d.n_samples} → {max_samples} rows"
            )
            d.X, d.y = resample(d.X, d.y, n_samples=max_samples, random_state=42)
            d.n_samples = max_samples
        result.append(d)
    return result


def main():
    args = _parse_args()
    _setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    hep_params = DEFAULT_HEP_PARAMS.copy()
    hep_params['n_generations']    = args.hep_generations
    hep_params['pop_size']         = args.hep_pop_size
    hep_params['timeout']          = args.hep_timeout
    hep_params['complexity_penalty'] = args.hep_complexity_penalty

    seeds = args.seeds or SEEDS

    # Load datasets
    if args.datasets:
        datasets = [get_dataset(name) for name in args.datasets]
    else:
        datasets = get_all_datasets()
    datasets = _subsample(datasets, args.max_samples)

    # ------------------------------------------------------------------ B1
    if args.benchmark in ('1', 'all'):
        from benchmarks.benchmark_1_baselines import run_benchmark_1
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK 1: Baseline models WITH vs WITHOUT HEP")
        logger.info("=" * 60)

        df1 = run_benchmark_1(datasets, hep_params, seeds, args.output_dir)
        save_results_csv(df1, os.path.join(args.output_dir, 'benchmark_1_results.csv'))
        print_results_table(df1, "Benchmark 1 Results")
        plot_benchmark_1(df1, os.path.join(args.output_dir, 'benchmark_1_comparison.png'))

    # ------------------------------------------------------------------ B2
    if args.benchmark in ('2', 'all'):
        from benchmarks.benchmark_2_fe_methods import run_benchmark_2
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK 2: HEP vs other Feature Engineering methods")
        logger.info("=" * 60)

        df2 = run_benchmark_2(datasets, hep_params, seeds)
        save_results_csv(df2, os.path.join(args.output_dir, 'benchmark_2_results.csv'))
        print_results_table(df2, "Benchmark 2 Results")
        print_summary_table(df2)
        plot_benchmark_2(df2, os.path.join(args.output_dir, 'benchmark_2_comparison.png'))
        plot_feature_count_vs_score(df2, os.path.join(args.output_dir, 'benchmark_2_scatter.png'))
        plot_pareto_memory_score(df2, os.path.join(args.output_dir, 'benchmark_2_pareto.png'))

        # Convergence curves for HEP runs (one history file per seed)
        if 'method' in df2.columns and 'HEP' in df2['method'].values:
            hep_rows = df2[(df2['method'] == 'HEP') & (df2.get('status', 'ok') == 'ok')]
            history_paths = [p for p in hep_rows.get('history_path', pd.Series()).tolist() if p]
            if history_paths:
                plot_convergence_curve(
                    history_paths,
                    os.path.join(args.output_dir, 'convergence_curve.png'),
                    title='HEP Convergence (across seeds)',
                )
            else:
                logger.info("No HEP history paths found — skipping convergence curve.")

    # ------------------------------------------------------------------ B3
    if args.benchmark in ('3', 'all'):
        from benchmarks.benchmark_3_ablation import run_benchmark_3
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK 3: HEP Hyperparameter Sensitivity (OFAT)")
        logger.info("=" * 60)

        # B3 always runs on the synthetic dataset regardless of --datasets
        synth = next((d for d in datasets if d.name == 'synthetic_regression'), None)
        if synth is None:
            from benchmarks.datasets import get_dataset as _gd
            synth = _gd('synthetic_regression')

        df3 = run_benchmark_3(synth, seeds, base_params=hep_params)
        save_results_csv(df3, os.path.join(args.output_dir, 'benchmark_3_ablation.csv'))
        print_results_table(df3, "Benchmark 3 Ablation Results")
        plot_ablation(df3, os.path.join(args.output_dir, 'benchmark_3_ablation.png'))

    logger.info(f"\nAll outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
