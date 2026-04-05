"""HEP Benchmark Suite.

Public API for programmatic use:

    from benchmarks import get_all_datasets, HEPTransformer
    from benchmarks import save_results_csv, plot_benchmark_1, print_summary_table
"""

from benchmarks.datasets import get_all_datasets, get_dataset
from benchmarks.hep_wrapper import HEPTransformer
from benchmarks.reporting import (
    save_results_csv,
    print_results_table,
    print_summary_table,
    plot_benchmark_1,
    plot_benchmark_2,
    plot_convergence_curve,
    plot_ablation,
)

__all__ = [
    'get_all_datasets',
    'get_dataset',
    'HEPTransformer',
    'save_results_csv',
    'print_results_table',
    'print_summary_table',
    'plot_benchmark_1',
    'plot_benchmark_2',
    'plot_convergence_curve',
    'plot_ablation',
]
