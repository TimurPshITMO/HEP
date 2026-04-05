"""Tables, plots, and statistical summaries for HEP benchmark results."""

from __future__ import annotations
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_results_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Results saved to {path}")


def print_results_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    numeric_cols = df.select_dtypes(include='number').columns
    fmt = {c: '{:.4f}'.format for c in numeric_cols}
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Summary statistics (win rate, average rank, Friedman test)
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame, score_col: str = 'test_score') -> None:
    """Print win rate, average rank, and Friedman test for Benchmark 2 results."""
    if score_col not in df.columns or 'method' not in df.columns:
        logger.warning("print_summary_table: required columns missing, skipping.")
        return

    # Aggregate mean score per (dataset, method) across seeds
    agg = df.groupby(['dataset', 'method'])[score_col].mean().reset_index()

    methods = agg['method'].unique()
    datasets = agg['dataset'].unique()

    # --- Rank per dataset (lower rank = better) ---
    rank_rows = []
    for ds in datasets:
        sub = agg[agg['dataset'] == ds].copy()
        sub['rank'] = sub[score_col].rank(ascending=False, method='min')
        rank_rows.append(sub)
    rank_df = pd.concat(rank_rows, ignore_index=True)

    avg_rank = rank_df.groupby('method')['rank'].mean().sort_values()

    # --- Win rate: fraction of datasets where method is best ---
    best_per_ds = agg.loc[agg.groupby('dataset')[score_col].idxmax(), ['dataset', 'method']]
    win_counts = best_per_ds['method'].value_counts()
    win_rate = (win_counts / len(datasets)).reindex(methods, fill_value=0.0)

    # --- Friedman test ---
    try:
        from scipy.stats import friedmanchisquare
        method_scores = [
            agg[agg['method'] == m][score_col].values for m in methods
        ]
        # Friedman requires equal-length groups; skip if unequal
        if len({len(s) for s in method_scores}) == 1 and len(method_scores) >= 3:
            stat, p_value = friedmanchisquare(*method_scores)
            friedman_str = f"Friedman χ²={stat:.3f}, p={p_value:.4f}"
            if p_value < 0.05:
                friedman_str += "  *** statistically significant"
        else:
            friedman_str = "Friedman test skipped (unequal group sizes or <3 methods)"
    except ImportError:
        friedman_str = "scipy not available — Friedman test skipped"

    # --- Print ---
    summary = pd.DataFrame({
        'avg_rank':  avg_rank,
        'win_rate':  win_rate,
    }).sort_values('avg_rank')
    summary['best_on'] = summary.index.map(
        lambda m: ', '.join(best_per_ds[best_per_ds['method'] == m]['dataset'].tolist()) or '—'
    )

    print(f"\n{'='*70}")
    print("  SUMMARY: Method Comparison")
    print(f"{'='*70}")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n{friedman_str}\n")


# ---------------------------------------------------------------------------
# Benchmark 1: grouped bar chart (raw vs HEP, per dataset per model)
# ---------------------------------------------------------------------------

def plot_benchmark_1(df: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart: Raw vs HEP test score, one subplot per dataset."""
    # Aggregate across seeds
    agg = df.groupby(['dataset', 'model']).agg(
        test_score_raw=('test_score_raw', 'mean'),
        test_score_hep=('test_score_hep', 'mean'),
        std_raw=('test_score_raw', 'std'),
        std_hep=('test_score_hep', 'std'),
        delta=('delta_test_score', 'mean'),
    ).reset_index()

    datasets = agg['dataset'].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for ax, ds in zip(axes[0], datasets):
        sub = agg[agg['dataset'] == ds].reset_index(drop=True)
        models = sub['model'].tolist()
        x = np.arange(len(models))
        w = 0.35

        ax.bar(x - w / 2, sub['test_score_raw'], w,
               yerr=sub['std_raw'].fillna(0), capsize=4,
               label='Raw', color='steelblue', alpha=0.85)
        bars_hep = ax.bar(x + w / 2, sub['test_score_hep'], w,
                          yerr=sub['std_hep'].fillna(0), capsize=4,
                          label='HEP', color='darkorange', alpha=0.85)

        # Annotate delta above HEP bars
        for bar, (_, row) in zip(bars_hep, sub.iterrows()):
            delta = row['delta']
            sign = '+' if delta >= 0 else ''
            color = 'darkgreen' if delta >= 0 else 'crimson'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(row['std_hep'] if not np.isnan(row['std_hep']) else 0, 0.005) + 0.005,
                f'{sign}{delta:.3f}',
                ha='center', va='bottom', fontsize=7, color=color, fontweight='bold',
            )

        ax.set_title(ds, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Test Score (mean ± std)')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.legend(fontsize=8)

    fig.suptitle('Benchmark 1: Raw Features vs HEP-Augmented Features',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Benchmark 1 plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Benchmark 2: FE methods comparison
# ---------------------------------------------------------------------------

def plot_benchmark_2(df: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart: FE methods per dataset, HEP highlighted."""
    agg = df[df['status'] == 'ok'].groupby(['dataset', 'method']).agg(
        mean_score=('test_score', 'mean'),
        std_score=('test_score', 'std'),
    ).reset_index()

    datasets = agg['dataset'].unique()
    methods = agg['method'].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(max(5, len(methods)) * n, 5), squeeze=False)

    cmap = plt.cm.tab10
    method_colors = {m: cmap(i / max(len(methods) - 1, 1)) for i, m in enumerate(methods)}
    method_colors['HEP'] = 'crimson'

    for ax, ds in zip(axes[0], datasets):
        sub = agg[agg['dataset'] == ds].reset_index(drop=True)
        x = np.arange(len(sub))
        colors = [method_colors.get(m, 'gray') for m in sub['method']]
        ax.bar(x, sub['mean_score'],
               yerr=sub['std_score'].fillna(0), capsize=4,
               color=colors, alpha=0.85)
        ax.set_title(ds, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sub['method'], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Test Score (mean ± std)')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    fig.suptitle('Benchmark 2: Feature Engineering Methods Comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Benchmark 2 plot saved to {output_path}")


def plot_feature_count_vs_score(df: pd.DataFrame, output_path: str) -> None:
    """Scatter: n_features produced vs mean test_score, one point per (dataset, method)."""
    agg = df[df['status'] == 'ok'].groupby(['dataset', 'method']).agg(
        mean_score=('test_score', 'mean'),
        mean_features=('n_features', 'mean'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = agg['method'].unique()
    cmap = plt.cm.tab10
    for i, m in enumerate(methods):
        sub = agg[agg['method'] == m]
        color = 'crimson' if m == 'HEP' else cmap(i / max(len(methods) - 1, 1))
        ax.scatter(sub['mean_features'], sub['mean_score'],
                   label=m, color=color, s=80, zorder=3,
                   edgecolors='black', linewidths=0.4)
        for _, row in sub.iterrows():
            ax.annotate(row['dataset'], (row['mean_features'], row['mean_score']),
                        fontsize=6, textcoords='offset points', xytext=(4, 2))

    ax.set_xlabel('Mean Number of Features')
    ax.set_ylabel('Mean Test Score')
    ax.set_title('Feature Count vs Score by Method', fontweight='bold')
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Feature count vs score plot saved to {output_path}")


def plot_pareto_memory_score(df: pd.DataFrame, output_path: str) -> None:
    """Scatter: peak memory vs test score — memory-accuracy Pareto frontier."""
    if 'peak_mem_mb' not in df.columns:
        logger.warning("peak_mem_mb column missing, skipping Pareto plot.")
        return

    agg = df[df['status'] == 'ok'].groupby(['dataset', 'method']).agg(
        mean_score=('test_score', 'mean'),
        mean_mem=('peak_mem_mb', 'mean'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = agg['method'].unique()
    cmap = plt.cm.tab10
    for i, m in enumerate(methods):
        sub = agg[agg['method'] == m]
        color = 'crimson' if m == 'HEP' else cmap(i / max(len(methods) - 1, 1))
        ax.scatter(sub['mean_mem'], sub['mean_score'],
                   label=m, color=color, s=80, zorder=3,
                   edgecolors='black', linewidths=0.4)

    ax.set_xlabel('Peak Memory (MB, Python heap only)')
    ax.set_ylabel('Mean Test Score')
    ax.set_title('Memory vs Score (Pareto Frontier)', fontweight='bold')
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Pareto memory/score plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Convergence curve (from EvolutionTracker full_history.json files)
# ---------------------------------------------------------------------------

def plot_convergence_curve(history_paths: List[str], output_path: str,
                           title: str = "HEP Convergence") -> None:
    """Plot best and average fitness per generation, averaged across seeds.

    Parameters
    ----------
    history_paths : list of str
        Paths to full_history.json files, one per seed.
    output_path : str
        Where to save the PNG.
    """
    all_best: List[List[float]] = []
    all_avg: List[List[float]] = []

    for path in history_paths:
        if not os.path.exists(path):
            logger.warning(f"History file not found: {path}")
            continue
        with open(path) as f:
            history = json.load(f)
        best_curve = [max(ind['fitness'] for ind in gen['population']) for gen in history]
        avg_curve  = [np.mean([ind['fitness'] for ind in gen['population']]) for gen in history]
        all_best.append(best_curve)
        all_avg.append(avg_curve)

    if not all_best:
        logger.warning("No valid history files found for convergence plot.")
        return

    # Pad shorter runs to common length
    max_len = max(len(c) for c in all_best)
    def _pad(curves):
        return np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])

    best_arr = _pad(all_best)
    avg_arr  = _pad(all_avg)
    gens = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(8, 5))

    for arr, color, label in [
        (best_arr, 'steelblue', 'Best fitness'),
        (avg_arr,  'darkorange', 'Avg fitness'),
    ]:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(gens, mean, color=color, label=label, linewidth=2)
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Convergence plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Benchmark 3: OFAT ablation line charts
# ---------------------------------------------------------------------------

def plot_ablation(df: pd.DataFrame, output_path: str) -> None:
    """Line chart per hyperparameter showing mean best_fitness vs param value."""
    params = df['param_name'].unique()
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, param in zip(axes[0], params):
        sub = df[df['param_name'] == param].copy()
        # param_value may be a list serialised as string — keep as label
        sub['label'] = sub['param_value'].astype(str)
        grp = sub.groupby('label')['best_fitness'].agg(['mean', 'std']).reset_index()
        x = np.arange(len(grp))
        ax.errorbar(x, grp['mean'], yerr=grp['std'].fillna(0),
                    fmt='-o', color='steelblue', capsize=4, linewidth=2)
        ax.set_title(param, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grp['label'], rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Best Fitness (mean ± std)')

    fig.suptitle('Benchmark 3: HEP Hyperparameter Sensitivity (OFAT)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Ablation plot saved to {output_path}")
