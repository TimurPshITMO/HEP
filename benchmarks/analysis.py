"""Feature importance analysis for HEP-generated features."""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd


def extract_hep_feature_importance(
    rf_model,
    best_individual,
    feature_names: List[str],
    use_permutation: bool = False,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Return a DataFrame of feature importances for original + HEP features.

    Parameters
    ----------
    rf_model : fitted RandomForest (or any model with feature_importances_)
        Must have been trained on the augmented matrix [X_original | X_hep].
    best_individual : Individual
        The best evolved individual from HEPTransformer.best_individual_.
    feature_names : list of str
        Names of the *original* features (len == n_original_features).
    use_permutation : bool
        If True, use sklearn permutation_importance instead of impurity-based
        importance. Less biased but requires X_val and y_val.

        Warning: RF impurity importance is biased toward high-cardinality
        features. For final conclusions, prefer use_permutation=True.
    X_val, y_val : array-like, optional
        Validation data required when use_permutation=True.

    Returns
    -------
    pd.DataFrame with columns: feature_name, importance, feature_type
        feature_type is 'original' or 'hep'.
    """
    hep_edges = list(best_individual.genome.edges.values())

    hep_names = []
    for edge in hep_edges:
        node_labels = [feature_names[i] if i < len(feature_names) else f"x{i}"
                       for i in edge.node_indices]
        hep_names.append(f"{edge.function_name}([{', '.join(node_labels)}])")

    all_names = list(feature_names) + hep_names
    types = ['original'] * len(feature_names) + ['hep'] * len(hep_names)

    if use_permutation:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val are required when use_permutation=True")
        from sklearn.inspection import permutation_importance
        result = permutation_importance(rf_model, X_val, y_val, n_repeats=10, random_state=42)
        importances = result.importances_mean
    else:
        importances = rf_model.feature_importances_

    # Guard against mismatch (e.g. empty genome → model trained on X only)
    n_model_features = len(importances)
    all_names = all_names[:n_model_features]
    types = types[:n_model_features]

    df = pd.DataFrame({
        'feature_name': all_names,
        'importance': importances[:len(all_names)],
        'feature_type': types,
    })
    return df.sort_values('importance', ascending=False).reset_index(drop=True)
