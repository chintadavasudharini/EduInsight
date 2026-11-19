"""
CO-5: Model Evaluation and Selection - Advanced Cross-Validation Schemes Module.
Demonstrates:
- Stratified K-Fold Cross-Validation (preserves class label balance)
- Group-Aware K-Fold Cross-Validation (prevents cluster/campus leakage across folds)
- Time-Series Cross-Validation (forward chaining temporal evaluation)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.config import FEATURE_NAMES, DEFAULT_RANDOM_SEED


def demonstrate_stratified_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Demonstrates Stratified K-Fold Cross-Validation for classification."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    fold_distributions = []

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=seed))
    ])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X[FEATURE_NAMES], y)):
        X_tr, y_tr = X.iloc[train_idx][FEATURE_NAMES], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx][FEATURE_NAMES], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_va)
        acc = float(accuracy_score(y_va, val_pred))
        scores.append(acc)

        dist = y_va.value_counts(normalize=True).round(3).to_dict()
        fold_distributions.append({"fold": fold + 1, "accuracy": acc, "class_distribution": dist})

    return {
        "n_splits": n_splits,
        "fold_scores": scores,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "folds_detail": fold_distributions,
    }


def demonstrate_group_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 2,
) -> Dict[str, Any]:
    """Demonstrates Group-Aware K-Fold (no student group appears in both train and validation)."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    group_splits = []

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42))
    ])

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X[FEATURE_NAMES], y, groups)):
        X_tr, y_tr = X.iloc[train_idx][FEATURE_NAMES], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx][FEATURE_NAMES], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        acc = float(accuracy_score(y_va, model.predict(X_va)))
        scores.append(acc)

        train_groups = list(groups.iloc[train_idx].unique())
        val_groups = list(groups.iloc[val_idx].unique())
        group_splits.append({
            "fold": fold + 1,
            "train_groups": train_groups,
            "val_groups": val_groups,
            "accuracy": acc,
        })

    return {
        "mean_accuracy": float(np.mean(scores)),
        "fold_scores": scores,
        "splits": group_splits,
    }


def demonstrate_timeseries_cv(
    X: pd.DataFrame,
    y_reg: pd.Series,
    n_splits: int = 4,
) -> Dict[str, Any]:
    """Demonstrates TimeSeriesSplit (expanding window evaluation without lookahead bias)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    fold_details = []

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X[FEATURE_NAMES])):
        X_tr, y_tr = X.iloc[train_idx][FEATURE_NAMES], y_reg.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx][FEATURE_NAMES], y_reg.iloc[val_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        rmse = float(root_mean_squared_error(y_va, pred))
        rmse_scores.append(rmse)

        fold_details.append({
            "fold": fold + 1,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "rmse": rmse,
        })

    return {
        "mean_rmse": float(np.mean(rmse_scores)),
        "rmse_scores": rmse_scores,
        "fold_details": fold_details,
    }
