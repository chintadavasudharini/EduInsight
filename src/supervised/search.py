"""
CO-2: Supervised Learning - Hyperparameter Optimization Module.
Demonstrates:
- K-Fold & Stratified Cross-Validation
- GridSearchCV (Exhaustive search over specified parameter grid)
- RandomizedSearchCV (Stochastic parameter sampling over distributions)
- Hyperparameter comparison tables and validation curves
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.config import FEATURE_NAMES, DEFAULT_RANDOM_SEED


def tune_ridge_regression(
    X: pd.DataFrame,
    y: pd.Series,
    search_type: str = "grid",
    n_splits: int = 5,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Demonstrates hyperparameter tuning for Ridge Regression alpha.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=seed))
    ])

    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
    }

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if search_type == "random":
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=5,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=seed,
        )
    else:
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )

    search.fit(X[FEATURE_NAMES], y)

    results_df = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ]
    results_df["mean_rmse"] = -results_df["mean_test_score"]
    results_df = results_df.sort_values(by="rank_test_score").reset_index(drop=True)

    return {
        "search_type": search_type,
        "best_params": search.best_params_,
        "best_rmse": float(-search.best_score_),
        "results_table": results_df,
        "best_estimator": search.best_estimator_,
    }


def tune_random_forest_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    search_type: str = "grid",
    n_splits: int = 4,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Demonstrates hyperparameter tuning for Random Forest (n_estimators, max_depth).
    """
    pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=seed, class_weight="balanced"))
    ])

    param_grid = {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if search_type == "random":
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=6,
            cv=cv,
            scoring="f1_macro",
            random_state=seed,
        )
    else:
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
        )

    search.fit(X[FEATURE_NAMES], y)

    results_df = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ]
    results_df["mean_f1_macro"] = results_df["mean_test_score"]
    results_df = results_df.sort_values(by="rank_test_score").reset_index(drop=True)

    return {
        "search_type": search_type,
        "best_params": search.best_params_,
        "best_f1_macro": float(search.best_score_),
        "results_table": results_df,
        "best_estimator": search.best_estimator_,
    }
