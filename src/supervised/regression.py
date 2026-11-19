"""
CO-2: Supervised Learning - Regression Module.
Implements:
- Linear Regression
- Polynomial Regression (degree 2)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet Regression (L1 + L2 mixture)
Includes leak-free pipeline wrapping with StandardScaler.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import FEATURE_NAMES, TARGET_REGRESSION, DEFAULT_RANDOM_SEED


def get_regression_models(seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    Returns dictionary of sklearn Pipeline objects with built-in scaling.
    This guarantees zero leakage during cross-validation or fitting.
    """
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Polynomial Regression (Deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=seed))
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, random_state=seed, max_iter=2000))
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed, max_iter=2000))
        ]),
    }
    return models


def train_and_evaluate_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits each regression model on X_train and evaluates on X_test.
    Returns comparison summary DataFrame and fitted models dictionary.
    """
    models = get_regression_models(seed=seed)
    results = []
    fitted_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train[FEATURE_NAMES], y_train)
        y_pred = pipeline.predict(X_test[FEATURE_NAMES])

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R-squared": round(r2, 4),
        })
        fitted_models[name] = {
            "pipeline": pipeline,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    summary_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    return summary_df, fitted_models
