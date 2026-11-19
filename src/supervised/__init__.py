"""Supervised Learning (CO-2)."""

from .regression import get_regression_models, train_and_evaluate_regressors
from .classification import get_classification_models, train_and_evaluate_classifiers
from .bias_variance import compute_bias_variance_decomposition
from .search import tune_ridge_regression, tune_random_forest_classifier

__all__ = [
    "get_regression_models",
    "train_and_evaluate_regressors",
    "get_classification_models",
    "train_and_evaluate_classifiers",
    "compute_bias_variance_decomposition",
    "tune_ridge_regression",
    "tune_random_forest_classifier",
]
