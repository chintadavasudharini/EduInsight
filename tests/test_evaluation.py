"""Unit tests for CO-5 Model Evaluation and Fairness."""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_regression_metrics, compute_classification_metrics
from src.evaluation.calibration import compute_expected_calibration_error
from src.evaluation.statistical_tests import (
    run_paired_ttest_cv,
    run_mcnemar_test,
    compute_bootstrap_confidence_interval,
)
from src.evaluation.fairness import compute_fairness_metrics


def test_regression_metrics():
    y_true = np.array([50.0, 60.0, 70.0, 80.0])
    y_pred = np.array([52.0, 58.0, 71.0, 79.0])
    res = compute_regression_metrics(y_true, y_pred)
    assert res["MAE"] == 1.5
    assert res["R2"] > 0.90


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    res = compute_classification_metrics(y_true, y_pred)
    assert res["Accuracy"] == 1.0
    assert res["F1_Score"] == 1.0
    assert res["MCC"] == 1.0


def test_ece_calculation():
    y_true_bin = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    ece, acc, conf, _ = compute_expected_calibration_error(y_true_bin, y_prob, n_bins=5)
    assert ece >= 0.0
    assert ece <= 1.0


def test_paired_ttest():
    scores_a = [0.85, 0.86, 0.87, 0.85, 0.86]
    scores_b = [0.80, 0.81, 0.82, 0.79, 0.81]
    res = run_paired_ttest_cv(scores_a, scores_b)
    assert res["mean_difference"] > 0.0
    assert res["p_value"] < 0.05


def test_fairness_metrics():
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y_true": [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25,
        "y_pred": [1] * 20 + [0] * 30 + [1] * 22 + [0] * 28,
    })
    res = compute_fairness_metrics(df, "group", "y_true", "y_pred")
    assert "demographic_parity_ratio" in res
    assert res["four_fifths_rule_met"] is True
