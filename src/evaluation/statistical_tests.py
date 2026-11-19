"""
CO-5: Model Evaluation and Selection - Statistical Hypothesis Testing Module.
Implements:
- Paired Student's t-test on Cross-Validation folds
- McNemar's Test for comparing two classifier error patterns
- Non-parametric Bootstrap Confidence Intervals for evaluation metrics
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


def run_paired_ttest_cv(
    scores_model_a: List[float],
    scores_model_b: List[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> Dict[str, Any]:
    """
    Performs Paired t-test on K-fold cross-validation scores:
        H0: Mean difference d_bar = 0 (No performance difference)
        H1: Mean difference d_bar != 0
    """
    a = np.asarray(scores_model_a, dtype=float)
    b = np.asarray(scores_model_b, dtype=float)
    diff = a - b
    mean_diff = float(np.mean(diff))

    t_stat, p_val = stats.ttest_rel(a, b)

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "mean_score_a": float(np.mean(a)),
        "mean_score_b": float(np.mean(b)),
        "mean_difference": mean_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "statistically_significant_at_05": bool(p_val < 0.05),
    }


def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> Dict[str, Any]:
    """
    Performs McNemar's Test on paired classifier predictions:
    Contingency table:
               Model B Correct | Model B Incorrect
      Model A Correct     n00             n01
      Model A Incorrect   n10             n11

    Test Statistic with Edwards continuity correction:
        chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
    """
    y_t = np.asarray(y_true)
    correct_a = (y_pred_a == y_t)
    correct_b = (y_pred_b == y_t)

    # Contingency elements
    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
    n10 = int(np.sum(~correct_a & correct_b))  # A wrong, B correct
    n11 = int(np.sum(~correct_a & ~correct_b))

    contingency = [[n00, n01], [n10, n11]]

    # Edwards continuity correction
    total_discordant = n01 + n10
    if total_discordant > 0:
        chi2 = ((abs(n01 - n10) - 1.0) ** 2) / total_discordant
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
    else:
        chi2 = 0.0
        p_val = 1.0

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "contingency_table": contingency,
        "n01_a_only_correct": n01,
        "n10_b_only_correct": n10,
        "chi2_statistic": float(chi2),
        "p_value": float(p_val),
        "statistically_significant": bool(p_val < 0.05),
    }


def compute_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str = "accuracy",
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes non-parametric empirical bootstrap confidence interval for test metric.
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstraps):
        idx = rng.choice(N, size=N, replace=True)
        sample_true = y_true[idx]
        sample_pred = y_pred[idx]

        if metric_name == "accuracy":
            score = accuracy_score(sample_true, sample_pred)
        elif metric_name == "f1":
            score = f1_score(sample_true, sample_pred, average="macro", zero_division=0)
        else:
            score = accuracy_score(sample_true, sample_pred)
        bootstrap_scores.append(score)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_scores, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_scores, 100.0 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(bootstrap_scores))
    std_err = float(np.std(bootstrap_scores))

    return {
        "metric": metric_name,
        "mean": mean_val,
        "std_error": std_err,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "bootstrap_scores": bootstrap_scores[:100],  # sample for visualization
    }
