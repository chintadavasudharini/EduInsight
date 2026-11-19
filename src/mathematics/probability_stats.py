"""
CO-1: Mathematical Foundations for ML - Probability and Statistics Module.
Implements:
- Summary statistics (mean, median, sample/population variance, standard deviation)
- Continuous distributions (PDF, CDF)
- Joint, marginal, and conditional probabilities
- Bayes' Theorem with complete step-by-step numerical breakdown
- Expectation and variance derivation: Var(X) = E[X^2] - (E[X])^2
- Confidence intervals (Student's t) and Hypothesis testing (p-values)
- Maximum Likelihood Estimation (MLE) vs Maximum A Posteriori (MAP)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats


def compute_descriptive_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Computes sample and population summary statistics:
    - Mean: mu = 1/N sum(x_i)
    - Median
    - Sample variance (ddof=1, Bessel's correction)
    - Population variance (ddof=0)
    - Sample and population standard deviation
    - Skewness and Kurtosis
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    sample_var = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    pop_var = float(np.var(arr, ddof=0))
    sample_std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    pop_std = float(np.std(arr, ddof=0))
    skew_val = float(stats.skew(arr))
    kurt_val = float(stats.kurtosis(arr))

    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "sample_variance": sample_var,
        "population_variance": pop_var,
        "sample_std": sample_std,
        "population_std": pop_std,
        "skewness": skew_val,
        "kurtosis": kurt_val,
    }


def compute_expectation_and_variance_proof(
    values: np.ndarray, probabilities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Demonstrates Expectation and the fundamental variance identity:
        Var(X) = E[X^2] - (E[X])^2
    """
    val = np.asarray(values, dtype=float)
    if probabilities is None:
        p = np.ones_like(val) / len(val)
    else:
        p = np.asarray(probabilities, dtype=float)
        p = p / np.sum(p)  # normalize

    e_x = float(np.sum(val * p))
    e_x2 = float(np.sum((val ** 2) * p))
    var_identity = e_x2 - (e_x ** 2)
    var_direct = float(np.sum(((val - e_x) ** 2) * p))

    return {
        "E_X": e_x,
        "E_X2": e_x2,
        "E_X_squared": e_x ** 2,
        "Var_identity": var_identity,
        "Var_direct": var_direct,
        "identity_verified": bool(np.isclose(var_identity, var_direct, atol=1e-8)),
    }


def compute_bayes_theorem_student_case(
    study_hours: np.ndarray,
    exam_scores: np.ndarray,
    high_study_threshold: float = 6.0,
    excellent_threshold: float = 80.0,
) -> Dict[str, Any]:
    """
    Demonstrates Bayes' Theorem using student records:
        Event A: Student scores >= excellent_threshold (Excellent)
        Event B: Student studies >= high_study_threshold (High Study)

    Formula:
        P(A | B) = [ P(B | A) * P(A) ] / P(B)
        Posterior = [ Likelihood * Prior ] / Marginal Evidence
    """
    study_arr = np.asarray(study_hours, dtype=float)
    score_arr = np.asarray(exam_scores, dtype=float)
    total_n = len(study_arr)

    event_a = score_arr >= excellent_threshold     # Excellent
    event_b = study_arr >= high_study_threshold    # High Study
    event_ab = event_a & event_b

    # Probabilities
    p_a = float(np.mean(event_a))                    # Prior P(A)
    p_b = float(np.mean(event_b))                    # Marginal P(B)
    p_ab = float(np.mean(event_ab))                  # Joint P(A and B)

    # Likelihood P(B | A) = P(A and B) / P(A)
    p_b_given_a = float(p_ab / p_a) if p_a > 0 else 0.0

    # Empirical Conditional P(A | B) = P(A and B) / P(B)
    empirical_p_a_given_b = float(p_ab / p_b) if p_b > 0 else 0.0

    # Bayes Calculation: [P(B | A) * P(A)] / P(B)
    bayes_p_a_given_b = float((p_b_given_a * p_a) / p_b) if p_b > 0 else 0.0

    return {
        "total_records": total_n,
        "prior_P_A": p_a,
        "marginal_P_B": p_b,
        "joint_P_A_and_B": p_ab,
        "likelihood_P_B_given_A": p_b_given_a,
        "posterior_bayes_P_A_given_B": bayes_p_a_given_b,
        "empirical_P_A_given_B": empirical_p_a_given_b,
        "formula_matched": bool(np.isclose(bayes_p_a_given_b, empirical_p_a_given_b, atol=1e-8)),
    }


def compute_confidence_intervals_and_ttest(
    sample_data: np.ndarray,
    null_mean: float = 65.0,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Computes:
    1. Confidence Interval using Student's t-distribution:
       x_bar +/- t_(alpha/2, n-1) * (s / sqrt(n))
    2. One-Sample Hypothesis Test (H0: mu = null_mean vs H1: mu != null_mean)
    """
    arr = np.asarray(sample_data, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_err = float(stats.sem(arr))  # s / sqrt(n)
    alpha = 1.0 - confidence_level

    # t-critical value
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin_of_error = t_crit * std_err
    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error

    # One-sample t-test
    t_stat, p_val = stats.ttest_1samp(arr, null_mean)

    return {
        "n": n,
        "mean": mean_val,
        "standard_error": std_err,
        "confidence_level": confidence_level,
        "t_critical": t_crit,
        "margin_of_error": margin_of_error,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "null_hypothesis_mean": null_mean,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "reject_null_at_05": bool(p_val < 0.05),
    }


def compute_mle_vs_map(
    data: np.ndarray,
    prior_mean: float = 65.0,
    prior_std: float = 5.0,
    assumed_data_std: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Demonstrates Maximum Likelihood Estimation (MLE) vs
    Maximum A Posteriori (MAP) for Gaussian parameter estimation.

    Likelihood: X_i ~ N(mu, sigma^2)
    Prior:      mu ~ N(mu_0, sigma_0^2)

    MLE:
        mu_hat_MLE = 1/N sum(x_i)

    MAP:
        mu_hat_MAP = ( (N / sigma^2) * x_bar + (1 / sigma_0^2) * mu_0 ) /
                     ( (N / sigma^2) + (1 / sigma_0^2) )
    Demonstrates regularization effect of the Bayesian prior.
    """
    arr = np.asarray(data, dtype=float)
    N = len(arr)
    sample_mean = float(np.mean(arr))
    sample_var = float(np.var(arr, ddof=1)) if N > 1 else 1.0

    sigma_sq = (assumed_data_std ** 2) if assumed_data_std else sample_var
    prior_var = prior_std ** 2

    # Precision weights
    data_precision = N / sigma_sq
    prior_precision = 1.0 / prior_var
    total_precision = data_precision + prior_precision

    map_mean = (data_precision * sample_mean + prior_precision * prior_mean) / total_precision
    map_variance = 1.0 / total_precision
    map_std = float(np.sqrt(map_variance))

    # Shrinkage weight towards prior
    prior_weight = prior_precision / total_precision
    data_weight = data_precision / total_precision

    return {
        "N": N,
        "sample_mean_MLE": sample_mean,
        "sample_std": float(np.sqrt(sample_var)),
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "MAP_mean": float(map_mean),
        "MAP_std": map_std,
        "data_weight": float(data_weight),
        "prior_weight": float(prior_weight),
        "shrinkage_effect": float(map_mean - sample_mean),
    }
