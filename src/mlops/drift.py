"""
CO-6: Production ML Engineering / MLOps - Data Drift Monitoring Module.
Implements:
- Kolmogorov-Smirnov (KS) Two-Sample Test for continuous feature distribution drift
- Population Stability Index (PSI) calculation
- Synthetic drift generator to simulate covariate shift in student populations
NOTE: This is an educational demonstration of statistical drift detection concepts.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import FEATURE_NAMES


def compute_ks_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: List[str] = FEATURE_NAMES,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes two-sample Kolmogorov-Smirnov test for each feature:
        H0: Current data is drawn from the same continuous distribution as Reference.
        H1: Distribution has shifted (p-value < alpha).
    """
    drift_report = {}
    n_drifted = 0

    for feat in features:
        ref_vals = reference_data[feat].dropna().values
        curr_vals = current_data[feat].dropna().values

        ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
        is_drifted = bool(p_val < alpha)
        if is_drifted:
            n_drifted += 1

        drift_report[feat] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "drift_detected": is_drifted,
            "ref_mean": round(float(np.mean(ref_vals)), 2),
            "curr_mean": round(float(np.mean(curr_vals)), 2),
            "ref_std": round(float(np.std(ref_vals)), 2),
            "curr_std": round(float(np.std(curr_vals)), 2),
        }

    return {
        "features": drift_report,
        "n_features_evaluated": len(features),
        "n_drifted_features": n_drifted,
        "drift_detected_overall": bool(n_drifted > 0),
    }


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Computes Population Stability Index (PSI):
        PSI = sum ( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
    Thresholds:
        PSI < 0.10: Insignificant change (No shift)
        0.10 <= PSI < 0.20: Moderate shift (Monitor closely)
        PSI >= 0.20: Significant shift (Retraining recommended)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    # Use quantile bins from reference to define buckets
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    # Ensure distinct bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Calculate proportions
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    ref_pct = ref_counts / (len(ref) + 1e-12)
    curr_pct = curr_counts / (len(curr) + 1e-12)

    # Smoothing to avoid zero division / log(0)
    ref_pct = np.clip(ref_pct, 1e-4, 1.0)
    curr_pct = np.clip(curr_pct, 1e-4, 1.0)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi)


def generate_drifted_student_data(
    base_df: pd.DataFrame,
    drift_type: str = "severe_covid_shift",
) -> pd.DataFrame:
    """
    Synthesizes shifted data distributions for educational monitoring demonstrations:
    - 'severe_covid_shift': Attendance drops significantly, sleep hours erratic, study hours decrease.
    - 'exam_cramming_shift': Study hours spike, sleep drops sharply.
    """
    drifted = base_df.copy()

    if drift_type == "severe_covid_shift":
        # Attendance drops by 15-25%
        drifted["attendance_percentage"] = np.clip(drifted["attendance_percentage"] * 0.82, 35.0, 100.0)
        # Study hours drop
        drifted["study_hours_per_day"] = np.clip(drifted["study_hours_per_day"] * 0.85, 1.0, 10.0)
    elif drift_type == "exam_cramming_shift":
        # Study hours increase by 30%
        drifted["study_hours_per_day"] = np.clip(drifted["study_hours_per_day"] * 1.30, 1.0, 10.0)
        # Sleep hours drop
        drifted["sleep_hours"] = np.clip(drifted["sleep_hours"] * 0.75, 3.5, 10.0)

    return drifted
