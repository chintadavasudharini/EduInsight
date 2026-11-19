"""
CO-3: Unsupervised Learning - Anomaly Detection Module.
Implements:
- Isolation Forest (Recursive partitioning path length)
- One-Class Support Vector Machine (Kernel boundary enclosing inlier density)
Identifies unusual academic profiles and outputs anomaly scores with explanations.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURE_NAMES, DEFAULT_RANDOM_SEED


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Fits Isolation Forest.
    Shorter average tree path length -> higher anomaly score.
    Returns:
      - is_anomaly (boolean)
      - anomaly_score (continuous negative score; lower = more anomalous)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=100,
    )
    # 1 for inlier, -1 for outlier
    raw_preds = iso_forest.fit_predict(X_scaled)
    # decision_function: negative scores are outliers
    scores = iso_forest.decision_function(X_scaled)

    is_anomaly = (raw_preds == -1)

    df_anomalies = df[is_anomaly].copy()
    df_anomalies["anomaly_score"] = scores[is_anomaly]
    df_anomalies = df_anomalies.sort_values(by="anomaly_score").reset_index(drop=True)

    return {
        "model": iso_forest,
        "is_anomaly": is_anomaly,
        "anomaly_scores": scores,
        "n_anomalies": int(np.sum(is_anomaly)),
        "anomaly_records": df_anomalies,
        "contamination": contamination,
    }


def detect_anomalies_one_class_svm(
    df: pd.DataFrame,
    nu: float = 0.05,
    gamma: str = "scale",
) -> Dict[str, Any]:
    """
    Fits One-Class SVM with RBF kernel to enclose dense inlier region.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    oc_svm = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    raw_preds = oc_svm.fit_predict(X_scaled)
    scores = oc_svm.decision_function(X_scaled)

    is_anomaly = (raw_preds == -1)

    df_anomalies = df[is_anomaly].copy()
    df_anomalies["svm_score"] = scores[is_anomaly]
    df_anomalies = df_anomalies.sort_values(by="svm_score").reset_index(drop=True)

    return {
        "model": oc_svm,
        "is_anomaly": is_anomaly,
        "svm_scores": scores,
        "n_anomalies": int(np.sum(is_anomaly)),
        "anomaly_records": df_anomalies,
        "nu": nu,
    }


def explain_student_anomaly(row: pd.Series, pop_means: pd.Series, pop_stds: pd.Series) -> str:
    """
    Generates academic plain-English explanation for why a student record is anomalous.
    Calculates z-scores for each feature.
    """
    reasons = []
    for col in FEATURE_NAMES:
        val = row[col]
        mean = pop_means[col]
        std = pop_stds[col]
        z = (val - mean) / (std + 1e-12)
        if abs(z) >= 2.2:
            direction = "unusually high" if z > 0 else "unusually low"
            reasons.append(f"{col.replace('_', ' ').title()} is {direction} ({val:.1f}, z={z:.2f})")

    # Domain specific interaction checks
    if row["attendance_percentage"] > 90.0 and row.get("final_exam_score", 50.0) < 40.0:
        reasons.append("High attendance (>=90%) but very low exam outcome (<40%) - atypical decoupling.")
    if row["study_hours_per_day"] > 8.5 and row["sleep_hours"] < 5.0:
        reasons.append("Extreme study hours with severe sleep deprivation (<5 hrs) - high fatigue risk.")

    if not reasons:
        reasons.append("Multivariate anomaly: combination of features lies in low-density region of feature space.")

    return "; ".join(reasons)
