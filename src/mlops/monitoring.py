"""
CO-6: Production ML Engineering / MLOps - Performance Monitoring & Retraining Module.
Implements:
- Rolling window model performance tracking
- Concept drift vs Data drift conceptualization
- A/B Model Deployment comparison (Champion vs Challenger)
- Automated trigger evaluation for controlled retraining
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score


def simulate_ab_testing(
    df: pd.DataFrame,
    champion_pipeline,
    challenger_pipeline,
    target_col: str = "final_exam_score",
    traffic_split: float = 0.50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulates A/B Testing:
    Routes incoming live inference requests between Champion (A) and Challenger (B) models.
    Measures operational throughput and test error on real evaluation batches.
    """
    rng = np.random.default_rng(seed)
    N = len(df)

    # Route traffic
    route_to_b = rng.random(N) < traffic_split

    df_a = df[~route_to_b].copy()
    df_b = df[route_to_b].copy()

    from src.utils.config import FEATURE_NAMES

    # Champion predictions
    pred_a = champion_pipeline.predict(df_a[FEATURE_NAMES])
    rmse_a = float(np.sqrt(mean_squared_error(df_a[target_col], pred_a)))

    # Challenger predictions
    pred_b = challenger_pipeline.predict(df_b[FEATURE_NAMES])
    rmse_b = float(np.sqrt(mean_squared_error(df_b[target_col], pred_b)))

    improvement_pct = ((rmse_a - rmse_b) / (rmse_a + 1e-12)) * 100.0

    return {
        "champion_samples": len(df_a),
        "challenger_samples": len(df_b),
        "champion_rmse": round(rmse_a, 4),
        "challenger_rmse": round(rmse_b, 4),
        "challenger_improvement_pct": round(improvement_pct, 2),
        "recommended_action": (
            "Promote Challenger to Production" if improvement_pct > 3.0 and len(df_b) > 50
            else "Maintain Champion (Difference not statistically substantial)"
        ),
    }


def evaluate_retraining_trigger(
    drift_report: Dict[str, Any],
    rolling_rmse: float,
    baseline_rmse: float,
    rmse_threshold_multiplier: float = 1.15,
) -> Dict[str, Any]:
    """
    Evaluates whether an automated retraining pipeline should be triggered.
    Trigger Condition:
      - 2 or more features have drifted (KS test p < 0.05) OR
      - Rolling RMSE exceeds baseline by > 15%.
    """
    n_drifted = drift_report.get("n_drifted_features", 0)
    drift_trigger = n_drifted >= 2

    rmse_ratio = rolling_rmse / (baseline_rmse + 1e-9)
    perf_trigger = rmse_ratio > rmse_threshold_multiplier

    should_retrain = drift_trigger or perf_trigger

    reasons = []
    if drift_trigger:
        reasons.append(f"{n_drifted} features showed significant distribution drift (KS test).")
    if perf_trigger:
        reasons.append(f"Model RMSE increased by {(rmse_ratio - 1.0) * 100:.1f}%, exceeding {rmse_threshold_multiplier}x threshold.")

    if not reasons:
        reasons.append("Model within normal drift and error tolerance bounds. Retraining not required.")

    return {
        "should_retrain": should_retrain,
        "n_drifted_features": n_drifted,
        "rolling_rmse": round(rolling_rmse, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "rmse_ratio": round(rmse_ratio, 3),
        "reasons": reasons,
    }
