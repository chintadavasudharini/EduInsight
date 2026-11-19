"""
CO-5: Model Evaluation and Selection - Metrics and Error Taxonomy Module.
Implements:
- Regression Metrics: MAE, RMSE, R-squared
- Classification Metrics: Accuracy, Macro/Micro Precision, Recall, F1-score,
  ROC-AUC, PR-AUC, Matthews Correlation Coefficient (MCC)
- Error Analysis Taxonomy: Breakdown of False Positives, False Negatives,
  boundary confusion, and residual distributions.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from src.utils.config import CATEGORY_LABELS


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard regression performance metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # Mean Absolute Percentage Error (avoid division by 0)
    non_zero = y_t > 1.0
    mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero]))) * 100.0 if np.any(non_zero) else 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics.
    Handles multiclass ROC-AUC and PR-AUC.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    class_labels = labels or CATEGORY_LABELS

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_p))

    roc_auc = float("nan")
    pr_auc_macro = float("nan")

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_t, y_prob, multi_class="ovr", average="macro"))
            # PR-AUC via one-vs-rest binarization
            pr_aucs = []
            for i in range(len(class_labels)):
                y_bin = (y_t == i).astype(int)
                if np.sum(y_bin) > 0:
                    p_curve, r_curve, _ = precision_recall_curve(y_bin, y_prob[:, i])
                    pr_aucs.append(auc(r_curve, p_curve))
            if pr_aucs:
                pr_auc_macro = float(np.mean(pr_aucs))
        except Exception:
            pass

    cm = confusion_matrix(y_t, y_p, labels=list(range(len(class_labels))))

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
        "PR_AUC": round(pr_auc_macro, 4) if not np.isnan(pr_auc_macro) else "N/A",
        "MCC": round(mcc, 4),
        "confusion_matrix": cm,
    }


def analyze_prediction_errors(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Error-analysis taxonomy: categorizes residuals into mild, moderate, and severe errors.
    """
    df_err = df.copy()
    df_err["residual"] = df_err[y_pred_col] - df_err[y_true_col]
    df_err["abs_error"] = np.abs(df_err["residual"])

    # Categorize error magnitude
    bins = [0, 5, 12, np.inf]
    labels = ["Accurate (<=5)", "Moderate Error (5-12)", "Severe Error (>12)"]
    df_err["error_severity"] = pd.cut(df_err["abs_error"], bins=bins, labels=labels, right=False)

    severity_counts = df_err["error_severity"].value_counts().to_dict()

    worst_errors = df_err.sort_values(by="abs_error", ascending=False).head(10)[
        feature_cols + [y_true_col, y_pred_col, "residual", "abs_error"]
    ]

    return {
        "severity_distribution": severity_counts,
        "worst_predictions": worst_errors,
        "mean_absolute_error": float(df_err["abs_error"].mean()),
        "max_absolute_error": float(df_err["abs_error"].max()),
    }
