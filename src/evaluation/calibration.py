"""
CO-5: Model Evaluation and Selection - Probability Calibration Module.
Implements:
- Reliability diagrams (calibration curve: predicted confidence vs empirical accuracy)
- Expected Calibration Error (ECE)
- Platt Scaling (Logistic Sigmoid calibration)
- Isotonic Regression (Non-parametric isotonic calibration)
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


def compute_expected_calibration_error(
    y_true_binary: np.ndarray,
    y_prob_positive: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE):
        ECE = sum_{m=1}^M ( |B_m| / N ) * | acc(B_m) - conf(B_m) |
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    y_prob = np.asarray(y_prob_positive, dtype=float)
    N = len(y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    prob_true_list = []
    prob_pred_list = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / N) * np.abs(bin_acc - bin_conf)

            prob_true_list.append(bin_acc)
            prob_pred_list.append(bin_conf)
            bin_counts.append(bin_size)
        else:
            prob_true_list.append(0.0)
            prob_pred_list.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    return float(ece), np.array(prob_true_list), np.array(prob_pred_list), np.array(bin_counts)


def calibrate_classifier(
    base_model,
    X_train,
    y_train,
    X_test,
    y_test,
    method: str = "sigmoid",  # 'sigmoid' for Platt, 'isotonic' for Isotonic
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Applies Platt Scaling ('sigmoid') or Isotonic Regression ('isotonic') calibration.
    Calculates pre-calibration vs post-calibration ECE.
    """
    # Binary calibration demonstration on top performance tier (e.g. Good or Excellent: >= 2)
    y_train_bin = (y_train >= 2).astype(int)
    y_test_bin = (y_test >= 2).astype(int)

    # Base predictions
    if hasattr(base_model, "predict_proba"):
        base_probs_train = base_model.predict_proba(X_train)
        base_probs_test = base_model.predict_proba(X_test)
        # Probability of class 2 or 3
        prob_uncal_test = np.sum(base_probs_test[:, 2:], axis=1) if base_probs_test.shape[1] > 2 else base_probs_test[:, 1]
    else:
        prob_uncal_test = np.clip(base_model.predict(X_test) / 3.0, 0.0, 1.0)

    ece_before, acc_before, conf_before, _ = compute_expected_calibration_error(
        y_test_bin, prob_uncal_test, n_bins=n_bins
    )

    # Fit calibrated model
    calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv=3)
    calibrated.fit(X_train, y_train)

    cal_probs_test = calibrated.predict_proba(X_test)
    prob_cal_test = np.sum(cal_probs_test[:, 2:], axis=1) if cal_probs_test.shape[1] > 2 else cal_probs_test[:, 1]

    ece_after, acc_after, conf_after, _ = compute_expected_calibration_error(
        y_test_bin, prob_cal_test, n_bins=n_bins
    )

    return {
        "method": "Platt Scaling (Sigmoid)" if method == "sigmoid" else "Isotonic Regression",
        "ece_before": ece_before,
        "ece_after": ece_after,
        "prob_uncal_test": prob_uncal_test,
        "prob_cal_test": prob_cal_test,
        "acc_before": acc_before.tolist(),
        "conf_before": conf_before.tolist(),
        "acc_after": acc_after.tolist(),
        "conf_after": conf_after.tolist(),
        "calibrated_model": calibrated,
    }
