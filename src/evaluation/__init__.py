"""Model Evaluation and Selection (CO-5)."""

from .metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    analyze_prediction_errors,
)
from .calibration import (
    compute_expected_calibration_error,
    calibrate_classifier,
)
from .statistical_tests import (
    run_paired_ttest_cv,
    run_mcnemar_test,
    compute_bootstrap_confidence_interval,
)
from .cross_validation import (
    demonstrate_stratified_cv,
    demonstrate_group_cv,
    demonstrate_timeseries_cv,
)
from .fairness import compute_fairness_metrics

__all__ = [
    "compute_regression_metrics",
    "compute_classification_metrics",
    "analyze_prediction_errors",
    "compute_expected_calibration_error",
    "calibrate_classifier",
    "run_paired_ttest_cv",
    "run_mcnemar_test",
    "compute_bootstrap_confidence_interval",
    "demonstrate_stratified_cv",
    "demonstrate_group_cv",
    "demonstrate_timeseries_cv",
    "compute_fairness_metrics",
]
