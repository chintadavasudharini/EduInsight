"""Applied Mathematics Foundations for ML (CO-1)."""

from .linear_algebra import (
    compute_vector_operations,
    compute_matrix_properties,
    compute_covariance_and_eigendecomposition,
    compute_svd,
)
from .probability_stats import (
    compute_descriptive_statistics,
    compute_expectation_and_variance_proof,
    compute_bayes_theorem_student_case,
    compute_confidence_intervals_and_ttest,
    compute_mle_vs_map,
)
from .calculus_optimization import (
    compute_gradient_and_hessian,
    verify_convexity_chord_inequality,
    simulate_gradient_descent_1d_or_2d,
    demonstrate_lagrange_and_kkt,
)

__all__ = [
    "compute_vector_operations",
    "compute_matrix_properties",
    "compute_covariance_and_eigendecomposition",
    "compute_svd",
    "compute_descriptive_statistics",
    "compute_expectation_and_variance_proof",
    "compute_bayes_theorem_student_case",
    "compute_confidence_intervals_and_ttest",
    "compute_mle_vs_map",
    "compute_gradient_and_hessian",
    "verify_convexity_chord_inequality",
    "simulate_gradient_descent_1d_or_2d",
    "demonstrate_lagrange_and_kkt",
]
