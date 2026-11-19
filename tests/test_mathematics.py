"""Unit tests for CO-1 Applied Mathematics module."""

import numpy as np
import pytest

from src.mathematics.linear_algebra import (
    compute_vector_operations,
    compute_matrix_properties,
    compute_covariance_and_eigendecomposition,
    compute_svd,
)
from src.mathematics.probability_stats import (
    compute_descriptive_statistics,
    compute_expectation_and_variance_proof,
    compute_bayes_theorem_student_case,
    compute_confidence_intervals_and_ttest,
    compute_mle_vs_map,
)
from src.mathematics.calculus_optimization import (
    compute_gradient_and_hessian,
    verify_convexity_chord_inequality,
    simulate_gradient_descent_1d_or_2d,
    demonstrate_lagrange_and_kkt,
)


def test_vector_operations():
    v1 = np.array([3.0, 4.0])
    v2 = np.array([3.0, 4.0])
    res = compute_vector_operations(v1, v2)
    assert np.isclose(res["dot_product"], 25.0)
    assert np.isclose(res["norm_l2_v1"], 5.0)
    assert np.isclose(res["cosine_similarity"], 1.0)
    assert np.isclose(res["euclidean_distance"], 0.0)


def test_covariance_and_eigendecomposition():
    X = np.random.normal(0, 1, size=(100, 4))
    res = compute_covariance_and_eigendecomposition(X, ["A", "B", "C", "D"])
    assert len(res["eigenvalues"]) == 4
    # Eigenvalues must be in descending order
    assert all(res["eigenvalues"][i] >= res["eigenvalues"][i+1] for i in range(3))
    assert np.isclose(np.sum(res["explained_variance_ratio"]), 1.0)


def test_svd_reconstruction():
    X = np.random.normal(0, 1, size=(50, 4))
    res = compute_svd(X, k=4)
    # Full rank SVD reconstruction error should be near zero
    assert res["relative_error"] < 1e-4


def test_variance_identity_proof():
    vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    res = compute_expectation_and_variance_proof(vals)
    assert res["identity_verified"] is True
    assert np.isclose(res["Var_identity"], res["Var_direct"])


def test_bayes_theorem():
    study = np.array([2.0, 3.0, 7.0, 8.0, 9.0])
    scores = np.array([45.0, 55.0, 82.0, 88.0, 92.0])
    res = compute_bayes_theorem_student_case(study, scores, high_study_threshold=6.0, excellent_threshold=80.0)
    assert res["formula_matched"] is True


def test_gradient_hessian_convexity():
    X = np.random.normal(0, 1, size=(40, 3))
    y = np.random.normal(0, 1, size=40)
    w = np.zeros(3)
    res = compute_gradient_and_hessian(w, X, y, lambda_reg=0.1)
    assert len(res["gradient"]) == 3
    assert np.array(res["hessian"]).shape == (3, 3)
    assert res["is_positive_definite"] is True


def test_kkt_conditions():
    res = demonstrate_lagrange_and_kkt()
    assert res["kkt_conditions"]["primal_feasibility_g_le_0"] is True
    assert res["kkt_conditions"]["dual_feasibility_lambda_ge_0"] is True
    assert res["kkt_conditions"]["complementary_slackness_lambda_g_eq_0"] is True
