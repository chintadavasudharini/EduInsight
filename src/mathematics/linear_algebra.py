"""
CO-1: Mathematical Foundations for ML - Linear Algebra Module.
Demonstrates vectors, matrices, dot products, norms, covariance decomposition,
eigenvalues/eigenvectors, and Singular Value Decomposition (SVD) on student data.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_vector_operations(v1: np.ndarray, v2: np.ndarray) -> Dict[str, Any]:
    """
    Computes vector metrics between two student profiles:
    - Dot product
    - L1, L2 (Euclidean), and Linf norms
    - Cosine similarity: (v1 . v2) / (||v1|| * ||v2||)
    - Euclidean distance
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    dot_prod = float(np.dot(v1, v2))
    norm_l1_v1 = float(np.linalg.norm(v1, ord=1))
    norm_l2_v1 = float(np.linalg.norm(v1, ord=2))
    norm_linf_v1 = float(np.linalg.norm(v1, ord=np.inf))

    norm_l2_v2 = float(np.linalg.norm(v2, ord=2))
    cos_sim = dot_prod / (norm_l2_v1 * norm_l2_v2 + 1e-12)
    euclidean_dist = float(np.linalg.norm(v1 - v2))

    return {
        "v1": v1.tolist(),
        "v2": v2.tolist(),
        "dot_product": dot_prod,
        "norm_l1_v1": norm_l1_v1,
        "norm_l2_v1": norm_l2_v1,
        "norm_linf_v1": norm_linf_v1,
        "norm_l2_v2": norm_l2_v2,
        "cosine_similarity": float(cos_sim),
        "euclidean_distance": euclidean_dist,
    }


def compute_matrix_properties(matrix: np.ndarray) -> Dict[str, Any]:
    """
    Computes fundamental matrix properties:
    - Shape, Rank, Trace, Determinant (if square), Condition Number
    """
    mat = np.asarray(matrix, dtype=float)
    rank = int(np.linalg.matrix_rank(mat))
    trace_val = float(np.trace(mat)) if mat.ndim == 2 and mat.shape[0] == mat.shape[1] else None
    
    det_val = None
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        try:
            det_val = float(np.linalg.det(mat))
        except Exception:
            det_val = None

    cond_num = float(np.linalg.cond(mat)) if mat.ndim == 2 else None

    return {
        "shape": list(mat.shape),
        "rank": rank,
        "trace": trace_val,
        "determinant": det_val,
        "condition_number": cond_num,
    }


def compute_covariance_and_eigendecomposition(
    X: np.ndarray, feature_names: list
) -> Dict[str, Any]:
    """
    Computes:
    1. Sample Covariance Matrix Sigma = 1/(N-1) (X - mu)^T (X - mu)
    2. Eigendecomposition: Sigma * v_i = lambda_i * v_i
    3. Sorted eigenvalues, eigenvectors, and explained variance ratios
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape

    # Mean center
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Unbiased sample covariance matrix (ddof=1)
    cov_matrix = np.cov(X, rowvar=False)

    # Eigendecomposition of symmetric covariance matrix
    # eigh guarantees real eigenvalues for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    sorted_evals = eigenvalues[idx]
    sorted_evecs = eigenvectors[:, idx]

    total_variance = np.sum(sorted_evals)
    explained_variance_ratio = sorted_evals / (total_variance + 1e-12)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    return {
        "mean_vector": mu.tolist(),
        "covariance_matrix": cov_matrix.tolist(),
        "eigenvalues": sorted_evals.tolist(),
        "eigenvectors": sorted_evecs.tolist(),
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "cumulative_variance_ratio": cumulative_variance_ratio.tolist(),
        "feature_names": feature_names,
    }


def compute_svd(X: np.ndarray, k: int = 2) -> Dict[str, Any]:
    """
    Performs Singular Value Decomposition:
        X = U * Sigma * V^T
    Computes rank-k truncation and Frobenius reconstruction error.
    """
    X = np.asarray(X, dtype=float)
    # Mean-center for PCA-style SVD
    X_centered = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Truncated reconstruction
    k = min(k, len(S))
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])
    X_reconstructed = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))

    # Frobenius norm error
    frobenius_error = float(np.linalg.norm(X_centered - X_reconstructed, ord="fro"))
    relative_error = frobenius_error / (float(np.linalg.norm(X_centered, ord="fro")) + 1e-12)

    return {
        "singular_values": S.tolist(),
        "singular_values_top_k": S[:k].tolist(),
        "U_shape": list(U.shape),
        "Vt_shape": list(Vt.shape),
        "k": k,
        "frobenius_error": frobenius_error,
        "relative_error": relative_error,
        "components": Vt[:k, :].tolist(),
    }
