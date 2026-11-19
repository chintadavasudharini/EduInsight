"""
CO-3: Unsupervised Learning - Student Clustering Module.
Implements:
- K-Means Clustering (inertia elbow & silhouette)
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Hierarchical (Agglomerative) Clustering
- Gaussian Mixture Models (GMM with BIC/AIC)
Includes feature scaling, cluster profiling, and academic disclaimer.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.utils.config import FEATURE_NAMES, DEFAULT_RANDOM_SEED


def perform_kmeans_clustering(
    df: pd.DataFrame,
    k: int = 3,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Fits K-Means on scaled student features.
    Computes silhouette score, Davies-Bouldin index, and cluster centroids.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = float(silhouette_score(X_scaled, labels)) if k > 1 and len(set(labels)) > 1 else 0.0
    db_idx = float(davies_bouldin_score(X_scaled, labels)) if k > 1 and len(set(labels)) > 1 else 0.0
    ch_idx = float(calinski_harabasz_score(X_scaled, labels)) if k > 1 and len(set(labels)) > 1 else 0.0

    # Cluster profiles in original scale
    df_profile = df[FEATURE_NAMES].copy()
    df_profile["Cluster"] = labels
    profiles = df_profile.groupby("Cluster").mean().round(2)

    return {
        "model": kmeans,
        "scaler": scaler,
        "labels": labels,
        "k": k,
        "inertia": float(kmeans.inertia_),
        "silhouette_score": sil,
        "davies_bouldin_index": db_idx,
        "calinski_harabasz_score": ch_idx,
        "profiles": profiles,
        "centroids_scaled": kmeans.cluster_centers_.tolist(),
    }


def perform_dbscan_clustering(
    df: pd.DataFrame,
    eps: float = 0.8,
    min_samples: int = 8,
) -> Dict[str, Any]:
    """
    Fits DBSCAN on scaled student features to detect arbitrary shape clusters and noise.
    Points labelled -1 are considered density outliers (noise).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    sil = (
        float(silhouette_score(X_scaled, labels))
        if n_clusters > 1 and len(np.unique(labels[labels != -1])) > 1
        else 0.0
    )

    return {
        "model": dbscan,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "silhouette_score": sil,
    }


def perform_hierarchical_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    linkage: str = "ward",
) -> Dict[str, Any]:
    """
    Fits Agglomerative Hierarchical Clustering using Ward linkage.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X_scaled)

    sil = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 else 0.0

    return {
        "model": agg,
        "labels": labels,
        "n_clusters": n_clusters,
        "silhouette_score": sil,
    }


def perform_gmm_clustering(
    df: pd.DataFrame,
    n_components: int = 3,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Fits Gaussian Mixture Models (GMM) with full covariance matrices.
    Returns cluster soft responsibilities, AIC, and BIC.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=seed)
    labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    bic_val = float(gmm.bic(X_scaled))
    aic_val = float(gmm.aic(X_scaled))
    sil = float(silhouette_score(X_scaled, labels)) if n_components > 1 else 0.0

    return {
        "model": gmm,
        "labels": labels,
        "probabilities": probs,
        "n_components": n_components,
        "bic": bic_val,
        "aic": aic_val,
        "silhouette_score": sil,
    }
