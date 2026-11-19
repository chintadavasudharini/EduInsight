"""
CO-3: Unsupervised Learning - Dimensionality Reduction Module.
Implements:
- Principal Component Analysis (PCA) with Scree Plot analysis
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection (UMAP) as optional dependency
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURE_NAMES, DEFAULT_RANDOM_SEED


def compute_pca_projection(
    df: pd.DataFrame,
    n_components: int = 2,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Fits PCA on standardized continuous features.
    Computes 2D/3D coordinates and explained variance ratios.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(X_scaled)

    # Full PCA for scree plot
    pca_full = PCA(n_components=len(FEATURE_NAMES), random_state=seed)
    pca_full.fit(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=FEATURE_NAMES,
    )

    return {
        "coords": coords,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "scree_variance_ratio": pca_full.explained_variance_ratio_.tolist(),
        "scree_cumulative_ratio": np.cumsum(pca_full.explained_variance_ratio_).tolist(),
        "loadings": loadings,
        "model": pca,
    }


def compute_tsne_projection(
    df: pd.DataFrame,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Computes 2D t-SNE embedding for non-linear manifold visualization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, len(df) / 4.0)),
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=seed,
        init="pca",
    )
    coords = tsne.fit_transform(X_scaled)

    return {
        "coords": coords,
        "perplexity": perplexity,
        "kl_divergence": float(tsne.kl_divergence_) if hasattr(tsne, "kl_divergence_") else None,
    }


def compute_umap_projection(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Computes 2D UMAP embedding if umap-learn is installed.
    Provides clear fallback if not installed.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES])

    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=seed,
        )
        coords = reducer.fit_transform(X_scaled)
        return {
            "available": True,
            "coords": coords,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }
    except ImportError:
        # Fallback to PCA with notification
        pca_fallback = PCA(n_components=2, random_state=seed).fit_transform(X_scaled)
        return {
            "available": False,
            "coords": pca_fallback,
            "message": "UMAP package is not installed. Showing PCA fallback projection.",
        }
