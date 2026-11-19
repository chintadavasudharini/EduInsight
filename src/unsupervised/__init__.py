"""Unsupervised Learning, Dimensionality Reduction, and Anomaly Detection (CO-3)."""

from .clustering import (
    perform_kmeans_clustering,
    perform_dbscan_clustering,
    perform_hierarchical_clustering,
    perform_gmm_clustering,
)
from .dim_reduction import (
    compute_pca_projection,
    compute_tsne_projection,
    compute_umap_projection,
)
from .anomaly import (
    detect_anomalies_isolation_forest,
    detect_anomalies_one_class_svm,
    explain_student_anomaly,
)
from .embeddings import (
    demonstrate_tabular_autoencoder_latents,
    demonstrate_educational_text_embeddings,
)

__all__ = [
    "perform_kmeans_clustering",
    "perform_dbscan_clustering",
    "perform_hierarchical_clustering",
    "perform_gmm_clustering",
    "compute_pca_projection",
    "compute_tsne_projection",
    "compute_umap_projection",
    "detect_anomalies_isolation_forest",
    "detect_anomalies_one_class_svm",
    "explain_student_anomaly",
    "demonstrate_tabular_autoencoder_latents",
    "demonstrate_educational_text_embeddings",
]
