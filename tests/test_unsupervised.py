"""Unit tests for CO-3 Unsupervised Learning (Clustering, PCA, Anomaly)."""

import pytest
import numpy as np

from src.data.generator import generate_student_dataset
from src.unsupervised.clustering import (
    perform_kmeans_clustering,
    perform_dbscan_clustering,
    perform_hierarchical_clustering,
    perform_gmm_clustering,
)
from src.unsupervised.dim_reduction import compute_pca_projection, compute_tsne_projection
from src.unsupervised.anomaly import detect_anomalies_isolation_forest, detect_anomalies_one_class_svm
from src.unsupervised.embeddings import demonstrate_tabular_autoencoder_latents, demonstrate_educational_text_embeddings


@pytest.fixture
def cohort():
    return generate_student_dataset(n_samples=150, seed=42)


def test_kmeans_clustering(cohort):
    res = perform_kmeans_clustering(cohort, k=3)
    assert len(res["labels"]) == len(cohort)
    assert len(set(res["labels"])) == 3
    assert res["silhouette_score"] > 0.10


def test_dbscan_clustering(cohort):
    res = perform_dbscan_clustering(cohort, eps=1.0, min_samples=5)
    assert len(res["labels"]) == len(cohort)
    assert "n_clusters" in res


def test_pca_projection(cohort):
    res = compute_pca_projection(cohort, n_components=2)
    assert res["coords"].shape == (len(cohort), 2)
    assert len(res["explained_variance_ratio"]) == 2
    assert res["total_variance_explained"] > 0.40


def test_isolation_forest_anomalies(cohort):
    res = detect_anomalies_isolation_forest(cohort, contamination=0.08)
    assert res["n_anomalies"] > 0
    assert len(res["anomaly_records"]) == res["n_anomalies"]


def test_tabular_autoencoder(cohort):
    res = demonstrate_tabular_autoencoder_latents(cohort, latent_dim=2, epochs=5)
    assert res["latent_embeddings"].shape == (len(cohort), 2)


def test_text_embeddings_demo():
    res = demonstrate_educational_text_embeddings()
    assert len(res["corpus"]) >= 5
    assert np.array(res["similarity_matrix"]).shape == (len(res["corpus"]), len(res["corpus"]))
