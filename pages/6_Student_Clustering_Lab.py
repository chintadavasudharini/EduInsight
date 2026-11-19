"""
Page 6: Student Clustering Lab (CO-3: Unsupervised Learning).
Discovers underlying academic learning patterns and behavioral groupings:
- K-Means Clustering (with Elbow & Silhouette analysis)
- DBSCAN (Density-based spatial clustering & noise detection)
- Agglomerative Hierarchical Clustering
- Gaussian Mixture Models (GMM with BIC/AIC)
Includes cluster profiling and academic disclaimer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURE_NAMES, DEFAULT_RANDOM_SEED
from src.data.loader import prepare_dataset
from src.unsupervised.clustering import (
    perform_kmeans_clustering,
    perform_dbscan_clustering,
    perform_hierarchical_clustering,
    perform_gmm_clustering,
)


def render_page():
    st.markdown('<div class="main-header">👥 Student Clustering & Learning Patterns (CO-3)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Unsupervised discovery of behavioral cohorts, study habit archetypes, and density clusters.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ **Academic Disclaimer:** Clusters represent statistical groupings based on study habits, attendance, and sleep. "
        "They do not define innate student ability or fixed academic potential."
    )

    df = prepare_dataset()

    tab_kmeans, tab_dbscan, tab_hierarchical, tab_gmm = st.tabs([
        "📍 K-Means Clustering",
        "🌐 DBSCAN (Density-Based)",
        "🌳 Hierarchical Clustering",
        "🌊 Gaussian Mixture Models (GMM)",
    ])

    with tab_kmeans:
        st.subheader("K-Means Clustering & Centroid Profiling")

        c1, c2 = st.columns([1, 2])
        with c1:
            k_val = st.slider("Select Number of Clusters (K):", 2, 7, 3, 1)
            kmeans_res = perform_kmeans_clustering(df, k=k_val)

            st.metric("Silhouette Score", f"{kmeans_res['silhouette_score']:.3f}", help="Higher is better (>0.25 indicates structure)")
            st.metric("Davies-Bouldin Index", f"{kmeans_res['davies_bouldin_index']:.3f}", help="Lower is better")
            st.metric("Inertia (WCSS)", f"{kmeans_res['inertia']:,.0f}")

        with c2:
            # 2D PCA projection for visualization
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(kmeans_res["scaler"].transform(df[FEATURE_NAMES]))

            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=kmeans_res["labels"],
                cmap="viridis",
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
            ax.set_title(f"K-Means (K={k_val}) Projected onto 2D PCA Space", fontsize=10, fontweight="bold")
            plt.colorbar(scatter, ax=ax, label="Cluster ID")
            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("##### 📊 Discovered Cluster Profiles (Cohort Averages)")
        st.dataframe(kmeans_res["profiles"], use_container_width=True)

        # Elbow curve demonstration
        with st.expander("📈 View Elbow Method & Silhouette Curve across K=2 to 8"):
            k_range = list(range(2, 9))
            inertias = []
            sils = []
            for k_i in k_range:
                res_i = perform_kmeans_clustering(df, k=k_i)
                inertias.append(res_i["inertia"])
                sils.append(res_i["silhouette_score"])

            fig, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(10, 3.5))
            ax_e1.plot(k_range, inertias, "bo-", linewidth=2)
            ax_e1.set_xlabel("Number of Clusters K")
            ax_e1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
            ax_e1.set_title("Elbow Method for Optimal K", fontsize=10, fontweight="bold")
            ax_e1.grid(True, linestyle=":", alpha=0.5)

            ax_e2.plot(k_range, sils, "go-", linewidth=2)
            ax_e2.set_xlabel("Number of Clusters K")
            ax_e2.set_ylabel("Silhouette Score")
            ax_e2.set_title("Silhouette Score vs K", fontsize=10, fontweight="bold")
            ax_e2.grid(True, linestyle=":", alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_dbscan:
        st.subheader("DBSCAN Density-Based Clustering")
        c1, c2 = st.columns(2)
        with c1:
            eps_val = st.slider("Epsilon (neighborhood radius):", 0.3, 2.0, 0.85, 0.05)
        with c2:
            min_pts = st.slider("Min Samples (core points):", 3, 20, 8, 1)

        db_res = perform_dbscan_clustering(df, eps=eps_val, min_samples=min_pts)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Clusters Formed", db_res["n_clusters"])
        with m2:
            st.metric("Noise Points (-1)", db_res["n_noise_points"])
        with m3:
            st.metric("Silhouette Score", f"{db_res['silhouette_score']:.3f}")

        # 2D projection
        pca = PCA(n_components=2, random_state=42)
        scaler = StandardScaler()
        coords = pca.fit_transform(scaler.fit_transform(df[FEATURE_NAMES]))

        fig, ax = plt.subplots(figsize=(7, 4))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=db_res["labels"],
            cmap="tab10",
            alpha=0.6,
        )
        ax.set_title(f"DBSCAN Clusters & Outliers (eps={eps_val}, min_samples={min_pts})", fontsize=10, fontweight="bold")
        plt.colorbar(scatter, ax=ax, label="Cluster (-1 = Noise)")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_hierarchical:
        st.subheader("Agglomerative Hierarchical Clustering")
        n_clusters_h = st.slider("Select Hierarchical Clusters:", 2, 6, 3, 1)
        h_res = perform_hierarchical_clustering(df, n_clusters=n_clusters_h)

        st.metric("Silhouette Score", f"{h_res['silhouette_score']:.3f}")

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(coords[:, 0], coords[:, 1], c=h_res["labels"], cmap="coolwarm", alpha=0.6)
        ax.set_title(f"Agglomerative Clustering (K={n_clusters_h}, Ward Linkage)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_gmm:
        st.subheader("Gaussian Mixture Models (Soft Clustering)")
        n_comp = st.slider("Select GMM Components:", 2, 6, 3, 1)
        gmm_res = perform_gmm_clustering(df, n_components=n_comp)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Silhouette Score", f"{gmm_res['silhouette_score']:.3f}")
        with c2:
            st.metric("BIC (Bayesian Information Criterion)", f"{gmm_res['bic']:,.0f}")
        with c3:
            st.metric("AIC (Akaike Information Criterion)", f"{gmm_res['aic']:,.0f}")

        # Show probability assignment table preview
        probs_df = pd.DataFrame(
            gmm_res["probabilities"],
            columns=[f"Component {i+1} Prob" for i in range(n_comp)]
        ).round(3)
        st.markdown("##### Soft Membership Responsibilities (First 10 Students)")
        st.dataframe(probs_df.head(10), use_container_width=True)


if __name__ == "__main__":
    render_page()
