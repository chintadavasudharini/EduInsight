"""
Page 7: Dimensionality Reduction & Manifold Visualization (CO-3).
Implements:
- Principal Component Analysis (PCA) with Scree Plot and Feature Loadings
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- UMAP projection (with clear fallback if uninstalled)
- Representation Learning (Autoencoder Embeddings)
- Educational Text Embedding & Cosine Similarity Demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import FEATURE_NAMES, TARGET_CLASSIFICATION, CATEGORY_LABELS
from src.data.loader import prepare_dataset
from src.unsupervised.dim_reduction import (
    compute_pca_projection,
    compute_tsne_projection,
    compute_umap_projection,
)
from src.unsupervised.embeddings import (
    demonstrate_tabular_autoencoder_latents,
    demonstrate_educational_text_embeddings,
)


def render_page():
    st.markdown('<div class="main-header">🌌 Dimensionality Reduction & Embeddings (CO-3)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Compressing 5D student academic feature space into 2D manifolds & representation embeddings.</div>',
        unsafe_allow_html=True,
    )

    df = prepare_dataset()

    tab_pca, tab_tsne, tab_umap, tab_autoencoder, tab_text_emb = st.tabs([
        "📐 Principal Component Analysis (PCA)",
        "🌀 t-SNE Manifold",
        "🗺️ UMAP Embedding",
        "🧠 Autoencoder Latents",
        "📚 Educational Text Embedding Demo",
    ])

    with tab_pca:
        st.subheader("Principal Component Analysis (Orthogonal Variance Maximization)")
        pca_res = compute_pca_projection(df, n_components=2)
        coords = pca_res["coords"]

        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            cat_palette = {"Needs Support": "#EF4444", "Average": "#F59E0B", "Good": "#3B82F6", "Excellent": "#10B981"}
            for cat in CATEGORY_LABELS:
                mask = (df[TARGET_CLASSIFICATION] == cat)
                ax.scatter(coords[mask, 0], coords[mask, 1], label=cat, color=cat_palette.get(cat, "gray"), alpha=0.6, edgecolors="none")
            ax.set_xlabel(f"PC1 ({pca_res['explained_variance_ratio'][0]*100:.1f}% Variance)")
            ax.set_ylabel(f"PC2 ({pca_res['explained_variance_ratio'][1]*100:.1f}% Variance)")
            ax.set_title("2D PCA Projection of Student Space", fontsize=11, fontweight="bold")
            ax.legend(title="Academic Tier")
            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("##### Scree Plot (Cumulative Variance)")
            fig_s, ax_s = plt.subplots(figsize=(5, 3.8))
            pcs = [f"PC{i+1}" for i in range(len(pca_res["scree_variance_ratio"]))]
            ax_s.bar(pcs, [v*100 for v in pca_res["scree_variance_ratio"]], color="#6366F1", label="Individual %")
            ax_s.plot(pcs, [v*100 for v in pca_res["scree_cumulative_ratio"]], "ro-", label="Cumulative %", linewidth=2)
            ax_s.set_ylabel("Variance Explained (%)")
            ax_s.set_title("Scree Plot", fontsize=10, fontweight="bold")
            ax_s.legend()
            ax_s.grid(axis="y", linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

        st.markdown("##### Principal Component Feature Loadings (Eigenvector Weights)")
        st.dataframe(pca_res["loadings"].round(3).T, use_container_width=True)

    with tab_tsne:
        st.subheader("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        c1, c2 = st.columns([1, 2])
        with c1:
            perp = st.slider("Perplexity:", 5.0, 50.0, 30.0, 5.0, help="Balances local vs global manifold preservation")
            lr_val = st.slider("Learning Rate:", 50.0, 400.0, 200.0, 25.0)

            with st.spinner("Computing t-SNE optimization (1000 iterations)..."):
                tsne_res = compute_tsne_projection(df, perplexity=perp, learning_rate=lr_val)
                t_coords = tsne_res["coords"]

        with c2:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for cat in CATEGORY_LABELS:
                mask = (df[TARGET_CLASSIFICATION] == cat)
                ax.scatter(t_coords[mask, 0], t_coords[mask, 1], label=cat, color=cat_palette.get(cat, "gray"), alpha=0.6, edgecolors="none")
            ax.set_title(f"t-SNE Projection (Perplexity={perp})", fontsize=11, fontweight="bold")
            ax.legend(title="Academic Tier")
            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_umap:
        st.subheader("Uniform Manifold Approximation and Projection (UMAP)")
        umap_res = compute_umap_projection(df)
        u_coords = umap_res["coords"]

        if not umap_res.get("available", True):
            st.info(f"ℹ️ {umap_res.get('message', 'UMAP fallback active.')}")

        fig, ax = plt.subplots(figsize=(6, 4))
        for cat in CATEGORY_LABELS:
            mask = (df[TARGET_CLASSIFICATION] == cat)
            ax.scatter(u_coords[mask, 0], u_coords[mask, 1], label=cat, color=cat_palette.get(cat, "gray"), alpha=0.6, edgecolors="none")
        ax.set_title("Manifold Projection (UMAP / Fallback)", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_autoencoder:
        st.subheader("Representation Learning via Neural Autoencoder Bottleneck")
        st.markdown(
            "Shallow PyTorch Autoencoder ($5 \\to 4 \\to 2 \\to 4 \\to 5$) learning a compressed "
            "non-linear 2D representation directly from reconstruction objective."
        )

        with st.spinner("Training Autoencoder for 40 epochs..."):
            ae_res = demonstrate_tabular_autoencoder_latents(df, latent_dim=2, epochs=40)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Final Reconstruction Loss (MSE)", f"{ae_res['final_reconstruction_loss']:.4f}")

            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(ae_res["loss_history"], color="#EF4444", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Reconstruction MSE Loss")
            ax.set_title("Autoencoder Training Curve", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            z = ae_res["latent_embeddings"]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for cat in CATEGORY_LABELS:
                mask = (df[TARGET_CLASSIFICATION] == cat)
                ax.scatter(z[mask, 0], z[mask, 1], label=cat, color=cat_palette.get(cat, "gray"), alpha=0.6, edgecolors="none")
            ax.set_title("Learned Latent Space $z \\in \\mathbb{R}^2$", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_text_emb:
        st.subheader("Educational NLP Demonstration: Word & Document Embeddings")
        text_demo = demonstrate_educational_text_embeddings()

        st.info(text_demo["explanation"])

        st.markdown("##### Academic Feedback Corpus & Cosine Similarity Heatmap")
        sim_df = pd.DataFrame(
            text_demo["similarity_matrix"],
            index=[f"Doc {i+1}" for i in range(len(text_demo["corpus"]))],
            columns=[f"Doc {i+1}" for i in range(len(text_demo["corpus"]))],
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(sim_df, annot=True, cmap="YlGnBu", cbar=True, ax=ax)
        plt.title("Semantic Document Cosine Similarity Matrix", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📄 View Educational Corpus Sentences"):
            for idx, sent in enumerate(text_demo["corpus"]):
                st.markdown(f"**Doc {idx+1}:** *\"{sent}\"*")


if __name__ == "__main__":
    render_page()
