"""
Page 3: Dataset Explorer.
Provides interactive inspection, statistical summaries, correlation heatmaps,
distribution plots, and reproducible synthetic dataset regeneration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import (
    FEATURE_NAMES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    CATEGORY_LABELS,
    DEFAULT_RANDOM_SEED,
)
from src.data.loader import prepare_dataset
from src.data.generator import generate_student_dataset


def render_page():
    st.markdown('<div class="main-header">📋 Dataset Explorer & Cohort Statistics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive inspection of reproducible student academic records and feature distributions.</div>',
        unsafe_allow_html=True,
    )

    df = prepare_dataset()

    # Top Control Bar
    col1, col2, col3 = st.columns([1.5, 1.2, 1.3])
    with col1:
        selected_tier = st.multiselect(
            "Filter by Performance Category:",
            options=CATEGORY_LABELS,
            default=CATEGORY_LABELS,
        )
    with col2:
        selected_campus = st.multiselect(
            "Filter by Campus Subgroup:",
            options=list(df["study_group"].unique()) if "study_group" in df.columns else [],
            default=list(df["study_group"].unique()) if "study_group" in df.columns else [],
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_bytes,
            file_name="student_performance_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Apply filters
    filtered_df = df[df[TARGET_CLASSIFICATION].isin(selected_tier)]
    if selected_campus and "study_group" in df.columns:
        filtered_df = filtered_df[filtered_df["study_group"].isin(selected_campus)]

    st.caption(f"Displaying **{len(filtered_df):,}** of **{len(df):,}** student records.")

    # Data Preview & Summary Statistics Tabs
    tab_preview, tab_stats, tab_correlations, tab_distributions, tab_generator = st.tabs([
        "📄 Data Preview",
        "📈 Statistical Summary",
        "🔥 Correlation Heatmap",
        "📊 Distributions & Boxplots",
        "⚙️ Dataset Generator Settings",
    ])

    with tab_preview:
        st.dataframe(filtered_df.head(100), use_container_width=True, height=350)

    with tab_stats:
        num_cols = FEATURE_NAMES + [TARGET_REGRESSION]
        desc_df = filtered_df[num_cols].describe().round(2).T
        desc_df["skewness"] = filtered_df[num_cols].skew().round(2)
        desc_df["kurtosis"] = filtered_df[num_cols].kurtosis().round(2)
        st.dataframe(desc_df, use_container_width=True)

    with tab_correlations:
        st.markdown("##### Pearson Correlation Matrix")
        corr = filtered_df[FEATURE_NAMES + [TARGET_REGRESSION]].corr().round(3)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(
            corr,
            annot=True,
            cmap="Blues",
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        plt.title("Academic Feature Correlation Matrix", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_distributions:
        selected_feat = st.selectbox("Select Feature for Distribution Plot:", FEATURE_NAMES + [TARGET_REGRESSION])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

        # Histogram with KDE
        sns.histplot(filtered_df[selected_feat], kde=True, color="#3B82F6", ax=ax1, edgecolor="#1E293B")
        ax1.set_title(f"Histogram & KDE: {selected_feat}", fontsize=10, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Boxplot across Performance Category
        sns.boxplot(
            data=filtered_df,
            x=TARGET_CLASSIFICATION,
            y=selected_feat,
            order=CATEGORY_LABELS,
            palette=["#EF4444", "#F59E0B", "#3B82F6", "#10B981"],
            ax=ax2,
        )
        ax2.set_title(f"{selected_feat} Across Performance Tiers", fontsize=10, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=15)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_generator:
        st.markdown("##### 🧪 Synthetic Dataset Configuration (Reproducibility Controls)")
        st.info(
            "NOTE: This synthetic generator uses NumPy Gaussian and Beta distributions with controlled interaction "
            "terms to reproduce realistic academic metrics without using real private student data."
        )

        with st.form("regen_form"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                n_new = st.number_input("Cohort Size (N)", min_value=1200, max_value=5000, value=1500, step=100)
            with col_b:
                seed_new = st.number_input("Random Seed", min_value=1, max_value=9999, value=DEFAULT_RANDOM_SEED)
            with col_c:
                noise_new = st.slider("Gaussian Noise Level (sigma)", 1.0, 8.0, 3.5, 0.5)

            regen_submit = st.form_submit_button("🔄 Regenerate & Save Dataset", use_container_width=True)

            if regen_submit:
                new_df = generate_student_dataset(n_samples=n_new, seed=seed_new, noise_level=noise_new)
                from src.utils.config import RAW_DATA_PATH
                new_df.to_csv(RAW_DATA_PATH, index=False)
                st.success(f"Successfully regenerated {len(new_df)} records (Seed={seed_new}, Noise={noise_new}).")
                st.rerun()


if __name__ == "__main__":
    render_page()
