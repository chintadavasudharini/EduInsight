"""
Page 8: Anomaly Detection Lab (CO-3: Unsupervised Learning).
Implements outlier and anomaly detection to identify atypical academic records:
- Isolation Forest (Contamination thresholding)
- One-Class Support Vector Machine (Kernel density boundary)
Provides anomaly score histograms, interactive 2D projections, and plain-English explainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURE_NAMES, TARGET_REGRESSION, TARGET_CLASSIFICATION
from src.data.loader import prepare_dataset
from src.unsupervised.anomaly import (
    detect_anomalies_isolation_forest,
    detect_anomalies_one_class_svm,
    explain_student_anomaly,
)


def render_page():
    st.markdown('<div class="main-header">🚨 Anomaly Detection Lab (CO-3)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detect unusual student behaviors, exam score divergences, and severe burnout profiles.</div>',
        unsafe_allow_html=True,
    )

    df = prepare_dataset()
    pop_means = df[FEATURE_NAMES].mean()
    pop_stds = df[FEATURE_NAMES].std()

    tab_iso, tab_ocsvm, tab_inspect = st.tabs([
        "🌲 Isolation Forest",
        "🎯 One-Class SVM",
        "🔍 Anomaly Explanation Inspector",
    ])

    with tab_iso:
        st.subheader("Isolation Forest Outlier Detection")
        c1, c2 = st.columns([1, 2])
        with c1:
            contam = st.slider("Contamination Factor (% expected outliers):", 0.01, 0.15, 0.05, 0.01)
            iso_res = detect_anomalies_isolation_forest(df, contamination=contam)

            st.metric("Total Anomalies Flagged", iso_res["n_anomalies"])
            st.metric("Inlier Population", len(df) - iso_res["n_anomalies"])

        with c2:
            # 2D PCA scatter of inliers vs outliers
            pca = PCA(n_components=2, random_state=42)
            scaler = StandardScaler()
            coords = pca.fit_transform(scaler.fit_transform(df[FEATURE_NAMES]))

            fig, ax = plt.subplots(figsize=(6, 3.8))
            # Inliers
            ax.scatter(coords[~iso_res["is_anomaly"], 0], coords[~iso_res["is_anomaly"], 1],
                       c="#10B981", alpha=0.5, label="Normal Inliers", edgecolors="none")
            # Outliers
            ax.scatter(coords[iso_res["is_anomaly"], 0], coords[iso_res["is_anomaly"], 1],
                       c="#EF4444", alpha=0.9, label="Flagged Anomalies", edgecolors="black", s=45)
            ax.set_title(f"Isolation Forest (Contamination={contam*100:.0f}%)", fontsize=10, fontweight="bold")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Anomaly score histogram
        fig_h, ax_h = plt.subplots(figsize=(7, 2.8))
        sns.histplot(iso_res["anomaly_scores"], bins=40, kde=True, color="#F59E0B", ax=ax_h)
        ax_h.axvline(0, color="red", linestyle="--", label="Decision Boundary (Score < 0 = Anomaly)")
        ax_h.set_xlabel("Anomaly Score (Lower = More Anomalous)")
        ax_h.set_title("Isolation Forest Decision Score Distribution", fontsize=10, fontweight="bold")
        ax_h.legend()
        ax_h.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

        st.markdown("##### 🚨 Top Flagged Anomalous Student Profiles")
        st.dataframe(iso_res["anomaly_records"].head(15), use_container_width=True)

    with tab_ocsvm:
        st.subheader("One-Class Support Vector Machine")
        nu_val = st.slider("Nu (Upper bound on training margin errors):", 0.01, 0.15, 0.05, 0.01)
        svm_res = detect_anomalies_one_class_svm(df, nu=nu_val)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("SVM Anomalies Flagged", svm_res["n_anomalies"])
        with m2:
            st.metric("Proportion Outliers", f"{(svm_res['n_anomalies'] / len(df))*100:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(coords[~svm_res["is_anomaly"], 0], coords[~svm_res["is_anomaly"], 1],
                   c="#3B82F6", alpha=0.5, label="Inliers", edgecolors="none")
        ax.scatter(coords[svm_res["is_anomaly"], 0], coords[svm_res["is_anomaly"], 1],
                   c="#DC2626", alpha=0.9, label="Outliers", edgecolors="black", s=40)
        ax.set_title(f"One-Class SVM Frontier (Nu={nu_val})", fontsize=10, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_inspect:
        st.subheader("Pedagogical Root-Cause Explanations for Anomalies")
        st.caption("Inspect why specific student profiles triggered anomaly flags based on multi-variate z-scores.")

        anom_df = iso_res["anomaly_records"].head(20).copy()
        anom_df["Diagnostic Reason"] = [
            explain_student_anomaly(row, pop_means, pop_stds) for _, row in anom_df.iterrows()
        ]

        st.dataframe(
            anom_df[["student_id", "study_hours_per_day", "attendance_percentage",
                     "sleep_hours", "final_exam_score", "Diagnostic Reason"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    render_page()
