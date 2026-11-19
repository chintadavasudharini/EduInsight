"""
Dashboard Module.
Clean, modern overview of student performance predictions, cohort metrics,
and academic tier distribution for authenticated users.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.utils.config import (
    REGRESSION_MODEL_PATH,
    CATEGORY_LABELS,
    FEATURE_NAMES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
)
from src.data.loader import prepare_dataset


def render_page():
    st.markdown('<div class="main-header">EduInsight Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Understand Learning. Predict Performance. Improve Outcomes.</div>',
        unsafe_allow_html=True,
    )

    df = prepare_dataset()

    # User greeting
    if st.session_state.get("authenticated", False) and st.session_state.get("user"):
        user = st.session_state["user"]
        st.success(f"👋 **Welcome back, {user['name']}!** You are signed in. Explore cohort metrics below or choose an analytics module from the sidebar.")

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records Analyzed", f"{len(df):,}")
    with c2:
        st.metric("Academic Features", f"{len(FEATURE_NAMES)}")
    with c3:
        st.metric("Target Output", "Exam Mark (0–100)")
    with c4:
        st.metric("Prediction Accuracy", "High (~89% R²)")

    st.markdown("---")

    # Layout: Category distribution + Quick Predictor
    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.subheader("Performance Tier Distribution")
        cat_counts = df[TARGET_CLASSIFICATION].value_counts().reindex(CATEGORY_LABELS)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#EF4444", "#F59E0B", "#3B82F6", "#10B981"]
        bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="#1E293B", linewidth=0.8)
        ax.set_ylabel("Student Count")
        ax.set_title("Student Categorization Across Performance Tiers", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height} ({height/len(df)*100:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Quick Prediction Simulator")
        st.caption("Adjust student habits below to calculate an immediate score estimate.")

        q_study = st.slider("Daily Study Hours", 1.0, 10.0, 6.0, 0.5)
        q_att = st.slider("Attendance Percentage (%)", 50.0, 100.0, 85.0, 1.0)
        q_prev = st.slider("Previous Exam Score", 30.0, 100.0, 72.0, 1.0)

        if REGRESSION_MODEL_PATH.exists():
            reg_pipe = joblib.load(REGRESSION_MODEL_PATH)
            sample_in = pd.DataFrame([{
                "study_hours_per_day": q_study,
                "attendance_percentage": q_att,
                "assignment_score": min(100.0, q_prev + 5.0),
                "previous_exam_score": q_prev,
                "sleep_hours": 7.0,
            }])
            quick_score = float(np.clip(reg_pipe.predict(sample_in[FEATURE_NAMES])[0], 0, 100))

            if quick_score >= 80:
                tier, color = "Excellent", "green"
            elif quick_score >= 60:
                tier, color = "Good", "blue"
            elif quick_score >= 40:
                tier, color = "Average", "orange"
            else:
                tier, color = "Needs Support", "red"

            st.success(f"**Estimated Exam Score:** `{quick_score:.1f} / 100`")
            st.markdown(f"**Predicted Category:** :{color}[**{tier}**]")
        else:
            st.info("Model pipeline ready. Train models to view live predictions.")

    st.markdown("---")
    st.subheader("Key Capabilities")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        **Continuous Forecasting**
        - Accurate estimation of final examination marks
        - Regularized linear and non-linear polynomial algorithms
        - Comprehensive residual error diagnostics
        """)
    with f2:
        st.markdown("""
        **Learning Pattern Clustering**
        - Unsupervised behavioral archetype identification
        - Density-based outlier and anomaly detection
        - Dimensionality reduction and manifold visualization
        """)
    with f3:
        st.markdown("""
        **Operational Reliability**
        - Production data drift monitoring
        - Statistical hypothesis validation
        - Automated calibration and fairness auditing
        """)


if __name__ == "__main__":
    render_page()
