"""
Page 1: Home Dashboard.
Provides high-level project telemetry, key metrics, summary charts, and CO mapping cards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.utils.config import (
    RAW_DATA_PATH,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    CATEGORY_LABELS,
    FEATURE_NAMES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
)
from src.data.loader import prepare_dataset


def render_page():
    st.markdown('<div class="main-header">🎓 Student Performance & Learning Pattern Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">M.Tech Academic Project integrating Applied Mathematics and Machine Learning (CO-1 to CO-6).</div>',
        unsafe_allow_html=True,
    )

    df = prepare_dataset()

    # Metric Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Records", f"{len(df):,}")
    with c2:
        st.metric("Features Analyzed", f"{len(FEATURE_NAMES)}")
    with c3:
        st.metric("Target Output", "Exam (0-100)")
    with c4:
        st.metric("Models Trained", "14 Architectures")
    with c5:
        st.metric("Syllabus Scope", "CO-1 to CO-6")

    st.markdown("---")

    # Layout: Category distribution + Quick Predictor
    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.subheader("📊 Academic Performance Distribution")
        cat_counts = df[TARGET_CLASSIFICATION].value_counts().reindex(CATEGORY_LABELS)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#EF4444", "#F59E0B", "#3B82F6", "#10B981"]
        bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="#1E293B", linewidth=0.8)
        ax.set_ylabel("Student Count")
        ax.set_title("Student Categorization Across Academic Tiers", fontsize=11, fontweight="bold")
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
        st.subheader("⚡ Quick Prediction Simulator")
        st.caption("Adjust student inputs to compute real-time marks & tier.")

        q_study = st.slider("Daily Study Hours", 1.0, 10.0, 6.0, 0.5)
        q_att = st.slider("Attendance %", 50.0, 100.0, 85.0, 1.0)
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

            st.success(f"**Predicted Final Score:** `{quick_score:.1f} / 100`")
            st.markdown(f"**Performance Tier:** :{color}[**{tier}**]")
        else:
            st.info("Run `python -m src.train` to train and inspect baseline models.")

    st.markdown("---")
    st.subheader("🎯 Syllabus Course Outcome (CO) Mapping")

    co_col1, co_col2 = st.columns(2)

    with co_col1:
        st.markdown("""
        - **CO-1: Mathematical Foundations for ML**
          Vectors, covariance eigendecomposition, SVD, Bayes theorem, expectation, hypothesis testing, MLE/MAP, gradients & Hessians, convexity, Lagrange & KKT conditions.
        - **CO-2: Supervised Learning**
          Linear, Polynomial, Ridge, Lasso, ElasticNet; Logistic, Naive Bayes, KNN, Decision Tree, Random Forest, SVM (Linear, RBF, Poly), XGBoost; Bias-Variance tradeoff, Grid & Random Search.
        - **CO-3: Unsupervised Learning & Embeddings**
          K-Means, DBSCAN, Agglomerative, GMM; PCA, t-SNE, UMAP; Isolation Forest, One-Class SVM; Autoencoder embeddings, NLP semantic embedding demo.
        """)

    with co_col2:
        st.markdown("""
        - **CO-4: Optimization and Training**
          PyTorch neural network, SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, Lion; learning rate schedules, L1/L2, Dropout, BatchNorm, Early Stopping, Xavier & He init.
        - **CO-5: Model Evaluation and Selection**
          MAE, RMSE, R-squared; Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC; Reliability calibration & ECE; Paired t-test, McNemar test, Bootstrap CIs, Stratified/Group/TimeSeries CV, Fairness analysis.
        - **CO-6: Production ML Engineering / MLOps**
          Joblib persistence, Model Cards, FastAPI REST endpoints, KS & PSI Data Drift monitoring, Retraining triggers, A/B Testing, CI/CD pipeline.
        """)


if __name__ == "__main__":
    render_page()
