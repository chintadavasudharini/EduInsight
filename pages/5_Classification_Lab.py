"""
Page 5: Classification Lab (CO-2: Supervised Learning).
Implements multi-class academic performance tier classification:
- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (Linear, RBF, Polynomial kernels)
- XGBoost Classifier
Includes Confusion Matrices, ROC curves, and Hyperparameter Tuning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import (
    FEATURE_NAMES,
    TARGET_CLASSIFICATION,
    CATEGORY_LABELS,
    LABEL_TO_INT,
    INT_TO_LABEL,
)
from src.data.loader import load_splits
from src.supervised.classification import (
    train_and_evaluate_classifiers,
    get_classification_models,
)
from src.supervised.search import tune_random_forest_classifier


def render_page():
    st.markdown('<div class="main-header">🎯 Supervised Classification Lab (CO-2)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-class classification into academic tiers: Needs Support, Average, Good, and Excellent.</div>',
        unsafe_allow_html=True,
    )

    train_df, val_df, test_df = load_splits()

    tab_benchmark, tab_matrix, tab_interactive, tab_tuning = st.tabs([
        "🏆 Model Comparison Benchmark",
        "🔲 Confusion Matrix & Detailed Metrics",
        "🎛️ Interactive Model Explorer",
        "🔍 Hyperparameter Search (Random Forest)",
    ])

    with tab_benchmark:
        st.subheader("Classification Models Benchmark on Unseen Test Set")
        summary_df, fitted_models = train_and_evaluate_classifiers(
            train_df, train_df[TARGET_CLASSIFICATION],
            test_df, test_df[TARGET_CLASSIFICATION],
        )

        st.dataframe(summary_df, use_container_width=True)

        # Plot Macro F1 and Accuracy
        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = np.arange(len(summary_df))
        width = 0.35

        ax.bar(x - width/2, summary_df["Accuracy"], width, label="Accuracy", color="#3B82F6", edgecolor="#1E293B")
        ax.bar(x + width/2, summary_df["F1-Score (Macro)"], width, label="F1-Score (Macro)", color="#10B981", edgecolor="#1E293B")

        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["Model"], rotation=25, ha="right")
        ax.set_ylabel("Score (0.0 to 1.0)")
        ax.set_title("Classification Benchmark: Accuracy vs Macro F1", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_matrix:
        st.subheader("Confusion Matrix Analysis")
        model_choice = st.selectbox("Select Model for Confusion Matrix:", list(fitted_models.keys()))

        cm = fitted_models[model_choice]["confusion_matrix"]
        acc = fitted_models[model_choice]["accuracy"]
        f1 = fitted_models[model_choice]["f1"]
        mcc = fitted_models[model_choice]["mcc"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{acc*100:.2f}%")
        with col2:
            st.metric("Macro F1-Score", f"{f1:.4f}")
        with col3:
            st.metric("Matthews Correlation (MCC)", f"{mcc:.4f}")

        fig, ax = plt.subplots(figsize=(6, 4.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CATEGORY_LABELS,
            yticklabels=CATEGORY_LABELS,
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Academic Tier", fontweight="bold")
        ax.set_ylabel("True Academic Tier", fontweight="bold")
        ax.set_title(f"Confusion Matrix: {model_choice}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_interactive:
        st.subheader("Interactive Classifier Hyperparameters")
        selected_clf = st.selectbox(
            "Select Architecture to Fine-Tune:",
            ["K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest", "SVM (RBF Kernel)"]
        )

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report

        if selected_clf == "K-Nearest Neighbors (KNN)":
            k_val = st.slider("Number of Neighbors (K):", 1, 25, 7, 2)
            pipe_interactive = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=k_val))])
        elif selected_clf == "Decision Tree":
            max_d = st.slider("Max Tree Depth:", 2, 15, 5, 1)
            pipe_interactive = Pipeline([("clf", DecisionTreeClassifier(max_depth=max_d, random_state=42))])
        elif selected_clf == "Random Forest":
            n_est = st.slider("Number of Estimators:", 20, 200, 100, 20)
            max_d = st.slider("Max Tree Depth:", 2, 12, 6, 1)
            pipe_interactive = Pipeline([("clf", RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42))])
        else:
            c_val = st.slider("Penalty Parameter C:", 0.1, 10.0, 1.0, 0.5)
            pipe_interactive = Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=c_val, kernel="rbf", random_state=42))])

        y_tr_num = train_df[TARGET_CLASSIFICATION].map(LABEL_TO_INT).values
        y_te_num = test_df[TARGET_CLASSIFICATION].map(LABEL_TO_INT).values

        pipe_interactive.fit(train_df[FEATURE_NAMES], y_tr_num)
        y_pred = pipe_interactive.predict(test_df[FEATURE_NAMES])

        int_acc = accuracy_score(y_te_num, y_pred)
        st.success(f"**Test Accuracy achieved:** `{int_acc * 100:.2f}%`")

        report_dict = classification_report(
            y_te_num, y_pred, target_names=CATEGORY_LABELS, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with tab_tuning:
        st.subheader("Grid & Random Search on Random Forest")
        search_mode = st.radio("Search Algorithm:", ["Grid Search", "Random Search"], horizontal=True)

        with st.spinner("Running cross-validated parameter optimization..."):
            rf_res = tune_random_forest_classifier(
                train_df,
                train_df[TARGET_CLASSIFICATION].map(LABEL_TO_INT),
                search_type="grid" if search_mode == "Grid Search" else "random",
                n_splits=3,
            )

        st.success(f"**Best Parameters:** `{rf_res['best_params']}` | **Best Macro F1:** `{rf_res['best_f1_macro']:.4f}`")
        st.dataframe(rf_res["results_table"], use_container_width=True)


if __name__ == "__main__":
    render_page()
