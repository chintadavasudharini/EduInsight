"""
Page 4: Regression Lab (CO-2: Supervised Learning).
Implements interactive training, comparative benchmarking, and error diagnostics for:
- Linear Regression
- Polynomial Regression (Degree 2)
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- ElasticNet Regression (L1 + L2 Regularization)
Includes Residual Analysis and Bias-Variance Tradeoff Decomposition.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import FEATURE_NAMES, TARGET_REGRESSION
from src.data.loader import load_splits
from src.supervised.regression import train_and_evaluate_regressors, get_regression_models
from src.supervised.bias_variance import compute_bias_variance_decomposition
from src.supervised.search import tune_ridge_regression


def render_page():
    st.markdown('<div class="main-header">📈 Supervised Regression Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Continuous final exam score forecasting using regularized and polynomial linear models.</div>',
        unsafe_allow_html=True,
    )

    train_df, val_df, test_df = load_splits()

    tab_benchmark, tab_interactive, tab_residuals, tab_bias_variance, tab_tuning = st.tabs([
        "🏆 Model Comparison Benchmark",
        "🎛️ Interactive Model Trainer",
        "📉 Residuals & Diagnostics",
        "⚖️ Bias-Variance Tradeoff",
        "🔍 Hyperparameter Search",
    ])

    with tab_benchmark:
        st.subheader("Model Benchmark Comparison on Test Partition")
        st.caption(f"Evaluated on {len(test_df)} unseen test samples.")

        summary_df, fitted_dict = train_and_evaluate_regressors(
            train_df, train_df[TARGET_REGRESSION],
            test_df, test_df[TARGET_REGRESSION]
        )

        st.dataframe(summary_df, use_container_width=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        # RMSE comparison
        ax1.barh(summary_df["Model"], summary_df["RMSE"], color="#3B82F6", edgecolor="#1E293B")
        ax1.set_xlabel("Root Mean Squared Error (Lower is better)")
        ax1.set_title("RMSE Across Models", fontsize=10, fontweight="bold")
        ax1.grid(axis="x", linestyle="--", alpha=0.5)

        # R-squared comparison
        ax2.barh(summary_df["Model"], summary_df["R-squared"], color="#10B981", edgecolor="#1E293B")
        ax2.set_xlabel("R-squared Score (Higher is better)")
        ax2.set_title("R² Score Across Models", fontsize=10, fontweight="bold")
        ax2.set_xlim(0.8, 1.0)
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_interactive:
        st.subheader("Interactive Model Fitting")
        selected_model = st.selectbox(
            "Select Regression Architecture:",
            ["Linear Regression", "Polynomial Regression (Deg 2)", "Ridge Regression", "Lasso Regression", "ElasticNet"]
        )

        alpha_val = 1.0
        l1_ratio_val = 0.5
        if selected_model in ["Ridge Regression", "Lasso Regression", "ElasticNet"]:
            alpha_val = st.slider("Regularization Strength (alpha):", 0.001, 10.0, 1.0, 0.05)
        if selected_model == "ElasticNet":
            l1_ratio_val = st.slider("L1 Ratio (rho):", 0.0, 1.0, 0.5, 0.05)

        # Fit model on demand
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if selected_model == "Linear Regression":
            pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        elif selected_model == "Polynomial Regression (Deg 2)":
            pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scaler", StandardScaler()), ("model", LinearRegression())])
        elif selected_model == "Ridge Regression":
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha_val))])
        elif selected_model == "Lasso Regression":
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha_val, max_iter=2000))])
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=alpha_val, l1_ratio=l1_ratio_val, max_iter=2000))])

        pipe.fit(train_df[FEATURE_NAMES], train_df[TARGET_REGRESSION])
        preds = pipe.predict(test_df[FEATURE_NAMES])

        mae = mean_absolute_error(test_df[TARGET_REGRESSION], preds)
        rmse = float(np.sqrt(mean_squared_error(test_df[TARGET_REGRESSION], preds)))
        r2 = r2_score(test_df[TARGET_REGRESSION], preds)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Test MAE", f"{mae:.3f}")
        with m2:
            st.metric("Test RMSE", f"{rmse:.3f}")
        with m3:
            st.metric("Test R²", f"{r2:.4f}")

        # Actual vs Predicted Scatter
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(test_df[TARGET_REGRESSION], preds, alpha=0.6, color="#2563EB", edgecolors="none")
        min_v = min(test_df[TARGET_REGRESSION].min(), preds.min())
        max_v = max(test_df[TARGET_REGRESSION].max(), preds.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", label="Ideal 45° Line")
        ax.set_xlabel("Actual Exam Marks")
        ax.set_ylabel("Predicted Exam Marks")
        ax.set_title(f"Actual vs Predicted: {selected_model}", fontsize=10, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_residuals:
        st.subheader("Residual Diagnostics & Error Distribution")
        # Compute residuals using Linear model
        pipe_lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        pipe_lin.fit(train_df[FEATURE_NAMES], train_df[TARGET_REGRESSION])
        y_test_arr = test_df[TARGET_REGRESSION].values
        test_preds = pipe_lin.predict(test_df[FEATURE_NAMES])
        residuals = y_test_arr - test_preds

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        # Residuals vs Predicted
        ax1.scatter(test_preds, residuals, alpha=0.6, color="#4B5563")
        ax1.axhline(0, color="red", linestyle="--")
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals (Actual - Predicted)")
        ax1.set_title("Residuals vs Fitted (Homoscedasticity Check)", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Residuals Histogram
        sns.histplot(residuals, kde=True, color="#8B5CF6", ax=ax2, edgecolor="#1E293B")
        ax2.set_xlabel("Residual Value")
        ax2.set_title("Residual Distribution (Normality Check)", fontsize=10, fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_bias_variance:
        st.subheader("Empirical Bias-Variance Decomposition")
        st.markdown(
            "Decomposing test MSE across polynomial degrees using bootstrap resampling: "
            "$$\\mathbb{E}[\\text{MSE}] = \\text{Bias}^2 + \\text{Variance} + \\sigma^2$$"
        )

        with st.spinner("Computing bootstrap bias-variance curves across degrees 1 to 5..."):
            bv_res = compute_bias_variance_decomposition(
                train_df[FEATURE_NAMES].values,
                train_df[TARGET_REGRESSION].values,
                test_df[FEATURE_NAMES].values,
                test_df[TARGET_REGRESSION].values,
                degrees=[1, 2, 3, 4, 5],
                n_bootstraps=25,
            )

        st.dataframe(bv_res["summary_table"], use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3.8))
        degs = bv_res["degrees"]
        ax.plot(degs, bv_res["bias_squared"], "o-", label="Bias² (Model Inflexibility)", color="#EF4444", linewidth=2)
        ax.plot(degs, bv_res["variance"], "s-", label="Variance (Overfitting Sensitivity)", color="#3B82F6", linewidth=2)
        ax.plot(degs, bv_res["total_mse"], "^-", label="Total Expected MSE", color="#10B981", linewidth=2.5)
        ax.set_xlabel("Polynomial Model Degree (Complexity)")
        ax.set_ylabel("Error Metric")
        ax.set_title("Bias-Variance Tradeoff Curve", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(bv_res["explanation"])

    with tab_tuning:
        st.subheader("Automated Hyperparameter Tuning (Grid vs Random Search)")
        search_method = st.radio("Search Strategy:", ["Grid Search", "Random Search"], horizontal=True)

        tune_res = tune_ridge_regression(
            train_df,
            train_df[TARGET_REGRESSION],
            search_type="grid" if search_method == "Grid Search" else "random",
            n_splits=5,
        )

        st.success(f"**Optimal Parameters:** `{tune_res['best_params']}` | **Best CV RMSE:** `{tune_res['best_rmse']:.4f}`")
        st.dataframe(tune_res["results_table"], use_container_width=True)


if __name__ == "__main__":
    render_page()
