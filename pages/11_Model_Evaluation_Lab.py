"""
Page 11: Model Evaluation Lab (CO-5: Model Evaluation and Selection).
Implements comprehensive evaluation telemetry:
- Standard Regression Metrics: MAE, RMSE, R-squared, MAPE
- Classification Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC
- Probability Calibration: Reliability Diagrams, Platt Scaling, Isotonic Calibration, ECE
- Error Analysis Taxonomy: Residual distribution and severity bucketing
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.utils.config import (
    FEATURE_NAMES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    CATEGORY_LABELS,
    LABEL_TO_INT,
)
from src.data.loader import load_splits
from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    analyze_prediction_errors,
)
from src.evaluation.calibration import calibrate_classifier, compute_expected_calibration_error


def render_page():
    st.markdown('<div class="main-header">🎯 Model Evaluation Lab (CO-5)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Rigorous performance benchmarking, probability calibration diagrams, and error taxonomies.</div>',
        unsafe_allow_html=True,
    )

    _, _, test_df = load_splits()

    if not REGRESSION_MODEL_PATH.exists() or not CLASSIFICATION_MODEL_PATH.exists():
        st.warning("Please train models first via `python -m src.train`.")
        return

    reg_model = joblib.load(REGRESSION_MODEL_PATH)
    clf_model = joblib.load(CLASSIFICATION_MODEL_PATH)

    tab_reg_eval, tab_clf_eval, tab_calibration, tab_taxonomy = st.tabs([
        "📏 Regression Evaluation",
        "🎯 Classification Evaluation",
        "📊 Probability Calibration & ECE",
        "🔍 Error Taxonomy & Residuals",
    ])

    with tab_reg_eval:
        st.subheader("Regression Performance Telemetry on Test Set")
        y_test_reg = test_df[TARGET_REGRESSION].values
        pred_reg = reg_model.predict(test_df[FEATURE_NAMES])

        reg_metrics = compute_regression_metrics(y_test_reg, pred_reg)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean Absolute Error (MAE)", f"{reg_metrics['MAE']:.3f} marks")
        with c2:
            st.metric("Root Mean Squared Error (RMSE)", f"{reg_metrics['RMSE']:.3f} marks")
        with c3:
            st.metric("Coefficient of Determination (R²)", f"{reg_metrics['R2']:.4f}")
        with c4:
            st.metric("Mean Abs % Error (MAPE)", f"{reg_metrics['MAPE']:.2f}%")

        st.markdown("##### Metric Interpretations")
        st.markdown("""
        - **MAE:** Average magnitude of prediction errors in raw marks. Unlike RMSE, it weights all individual discrepancies equally without squaring.
        - **RMSE:** Penalizes large outliers more heavily due to the square term. Standard metric for academic grading models.
        - **R²:** Proportional variance explained by the model compared to predicting the simple mean baseline.
        """)

    with tab_clf_eval:
        st.subheader("Classification Performance Telemetry")
        y_test_clf = test_df[TARGET_CLASSIFICATION].map(LABEL_TO_INT).values
        pred_clf = clf_model.predict(test_df[FEATURE_NAMES])

        probs_clf = None
        if hasattr(clf_model, "predict_proba"):
            try:
                probs_clf = clf_model.predict_proba(test_df[FEATURE_NAMES])
            except Exception:
                pass

        clf_metrics = compute_classification_metrics(y_test_clf, pred_clf, probs_clf)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Accuracy", f"{clf_metrics['Accuracy']*100:.2f}%")
        with m2:
            st.metric("Macro Precision", f"{clf_metrics['Precision']:.4f}")
        with m3:
            st.metric("Macro Recall", f"{clf_metrics['Recall']:.4f}")
        with m4:
            st.metric("Macro F1-Score", f"{clf_metrics['F1_Score']:.4f}")
        with m5:
            st.metric("Matthews Corr (MCC)", f"{clf_metrics['MCC']:.4f}")

        st.markdown(f"**Multi-Class ROC-AUC (OVR Macro):** `{clf_metrics['ROC_AUC']}` | **PR-AUC (Macro):** `{clf_metrics['PR_AUC']}`")

    with tab_calibration:
        st.subheader("Probability Calibration & Reliability Diagrams")
        st.markdown(
            "Measures whether predicted confidence matches empirical posterior correctness: "
            "$$\\text{ECE} = \\sum_{m=1}^M \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$"
        )

        cal_mode = st.radio("Calibration Algorithm:", ["Platt Scaling (Sigmoid)", "Isotonic Regression"], horizontal=True)
        method_key = "sigmoid" if "Platt" in cal_mode else "isotonic"

        train_df, _, _ = load_splits()
        y_train_clf = train_df[TARGET_CLASSIFICATION].map(LABEL_TO_INT).values

        cal_res = calibrate_classifier(
            clf_model,
            train_df[FEATURE_NAMES], y_train_clf,
            test_df[FEATURE_NAMES], y_test_clf,
            method=method_key,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pre-Calibration ECE", f"{cal_res['ece_before']*100:.2f}%")
        with c2:
            st.metric(f"Post-{cal_mode} ECE", f"{cal_res['ece_after']*100:.2f}%")

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(cal_res["conf_before"], cal_res["acc_before"], "s-", color="#EF4444", label=f"Uncalibrated (ECE={cal_res['ece_before']*100:.1f}%)")
        ax.plot(cal_res["conf_after"], cal_res["acc_after"], "o-", color="#10B981", label=f"Calibrated (ECE={cal_res['ece_after']*100:.1f}%)")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title("Reliability Diagram (Calibration Curve)", fontsize=10, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_taxonomy:
        st.subheader("Error-Analysis Taxonomy & Severe Outliers")
        test_eval_df = test_df.copy()
        test_eval_df["pred_score"] = pred_reg

        err_res = analyze_prediction_errors(
            test_eval_df,
            y_true_col=TARGET_REGRESSION,
            y_pred_col="pred_score",
            feature_cols=FEATURE_NAMES,
        )

        st.markdown("##### Residual Severity Breakdown")
        for k, v in err_res["severity_distribution"].items():
            st.write(f"- **{k}:** {v} student instances ({v/len(test_df)*100:.1f}%)")

        st.markdown("##### Top 10 Worst Predictions (Diagnostic Audit)")
        st.dataframe(err_res["worst_predictions"], use_container_width=True)


if __name__ == "__main__":
    render_page()
