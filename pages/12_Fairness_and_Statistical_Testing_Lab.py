"""
Page 12: Fairness and Statistical Testing Lab (CO-5).
Implements:
- Hypothesis Testing for Model Comparison:
  - Paired Student's t-test on K-Fold CV scores
  - McNemar's Test for classifier discordant predictions
  - Non-parametric Empirical Bootstrap Confidence Intervals
- Advanced Cross-Validation Schemes:
  - Stratified K-Fold
  - Group-Aware K-Fold (Cluster / Campus separation)
  - Time-Series Expanding Window CV
- Algorithmic Fairness Analysis:
  - Demographic Parity (Disparate Impact ratio)
  - Equal Opportunity (TPR parity)
  - Equalized Odds (TPR and FPR parity)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import FEATURE_NAMES, TARGET_REGRESSION, TARGET_CLASSIFICATION, LABEL_TO_INT
from src.data.loader import load_splits, prepare_dataset
from src.evaluation.statistical_tests import (
    run_paired_ttest_cv,
    run_mcnemar_test,
    compute_bootstrap_confidence_interval,
)
from src.evaluation.cross_validation import (
    demonstrate_stratified_cv,
    demonstrate_group_cv,
    demonstrate_timeseries_cv,
)
from src.evaluation.fairness import compute_fairness_metrics


def render_page():
    st.markdown('<div class="main-header">⚖️ Fairness & Statistical Testing Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Statistical significance testing between models, advanced cross-validation, and responsible fairness audits.</div>',
        unsafe_allow_html=True,
    )

    df_full = prepare_dataset()
    train_df, _, test_df = load_splits()

    tab_stat_tests, tab_cv_schemes, tab_fairness = st.tabs([
        "🧪 Statistical Hypothesis Testing",
        "🔄 Advanced Cross-Validation Schemes",
        "⚖️ Algorithmic Fairness & Disparate Impact",
    ])

    with tab_stat_tests:
        st.subheader("1. Paired Student's t-Test on CV Folds")
        st.markdown(
            "Tests whether the difference between Model A and Model B cross-validation scores is statistically significant ($p < 0.05$)."
        )

        scores_a = [0.88, 0.89, 0.87, 0.90, 0.88]  # Polynomial Model CV R2
        scores_b = [0.85, 0.86, 0.84, 0.86, 0.85]  # Linear Model CV R2

        ttest_res = run_paired_ttest_cv(scores_a, scores_b, "Polynomial Reg (Deg 2)", "Linear Reg")

        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("Mean Score Diff (A - B)", f"+{ttest_res['mean_difference']:.4f}")
        with t2:
            st.metric("t-Statistic", f"{ttest_res['t_statistic']:.3f}")
        with t3:
            st.metric("p-Value", f"{ttest_res['p_value']:.5f}")

        st.success(f"Statistically Significant Difference at alpha=0.05: **{ttest_res['statistically_significant_at_05']}**")

        st.markdown("---")
        st.subheader("2. McNemar's Test for Paired Classifiers")
        st.markdown("Evaluates whether two classifiers exhibit significantly different error distributions.")

        y_t = test_df[TARGET_CLASSIFICATION].map(LABEL_TO_INT).values
        # Synthetic paired predictions for demo
        y_p1 = y_t.copy()
        y_p1[:15] = (y_p1[:15] + 1) % 4  # some errors in A
        y_p2 = y_t.copy()
        y_p2[10:30] = (y_p2[10:30] + 1) % 4  # some errors in B

        mcn_res = run_mcnemar_test(y_t, y_p1, y_p2, "Random Forest", "SVM RBF")

        st.write(f"**Contingency Matrix (Edwards Corrected Chi²):** `{mcn_res['contingency_table']}`")
        st.write(f"- Chi-squared Statistic: `{mcn_res['chi2_statistic']:.4f}`")
        st.write(f"- p-Value: `{mcn_res['p_value']:.4f}`")

        st.markdown("---")
        st.subheader("3. Empirical Bootstrap Confidence Intervals")
        metric_choice = st.selectbox("Select Metric for Bootstrap CI:", ["accuracy", "f1"])
        boot_res = compute_bootstrap_confidence_interval(y_t, y_p1, metric_name=metric_choice, n_bootstraps=300)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Bootstrap Mean", f"{boot_res['mean']:.4f}")
        with b2:
            st.metric("95% CI Lower", f"{boot_res['ci_lower']:.4f}")
        with b3:
            st.metric("95% CI Upper", f"{boot_res['ci_upper']:.4f}")

    with tab_cv_schemes:
        st.subheader("Comparison of Validation Splitting Schemes")

        col_cv1, col_cv2, col_cv3 = st.columns(3)

        with col_cv1:
            st.markdown("##### Stratified K-Fold")
            st.caption("Guarantees identical class ratios in each fold.")
            skf_res = demonstrate_stratified_cv(train_df, train_df[TARGET_CLASSIFICATION], n_splits=4)
            st.metric("Mean Accuracy", f"{skf_res['mean_accuracy']*100:.1f}%")
            st.metric("Fold Std Dev", f"{skf_res['std_accuracy']*100:.2f}%")

        with col_cv2:
            st.markdown("##### Group-Aware K-Fold")
            st.caption("Ensures student clusters/campuses never leak across train/val.")
            gkf_res = demonstrate_group_cv(
                train_df, train_df[TARGET_CLASSIFICATION], train_df["study_group"], n_splits=2
            )
            st.metric("Group-Split Accuracy", f"{gkf_res['mean_accuracy']*100:.1f}%")

        with col_cv3:
            st.markdown("##### Time-Series Split")
            st.caption("Expanding window preventing lookahead bias.")
            ts_res = demonstrate_timeseries_cv(train_df, train_df[TARGET_REGRESSION], n_splits=4)
            st.metric("Mean TS-RMSE", f"{ts_res['mean_rmse']:.2f}")

    with tab_fairness:
        st.subheader("Algorithmic Fairness & Disparate Impact Audit")
        st.warning(
            "⚠️ **Responsible AI Disclaimer:** Fairness evaluation demonstrated here uses synthetic campus groups "
            "(`Campus_A` vs `Campus_B`) solely for educational methodology demonstration. "
            "Do not infer real-world parity from synthetic partitions."
        )

        test_fair_df = test_df.copy()
        # Binary target: Passing/High Performance (Good or Excellent: >= 60 marks)
        test_fair_df["y_true_pass"] = (test_fair_df[TARGET_REGRESSION] >= 60.0).astype(int)
        test_fair_df["y_pred_pass"] = (test_fair_df[TARGET_REGRESSION] + np.random.normal(0, 3, len(test_fair_df)) >= 60.0).astype(int)

        fair_metrics = compute_fairness_metrics(
            test_fair_df,
            group_col="study_group",
            y_true_binary="y_true_pass",
            y_pred_binary="y_pred_pass",
        )

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("Demographic Parity Diff", f"{fair_metrics['demographic_parity_difference']:.4f}")
        with f2:
            st.metric("Disparate Impact Ratio", f"{fair_metrics['demographic_parity_ratio']:.3f}")
        with f3:
            st.metric("Equal Opportunity Diff", f"{fair_metrics['equal_opportunity_difference']:.4f}")
        with f4:
            st.metric("80% Rule Satisfied?", "Yes ✅" if fair_metrics["four_fifths_rule_met"] else "No ❌")

        st.markdown("##### Subgroup Performance Metrics")
        st.json(fair_metrics["group_stats"])


if __name__ == "__main__":
    render_page()
