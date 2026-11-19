"""
Page 13: MLOps and Model Monitoring (CO-6: Production ML Engineering).
Demonstrates:
- Feature Distribution Drift Monitoring (Kolmogorov-Smirnov test & Population Stability Index)
- Synthetic Drift Simulator (COVID disruptions, cramming behavior)
- Automated Retraining Pipeline Trigger Evaluation
- A/B Model Deployment Testing (Champion vs Challenger)
- Mitchell et al. Model Card Viewer & Cryptographic SHA-256 Provenance
- FastAPI REST Integration & Health Telemetry
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

from src.utils.config import (
    FEATURE_NAMES,
    TARGET_REGRESSION,
    METADATA_DIR,
    RAW_DATA_PATH,
    REGRESSION_MODEL_PATH,
)
from src.data.loader import load_splits, prepare_dataset
from src.mlops.drift import (
    compute_ks_drift,
    compute_psi,
    generate_drifted_student_data,
)
from src.mlops.monitoring import (
    simulate_ab_testing,
    evaluate_retraining_trigger,
)
from src.mlops.versioning import (
    compute_file_sha256,
    get_git_commit_hash,
)


def render_page():
    st.markdown('<div class="main-header">🛠️ MLOps & Production Engineering (CO-6)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Data drift telemetry, retraining triggers, A/B deployment tests, and model governance cards.</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "ℹ️ **Educational Architecture:** Lightweight reference implementation of statistical monitoring, "
        "reproducibility tracking, and automated retraining triggers."
    )

    train_df, _, test_df = load_splits()

    tab_drift, tab_retrain, tab_ab_test, tab_card, tab_provenance = st.tabs([
        "📡 Data Drift Telemetry",
        "🔄 Retraining Trigger Workflow",
        "🅰️/🅱️ Model Comparison",
        "📄 Model Card Viewer",
        "🔐 Cryptographic Provenance",
    ])

    with tab_drift:
        st.subheader("Data Distribution & Feature Drift Monitoring")
        st.markdown(
            "Uses the two-sample **Kolmogorov-Smirnov (KS) Test** and **Population Stability Index (PSI)** "
            "to detect covariate shift between reference training data and incoming live inference streams."
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            drift_scenario = st.selectbox(
                "Simulate Live Data Scenario:",
                ["Stationary Stream (No Drift)", "Severe COVID Disruption (Shifted)", "Exam Cramming Burst (Shifted)"],
            )

            if "Stationary" in drift_scenario:
                live_df = test_df.copy()
            elif "COVID" in drift_scenario:
                live_df = generate_drifted_student_data(test_df, "severe_covid_shift")
            else:
                live_df = generate_drifted_student_data(test_df, "exam_cramming_shift")

            ks_results = compute_ks_drift(train_df, live_df)

            st.metric("Features Drifted", f"{ks_results['n_drifted_features']} / {ks_results['n_features_evaluated']}")
            st.metric("System Drift Flag", "DRIFT DETECTED ⚠️" if ks_results["drift_detected_overall"] else "NORMAL ✅")

        with c2:
            st.markdown("##### Feature-Level KS Drift Audit")
            drift_rows = []
            for feat, d in ks_results["features"].items():
                psi_val = compute_psi(train_df[feat].values, live_df[feat].values)
                drift_rows.append({
                    "Feature": feat,
                    "KS Stat": d["ks_statistic"],
                    "p-Value": d["p_value"],
                    "PSI": round(psi_val, 4),
                    "Ref Mean": d["ref_mean"],
                    "Live Mean": d["curr_mean"],
                    "Drift Status": "DRIFT 🚨" if d["drift_detected"] else "STABLE 🟢",
                })
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

    with tab_retrain:
        st.subheader("Automated Controlled Retraining Workflow")
        st.markdown(
            "Retraining trigger logic: Triggers automated pipeline if **>= 2 features drift** OR "
            "rolling validation RMSE degrades by **> 15%** from baseline."
        )

        base_rmse = 3.50
        sim_rolling_rmse = st.slider("Simulate Live Rolling RMSE:", 2.50, 6.00, 4.30, 0.10)

        trigger_res = evaluate_retraining_trigger(
            ks_results,
            rolling_rmse=sim_rolling_rmse,
            baseline_rmse=base_rmse,
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Baseline RMSE", f"{trigger_res['baseline_rmse']:.2f}")
        with r2:
            st.metric("Live Rolling RMSE", f"{trigger_res['rolling_rmse']:.2f}")
        with r3:
            st.metric("Degradation Ratio", f"{trigger_res['rmse_ratio']:.2f}x")

        if trigger_res["should_retrain"]:
            st.error("🚨 **RETRAINING TRIGGER ACTIVATED!**")
            for r in trigger_res["reasons"]:
                st.write(f"- {r}")
            st.markdown("```bash\n# Automated GitHub Actions or Airflow job triggered:\npython -m src.train\n```")
        else:
            st.success("✅ **PIPELINE HEALTHY:** Model operating within established tolerances.")

    with tab_ab_test:
        st.subheader("A/B Deployment Simulator (Champion vs Challenger)")
        st.markdown(
            "Routes live inference traffic 50/50 between current production Champion (e.g., Polynomial Regression) "
            "and candidate Challenger (e.g., Ridge Regression) to measure operational improvements."
        )

        if REGRESSION_MODEL_PATH.exists():
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge

            champ_pipe = joblib.load(REGRESSION_MODEL_PATH)
            chall_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
            chall_pipe.fit(train_df[FEATURE_NAMES], train_df[TARGET_REGRESSION])

            ab_res = simulate_ab_testing(test_df, champ_pipe, chall_pipe)

            c_a, c_b, c_action = st.columns(3)
            with c_a:
                st.metric("Champion RMSE (A)", f"{ab_res['champion_rmse']:.3f} marks")
            with c_b:
                st.metric("Challenger RMSE (B)", f"{ab_res['challenger_rmse']:.3f} marks")
            with c_action:
                st.metric("Challenger Gain", f"{ab_res['challenger_improvement_pct']:+.2f}%")

            st.info(f"**Deployment Decision:** {ab_res['recommended_action']}")

    with tab_card:
        st.subheader("Standardized Academic Model Card")
        card_files = list(METADATA_DIR.glob("*model_card.json"))
        if card_files:
            with open(card_files[0], "r", encoding="utf-8") as f:
                card_json = json.load(f)
            st.json(card_json)
        else:
            st.info("Run `python -m src.train` to generate formal model card metadata.")

    with tab_provenance:
        st.subheader("Data & Code Cryptographic Provenance")
        data_hash = compute_file_sha256(RAW_DATA_PATH)
        git_hash = get_git_commit_hash()

        st.write(f"- **Git Commit Provenance:** `{git_hash}`")
        st.write(f"- **Dataset SHA-256 Checksum:** `{data_hash}`")
        st.write("- **Reproducibility Seed:** `42` (Fixed pseudo-random state)")
        st.write("- **FastAPI REST Service:** Available at `http://127.0.0.1:8000/docs`")


if __name__ == "__main__":
    render_page()
