"""
Page 2: Student Performance Prediction (Inference Lab).
Integrates:
- Continuous Final Exam Mark Regression
- Multi-class Academic Tier Classification
- Outlier & Anomaly Verification (Isolation Forest)
- Interpretability and Pedagogical Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.utils.config import (
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    FEATURE_NAMES,
    CATEGORY_LABELS,
    INT_TO_LABEL,
)
from src.unsupervised.anomaly import explain_student_anomaly
from src.data.loader import prepare_dataset


def render_page():
    st.markdown('<div class="main-header">🎯 Student Performance Prediction & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Input academic habits to forecast final exam marks, classify performance, and detect behavioral anomalies.</div>',
        unsafe_allow_html=True,
    )

    if not REGRESSION_MODEL_PATH.exists() or not CLASSIFICATION_MODEL_PATH.exists():
        st.warning("⚠️ Serialized models not found. Please train models first using `python -m src.train`.")
        return

    reg_pipeline = joblib.load(REGRESSION_MODEL_PATH)
    clf_pipeline = joblib.load(CLASSIFICATION_MODEL_PATH)
    iso_model = joblib.load(ANOMALY_MODEL_PATH) if ANOMALY_MODEL_PATH.exists() else None

    # Two column layout: Input Form and Output Diagnostics
    col_input, col_output = st.columns([1.1, 1.2])

    with col_input:
        st.subheader("📝 Student Academic Profile Input")
        with st.form("prediction_form"):
            study_hours = st.slider(
                "Study Hours per Day",
                min_value=1.0, max_value=10.0, value=5.5, step=0.25,
                help="Average daily independent study hours outside class."
            )
            attendance = st.slider(
                "Attendance Percentage (%)",
                min_value=45.0, max_value=100.0, value=82.0, step=0.5,
                help="Class lecture attendance rate over the semester."
            )
            assignment_score = st.slider(
                "Assignment & Lab Score (out of 100)",
                min_value=30.0, max_value=100.0, value=75.0, step=0.5,
                help="Average score achieved across continuous assessment assignments."
            )
            previous_exam_score = st.slider(
                "Previous Examination Score (out of 100)",
                min_value=25.0, max_value=100.0, value=68.0, step=0.5,
                help="Marks scored in the midterm/previous semester examination."
            )
            sleep_hours = st.slider(
                "Sleep Hours per Day",
                min_value=4.0, max_value=10.0, value=7.0, step=0.25,
                help="Average daily nocturnal sleep duration."
            )

            submit_btn = st.form_submit_button("🔮 Run Academic Prediction & Diagnostics", use_container_width=True)

    with col_output:
        st.subheader("📊 Forecast Results & Diagnostic Signals")

        if submit_btn or "last_pred" in st.session_state:
            input_df = pd.DataFrame([{
                "study_hours_per_day": study_hours,
                "attendance_percentage": attendance,
                "assignment_score": assignment_score,
                "previous_exam_score": previous_exam_score,
                "sleep_hours": sleep_hours,
            }])

            # 1. Regression Score
            pred_score = float(np.clip(reg_pipeline.predict(input_df[FEATURE_NAMES])[0], 0.0, 100.0))

            # 2. Classification Tier
            pred_class_idx = clf_pipeline.predict(input_df[FEATURE_NAMES])[0]
            cat_label = INT_TO_LABEL.get(int(pred_class_idx), str(pred_class_idx))

            # 3. Anomaly Detection
            is_anomaly = False
            anomaly_score = 0.0
            anomaly_explanation = "Normal academic profile consistent with general cohort distributions."
            if iso_model is not None:
                iso_pred = iso_model.predict(input_df[FEATURE_NAMES])[0]
                is_anomaly = bool(iso_pred == -1)
                anomaly_score = float(iso_model.decision_function(input_df[FEATURE_NAMES])[0])
                if is_anomaly:
                    df_pop = prepare_dataset()
                    anomaly_explanation = explain_student_anomaly(
                        input_df.iloc[0],
                        df_pop[FEATURE_NAMES].mean(),
                        df_pop[FEATURE_NAMES].std(),
                    )

            # Display Output Cards
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Exam Score", f"{pred_score:.1f} / 100")
            with m2:
                tier_color = {
                    "Excellent": "#10B981",
                    "Good": "#3B82F6",
                    "Average": "#F59E0B",
                    "Needs Support": "#EF4444",
                }.get(cat_label, "#6B7280")
                st.markdown(
                    f'<div style="background:{tier_color}15; border:1px solid {tier_color}; '
                    f'border-radius:8px; padding:12px; text-align:center;">'
                    f'<div style="font-size:0.8rem; color:#6B7280; font-weight:600;">PERFORMANCE TIER</div>'
                    f'<div style="font-size:1.5rem; font-weight:700; color:{tier_color};">{cat_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Anomaly Banner
            if is_anomaly:
                st.error(f"🚨 **Atypical Learning Profile Detected!** (Score: `{anomaly_score:.3f}`)\n\n*{anomaly_explanation}*")
            else:
                st.success(f"✅ **Cohort Congruence:** Normal learning pattern (Inlier Score: `{anomaly_score:.3f}`).")

            # Feature Impact Visual
            st.markdown("##### Feature Impact on Predicted Outcome")
            # Baseline mean student vector
            df_pop = prepare_dataset()
            pop_means = df_pop[FEATURE_NAMES].mean()
            diffs = {
                "Study Hours": (study_hours - pop_means["study_hours_per_day"]) * 3.5,
                "Attendance": (attendance - pop_means["attendance_percentage"]) * 0.26,
                "Assignment": (assignment_score - pop_means["assignment_score"]) * 0.20,
                "Previous Exam": (previous_exam_score - pop_means["previous_exam_score"]) * 0.28,
                "Sleep Factor": -3.0 if sleep_hours < 5.5 else (+1.0 if 6.5 <= sleep_hours <= 8.5 else -1.5),
            }

            fig, ax = plt.subplots(figsize=(6, 2.8))
            names = list(diffs.keys())
            vals = list(diffs.values())
            colors = ["#10B981" if v >= 0 else "#EF4444" for v in vals]
            ax.barh(names, vals, color=colors, edgecolor="#1E293B", linewidth=0.7)
            ax.axvline(0, color="#1E293B", linestyle="--", alpha=0.5)
            ax.set_xlabel("Relative Deviation from Cohort Average (Points)")
            ax.grid(axis="x", linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Pedagogical Advisory
            st.markdown("##### 💡 Academic Advising Insights")
            recommendations = []
            if study_hours < 4.0:
                recommendations.append("Increase daily study hours to at least 5.0 hours with spaced repetition.")
            if attendance < 75.0:
                recommendations.append("Attendance is below standard 75% requirement; direct lecture engagement is needed.")
            if sleep_hours < 6.0:
                recommendations.append("Severe sleep deprivation (<6 hours) compromises memory consolidation and exam recall.")
            if previous_exam_score < 50.0:
                recommendations.append("Enroll in remedial tutorial sessions to strengthen fundamental prerequisites.")
            if not recommendations:
                recommendations.append("Outstanding academic habit balance! Maintain consistent revision schedule.")

            for rec in recommendations:
                st.info(f"📌 {rec}")
        else:
            st.info("👈 Enter student parameters and click **Run Academic Prediction & Diagnostics**.")


if __name__ == "__main__":
    render_page()
