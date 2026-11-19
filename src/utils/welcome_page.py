"""
Dedicated Welcome / Home Page Module.
Provides a clean, welcoming landing page for visitors before login.
Separate from the authenticated analytical Dashboard.
"""

import streamlit as st


def render_welcome_page():
    st.markdown('<div class="main-header">EduInsight</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Understand Learning. Predict Performance. Improve Outcomes.</div>',
        unsafe_allow_html=True,
    )

    # Hero Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); color: white; padding: 36px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);">
            <h2 style="color: white; margin-top: 0; font-size: 1.8rem;">Empowering Academic Success Through Data-Driven Insights</h2>
            <p style="font-size: 1.05rem; opacity: 0.95; max-width: 800px; line-height: 1.6;">
                EduInsight bridges predictive machine learning with transparent statistical diagnostics. 
                Forecast final examination marks, categorize academic performance tiers, discover behavioral learning habits, 
                and receive early alerts for atypical study profiles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Call-to-Action Action Buttons
    if st.session_state.get("authenticated", False):
        c_btn1, _ = st.columns([1.5, 4])
        with c_btn1:
            if st.button("🚀 Go to Analytics Dashboard", use_container_width=True, type="primary"):
                st.session_state["nav_selection"] = "Dashboard"
                st.rerun()
    else:
        c_btn1, c_btn2, _ = st.columns([1.2, 1.2, 3])
        with c_btn1:
            if st.button("🔑 Sign In to Account", use_container_width=True, type="primary"):
                st.session_state["nav_selection"] = "Login"
                st.rerun()
        with c_btn2:
            if st.button("📝 Create New Account", use_container_width=True):
                st.session_state["nav_selection"] = "Register"
                st.rerun()

    st.markdown("---")

    # Core System Capabilities Grid
    st.subheader("Platform Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="color: #1E3A8A; margin-top: 0;">🎯 Outcome Forecasting</h4>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    Predict final examination scores using regularized linear and polynomial models calibrated with zero preprocessing data leakage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="color: #1E3A8A; margin-top: 0;">👥 Pattern Clustering</h4>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    Identify unsupervised behavioral archetypes using K-Means, DBSCAN, Hierarchical clustering, and Gaussian Mixture Models.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="color: #1E3A8A; margin-top: 0;">🚨 Anomaly Early Alerts</h4>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    Detect unusual student habit combinations such as acute sleep deprivation or unexpected performance decoupling with clear root-cause explanations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="color: #1E3A8A; margin-top: 0;">🌌 Manifold Projections</h4>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    Visualize high-dimensional student academic feature spaces in 2D using PCA, t-SNE, and neural representation autoencoders.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="color: #1E3A8A; margin-top: 0;">⚡ Optimization Lab</h4>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    Benchmark 8 neural network optimizers (SGD, Adam, AdamW, Lion) and learning-rate schedules with live CPU convergence curves.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="color: #1E3A8A; margin-top: 0;">📡 Model Monitoring</h4>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    Monitor production data distribution drift using Kolmogorov-Smirnov tests and PSI with automated retraining triggers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Workflow Steps
    st.subheader("How It Works")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("#### 1. Sign In or Register\nCreate your user credentials or log in with your email to access the interactive system.")
    with s2:
        st.markdown("#### 2. Input Academic Metrics\nEnter study hours, lecture attendance, assignment marks, and sleep patterns.")
    with s3:
        st.markdown("#### 3. View Analytics & Insights\nInspect predicted final marks, behavioral clusters, reliability diagrams, and advisory recommendations.")
