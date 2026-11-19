"""
System Architecture and Documentation Module.
Clean overview of technical specifications, modular architecture,
and technology stack without academic/curriculum jargon.
"""

import streamlit as st


def render_page():
    st.markdown('<div class="main-header">System Architecture & Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Technical specifications, functional modules, and design principles.</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_modules, tab_tech, tab_refs = st.tabs([
        "System Summary",
        "Functional Modules",
        "Technology Stack",
        "Scientific References",
    ])

    with tab_overview:
        st.subheader("System Overview")
        st.markdown("""
        **Platform Name:** EduInsight  
        **Tagline:** *Understand Learning. Predict Performance. Improve Outcomes.*  
        **Project Title:** Student Performance Prediction and Learning Pattern Analysis

        **System Purpose:**
        An integrated predictive analytics and decision-support platform designed to forecast student exam outcomes,
        diagnose study habits, detect atypical learning patterns, and provide transparent model diagnostics.

        **Core Pillars:**
        1. **Forecasting (Regression):** Continuous prediction of examination marks using regularized linear and polynomial models.
        2. **Classification:** Categorization into performance tiers (*Needs Support, Average, Good, Excellent*).
        3. **Pattern Clustering:** Unsupervised grouping of learning habits using K-Means, DBSCAN, Hierarchical, and GMM.
        4. **Manifold Projections:** Visual representation of student space via PCA, t-SNE, and neural autoencoders.
        5. **Anomaly Detection:** Flagging atypical student profiles with root-cause diagnostic explanations.
        6. **Optimization & Training:** PyTorch neural network training benchmarking multiple modern optimizers.
        7. **Model Governance:** Real-time data drift monitoring, calibration curves, and automated retraining triggers.
        """)

    with tab_modules:
        st.subheader("Functional Module Architecture")
        st.markdown("""
        | Module | Component Description | Core Techniques |
        | :--- | :--- | :--- |
        | **Foundations** | Mathematical operations | Vector norms, Covariance Eigendecomposition, SVD, Bayes theorem, Gradient & Hessian |
        | **Supervised Models** | Continuous & discrete prediction | Linear, Polynomial, Ridge, Lasso, ElasticNet, Logistic, Naive Bayes, KNN, Decision Tree, Random Forest, SVM, XGBoost |
        | **Unsupervised Patterns** | Habit clustering & projection | K-Means, DBSCAN, Hierarchical, GMM, PCA, t-SNE, Isolation Forest, One-Class SVM |
        | **Neural Optimization** | Deep learning training | PyTorch MLP, 8 Optimizers (SGD, Adam, AdamW, Lion, etc.), Learning rate schedulers, Regularization |
        | **Evaluation** | Quality & fairness diagnostics | Reliability diagrams, Platt scaling, Paired t-tests, McNemar's test, Demographic Parity |
        | **Production MLOps** | Serving & lifecycle | FastAPI REST API, Kolmogorov-Smirnov drift detection, PSI, Automated retraining triggers |
        """)

    with tab_tech:
        st.subheader("Technology Stack")
        st.markdown("""
        - **Core Language:** Python 3.11+
        - **Frontend Interface:** Streamlit with custom responsive styling
        - **REST Microservice:** FastAPI & Uvicorn
        - **Data Processing:** Pandas & NumPy
        - **Mathematical Modeling:** SciPy
        - **Machine Learning:** Scikit-Learn & XGBoost
        - **Deep Learning:** PyTorch (CPU-optimized)
        - **Data Visualization:** Matplotlib & Seaborn
        - **Model Persistence:** Joblib
        - **Input Validation:** Pydantic V2
        - **Automated Testing:** Pytest
        """)

    with tab_refs:
        st.subheader("Scientific References")
        st.markdown("""
        1. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer.
        2. **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization.* Cambridge University Press.
        3. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.
        4. **Chen, X., et al. (2023).** *Symbolic Discovery of Optimization Algorithms.* arXiv:2302.06675.
        5. **Guo, C., et al. (2017).** *On Calibration of Modern Neural Networks.* ICML.
        6. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** *Isolation Forest.* IEEE ICDM.
        """)


if __name__ == "__main__":
    render_page()
