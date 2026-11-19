"""
Page 14: About Project and CO Mapping.
Complete M.Tech academic project overview, Course Outcome mapping matrix,
technology stack specifications, execution guide, and verified academic references.
"""

import streamlit as st


def render_page():
    st.markdown('<div class="main-header">📖 About Project & CO Syllabus Mapping</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">M.Tech Academic Project: Student Performance Prediction and Learning Pattern Analysis Using Applied Mathematics and Machine Learning.</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_co_map, tab_tech, tab_refs = st.tabs([
        "🎓 Project Overview",
        "🎯 CO-1 to CO-6 Mapping Matrix",
        "💻 Technology Stack",
        "📚 Academic References",
    ])

    with tab_overview:
        st.subheader("Executive Project Summary")
        st.markdown("""
        **Project Title:** Student Performance Prediction and Learning Pattern Analysis Using Applied Mathematics and Machine Learning

        **Academic Context:**
        Developed for an advanced M.Tech curriculum bridging rigorous **Applied Mathematics** with modern **Machine Learning & MLOps**.
        The project implements an end-to-end, fully reproducible, educational decision-support platform designed to forecast student exam outcomes,
        diagnose learning habits, detect atypical burnout patterns, and demonstrate the foundational mathematics underlying ML algorithms.

        **Core Functional Pillars:**
        1. **Forecasting (Regression):** Continuous prediction of final examination marks (0-100) using regularized linear and polynomial models.
        2. **Grading (Classification):** Categorization into standardized academic tiers (*Needs Support, Average, Good, Excellent*).
        3. **Behavioral Archetyping (Clustering):** Unsupervised grouping of learning habits using K-Means, DBSCAN, Hierarchical, and GMM.
        4. **Manifold Analysis (Dimensionality Reduction):** 2D visual projection of student spaces via PCA, t-SNE, and representation autoencoders.
        5. **Early Warning (Anomaly Detection):** Isolation Forest and One-Class SVM to flag atypical student records and burnout signals.
        6. **Mathematical Foundations (Applied Mathematics Lab):** Derivations and live code for linear algebra, probability, Bayesian inference, and constrained optimization.
        7. **Deep Learning Optimization (PyTorch Lab):** Multi-optimizer convergence benchmark (SGD, Adam, AdamW, Lion, schedulers).
        8. **Governance & Production (MLOps):** Data drift detection (KS test, PSI), model cards, and FastAPI endpoints.
        """)

    with tab_co_map:
        st.subheader("Course Outcome (CO-1 to CO-6) Alignment Matrix")
        st.markdown("""
        | Course Outcome | Syllabus Topic Description | Implemented Module & Location | Working Demonstration |
        | :--- | :--- | :--- | :--- |
        | **CO-1** | **Mathematical Foundations for ML** | `src/mathematics/`, Page 9 | Vector dot products, SVD, covariance eigendecomposition, Bayes Theorem, expectation variance identity, MLE/MAP, Hessians, convexity chord test, Gradient Descent, Lagrange & KKT conditions. |
        | **CO-2** | **Supervised Learning** | `src/supervised/`, Pages 4 & 5 | Linear, Poly (Deg 2), Ridge, Lasso, ElasticNet regression; Logistic, Naive Bayes, KNN, Decision Tree, Random Forest, SVM (Linear, RBF, Poly), XGBoost; Bias-Variance tradeoff, Grid & Random search. |
        | **CO-3** | **Unsupervised Learning & Embeddings** | `src/unsupervised/`, Pages 6, 7 & 8 | K-Means, DBSCAN, Hierarchical, GMM; PCA scree plot, t-SNE, UMAP; Isolation Forest, One-Class SVM; PyTorch Autoencoder embeddings, NLP cosine similarity text demo. |
        | **CO-4** | **Optimization and Training** | `src/optimization/`, Page 10 | PyTorch StudentMLP neural network, benchmark of 8 optimizers (SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW, Lion); Step, Cosine, Warm Restarts, OneCycle schedulers; L1/L2, Dropout, BatchNorm, Early Stopping, Xavier & He init. |
        | **CO-5** | **Model Evaluation & Selection** | `src/evaluation/`, Pages 11 & 12 | MAE, RMSE, R², Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC; Reliability diagrams, Platt & Isotonic calibration, ECE; Paired t-test, McNemar test, Bootstrap CIs, Stratified/Group/TimeSeries CV, Fairness analysis. |
        | **CO-6** | **Production ML / MLOps** | `src/mlops/`, `api.py`, Page 13 | Joblib persistence, Pydantic input validation, FastAPI REST endpoints, Kolmogorov-Smirnov & PSI drift detection, automated retraining trigger, A/B testing simulator, Mitchell et al. Model Cards, CI/CD workflow. |
        """)

    with tab_tech:
        st.subheader("Pure Open-Source Technology Stack")
        st.markdown("""
        - **Python 3.11+ / 3.14**: Core runtime environment
        - **Frontend User Interface:** Streamlit (v1.65.0) with responsive custom CSS styling
        - **REST API Microservice:** FastAPI (v0.141.1) and Uvicorn
        - **Data Processing & Tabular Ops:** Pandas (v3.0.6) and NumPy (v2.5.3)
        - **Applied Mathematics & Optimization:** SciPy (v1.18.1)
        - **Classical Machine Learning:** Scikit-Learn (v1.9.1) and XGBoost (v3.4.1)
        - **Deep Learning & Gradient Lab:** PyTorch (v2.15.0 CPU-friendly)
        - **Scientific Visualization:** Matplotlib (v3.11.2) and Seaborn (v0.13.2)
        - **Model Persistence:** Joblib (v1.6.0)
        - **Input Validation & Schemas:** Pydantic (v2.13.4)
        - **Automated Unit Testing:** Pytest (v9.1.1)
        - **Continuous Integration:** GitHub Actions CI workflow
        """)

    with tab_refs:
        st.subheader("Academic References & Foundational Literature")
        st.markdown("""
        1. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer, New York. (Foundations of Bayesian estimation, linear models, and eigendecomposition).
        2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press, Cambridge, MA. (Neural network optimization, gradient descent dynamics, and regularization).
        3. **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization.* Cambridge University Press. (Lagrange duality, KKT conditions, and convex analysis).
        4. **Mitchell, M., et al. (2019).** *Model Cards for Model Reporting.* Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*), pp. 220–229.
        5. **Chen, X., et al. (2023).** *Symbolic Discovery of Optimization Algorithms.* arXiv preprint arXiv:2302.06675. (Introduction and formulation of the Lion optimizer).
        6. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* International Conference on Machine Learning (ICML), pp. 1321–1330.
        7. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** *Isolation Forest.* IEEE International Conference on Data Mining (ICDM), pp. 413–422.
        8. **Dwork, C., et al. (2012).** *Fairness through Awareness.* Innovations in Theoretical Computer Science (ITCS), pp. 214–226.
        """)


if __name__ == "__main__":
    render_page()
