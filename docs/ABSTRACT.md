# ABSTRACT

**Project Title:** Student Performance Prediction and Learning Pattern Analysis Using Applied Mathematics and Machine Learning  
**Degree Program:** Master of Technology (M.Tech) in Computer Science & Engineering / Artificial Intelligence and Machine Learning  
**Domain:** Applied Mathematics, Supervised Learning, Unsupervised Pattern Recognition, Deep Learning Optimization, and Production ML Engineering (MLOps)

---

### Abstract
Higher education institutions face substantial challenges in identifying at-risk students sufficiently early in an academic semester to deploy targeted pedagogical interventions. While modern Machine Learning (ML) techniques demonstrate remarkable predictive capabilities, machine learning curricula and academic projects frequently suffer from a disjointed divide: mathematical theory is often taught in abstraction without working computational grounding, while empirical ML models are deployed as black-boxes without rigorous linear algebraic, probabilistic, and optimization rigor.

This project presents an end-to-end, reproducible, production-engineered educational analytics platform that unifies **Applied Mathematics (CO-1)**, **Supervised Learning (CO-2)**, **Unsupervised Learning and Manifold Embeddings (CO-3)**, **Deep Learning Training Optimization (CO-4)**, **Statistical Model Evaluation and Fairness (CO-5)**, and **Production MLOps (CO-6)**. 

Using a reproducible synthetic cohort of 1,500 student records characterized by five foundational academic habits (daily study hours, class attendance, continuous assessment assignment scores, prior examination scores, and sleep hours), the system achieves:
1. **Accurate Continuous Forecasting:** Polynomial Regression (Deg 2) and Regularized Linear Models (Ridge, Lasso, ElasticNet) predict continuous examination scores with an RMSE of 3.4986 marks and $R^2 = 0.8914$.
2. **Academic Tier Categorization:** A comparative suite of nine supervised classifiers (Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree, Random Forest, SVM with linear/RBF/polynomial kernels, and XGBoost) categorizes student performance into four tiers (*Needs Support, Average, Good, and Excellent*) achieving up to 84.00% accuracy and 0.7479 Macro F1-score.
3. **Behavioral Archetyping:** K-Means ($k=3$, Silhouette = 0.221), DBSCAN, Agglomerative Hierarchical, and Gaussian Mixture Models identify distinctive study habit cohorts.
4. **Manifold & Latent Compression:** 2D projections via PCA (preserving 68.3% variance in 2 components), t-SNE, and a neural autoencoder bottleneck reveal underlying behavioral manifolds.
5. **Early Anomaly Detection:** Isolation Forest and One-Class SVM flag atypical academic decouple patterns (e.g., severe sleep deprivation or anomalous attendance-grade divergence) with plain-English pedagogical root-cause explanations.
6. **Mathematical Foundations Laboratory:** Working computational modules derive vector geometry, covariance eigendecomposition, Singular Value Decomposition (SVD), expectation-variance identity proofs, Bayes' Theorem, Student's t hypothesis testing, Maximum Likelihood vs. Maximum A Posteriori (MAP) shrinkage, Hessian positive-definiteness for convexity, and Karush-Kuhn-Tucker (KKT) constrained optimization.
7. **Neural Optimization Benchmarking:** A CPU-friendly PyTorch neural network evaluates eight optimizers (SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, and Lion) alongside four dynamic learning-rate schedules (StepLR, CosineAnnealing, Warm Restarts, and OneCycle) and regularization strategies (L2, Dropout, BatchNorm, Early Stopping).
8. **Production MLOps:** Integrated with FastAPI for REST inference, Pydantic data boundary validation, Kolmogorov-Smirnov and Population Stability Index (PSI) data-drift monitoring, automated retraining triggers, A/B testing simulation, Mitchell et al. Model Cards, and 39 automated unit tests via Pytest.

The entire codebase is implemented in Python using free and open-source libraries, requires no proprietary APIs or GPU acceleration, and adheres strictly to academic integrity and responsible AI principles.

**Keywords:** Educational Data Mining, Student Performance Prediction, Applied Mathematics for Machine Learning, Convex Optimization, Eigendecomposition, Probability Calibration, Algorithmic Fairness, MLOps, Data Drift Monitoring.
