# Student Performance Prediction and Learning Pattern Analysis Using Machine Learning with Applied Mathematics

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
![API](https://img.shields.io/badge/API-FastAPI-green.svg)
![Tests](https://img.shields.io/badge/Tests-39%20Passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey.svg)

A complete, beginner-friendly, fully functional M.Tech academic project integrating **Applied Mathematics** and **Machine Learning** across Course Outcomes **CO-1 through CO-6**.

---

## 📌 Project Overview
Higher education institutions need transparent, explainable decision-support systems to forecast student academic outcomes and detect at-risk study patterns early in the semester. This repository provides an end-to-end educational machine learning system covering:
- **CO-1: Mathematical Foundations for ML:** Live derivations of vector algebra, covariance eigendecomposition, SVD, expectation/variance proofs, Bayes' Theorem, hypothesis testing, MLE vs. MAP shrinkage, Hessians, convexity verification, gradient descent step simulation, and KKT conditions.
- **CO-2: Supervised Learning:** Comparative benchmarking of 5 regression models (Linear, Polynomial Deg 2, Ridge, Lasso, ElasticNet) and 9 classifiers (Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree, Random Forest, SVM with linear/RBF/poly kernels, XGBoost) with leak-free pipelines and bootstrap bias-variance decomposition.
- **CO-3: Unsupervised Learning & Embeddings:** K-Means, DBSCAN, Hierarchical, GMM; PCA, t-SNE, UMAP; Isolation Forest and One-Class SVM with root-cause pedagogical explainability; PyTorch neural autoencoder representations and educational text embeddings.
- **CO-4: Deep Learning & Optimization:** CPU-friendly PyTorch `StudentMLP` architecture benchmarking 8 optimizers (SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, Lion), learning rate schedules (StepLR, Cosine, Warm Restarts, OneCycle), and regularization (L2, Dropout, BatchNorm, Early Stopping).
- **CO-5: Model Evaluation & Fairness:** MAE, RMSE, R², Accuracy, Precision, Recall, Macro F1, ROC-AUC, PR-AUC, MCC; reliability calibration diagrams and Expected Calibration Error (ECE); paired t-tests, McNemar's tests, bootstrap confidence intervals, Stratified/Group/TimeSeries CV, and demographic parity audits.
- **CO-6: Production ML Engineering / MLOps:** Model serialization with Joblib, FastAPI REST microservice with Pydantic V2 validation, Kolmogorov-Smirnov and PSI feature drift monitoring, automated retraining triggers, A/B testing simulation, Mitchell et al. Model Cards, and automated GitHub Actions CI workflow.

---

## 🗂️ Project Directory Structure

```text
student-performance-ml/
├── app.py                     # Main Streamlit web dashboard entrypoint
├── api.py                     # Production FastAPI REST microservice
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git ignore rules for virtualenvs, bytecode & artifacts
├── .env.example               # Environment variables template
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (Python 3.11 & 3.12)
├── src/
│   ├── __init__.py
│   ├── train.py               # Reproducible end-to-end model training script
│   ├── data/
│   │   ├── generator.py       # Configurable synthetic student dataset generator
│   │   └── loader.py          # Leak-free Train/Val/Test splitting & loading
│   ├── preprocessing/
│   │   └── pipeline.py        # StandardScaler transformer with fit/transform separation
│   ├── mathematics/
│   │   ├── linear_algebra.py  # Vectors, matrices, covariance, eigenvalues, SVD
│   │   ├── probability_stats.py # Statistics, Bayes' theorem, MLE/MAP, hypothesis testing
│   │   └── calculus_optimization.py # Gradients, Hessians, convexity, GD, Lagrange/KKT
│   ├── supervised/
│   │   ├── regression.py      # Linear, Poly, Ridge, Lasso, ElasticNet pipelines
│   │   ├── classification.py  # Logistic, Naive Bayes, KNN, Tree, Forest, SVM, XGBoost
│   │   ├── bias_variance.py   # Bootstrap empirical bias-variance decomposition
│   │   └── search.py          # GridSearchCV and RandomizedSearchCV
│   ├── unsupervised/
│   │   ├── clustering.py      # K-Means, DBSCAN, Hierarchical, GMM
│   │   ├── dim_reduction.py   # PCA (scree plot & loadings), t-SNE, UMAP
│   │   ├── anomaly.py         # Isolation Forest, One-Class SVM & root-cause explanations
│   │   └── embeddings.py      # PyTorch Autoencoder & educational NLP text embeddings
│   ├── optimization/
│   │   ├── torch_models.py    # PyTorch StudentMLP feedforward architecture
│   │   ├── optimizers_lab.py  # Benchmark of 8 optimizers (including Lion)
│   │   ├── schedulers_lab.py  # Dynamic learning rate schedules
│   │   └── regularization_lab.py # L2, Dropout, BatchNorm, Early Stopping, Xavier/He init
│   ├── evaluation/
│   │   ├── metrics.py         # Regression & classification metrics, error taxonomy
│   │   ├── calibration.py     # Reliability diagrams, Platt scaling, Isotonic, ECE
│   │   ├── statistical_tests.py # Paired t-test, McNemar's test, bootstrap CIs
│   │   ├── cross_validation.py # Stratified, Group-Aware, TimeSeries CV
│   │   └── fairness.py        # Demographic Parity, Equal Opportunity, Disparate Impact
│   ├── mlops/
│   │   ├── drift.py           # Kolmogorov-Smirnov test & Population Stability Index (PSI)
│   │   ├── monitoring.py      # Rolling RMSE tracking & automated retraining trigger
│   │   ├── model_card.py      # Mitchell et al. Model Card JSON generator
│   │   └── versioning.py      # SHA-256 data checksums & Git commit tracking
│   └── utils/
│       ├── config.py          # Constants, feature names, file paths
│       └── helpers.py         # Fixed random seeds & formatting utilities
├── pages/
│   ├── 1_Home_Dashboard.py
│   ├── 2_Student_Performance_Prediction.py
│   ├── 3_Dataset_Explorer.py
│   ├── 4_Regression_Lab.py
│   ├── 5_Classification_Lab.py
│   ├── 6_Student_Clustering_Lab.py
│   ├── 7_PCA_tSNE_UMAP_Visualization.py
│   ├── 8_Anomaly_Detection_Lab.py
│   ├── 9_Applied_Mathematics_Lab.py
│   ├── 10_Optimization_and_Training_Lab.py
│   ├── 11_Model_Evaluation_Lab.py
│   ├── 12_Fairness_and_Statistical_Testing_Lab.py
│   ├── 13_MLOps_and_Model_Monitoring.py
│   └── 14_About_Project_and_CO_Mapping.py
├── artifacts/
│   ├── data/                  # Generated CSV splits (train, val, test)
│   ├── models/                # Serialized Joblib & PyTorch model weights
│   └── metadata/              # Model cards & experiment tracking logs
├── tests/
│   ├── test_data.py           # Dataset generator, bounds, reproducibility
│   ├── test_preprocessing.py  # Scaler fitting, transformation, serialization
│   ├── test_mathematics.py    # Linear algebra, Bayes, variance identity, KKT
│   ├── test_supervised.py     # Regression, classification, bias-variance
│   ├── test_unsupervised.py   # Clustering, PCA, Isolation Forest, autoencoder
│   ├── test_optimization.py   # PyTorch forward pass, Lion optimizer, early stopping
│   ├── test_evaluation.py     # Metrics, ECE, paired t-test, fairness
│   ├── test_mlops.py          # KS drift, PSI, retraining triggers, model card
│   └── test_api.py            # FastAPI TestClient endpoints & boundary validation
└── docs/
    ├── ABSTRACT.md            # Formal academic abstract
    ├── CO_MAPPING.md          # Syllabus alignment matrix (CO-1 to CO-6)
    ├── METHODOLOGY.md         # Architecture diagrams & algorithmic details
    ├── PROJECT_REPORT.md      # Full academic project report template
    └── USER_GUIDE.md          # Installation, execution & API guide
```

---

## ⚡ Quick Start & Execution

### 1. Environment Setup
```bash
# Clone the repository and enter the directory
cd student-performance-ml

# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Reproducible Model Training
```bash
python -m src.train
```
*Output Summary:*
- Dataset generated ($N=1,500$ records, Seed = 42).
- Zero-leakage preprocessor fitted and saved to `artifacts/models/preprocessor.joblib`.
- Best Regression Model (`Polynomial Regression Deg 2`, RMSE = 3.4986 marks, $R^2 = 0.8914$) saved to `artifacts/models/best_regression_model.joblib`.
- Best Classification Model (`Naive Bayes`, Macro F1 = 0.7479, Accuracy = 84.00%) saved to `artifacts/models/best_classification_model.joblib`.
- Unsupervised K-Means ($k=3$) and Isolation Forest anomaly models saved.
- PyTorch `StudentMLP` trained on CPU and weights saved to `artifacts/models/best_torch_model.pt`.
- Mitchell et al. Model Card generated in `artifacts/metadata/`.

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 4. Launch FastAPI REST Service (Optional)
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation: **`http://127.0.0.1:8000/docs`**

---

## 🧪 Automated Testing

Execute the comprehensive test suite:
```bash
python -m pytest -v
```
**Test Results:**
- `39 passed in 20.09s` with zero failures across all modules.

---

## 📊 Experimental Results Summary

### Regression Benchmark (CO-2)
| Model | MAE | RMSE | $R^2$ |
| :--- | :---: | :---: | :---: |
| **Polynomial Regression (Deg 2)** | **2.7749** | **3.4986** | **0.8914** |
| Linear Regression | 2.8561 | 3.5693 | 0.8869 |
| Ridge Regression ($\alpha=1.0$) | 2.8567 | 3.5702 | 0.8869 |
| Lasso Regression ($\alpha=0.1$) | 2.8699 | 3.5893 | 0.8857 |
| ElasticNet ($\alpha=0.1, \rho=0.5$) | 2.9113 | 3.6413 | 0.8823 |

### Classification Benchmark (CO-2)
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes (Gaussian)** | 0.8400 | 0.7287 | 0.7863 | **0.7479** | 0.9638 | 0.7113 |
| **SVM (RBF Kernel)** | 0.8178 | 0.6639 | 0.9073 | 0.7101 | **0.9696** | 0.7056 |
| **XGBoost Classifier** | 0.8311 | 0.8164 | 0.6330 | 0.6889 | 0.9627 | 0.6852 |
| **K-Nearest Neighbors (KNN)** | **0.8444** | **0.9206** | 0.5727 | 0.6339 | 0.9391 | 0.7067 |
| **Random Forest** | 0.8089 | 0.6490 | 0.6913 | 0.6611 | 0.9579 | 0.6608 |

---

## 🚀 GitHub Repository Upload Instructions

To publish this project to your GitHub account, run the following exact Git commands:

```bash
# Initialize local git repository
git init -b main

# Stage all files (respects .gitignore)
git add .

# Create initial commit
git commit -m "feat: complete student performance prediction project (CO-1 to CO-6)"

# Create a new repository on https://github.com (e.g. named student-performance-ml)
# Then link your remote and push (replace <YOUR_USERNAME> and <REPO_NAME> with your details):
git remote add origin https://github.com/<YOUR_USERNAME>/student-performance-ml.git

# Push to main branch
git push -u origin main
```

---

## ⚖️ Academic Integrity & Ethical Use
- **Synthetic Data:** All student records are synthetically generated using fixed random seeds. No private student personal identifiable information (PII) is included or processed.
- **Decision Support Only:** Predicted marks and performance tiers are intended strictly for educational advising, diagnostic early alerts, and pedagogical research. They must never be used for autonomous grading or punitive administrative actions.
- **Fairness:** The included demographic parity and disparate impact audits demonstrate methodology on synthetic groups and must be re-evaluated responsibly before any institutional deployment.

---

## 📜 License
This academic project is distributed under the **Apache-2.0 License** for free and open educational use.
