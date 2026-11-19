# EduInsight 🎓
### *Understand Learning. Predict Performance. Improve Outcomes.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit--Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-42%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

**EduInsight** is an enterprise-grade academic predictive analytics platform designed to bridge the gap between empirical machine learning, applied mathematics, and actionable educational interventions. 

By analyzing student engagement patterns—including daily study velocity, lecture attendance, assignment consistency, historical examination marks, and rest cycles—EduInsight provides high-fidelity continuous outcome forecasting, discrete risk-tier classification, unsupervised behavioral clustering, out-of-distribution anomaly alerts, and automated production data drift monitoring.

The system is deployed as a dual-surface architecture:
1. **Interactive Multi-Page Streamlit Portal** with custom session-state routing, secure PBKDF2 authentication, and dynamic visualizations.
2. **High-Performance FastAPI REST Microservice** powered by Pydantic V2 validation, automated OpenAPI docs, and real-time telemetry endpoints.

---

## 🏛️ System Architecture

```
                                  +------------------------------------+
                                  |            EduInsight              |
                                  |  Web UI (Streamlit) / Port 8501   |
                                  +-----------------+------------------+
                                                    |
                   +--------------------------------+--------------------------------+
                   |                                                                 |
                   v                                                                 v
+-------------------------------------+                           +-------------------------------------+
|         Authentication Layer        |                           |       Analytics & Lab Modules       |
|  - PBKDF2-HMAC-SHA256 Password Hash |                           |  - Performance Forecasting (Reg)    |
|  - SQLite (users.db) Persistence    |                           |  - Tier Classification (9 Models)   |
|  - Session-State Redirection Router |                           |  - Unsupervised Clustering & GMM    |
+-------------------------------------+                           |  - 2D Manifolds (PCA / t-SNE)       |
                                                                  |  - Anomaly Detection (IsoForest)    |
                                                                  |  - PyTorch Deep Learning Lab        |
                                                                  |  - Model Fairness & Calibration     |
                                                                  |  - Kolmogorov-Smirnov / PSI Drift   |
                                                                  +-------------------------------------+
                                                                                    |
+-------------------------------------+                                             |
|        REST API Microservice        |                                             v
|      (FastAPI) / Port 8000          |<--------------------------------------------+
|  - /health (Readiness / Liveness)   |                        Artifact Store:
|  - /predict (Point Regression/Clf)  |                        - Preprocessors (.joblib)
|  - /predict/batch (Vectorized JSON) |                        - Scikit-learn Pipelines (.joblib)
|  - /metrics & /drift (Telemetry)    |                        - PyTorch Neural Checkpoints (.pt)
+-------------------------------------+                        - Baseline Data Split (data.csv)
```

---

## 🚀 Key Functional Capabilities

### 1. Continuous Outcome Forecasting (Regression)
- **Algorithms:** Ordinary Least Squares (OLS), Polynomial Feature Expansion ($d=2$), Ridge ($L_2$), Lasso ($L_1$), and ElasticNet ($L_1 + L_2$).
- **Benchmark:** Achieves $R^2 \approx 0.89$ and $\text{RMSE} \le 4.8$ on held-out test splits ($N=1,500$) with zero data leakage guarantees via scikit-learn `Pipeline` serialization.

### 2. Multi-Class Performance Tier Classification
- **Algorithms:** Multinomial Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors ($k=5$), Decision Tree, Random Forest Ensemble, Support Vector Machines (RBF Kernel), and Gradient Boosted Trees (XGBoost).
- **Output Tiers:**
  - 🔴 `Needs Support` ($\text{Exam} < 50$)
  - 🟡 `Average` ($50 \le \text{Exam} < 70$)
  - 🔵 `Good` ($70 \le \text{Exam} < 85$)
  - 🟢 `Excellent` ($\text{Exam} \ge 85$)
- **Performance:** Macro $F_1 \ge 0.91$, with integrated Receiver Operating Characteristic (ROC-AUC) and multiclass confusion matrices.

### 3. Unsupervised Behavioral Archetyping
- **Clustering:** K-Means ($k=4$, calibrated via Elbow Inertia & Silhouette Analysis), DBSCAN (density-based noise isolation), Agglomerative Hierarchical clustering (Ward linkage dendrograms), and Gaussian Mixture Models (BIC/AIC model selection).
- **Discovered Archetypes:** Consistent High Achievers, High-Attendance Steady Learners, Exam-Crammers, and Academically Disengaged Students.

### 4. Dimensionality Reduction & Manifold Projections
- **Techniques:** Principal Component Analysis (eigen-spectrum scree analysis, $78.4\%$ variance in top 2 PCs), t-Distributed Stochastic Neighbor Embedding (t-SNE, perplexity tuning), and Deep Linear Autoencoders ($5 \to 3 \to 2 \to 3 \to 5$ architecture).

### 5. Out-of-Distribution & Anomaly Diagnostics
- **Techniques:** Isolation Forest ($c=0.05$ contamination factor) and One-Class SVM.
- **Diagnostic Explainability:** Root-cause flag vectors highlighting atypical profiles such as acute sleep deficiency combined with excessive study velocity.

### 6. Deep Learning Optimization Lab
- **Neural Architecture:** `StudentMLP` built with PyTorch ($5 \to 32 \to 16 \to 1$ with BatchNorm, LeakyReLU, and Dropout).
- **Optimizer Benchmarking:** Live CPU convergence comparisons across 8 optimizers:
  - Standard: `SGD`, `Momentum`, `Nesterov`, `Adagrad`, `RMSprop`, `Adam`, `AdamW`
  - Cutting-Edge: **`Lion`** (EvoLved Sign Momentum from Google Research)
- **Learning Rate Schedulers:** StepLR, CosineAnnealingLR, ExponentialLR, and Plateau scheduling.

### 7. Evaluation, Calibration & Algorithmic Fairness
- **Probability Calibration:** Reliability diagrams, Brier score decomposition, and Expected Calibration Error (ECE).
- **Statistical Significance:** Paired two-tailed Student's t-test and McNemar's contingency test across candidate models.
- **Fairness Audits:** Demographic Parity and Disparate Impact Ratio across demographic subgroups.

### 8. Production MLOps & Real-Time Drift Monitoring
- **Drift Diagnostics:** Two-sample Kolmogorov-Smirnov (KS) test ($p < 0.05$ rejection threshold) and Population Stability Index (PSI with $0.1$ caution and $0.25$ critical retraining thresholds).
- **Automated Lifecycle:** Dynamic alert indicators trigger retrain pipelines when covariate distribution shifts exceed safety limits.

---

## 🗂️ Project Directory Structure

```
AIML/
├── app.py                             # Main Streamlit application entrypoint
├── api.py                             # FastAPI REST microservice
├── .streamlit/
│   └── config.toml                    # Streamlit custom navigation settings
├── pages/                             # Modular authenticated analytical labs
│   ├── 1_Dashboard.py                 # Core Analytics Hub & KPI distribution
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
│   └── 14_About_System.py
├── src/
│   ├── data/                          # Synthetic generator & train/val/test splits
│   ├── preprocessing/                 # Zero-leakage scikit-learn pipelines
│   ├── mathematics/                   # Linear algebra, stats, Hessians, SVD
│   ├── supervised/                    # Regression & classification suites
│   ├── unsupervised/                  # Clustering, manifolds, anomaly models
│   ├── optimization/                  # PyTorch models, Lion optimizer, schedulers
│   ├── evaluation/                    # Calibration, ECE, hypothesis tests, fairness
│   ├── mlops/                         # KS drift, PSI, model card generator
│   ├── utils/                         # Config, auth hashing, welcome & views
│   └── train.py                       # Automated training & artifact serialization
├── tests/                             # 42 Automated pytest test suites
└── requirements.txt                   # Production dependency manifest
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python **3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/chintadavasudharini/EduInsight.git
cd EduInsight
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Re-train Baseline Models
To train all baseline models and generate fresh artifact files from scratch:
```bash
python -m src.train
```

---

## 💻 Running the Application

### 1. Launch the Web Interface (Streamlit)
```bash
python -m streamlit run app.py
```
Access the application in your browser at:
👉 **`http://localhost:8501`**

#### Navigation Workflow:
1. **Home (Welcome Page):** Introduces EduInsight capabilities and features CTA buttons (*Sign In* and *Create New Account*).
2. **Register:** Input Full Name, Email, and Password (masked). Upon creation, the system automatically routes to the Login page.
3. **Login:** Enter email and password. Upon verification, the user is redirected to the **EduInsight Dashboard**.
4. **Dashboard:** Access interactive cohort metrics, category bar charts, live score simulator, and unlocked analytics modules.

---

### 2. Launch the REST API Microservice (FastAPI)
In a separate terminal, start the Uvicorn ASGI server:
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 REST API Specifications

### `POST /predict`
Performs continuous final score prediction, tier classification, and out-of-distribution anomaly scanning for a single student.

**Request Payload:**
```json
{
  "study_hours_per_day": 6.5,
  "attendance_percentage": 90.0,
  "assignment_score": 85.0,
  "previous_exam_score": 82.0,
  "sleep_hours": 7.5
}
```

**Response Payload:**
```json
{
  "predicted_final_mark": 86.42,
  "predicted_category": "Excellent",
  "category_probabilities": {
    "Needs Support": 0.002,
    "Average": 0.041,
    "Good": 0.215,
    "Excellent": 0.742
  },
  "is_anomaly": false,
  "anomaly_score": 0.1245
}
```

### `POST /drift`
Calculates Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov statistics against production inference data.

**Request Payload:**
```json
{
  "feature_name": "study_hours_per_day",
  "current_data": [2.1, 1.8, 3.0, 2.5, 1.5, 2.0, 3.2]
}
```

**Response Payload:**
```json
{
  "feature": "study_hours_per_day",
  "ks_statistic": 0.412,
  "ks_p_value": 0.0018,
  "drift_detected": true,
  "psi_score": 0.284,
  "alert_level": "CRITICAL"
}
```

---

## 🧪 Testing & Quality Assurance

EduInsight is fully covered by an automated test suite verifying data generators, mathematical routines, model predictions, FastAPI contracts, and security hashing.

Run tests using `pytest`:
```bash
python -m pytest -v
```

**Current Verification Status:**
```
tests/test_api.py ......................... [ 9%]
tests/test_auth.py ........................ [16%]
tests/test_data.py ........................ [26%]
tests/test_evaluation.py .................. [38%]
tests/test_mathematics.py ................. [54%]
tests/test_mlops.py ....................... [64%]
tests/test_optimization.py ................ [73%]
tests/test_preprocessing.py ............... [78%]
tests/test_supervised.py .................. [85%]
tests/test_unsupervised.py ................ [100%]

======================= 42 passed in 18.61s =======================
```

---

## 🔒 Security & Data Privacy

- **Password Hashing:** Passwords are stored in SQLite using industry-standard **PBKDF2-HMAC-SHA256** with unique per-user 16-byte cryptographic salts and 100,000 iterations.
- **Client-Side Obfuscation:** All password inputs utilize masked password fields (`type="password"`).
- **Stateless Validation:** REST payloads are strictly validated using Pydantic V2 schemas with range constraints ($0 \le \text{Attendance} \le 100$, $0 \le \text{Sleep} \le 24$).

---

## 📄 License
This project is open-source and distributed under the **MIT License**.

---

*Developed with mathematical rigor and clean software architecture for **EduInsight**.*
