# USER AND OPERATIONAL GUIDE

This guide provides step-by-step instructions for installing, configuring, training, running, and testing the Student Performance Machine Learning system.

---

## 1. Environment Setup

### 1.1 Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14 (CPython)
- Windows, macOS, or Linux operating system
- At least 4 GB RAM (CPU-only execution, no GPU required)

### 1.2 Virtual Environment Installation
Open your terminal (PowerShell or Bash) and execute:

```bash
# Navigate to the project root directory
cd student-performance-ml

# Create an isolated Python virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Upgrade pip and install all required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. End-to-End Model Training Pipeline

Before launching the web dashboard or REST API, execute the training pipeline to generate the synthetic dataset, fit all preprocessors, train regression, classification, clustering, and neural network models, and serialize artifacts:

```bash
python -m src.train
```

### Generated Artifacts
The training script populates the `artifacts/` directory with:
- `artifacts/data/`: `student_performance_data.csv`, `train_data.csv`, `val_data.csv`, `test_data.csv`
- `artifacts/models/`: `preprocessor.joblib`, `best_regression_model.joblib`, `best_classification_model.joblib`, `best_clustering_model.joblib`, `best_anomaly_model.joblib`, `best_torch_model.pt`
- `artifacts/metadata/`: `student_exam_regressor_model_card.json`, `baseline_academic_run_runs.jsonl`

---

## 3. Running the Streamlit Web Dashboard

Launch the interactive 14-page academic research dashboard:

```bash
streamlit run app.py
```
Open your web browser and navigate to: **`http://localhost:8501`**

### Navigating the Pages
Use the left sidebar navigation to access:
- **1. Home Dashboard:** High-level metrics, category distribution, and CO alignment cards.
- **2. Student Performance Prediction:** Interactive slider-based inference simulator with anomaly alerts and advising recommendations.
- **3. Dataset Explorer:** Raw data table, descriptive statistics, correlation heatmaps, and dataset generator controls.
- **4. Regression Lab (CO-2):** Model comparison table, residual plots, actual vs. predicted scatters, and bias-variance curves.
- **5. Classification Lab (CO-2):** Accuracy/F1 comparisons, confusion matrices, and interactive hyperparameter sliders.
- **6. Student Clustering Lab (CO-3):** K-Means, DBSCAN, Hierarchical, GMM, silhouette scores, and cluster profiles.
- **7. PCA, t-SNE & UMAP (CO-3):** 2D manifold projections, scree plot, feature loadings, and neural autoencoder latents.
- **8. Anomaly Detection Lab (CO-3):** Isolation Forest and One-Class SVM with root-cause diagnostic explainability.
- **9. Applied Mathematics Lab (CO-1):** Live mathematical derivations: vectors, covariance eigendecomposition, SVD, expectation identity proof, Bayes' theorem, hypothesis testing, MLE vs. MAP, gradients, Hessians, convexity verification, gradient descent step simulator, and Lagrange/KKT conditions.
- **10. Optimization & Training Lab (CO-4):** PyTorch StudentMLP training, benchmark of 8 optimizers, learning rate schedulers, regularization gap, and early stopping.
- **11. Model Evaluation Lab (CO-5):** MAE, RMSE, R², F1, ROC-AUC, reliability diagrams, Platt scaling, and ECE.
- **12. Fairness & Statistical Testing (CO-5):** Paired t-tests, McNemar's test, bootstrap CIs, advanced CV schemes, and algorithmic fairness audits.
- **13. MLOps & Model Monitoring (CO-6):** Kolmogorov-Smirnov drift tests, PSI, automated retraining triggers, A/B testing simulator, and model card viewer.
- **14. About Project & CO Mapping:** Syllabus matrix, tech stack, and academic references.

---

## 4. Running the FastAPI REST Microservice

Launch the production REST API server with live reloading:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation (Swagger UI) is immediately available at:
**`http://127.0.0.1:8000/docs`**

### 4.1 Sample REST API Requests

#### Health Check
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

#### Single Student Online Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "study_hours_per_day": 7.5,
       "attendance_percentage": 92.0,
       "assignment_score": 88.0,
       "previous_exam_score": 84.0,
       "sleep_hours": 7.5
     }'
```
**Sample Response:**
```json
{
  "predicted_final_exam_score": 83.42,
  "predicted_performance_category": "Excellent",
  "anomaly_detected": false,
  "anomaly_score": 0.0821,
  "confidence_tier": "High"
}
```

#### Batch Inference Request
```bash
curl -X POST "http://127.0.0.1:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{
       "students": [
         {
           "study_hours_per_day": 4.0,
           "attendance_percentage": 65.0,
           "assignment_score": 50.0,
           "previous_exam_score": 45.0,
           "sleep_hours": 6.0
         },
         {
           "study_hours_per_day": 8.0,
           "attendance_percentage": 95.0,
           "assignment_score": 90.0,
           "previous_exam_score": 85.0,
           "sleep_hours": 7.5
         }
       ]
     }'
```

---

## 5. Running Automated Unit Tests

Execute the comprehensive Pytest verification test suite:

```bash
python -m pytest -v
```

This executes all 39 unit tests across data generation, preprocessing, applied mathematics, supervised learning, unsupervised learning, deep learning optimization, model evaluation, MLOps drift detection, and FastAPI endpoints.
