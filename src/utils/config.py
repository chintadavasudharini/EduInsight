"""
Configuration settings and constants for Student Performance ML Project.
Aligned with Course Outcomes CO-1 through CO-6.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
for directory in [ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 1500  # >= 1200 as required

FEATURE_NAMES = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_score",
    "previous_exam_score",
    "sleep_hours",
]

TARGET_REGRESSION = "final_exam_score"
TARGET_CLASSIFICATION = "performance_category"

# Categorization Thresholds (Academic grading)
# 0-39: Needs Support, 40-59: Average, 60-79: Good, 80-100: Excellent
CATEGORY_BINS = [-1.0, 39.5, 59.5, 79.5, 100.0]
CATEGORY_LABELS = ["Needs Support", "Average", "Good", "Excellent"]

# Categorical mapping
LABEL_TO_INT = {label: i for i, label in enumerate(CATEGORY_LABELS)}
INT_TO_LABEL = {i: label for i, label in enumerate(CATEGORY_LABELS)}

# Dataset file paths
RAW_DATA_PATH = DATA_DIR / "student_performance_data.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
VAL_DATA_PATH = DATA_DIR / "val_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

# Model file paths
REGRESSION_MODEL_PATH = MODELS_DIR / "best_regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "best_classification_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
CLUSTERING_MODEL_PATH = MODELS_DIR / "best_clustering_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "best_anomaly_model.joblib"
OPTIMIZER_MODEL_PATH = MODELS_DIR / "best_torch_model.pt"

# API & Server Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
STREAMLIT_PORT = 8501
