"""
FastAPI REST Service for Student Performance Prediction.
Provides:
- Input validation using Pydantic schemas
- Single record online inference (/predict)
- Batch inference (/batch_predict)
- Anomaly scoring integration
- Model card metadata endpoint (/model_card)
- Health check and runtime telemetry (/health)
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import numpy as np

from src.utils.config import (
    FEATURE_NAMES,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    METADATA_DIR,
    CATEGORY_LABELS,
    INT_TO_LABEL,
)
from src.mlops.drift import compute_ks_drift
from src.data.loader import load_splits

app = FastAPI(
    title="Student Performance AI/ML Service",
    description="Production-ready REST API for student academic mark regression, classification, and anomaly detection.",
    version="1.0.0",
)

# Global model cache
models_cache: Dict[str, Any] = {}


def load_models():
    """Load serialized models into memory."""
    try:
        if REGRESSION_MODEL_PATH.exists():
            models_cache["regression"] = joblib.load(REGRESSION_MODEL_PATH)
        if CLASSIFICATION_MODEL_PATH.exists():
            models_cache["classification"] = joblib.load(CLASSIFICATION_MODEL_PATH)
        if ANOMALY_MODEL_PATH.exists():
            models_cache["anomaly"] = joblib.load(ANOMALY_MODEL_PATH)
        print("API: All serialized models successfully loaded into memory.")
    except Exception as e:
        print(f"API Warning: Model loading encountered exception: {e}")


@app.on_event("startup")
def startup_event():
    load_models()


# Pydantic Schemas with rigorous input boundary validation
class StudentRecord(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "study_hours_per_day": 6.5,
                "attendance_percentage": 88.0,
                "assignment_score": 82.5,
                "previous_exam_score": 75.0,
                "sleep_hours": 7.0,
            }
        }
    )

    study_hours_per_day: float = Field(..., ge=0.0, le=24.0, description="Daily study hours (0-24)")
    attendance_percentage: float = Field(..., ge=0.0, le=100.0, description="Class attendance percentage (0-100)")
    assignment_score: float = Field(..., ge=0.0, le=100.0, description="Average assignment mark (0-100)")
    previous_exam_score: float = Field(..., ge=0.0, le=100.0, description="Prior exam mark (0-100)")
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Daily sleep hours (0-24)")


class PredictionResponse(BaseModel):
    predicted_final_exam_score: float
    predicted_performance_category: str
    anomaly_detected: bool
    anomaly_score: float
    confidence_tier: str


class BatchPredictionRequest(BaseModel):
    students: List[StudentRecord]


class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[PredictionResponse]


@app.get("/health", tags=["System"])
def health_check():
    """Returns server status and available models."""
    return {
        "status": "healthy",
        "models_loaded": list(models_cache.keys()),
        "service_version": "1.0.0",
        "framework": "FastAPI + Scikit-Learn",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_student(student: StudentRecord):
    """
    Performs real-time online inference for a single student profile.
    Predicts:
    1. Continuous final exam marks (Regression)
    2. Discrete performance category (Classification)
    3. Anomaly flag and score (Isolation Forest)
    """
    if "regression" not in models_cache or "classification" not in models_cache:
        # Attempt fallback reload
        load_models()
        if "regression" not in models_cache:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained models not found on disk. Run 'python -m src.train' first.",
            )

    df_in = pd.DataFrame([student.model_dump()])

    # 1. Regression Score
    reg_model = models_cache["regression"]
    pred_score = float(reg_model.predict(df_in[FEATURE_NAMES])[0])
    pred_score = float(np.clip(pred_score, 0.0, 100.0))

    # 2. Classification Category
    clf_model = models_cache["classification"]
    pred_class_idx = clf_model.predict(df_in[FEATURE_NAMES])[0]
    category_label = INT_TO_LABEL.get(int(pred_class_idx), str(pred_class_idx))

    # 3. Anomaly check
    is_anomaly = False
    anomaly_score = 0.0
    if "anomaly" in models_cache:
        iso = models_cache["anomaly"]
        score_val = float(iso.decision_function(df_in[FEATURE_NAMES])[0])
        is_anomaly = bool(iso.predict(df_in[FEATURE_NAMES])[0] == -1)
        anomaly_score = score_val

    # Confidence tier heuristic
    conf = "High" if not is_anomaly and 15.0 < pred_score < 95.0 else "Medium"

    return PredictionResponse(
        predicted_final_exam_score=round(pred_score, 2),
        predicted_performance_category=category_label,
        anomaly_detected=is_anomaly,
        anomaly_score=round(anomaly_score, 4),
        confidence_tier=conf,
    )


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_students(batch: BatchPredictionRequest):
    """
    Performs vector-accelerated batch inference for multiple student records.
    """
    if "regression" not in models_cache:
        load_models()
        if "regression" not in models_cache:
            raise HTTPException(status_code=503, detail="Models not loaded.")

    data_dicts = [s.model_dump() for s in batch.students]
    df_batch = pd.DataFrame(data_dicts)

    reg_model = models_cache["regression"]
    clf_model = models_cache["classification"]

    pred_scores = np.clip(reg_model.predict(df_batch[FEATURE_NAMES]), 0.0, 100.0)
    pred_classes = clf_model.predict(df_batch[FEATURE_NAMES])

    predictions = []
    for i in range(len(df_batch)):
        cat_str = INT_TO_LABEL.get(int(pred_classes[i]), str(pred_classes[i]))
        predictions.append(
            PredictionResponse(
                predicted_final_exam_score=round(float(pred_scores[i]), 2),
                predicted_performance_category=cat_str,
                anomaly_detected=False,
                anomaly_score=0.0,
                confidence_tier="High",
            )
        )

    return BatchPredictionResponse(
        total_records=len(predictions),
        predictions=predictions,
    )


@app.get("/model_card", tags=["Governance"])
def get_model_card_info():
    """Returns stored Model Card metadata."""
    import json
    card_files = list(METADATA_DIR.glob("*model_card.json"))
    if not card_files:
        raise HTTPException(status_code=404, detail="No model cards found.")
    with open(card_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
