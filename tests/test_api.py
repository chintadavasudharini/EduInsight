"""Unit tests for FastAPI REST microservice."""

import pytest
from fastapi.testclient import TestClient

from api import app, load_models

client = TestClient(app)


@pytest.fixture(autouse=True)
def init_api():
    load_models()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service_version" in data


def test_predict_endpoint_valid():
    payload = {
        "study_hours_per_day": 7.0,
        "attendance_percentage": 90.0,
        "assignment_score": 85.0,
        "previous_exam_score": 80.0,
        "sleep_hours": 7.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_final_exam_score" in data
    assert 0.0 <= data["predicted_final_exam_score"] <= 100.0
    assert data["predicted_performance_category"] in ["Needs Support", "Average", "Good", "Excellent"]
    assert "anomaly_detected" in data


def test_predict_endpoint_boundary_validation():
    # Negative study hours should fail Pydantic validation (422)
    payload_invalid = {
        "study_hours_per_day": -2.0,
        "attendance_percentage": 90.0,
        "assignment_score": 85.0,
        "previous_exam_score": 80.0,
        "sleep_hours": 7.0,
    }
    response = client.post("/predict", json=payload_invalid)
    assert response.status_code == 422


def test_batch_predict_endpoint():
    payload = {
        "students": [
            {
                "study_hours_per_day": 5.0,
                "attendance_percentage": 80.0,
                "assignment_score": 75.0,
                "previous_exam_score": 70.0,
                "sleep_hours": 7.0,
            },
            {
                "study_hours_per_day": 8.5,
                "attendance_percentage": 95.0,
                "assignment_score": 92.0,
                "previous_exam_score": 88.0,
                "sleep_hours": 8.0,
            },
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
