"""Unit tests for CO-6 MLOps (Drift, Model Cards, Retraining Triggers)."""

import pytest
import numpy as np
import pandas as pd

from src.data.generator import generate_student_dataset
from src.mlops.drift import compute_ks_drift, compute_psi, generate_drifted_student_data
from src.mlops.monitoring import evaluate_retraining_trigger, simulate_ab_testing
from src.mlops.model_card import generate_model_card
from src.mlops.versioning import compute_file_sha256, track_experiment_run
from src.utils.config import RAW_DATA_PATH


@pytest.fixture
def base_data():
    return generate_student_dataset(n_samples=200, seed=42)


def test_ks_drift_detection(base_data):
    # Same distribution: should detect 0 drift
    res_stationary = compute_ks_drift(base_data, base_data)
    assert res_stationary["n_drifted_features"] == 0

    # Shifted distribution: should detect drift
    drifted = generate_drifted_student_data(base_data, "severe_covid_shift")
    res_shifted = compute_ks_drift(base_data, drifted)
    assert res_shifted["n_drifted_features"] >= 1


def test_psi_metric(base_data):
    # Same data: PSI near 0
    psi_zero = compute_psi(base_data["study_hours_per_day"].values, base_data["study_hours_per_day"].values)
    assert psi_zero < 0.05

    # Shifted data: PSI higher
    shifted_study = base_data["study_hours_per_day"].values * 1.5
    psi_shifted = compute_psi(base_data["study_hours_per_day"].values, shifted_study)
    assert psi_shifted > psi_zero


def test_retraining_trigger_activation():
    # Drifted report
    fake_drift = {"n_drifted_features": 3}
    res = evaluate_retraining_trigger(fake_drift, rolling_rmse=4.5, baseline_rmse=3.5)
    assert res["should_retrain"] is True
    assert len(res["reasons"]) >= 1


def test_model_card_generation():
    card = generate_model_card(
        model_name="Test_Regressor",
        model_type="Ridge",
        metrics={"RMSE": 3.45},
        hyperparameters={"alpha": 1.0},
    )
    assert card["model_details"]["name"] == "Test_Regressor"
    assert "intended_use" in card
