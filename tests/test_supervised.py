"""Unit tests for CO-2 Supervised Learning (Regression & Classification)."""

import pytest
import numpy as np
import pandas as pd

from src.data.generator import generate_student_dataset
from src.supervised.regression import train_and_evaluate_regressors, get_regression_models
from src.supervised.classification import train_and_evaluate_classifiers, get_classification_models
from src.supervised.bias_variance import compute_bias_variance_decomposition
from src.utils.config import TARGET_REGRESSION, TARGET_CLASSIFICATION


@pytest.fixture
def sample_data():
    df = generate_student_dataset(n_samples=250, seed=42)
    train_df = df.iloc[:180]
    test_df = df.iloc[180:]
    return train_df, test_df


def test_regression_models(sample_data):
    train_df, test_df = sample_data
    summary_df, fitted = train_and_evaluate_regressors(
        train_df, train_df[TARGET_REGRESSION],
        test_df, test_df[TARGET_REGRESSION],
    )
    assert len(summary_df) >= 5
    assert "RMSE" in summary_df.columns
    assert "R-squared" in summary_df.columns
    assert summary_df["R-squared"].max() > 0.70


def test_classification_models(sample_data):
    train_df, test_df = sample_data
    summary_df, fitted = train_and_evaluate_classifiers(
        train_df, train_df[TARGET_CLASSIFICATION],
        test_df, test_df[TARGET_CLASSIFICATION],
    )
    assert len(summary_df) >= 7
    assert "Accuracy" in summary_df.columns
    assert "F1-Score (Macro)" in summary_df.columns
    assert summary_df["Accuracy"].max() > 0.65


def test_bias_variance_decomposition(sample_data):
    train_df, test_df = sample_data
    res = compute_bias_variance_decomposition(
        train_df[["study_hours_per_day", "attendance_percentage"]].values,
        train_df[TARGET_REGRESSION].values,
        test_df[["study_hours_per_day", "attendance_percentage"]].values,
        test_df[TARGET_REGRESSION].values,
        degrees=[1, 2],
        n_bootstraps=5,
    )
    assert len(res["summary_table"]) == 2
    assert "Bias_Squared" in res["summary_table"].columns
    assert "Variance" in res["summary_table"].columns
