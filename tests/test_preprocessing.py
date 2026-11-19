"""Unit tests for feature preprocessing and leak-free pipelines."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing.pipeline import StudentFeaturePreprocessor
from src.data.generator import generate_student_dataset
from src.utils.config import FEATURE_NAMES


def test_preprocessor_fit_transform():
    df = generate_student_dataset(n_samples=200, seed=42)
    preprocessor = StudentFeaturePreprocessor()

    with pytest.raises(RuntimeError):
        preprocessor.transform(df)

    X_scaled = preprocessor.fit_transform(df)
    assert X_scaled.shape == (200, len(FEATURE_NAMES))
    # Standardized features should have mean approx 0 and std approx 1
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-2)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-2)


def test_preprocessor_serialization(tmp_path):
    df = generate_student_dataset(n_samples=100, seed=42)
    preprocessor = StudentFeaturePreprocessor()
    preprocessor.fit(df)

    save_file = tmp_path / "test_preprocessor.joblib"
    preprocessor.save(save_file)
    assert save_file.exists()

    loaded = StudentFeaturePreprocessor.load(save_file)
    assert loaded.is_fitted
    transformed = loaded.transform(df)
    assert transformed.shape == (100, 5)
