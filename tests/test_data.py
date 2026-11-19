"""Unit tests for dataset generation and loading."""

import pytest
import pandas as pd
import numpy as np

from src.data.generator import generate_student_dataset
from src.data.loader import prepare_dataset, split_and_save_data
from src.utils.config import (
    FEATURE_NAMES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    CATEGORY_LABELS,
)


def test_generate_student_dataset_shape_and_columns():
    df = generate_student_dataset(n_samples=1300, seed=42)
    assert len(df) == 1300
    for col in FEATURE_NAMES:
        assert col in df.columns
    assert TARGET_REGRESSION in df.columns
    assert TARGET_CLASSIFICATION in df.columns


def test_dataset_reproducibility():
    df1 = generate_student_dataset(n_samples=500, seed=123)
    df2 = generate_student_dataset(n_samples=500, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_target_value_bounds():
    df = generate_student_dataset(n_samples=500, seed=99)
    assert df[TARGET_REGRESSION].min() >= 0.0
    assert df[TARGET_REGRESSION].max() <= 100.0
    categories = set(df[TARGET_CLASSIFICATION].unique())
    assert categories.issubset(set(CATEGORY_LABELS))


def test_train_val_test_split_ratios():
    df = generate_student_dataset(n_samples=1000, seed=42)
    train_df, val_df, test_df = split_and_save_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150
    # No student overlap between train and test
    train_ids = set(train_df["student_id"])
    test_ids = set(test_df["student_id"])
    assert len(train_ids.intersection(test_ids)) == 0
