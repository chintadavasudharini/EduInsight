"""
Preprocessing and Feature Engineering Pipeline.
Ensures zero data leakage by fitting transformers strictly on training data.
Provides reusable transformation for validation, test, and live inference.
"""

from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.config import (
    FEATURE_NAMES,
    PREPROCESSOR_PATH,
)


class StudentFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Standard preprocessing transformer for Student Performance features.
    Standardizes continuous features:
      - study_hours_per_day
      - attendance_percentage
      - assignment_score
      - previous_exam_score
      - sleep_hours
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or FEATURE_NAMES
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler strictly on training feature matrix."""
        df_feat = X[self.feature_cols].copy()
        self.scaler.fit(df_feat.values)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform can be called.")
        df_feat = X[self.feature_cols].copy()
        return self.scaler.transform(df_feat.values)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath=PREPROCESSOR_PATH) -> None:
        """Save fitted preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath=PREPROCESSOR_PATH) -> "StudentFeaturePreprocessor":
        """Load saved preprocessor from disk."""
        return joblib.load(filepath)


def build_preprocessor() -> StudentFeaturePreprocessor:
    """Factory function for student preprocessor."""
    return StudentFeaturePreprocessor()
