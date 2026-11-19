"""
Data loading and dataset splitting utilities.
Implements reproducible Train, Validation, and Test separation.
Ensures zero data leakage between splits.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    DEFAULT_RANDOM_SEED,
    TARGET_CLASSIFICATION,
)
from src.data.generator import generate_student_dataset


def prepare_dataset(
    n_samples: int = 1500,
    seed: int = DEFAULT_RANDOM_SEED,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """
    Ensures the raw dataset exists on disk; creates it if missing.
    """
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if force_regenerate or not RAW_DATA_PATH.exists():
        df = generate_student_dataset(n_samples=n_samples, seed=seed)
        df.to_csv(RAW_DATA_PATH, index=False)
        return df
    return pd.read_csv(RAW_DATA_PATH)


def split_and_save_data(
    df: Optional[pd.DataFrame] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = DEFAULT_RANDOM_SEED,
    stratify_col: Optional[str] = TARGET_CLASSIFICATION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into Train, Validation, and Test partitions.
    Uses stratification on performance category to maintain label distribution.
    Saves all three partitions to artifacts/data/.
    """
    if df is None:
        df = prepare_dataset(seed=seed)

    stratify = df[stratify_col] if stratify_col in df.columns else None

    # Step 1: Split into Train and Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=stratify,
    )

    # Step 2: Split Temp into Validation and Test
    stratify_temp = temp_df[stratify_col] if stratify_col in temp_df.columns else None
    val_share = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share),
        random_state=seed,
        stratify=stratify_temp,
    )

    # Save to disk
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    return train_df, val_df, test_df


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test datasets from disk.
    If not already generated, triggers generation and splitting.
    """
    if not (TRAIN_DATA_PATH.exists() and VAL_DATA_PATH.exists() and TEST_DATA_PATH.exists()):
        return split_and_save_data()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    return train_df, val_df, test_df
