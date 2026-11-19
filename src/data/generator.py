"""
Reproducible Synthetic Dataset Generator for Student Performance Analysis.
Constructs educational data representing study habits, attendance, and exam marks.
NOTE: This is synthetic educational data designed for academic demonstration,
not actual student records.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_SIZE,
    CATEGORY_BINS,
    CATEGORY_LABELS,
    FEATURE_NAMES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
)


def generate_student_dataset(
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
    noise_level: float = 3.5,
    include_subgroups: bool = True,
) -> pd.DataFrame:
    """
    Generates a reproducible synthetic dataset of student academic metrics.

    Features:
        - study_hours_per_day (float: 1.0 - 10.0)
        - attendance_percentage (float: 45.0 - 100.0)
        - assignment_score (float: 30.0 - 100.0)
        - previous_exam_score (float: 30.0 - 100.0)
        - sleep_hours (float: 4.0 - 10.0)

    Target:
        - final_exam_score (float: 0.0 - 100.0)
        - performance_category (Needs Support, Average, Good, Excellent)
        - study_group (optional synthetic subgroup for fairness testing)

    Ground-truth relationship incorporates linear contributions, non-linear
    sleep/study interaction, diminishing returns, and Gaussian noise.
    """
    rng = np.random.default_rng(seed)

    # 1. Study hours per day: truncated normal around 5.5 hours, range [1, 10]
    study_hours = np.clip(rng.normal(loc=5.2, scale=1.8, size=n_samples), 1.0, 10.0)

    # 2. Attendance percentage: beta distribution scaled to [45, 100]
    # Students generally attend 70-95% of classes
    attendance = 45.0 + 55.0 * rng.beta(a=5.0, b=2.0, size=n_samples)
    attendance = np.clip(attendance, 45.0, 100.0)

    # 3. Assignment score: correlates with attendance and study hours
    assignment_latent = (
        0.35 * (attendance - 45.0) / 55.0 * 100.0
        + 0.35 * (study_hours / 10.0) * 100.0
        + 0.30 * rng.uniform(40.0, 95.0, size=n_samples)
        + rng.normal(0, 4.0, size=n_samples)
    )
    assignment_score = np.clip(assignment_latent, 30.0, 100.0)

    # 4. Previous exam score: prior academic baseline
    prev_latent = (
        0.40 * assignment_score
        + 0.30 * (study_hours * 8.0)
        + 0.30 * rng.normal(60.0, 15.0, size=n_samples)
    )
    previous_exam_score = np.clip(prev_latent, 25.0, 100.0)

    # 5. Sleep hours: normal around 7.0, range [4, 10]
    sleep_hours = np.clip(rng.normal(loc=7.1, scale=1.1, size=n_samples), 4.0, 10.0)

    # Nonlinear & Interaction modeling for final exam score
    # Optimal sleep is around 7-8 hours; severe sleep deprivation (<5.5) or oversleep (>9) incurs penalty
    sleep_penalty = np.where(sleep_hours < 5.5, (5.5 - sleep_hours) * 3.0, 0.0)
    sleep_penalty += np.where(sleep_hours > 8.8, (sleep_hours - 8.8) * 1.5, 0.0)

    # Diminishing returns on study hours beyond 8 hours
    effective_study = np.where(
        study_hours <= 7.5,
        study_hours * 3.5,
        7.5 * 3.5 + (study_hours - 7.5) * 1.2
    )

    # Linear baseline contributions (weights sum to ~1.0 scale)
    # attendance (25%), previous exam (30%), assignment (20%), study (25%)
    raw_final_score = (
        0.26 * attendance
        + 0.28 * previous_exam_score
        + 0.20 * assignment_score
        + effective_study
        - sleep_penalty
        + rng.normal(0, noise_level, size=n_samples)
        - 8.0  # calibration intercept
    )

    final_exam_score = np.clip(raw_final_score, 0.0, 100.0)
    final_exam_score = np.round(final_exam_score, 2)

    # Derive categorical performance grades:
    # 0-39: Needs Support, 40-59: Average, 60-79: Good, 80-100: Excellent
    performance_category = pd.cut(
        final_exam_score,
        bins=CATEGORY_BINS,
        labels=CATEGORY_LABELS,
        right=True,
    ).astype(str)

    df = pd.DataFrame({
        "student_id": [f"STD_{1000 + i}" for i in range(n_samples)],
        "study_hours_per_day": np.round(study_hours, 2),
        "attendance_percentage": np.round(attendance, 2),
        "assignment_score": np.round(assignment_score, 2),
        "previous_exam_score": np.round(previous_exam_score, 2),
        "sleep_hours": np.round(sleep_hours, 2),
        TARGET_REGRESSION: final_exam_score,
        TARGET_CLASSIFICATION: performance_category,
    })

    if include_subgroups:
        # Synthetic group partition for fairness testing demonstration
        # Note: completely synthetic partition (e.g. Campus A vs Campus B)
        # to demonstrate demographic parity / equalized odds without real protected attributes
        groups = rng.choice(["Campus_A", "Campus_B"], size=n_samples, p=[0.55, 0.45])
        df["study_group"] = groups

    return df
