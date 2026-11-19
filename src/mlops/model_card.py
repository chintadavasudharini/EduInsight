"""
CO-6: Production ML Engineering / MLOps - Model Card and Metadata Generator.
Generates structured academic model cards adhering to Mitchell et al. (2019) standards.
"""

from typing import Dict, Any
import json
from pathlib import Path
from src.utils.config import METADATA_DIR


def generate_model_card(
    model_name: str,
    model_type: str,
    metrics: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    training_samples: int = 1050,
    test_samples: int = 225,
) -> Dict[str, Any]:
    """
    Constructs a standardized Model Card dictionary and exports as JSON.
    """
    card = {
        "model_details": {
            "name": model_name,
            "type": model_type,
            "version": "1.0.0",
            "license": "Apache-2.0 / Open Educational Use",
            "framework": "Scikit-Learn / PyTorch",
        },
        "intended_use": {
            "primary_uses": "Academic performance prediction, early intervention advisement, and study pattern clustering.",
            "intended_users": "Course instructors, academic advisors, educational researchers.",
            "out_of_scope_uses": "Automated punitive grading, admissions rejection, high-stakes disciplinary decisions without human oversight.",
        },
        "factors_and_data": {
            "features": [
                "study_hours_per_day",
                "attendance_percentage",
                "assignment_score",
                "previous_exam_score",
                "sleep_hours",
            ],
            "dataset_nature": "Synthetic educational data generated with fixed random seeds for academic rigor.",
            "training_samples": training_samples,
            "test_samples": test_samples,
        },
        "quantitative_metrics": metrics,
        "hyperparameters": hyperparameters,
        "ethical_considerations": {
            "privacy": "Zero personal identifiable information (PII). All records are synthetically constructed.",
            "fairness_warning": "Fairness parity metrics on synthetic groups cannot guarantee parity on diverse real-world demographics.",
            "human_in_the_loop": "Predictions must serve as diagnostic suggestions for advisors, never autonomous grading.",
        },
        "caveats_and_limitations": {
            "linear_assumptions": "Linear models cannot capture extreme non-linear burnout dynamics.",
            "domain_shift": "Performance models trained on typical semesters fail during abrupt schedule shifts without retraining.",
        },
    }

    # Save to disk
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    card_path = METADATA_DIR / f"{model_name.lower().replace(' ', '_')}_model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    return card
