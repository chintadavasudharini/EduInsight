"""Dataset generation, splitting, and loading."""

from .generator import generate_student_dataset
from .loader import prepare_dataset, split_and_save_data, load_splits

__all__ = [
    "generate_student_dataset",
    "prepare_dataset",
    "split_and_save_data",
    "load_splits",
]
