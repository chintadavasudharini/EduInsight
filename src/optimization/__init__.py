"""Optimization and Neural Network Training (CO-4)."""

from .torch_models import StudentMLP
from .optimizers_lab import get_optimizer, run_optimizer_benchmark, Lion
from .schedulers_lab import run_scheduler_benchmark
from .regularization_lab import (
    compare_regularization_techniques,
    demonstrate_early_stopping,
    compare_initializations,
    demonstrate_gradient_accumulation,
)

__all__ = [
    "StudentMLP",
    "get_optimizer",
    "run_optimizer_benchmark",
    "Lion",
    "run_scheduler_benchmark",
    "compare_regularization_techniques",
    "demonstrate_early_stopping",
    "compare_initializations",
    "demonstrate_gradient_accumulation",
]
