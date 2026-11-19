"""
General helper utilities for Student Performance ML Project.
Includes logging, seed fixing, and formatting.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across standard library, numpy, and torch (if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_metric(val: float, precision: int = 4) -> str:
    """Format floating point numbers consistently."""
    if np.isnan(val):
        return "N/A"
    return f"{val:.{precision}f}"
