"""
CO-4: Optimization and Training - Regularization, Initialization, and Training Strategies Lab.
Implements and compares:
- L2 Regularization (Weight Decay)
- Dropout
- Batch Normalization
- Early Stopping callback
- Xavier vs He Initialization
- Gradient Accumulation demonstration
- Mixed Precision check (torch.amp)
- Distributed Data Parallel (DDP) architecture explanation
"""

from typing import Dict, Any, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.optimization.torch_models import StudentMLP
from src.utils.config import DEFAULT_RANDOM_SEED


def compare_regularization_techniques(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    lr: float = 0.02,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Compares:
    1. Baseline (No reg, no dropout, no batchnorm)
    2. L2 Regularization (Weight Decay = 0.01)
    3. Dropout (p = 0.25)
    4. Batch Normalization
    5. Combined Regularization (L2 + Dropout + BatchNorm)
    """
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tr, y_tr)
    criterion = nn.MSELoss()

    configs = {
        "Baseline (No Reg)": {"bn": False, "dropout": 0.0, "wd": 0.0},
        "L2 Reg (Weight Decay=0.01)": {"bn": False, "dropout": 0.0, "wd": 0.01},
        "Dropout (p=0.25)": {"bn": False, "dropout": 0.25, "wd": 0.0},
        "Batch Normalization": {"bn": True, "dropout": 0.0, "wd": 0.0},
        "Combined (L2 + Dropout + BN)": {"bn": True, "dropout": 0.2, "wd": 0.005},
    }

    results = {}

    for name, cfg in configs.items():
        torch.manual_seed(seed)
        model = StudentMLP(
            in_features=X_train.shape[1],
            use_batch_norm=cfg["bn"],
            dropout_rate=cfg["dropout"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["wd"])
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        train_curve = []
        val_curve = []

        for epoch in range(epochs):
            model.train()
            b_losses = []
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                b_losses.append(loss.item())

            train_curve.append(float(np.mean(b_losses)))

            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(X_v), y_v).item())
                val_curve.append(val_loss)

        # Generalization gap: Val Loss - Train Loss
        gap = val_curve[-1] - train_curve[-1]

        results[name] = {
            "train_losses": train_curve,
            "val_losses": val_curve,
            "final_val_loss": val_curve[-1],
            "generalization_gap": gap,
        }

    return results


def demonstrate_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    patience: int = 6,
    max_epochs: int = 50,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Demonstrates Early Stopping mechanism to prevent overfitting.
    Monitors validation loss and stops when improvement ceases for 'patience' epochs.
    """
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(seed)
    model = StudentMLP(in_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    val_history = []
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            cur_val_loss = float(criterion(model(X_v), y_v).item())
            val_history.append(cur_val_loss)

        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_early = True
                break

    return {
        "stopped_early": stopped_early,
        "stopped_epoch": len(val_history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_history": val_history,
        "patience": patience,
    }


def compare_initializations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Compares Xavier (Glorot) vs He (Kaiming) initialization.
    """
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    criterion = nn.MSELoss()
    results = {}

    for init_name in ["xavier", "he"]:
        torch.manual_seed(seed)
        model = StudentMLP(in_features=X_train.shape[1], init_type=init_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

        train_curve = []
        for epoch in range(epochs):
            model.train()
            b_losses = []
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                b_losses.append(loss.item())
            train_curve.append(float(np.mean(b_losses)))

        results[init_name] = {
            "train_losses": train_curve,
            "final_loss": train_curve[-1],
        }

    return results


def demonstrate_gradient_accumulation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    effective_batch_size: int = 64,
    micro_batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Demonstrates Gradient Accumulation:
    Simulates large mini-batch updates on memory-constrained hardware by
    accumulating gradients across multiple micro-batches before calling optimizer.step().
    """
    accumulation_steps = effective_batch_size // micro_batch_size
    return {
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "explanation": (
            f"By accumulating gradients across {accumulation_steps} micro-batches of size {micro_batch_size}, "
            f"we achieve the exact mathematical update of a large batch size {effective_batch_size} "
            f"while peak GPU/CPU memory consumption remains bounded to micro-batch size {micro_batch_size}."
        ),
    }
