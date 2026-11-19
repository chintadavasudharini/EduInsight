"""
CO-4: Optimization and Training - Learning Rate Schedulers Lab.
Implements and compares:
- Constant Learning Rate
- StepLR (Multiplicative step decay)
- CosineAnnealingLR (Smooth cosine annealing decay)
- CosineAnnealingWarmRestarts (Periodic SGDR warm restarts)
- OneCycleLR (Super-convergence cycle)
Tracks learning rate schedules and validation convergence curves.
"""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR

from src.optimization.torch_models import StudentMLP
from src.utils.config import DEFAULT_RANDOM_SEED


def run_scheduler_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scheduler_list: List[str] = ["Constant", "StepLR", "CosineAnnealing", "WarmRestarts", "OneCycle"],
    epochs: int = 35,
    base_lr: float = 0.03,
    batch_size: int = 32,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Trains StudentMLP using AdamW with various learning rate schedulers.
    """
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tr, y_tr)
    criterion = nn.MSELoss()

    results = {}

    for sched_name in scheduler_list:
        torch.manual_seed(seed)
        model = StudentMLP(in_features=X_train.shape[1], hidden_dim1=16, hidden_dim2=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        steps_per_epoch = len(loader)

        scheduler = None
        if sched_name == "StepLR":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        elif sched_name == "CosineAnnealing":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        elif sched_name == "WarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
        elif sched_name == "OneCycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=base_lr * 1.5,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            )

        lr_history = []
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

                if sched_name == "OneCycle":
                    lr_history.append(float(optimizer.param_groups[0]["lr"]))
                    scheduler.step()

            train_losses.append(float(np.mean(batch_losses)))

            if sched_name != "OneCycle":
                lr_history.append(float(optimizer.param_groups[0]["lr"]))
                if scheduler is not None:
                    scheduler.step()

            model.eval()
            with torch.no_grad():
                val_losses.append(float(criterion(model(X_v), y_v).item()))

        results[sched_name] = {
            "lr_history": lr_history,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_val_loss": val_losses[-1],
        }

    return results
