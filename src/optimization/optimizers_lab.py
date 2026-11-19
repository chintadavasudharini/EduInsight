"""
CO-4: Optimization and Training - Optimizers Comparison Lab.
Implements and compares:
- Stochastic Gradient Descent (SGD)
- SGD with Momentum
- Nesterov Accelerated Gradient (NAG)
- Adagrad (Adaptive Subgradient)
- RMSprop (Root Mean Square Propagation)
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with Decoupled Weight Decay)
- Lion (EvoLved Sign Momentum optimizer)
Runs CPU-friendly mini-batch training comparisons and records training and validation loss curves.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.optimization.torch_models import StudentMLP
from src.utils.config import DEFAULT_RANDOM_SEED


# Clean, lightweight implementation of the Google Lion optimizer (Chen et al., 2023)
class Lion(torch.optim.Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer.
    Uses sign function of interpolated momentum to determine parameter update direction.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Step 1: Compute update direction via sign(beta1 * m + (1-beta1) * g)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign()

                # Step 2: Apply weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Step 3: Update parameter
                p.data.add_(update, alpha=-lr)

                # Step 4: Update momentum buffer exp_avg = beta2 * m + (1-beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Factory function for PyTorch optimizers."""
    opt_lower = optimizer_name.lower()
    if opt_lower == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    elif opt_lower == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=False, weight_decay=weight_decay)
    elif opt_lower == "nesterov":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif opt_lower == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_lower == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, weight_decay=weight_decay)
    elif opt_lower == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_lower == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_lower == "lion":
        return Lion(model.parameters(), lr=lr * 0.3, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)


def run_optimizer_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimizer_list: Optional[List[str]] = None,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Trains StudentMLP with each optimizer under identical seeds and initializations.
    Records convergence trajectories for train and validation loss.
    """
    if optimizer_list is None:
        optimizer_list = ["SGD", "Momentum", "Nesterov", "Adagrad", "RMSprop", "Adam", "AdamW", "Lion"]

    # Convert to PyTorch tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tr, y_tr)
    criterion = nn.MSELoss()

    benchmark_results = {}

    for opt_name in optimizer_list:
        torch.manual_seed(seed)
        model = StudentMLP(in_features=X_train.shape[1], hidden_dim1=16, hidden_dim2=8, init_type="xavier")
        optimizer = get_optimizer(opt_name, model, lr=lr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

            train_epoch_loss = float(np.mean(batch_losses))
            train_losses.append(train_epoch_loss)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = float(criterion(val_pred, y_v).item())
                val_losses.append(val_loss)

        benchmark_results[opt_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_rmse": float(np.sqrt(train_losses[-1])),
            "final_val_rmse": float(np.sqrt(val_losses[-1])),
        }

    return benchmark_results
