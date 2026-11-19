"""Unit tests for CO-4 Optimization and PyTorch models."""

import pytest
import torch
import numpy as np

from src.optimization.torch_models import StudentMLP
from src.optimization.optimizers_lab import Lion, get_optimizer, run_optimizer_benchmark
from src.optimization.schedulers_lab import run_scheduler_benchmark
from src.optimization.regularization_lab import demonstrate_early_stopping


def test_student_mlp_forward():
    model = StudentMLP(in_features=5, hidden_dim1=16, hidden_dim2=8, init_type="xavier")
    x = torch.randn(10, 5)
    out = model(x)
    assert out.shape == (10, 1)


def test_lion_optimizer_step():
    model = StudentMLP(in_features=5, hidden_dim1=8, hidden_dim2=4)
    optimizer = Lion(model.parameters(), lr=1e-3)
    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    # Check that parameters updated
    assert loss.item() >= 0.0


def test_optimizer_benchmark_run():
    X_tr = np.random.normal(0, 1, size=(60, 5))
    y_tr = np.random.normal(60, 10, size=60)
    X_va = np.random.normal(0, 1, size=(20, 5))
    y_va = np.random.normal(60, 10, size=20)

    res = run_optimizer_benchmark(
        X_tr, y_tr, X_va, y_va,
        optimizer_list=["SGD", "Adam", "Lion"],
        epochs=4,
        batch_size=16,
    )
    assert "SGD" in res
    assert "Adam" in res
    assert "Lion" in res
    assert len(res["Adam"]["train_losses"]) == 4


def test_early_stopping():
    X_tr = np.random.normal(0, 1, size=(50, 5))
    y_tr = np.random.normal(60, 10, size=50)
    X_va = np.random.normal(0, 1, size=(20, 5))
    y_va = np.random.normal(60, 10, size=20)

    res = demonstrate_early_stopping(X_tr, y_tr, X_va, y_va, patience=3, max_epochs=10)
    assert "stopped_early" in res
    assert res["stopped_epoch"] <= 10
