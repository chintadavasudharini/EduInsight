"""
CO-4: Optimization and Training - PyTorch Student Neural Network Architecture.
Lightweight CPU-friendly Multi-Layer Perceptron (MLP) for student performance regression.
Includes configurable:
- Xavier / He initialization
- Batch Normalization
- Dropout
"""

from typing import Optional
import torch
import torch.nn as nn


class StudentMLP(nn.Module):
    """
    Lightweight feedforward neural network designed to predict exam scores
    from 5 student academic features.
    Architecture:
        Input (5) -> Linear (16) -> [BatchNorm] -> ReLU -> [Dropout] ->
        Linear (8) -> [BatchNorm] -> ReLU -> [Dropout] -> Linear (1)
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim1: int = 16,
        hidden_dim2: int = 8,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        init_type: str = "xavier",  # 'xavier', 'he', 'default'
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(in_features, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1) if use_batch_norm else nn.Identity()
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2) if use_batch_norm else nn.Identity()
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.fc_out = nn.Linear(hidden_dim2, 1)

        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type: str) -> None:
        """Applies Xavier or He weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
                elif init_type == "he":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        out = self.fc_out(x)
        return out
