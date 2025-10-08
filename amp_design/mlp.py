from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    """Simple two-layer perceptron used as a sequence-level classifier."""

    def __init__(self, input_dim: int = 320, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
