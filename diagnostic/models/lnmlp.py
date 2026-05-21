"""
diagnostic/models/lnmlp.py
===========================

LNMLP — DeepMLP variant with LayerNorm+ReLU (LNReLU) activations (E2).

Architecture: Flatten → fc1 → LNReLU → fc2 → LNReLU → ... → fc(n-1) → LNReLU → fcN

Module registration order (fc1, act1, fc2, act2, ...) ensures that
detect_following_activation finds the correct LNReLU for each Linear layer.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn

from diagnostic.layers import LNReLU


class LNMLP(nn.Module):
    def __init__(
        self,
        n_layers: int = 10,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.flatten = nn.Flatten()

        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        for i in range(n_layers):
            setattr(self, f"fc{i + 1}", nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers - 1:
                # LNReLU normalizes the output of fc{i+1} which has dims[i+1] features
                setattr(self, f"act{i + 1}", LNReLU(dims[i + 1]))

    def forward(self, x):
        x = self.flatten(x)
        for i in range(1, self.n_layers + 1):
            x = getattr(self, f"fc{i}")(x)
            if i < self.n_layers:
                x = getattr(self, f"act{i}")(x)
        return x

    def get_linear_layers(self) -> List[nn.Linear]:
        return [getattr(self, f"fc{i + 1}") for i in range(self.n_layers)]

    def get_layers_deepest_first(self) -> List[nn.Linear]:
        return list(reversed(self.get_linear_layers()))
