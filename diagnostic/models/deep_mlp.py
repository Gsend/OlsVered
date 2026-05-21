"""
diagnostic/models/deep_mlp.py
==============================

DeepMLP — configurable-depth fully-connected network for MNIST.

Architecture: Flatten → fc1 → ReLU → fc2 → ReLU → ... → fc(n-1) → ReLU → fcN

Module registration order (fc1, act1, fc2, act2, ...) ensures that
diagnostic/capture.py's detect_following_activation finds the right ReLU
for each Linear layer without requiring explicit activation_overrides.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn


class DeepMLP(nn.Module):
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
        # Register fc1,act1,fc2,act2,...,fc(n-1),act(n-1),fcN in that order so
        # named_modules() yields them interleaved and activation detection works.
        for i in range(n_layers):
            setattr(self, f"fc{i + 1}", nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers - 1:
                setattr(self, f"act{i + 1}", nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        for i in range(1, self.n_layers + 1):
            x = getattr(self, f"fc{i}")(x)
            if i < self.n_layers:
                x = getattr(self, f"act{i}")(x)
        return x

    def get_linear_layers(self) -> List[nn.Linear]:
        """Return all Linear layers in forward order (fc1 → fcN)."""
        return [getattr(self, f"fc{i + 1}") for i in range(self.n_layers)]

    def get_layers_deepest_first(self) -> List[nn.Linear]:
        """Return layers deepest-first (fcN → fc1) for TP back-propagation."""
        return list(reversed(self.get_linear_layers()))
