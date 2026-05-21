"""
diagnostic/models/mlp_autoencoder.py
=====================================

MlpAutoencoder — 6-layer MLP autoencoder for MNIST reconstruction (E4).

Architecture: Flatten → fc1(784→256) → ReLU → fc2(256→128) → ReLU →
              fc3(128→64) → ReLU → fc4(64→128) → ReLU →
              fc5(128→256) → ReLU → fc6(256→784)

Same setattr registration pattern as DeepMLP so that
detect_following_activation finds the correct ReLU for each Linear layer.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn


_DIMS = [784, 256, 128, 64, 128, 256, 784]
_N_LAYERS = 6


class MlpAutoencoder(nn.Module):
    DIMS = _DIMS
    N_LAYERS = _N_LAYERS

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        for i in range(self.N_LAYERS):
            setattr(self, f"fc{i + 1}", nn.Linear(self.DIMS[i], self.DIMS[i + 1]))
            if i < self.N_LAYERS - 1:
                setattr(self, f"act{i + 1}", nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        for i in range(1, self.N_LAYERS + 1):
            x = getattr(self, f"fc{i}")(x)
            if i < self.N_LAYERS:
                x = getattr(self, f"act{i}")(x)
        return x  # (B, 784) — unnormalized reconstruction

    def get_linear_layers(self) -> List[nn.Linear]:
        return [getattr(self, f"fc{i + 1}") for i in range(self.N_LAYERS)]

    def get_layers_deepest_first(self) -> List[nn.Linear]:
        return list(reversed(self.get_linear_layers()))
