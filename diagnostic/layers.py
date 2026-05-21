"""
diagnostic/layers.py
====================
Composite activation modules for architecture-coverage experiments.
Defined here (not in diagnostic.models) to avoid circular imports when
capture.py and inversion.py need to isinstance-check them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LNReLU(nn.Module):
    """LayerNorm → ReLU composite activation used in LNMLP (E2).

    Registered as a single module so that detect_following_activation finds
    it immediately after each nn.Linear in LNMLP's named_modules walk.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(x))
