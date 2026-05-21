"""
diagnostic/models/lenet.py
===========================

LeNet — classic LeNet-style CNN for MNIST classification (E3).

Architecture:
  conv1(1,6,5,pad=2) → ReLU → MaxPool(2,2)
  → conv2(6,16,5) → ReLU → MaxPool(2,2)
  → Flatten
  → fc1(400,120) → ReLU → fc2(120,84) → ReLU → fc3(84,10)

All modules registered via setattr so that named_modules() yields them
in declaration order, enabling detect_following_activation to work correctly
for both Conv2d and Linear layers.

Spatial dimensions (28×28 MNIST input):
  conv1 → (6,28,28), pool1 → (6,14,14)
  conv2 → (16,10,10), pool2 → (16,5,5)
  flatten → 400
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # Spatial output sizes (H_out, W_out) and input sizes (H_in, W_in) per conv layer
    CONV1_OUT_HW = (28, 28)   # after conv1 (padding=2, kernel=5)
    CONV2_OUT_HW = (10, 10)   # after conv2 (no padding, kernel=5)
    POOL1_OUT_HW = (14, 14)   # after pool1 (MaxPool 2×2)
    POOL2_OUT_HW = (5, 5)     # after pool2 (MaxPool 2×2)
    FC1_IN = 400              # 16 * 5 * 5

    def __init__(self):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv block 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # FC block
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.FC1_IN, 120)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        return self.fc3(x)

    def get_fc_layers(self) -> List[nn.Linear]:
        return [self.fc1, self.fc2, self.fc3]

    def get_conv_layers(self) -> List[nn.Conv2d]:
        return [self.conv1, self.conv2]

    def get_layers_deepest_first(self) -> List[Union[nn.Linear, nn.Conv2d]]:
        """All parametric layers deepest-first for TP back-propagation."""
        return [self.fc3, self.fc2, self.fc1, self.conv2, self.conv1]

    def forward_with_pool_state(self, x) -> Dict[str, "torch.Tensor"]:
        """Forward pass that also returns pre-pool activations and the MaxPool
        argmax switch indices, for switch-aware unpooling (Option 2).

        Returns a dict with:
          conv1_prepool : (B, 6, 28, 28)   ReLU(conv1(x)), before pool1
          pool1_indices : (B, 6, 14, 14)   argmax flat indices from pool1
          conv2_prepool : (B, 16, 10, 10)  ReLU(conv2(pool1)), before pool2
          pool2_indices : (B, 16, 5, 5)    argmax flat indices from pool2
          logits        : (B, 10)
        """
        c1 = self.act1(self.conv1(x))                       # (B,6,28,28)
        p1, idx1 = F.max_pool2d(c1, kernel_size=2, stride=2,
                                return_indices=True)        # (B,6,14,14)
        c2 = self.act2(self.conv2(p1))                      # (B,16,10,10)
        p2, idx2 = F.max_pool2d(c2, kernel_size=2, stride=2,
                                return_indices=True)        # (B,16,5,5)
        flat = self.flatten(p2)
        h1 = self.act3(self.fc1(flat))
        h2 = self.act4(self.fc2(h1))
        logits = self.fc3(h2)
        return {
            "conv1_prepool": c1, "pool1_indices": idx1,
            "conv2_prepool": c2, "pool2_indices": idx2,
            "logits": logits,
        }


class LeNetAvgPool(nn.Module):
    """LeNet variant with average-pooling instead of max-pooling (E3, Option 3).

    AvgPool2d is linear and exactly invertible (each pooled value is the mean
    of a 2x2 block; the natural pseudo-inverse broadcasts value/4 — but for
    target-prop we upsample by replication, which is the adjoint up to scale).
    This removes the lossy max-pool inversion that breaks the plain LeNet, so
    it isolates whether the conv TP path works once pooling is well-behaved.

    Identical layer shapes to LeNet so the same battery code applies.
    """
    CONV1_OUT_HW = (28, 28)
    CONV2_OUT_HW = (10, 10)
    POOL1_OUT_HW = (14, 14)
    POOL2_OUT_HW = (5, 5)
    FC1_IN = 400

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.FC1_IN, 120)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        return self.fc3(x)

    def get_fc_layers(self) -> List[nn.Linear]:
        return [self.fc1, self.fc2, self.fc3]

    def get_conv_layers(self) -> List[nn.Conv2d]:
        return [self.conv1, self.conv2]

    def get_layers_deepest_first(self) -> List[Union[nn.Linear, nn.Conv2d]]:
        return [self.fc3, self.fc2, self.fc1, self.conv2, self.conv1]
