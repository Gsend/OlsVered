"""
diagnostic/models/miniresnet.py
================================

MiniResNet — small BN-free residual network for CIFAR-10 classification (E5).

Design choices (Q1 resolution: decompose y = x + f(x)):
  - NO BatchNorm. BN would require a normalization-inversion primitive the
    framework does not yet have; omitting it isolates the *additive skip* —
    the structural lesson E5 is meant to probe — instead of confounding it
    with normalization handling.
  - Identity skips only (same channels, same spatial size inside a block), so
    the shortcut needs NO inversion: in chain back-prop the target for the
    residual branch f is simply  t_f = t_sum - x_block_input.
  - MaxPool for downsampling — reuses the switch-aware max-unpool inversion
    already built and validated in E3.

Architecture (input (3, 32, 32)):
  conv0(3,W,3,pad=1)  -> ReLU -> MaxPool(2)            -> (W, 16, 16)
  block1: f = conv1b(ReLU(conv1a(x)));  x = ReLU(x + f) -> (W, 16, 16)
          MaxPool(2)                                    -> (W,  8,  8)
  block2: f = conv2b(ReLU(conv2a(x)));  x = ReLU(x + f) -> (W,  8,  8)
  GlobalAvgPool -> Flatten -> fc(W, 10)

The residual-branch convs (conv1a/conv1b, conv2a/conv2b) have NO activation
directly after conv?b; the block's ReLU is applied AFTER the addition.

Module registration order is declaration order; activation detection for the
runner is done manually (the runner captures activations with an explicit
forward), so we do not rely on detect_following_activation here.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniResNet(nn.Module):
    # spatial sizes through the network (H == W everywhere)
    CONV0_OUT_HW = (32, 32)
    POOL0_OUT_HW = (16, 16)   # block1 operates here
    POOL1_OUT_HW = (8, 8)     # block2 operates here

    def __init__(self, width: int = 32, num_classes: int = 10):
        super().__init__()
        self.width = width
        self.num_classes = num_classes

        # stem
        self.conv0 = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.act0 = nn.ReLU()
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 1 (identity skip @ 16x16)
        self.b1_conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.b1_act1 = nn.ReLU()
        self.b1_conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.b1_act_out = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2 (identity skip @ 8x8)
        self.b2_conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.b2_act1 = nn.ReLU()
        self.b2_conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.b2_act_out = nn.ReLU()

        # head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(width, num_classes)

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self.pool0(self.act0(self.conv0(x)))      # (W,16,16)
        # block 1
        f = self.b1_conv2(self.b1_act1(self.b1_conv1(x)))
        x = self.b1_act_out(x + f)                    # (W,16,16)
        x = self.pool1(x)                             # (W,8,8)
        # block 2
        f = self.b2_conv2(self.b2_act1(self.b2_conv1(x)))
        x = self.b2_act_out(x + f)                    # (W,8,8)
        # head
        x = self.flatten(self.gap(x))                 # (W,)
        return self.fc(x)

    # ------------------------------------------------------------------
    def get_conv_layers(self) -> List[nn.Conv2d]:
        return [self.conv0, self.b1_conv1, self.b1_conv2,
                self.b2_conv1, self.b2_conv2]

    def get_layers_deepest_first(self) -> List[nn.Module]:
        """Parametric layers deepest-first for TP back-propagation."""
        return [self.fc, self.b2_conv2, self.b2_conv1,
                self.b1_conv2, self.b1_conv1, self.conv0]

    # ------------------------------------------------------------------
    def forward_with_state(self, x) -> Dict[str, torch.Tensor]:
        """Forward pass returning every tensor the TP retrain / distill needs:

          conv0_prepool : ReLU(conv0(x))            (B,W,32,32)  pre-pool0
          pool0_indices : argmax indices for pool0  (B,W,16,16)
          b1_in         : block-1 input  x          (B,W,16,16)  (= pool0 out)
          b1_conv1_pre  : conv1a(x)                 (B,W,16,16)  pre-activation
          b1_conv2_pre  : conv1b(ReLU(conv1a(x)))   (B,W,16,16)  pre-activation (= f)
          b1_sum_pre    : x + f                     (B,W,16,16)  pre out-ReLU
          b1_out        : ReLU(x + f)               (B,W,16,16)
          pool1_indices : argmax indices for pool1  (B,W,8,8)
          b2_in         : block-2 input  x          (B,W,8,8)    (= pool1 out)
          b2_conv1_pre  : conv2a(x)                 (B,W,8,8)
          b2_conv2_pre  : conv2b(ReLU(conv2a(x)))   (B,W,8,8)
          b2_sum_pre    : x + f                     (B,W,8,8)
          b2_out        : ReLU(x + f)               (B,W,8,8)
          gap_out       : flatten(GAP(b2_out))      (B,W)        fc input
          logits        : fc(gap_out)               (B,10)
        """
        s: Dict[str, torch.Tensor] = {}
        c0 = self.act0(self.conv0(x))                       # (W,32,32)
        s["conv0_prepool"] = c0
        p0, idx0 = F.max_pool2d(c0, 2, 2, return_indices=True)
        s["pool0_indices"] = idx0

        # block 1
        b1_in = p0                                          # (W,16,16)
        s["b1_in"] = b1_in
        c1a = self.b1_conv1(b1_in)
        s["b1_conv1_pre"] = c1a
        c1b = self.b1_conv2(self.b1_act1(c1a))              # = f
        s["b1_conv2_pre"] = c1b
        sum1 = b1_in + c1b
        s["b1_sum_pre"] = sum1
        b1_out = self.b1_act_out(sum1)
        s["b1_out"] = b1_out
        p1, idx1 = F.max_pool2d(b1_out, 2, 2, return_indices=True)
        s["pool1_indices"] = idx1

        # block 2
        b2_in = p1                                          # (W,8,8)
        s["b2_in"] = b2_in
        c2a = self.b2_conv1(b2_in)
        s["b2_conv1_pre"] = c2a
        c2b = self.b2_conv2(self.b2_act1(c2a))
        s["b2_conv2_pre"] = c2b
        sum2 = b2_in + c2b
        s["b2_sum_pre"] = sum2
        b2_out = self.b2_act_out(sum2)
        s["b2_out"] = b2_out

        gap_out = self.flatten(self.gap(b2_out))            # (W,)
        s["gap_out"] = gap_out
        s["logits"] = self.fc(gap_out)
        return s
