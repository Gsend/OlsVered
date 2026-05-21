"""
diagnostic/models/unet.py
=========================

SmallUNet — BN-free 2-level U-Net for MNIST denoising autoencoder (E6).

Images are padded to 32×32 (from MNIST 28×28, 2px border on each side).

Architecture:

  Encoder:
    enc1_conv1: Conv(1→16, 3×3, p=1) → ReLU
    enc1_conv2: Conv(16→16, 3×3, p=1) → ReLU  → enc1_post (B,16,32,32)
    pool1:      MaxPool2d(2, return_indices)     → (B,16,16,16) + pool1_indices

    enc2_conv1: Conv(16→32, 3×3, p=1) → ReLU
    enc2_conv2: Conv(32→32, 3×3, p=1) → ReLU  → enc2_post (B,32,16,16)
    pool2:      MaxPool2d(2, return_indices)     → (B,32,8,8) + pool2_indices

  Decoder (bottleneck = pool2 output):
    unpool2(pool2_out, pool2_indices):           → (B,32,16,16)
    cat(dec_up2, enc2_post):  dec2_in            (B,64,16,16)
    dec2_conv1: Conv(64→32, 3×3, p=1) → ReLU
    dec2_conv2: Conv(32→16, 3×3, p=1) → ReLU   → dec2_post (B,16,16,16)

    [dec2_post is 16ch to match pool1_indices (16ch from pool1(enc1_post))]

    unpool1(dec2_post, pool1_indices):           → (B,16,32,32)
    cat(dec_up1, enc1_post):  dec1_in            (B,32,32,32)
    dec1_conv1: Conv(32→16, 3×3, p=1) → ReLU
    dec1_conv2: Conv(16→16, 3×3, p=1) → ReLU   → dec1_post (B,16,32,32)

    final_conv: Conv(16→1, 1×1) — no activation  (B,1,32,32)

E6 retrain concat-skip splits:
  dec2_in (64ch): first 32ch from dec_up2 branch; last 32ch from enc2_post skip.
  dec1_in (32ch): first 16ch from dec_up1 branch; last 16ch from enc1_post skip.

Unpool adjoint (for retrain backward pass):
  Inverting MaxUnpool via adjoint (gather at argmax) maps
  (B,C,H_up,W_up) target → (B,C,H_pool,W_pool) target.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallUNet(nn.Module):
    """BN-free 2-level U-Net, 1-channel in/out, for 32×32 images."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1_conv1 = nn.Conv2d(1,  16, 3, padding=1)
        self.enc1_conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool1      = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc2_conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.enc2_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2      = nn.MaxPool2d(2, stride=2, return_indices=True)

        # Decoder
        self.unpool2    = nn.MaxUnpool2d(2, stride=2)
        self.dec2_conv1 = nn.Conv2d(64, 32, 3, padding=1)   # 32+32 concat
        self.dec2_conv2 = nn.Conv2d(32, 16, 3, padding=1)   # output 16ch to match pool1

        self.unpool1    = nn.MaxUnpool2d(2, stride=2)
        self.dec1_conv1 = nn.Conv2d(32, 16, 3, padding=1)   # 16+16 concat = 32ch
        self.dec1_conv2 = nn.Conv2d(16, 16, 3, padding=1)

        self.final_conv = nn.Conv2d(16, 1, 1)                # no activation
        self.relu       = nn.ReLU()

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.relu(self.enc1_conv2(self.relu(self.enc1_conv1(x))))
        p1, idx1 = self.pool1(e1)
        e2 = self.relu(self.enc2_conv2(self.relu(self.enc2_conv1(p1))))
        p2, idx2 = self.pool2(e2)

        up2  = self.unpool2(p2, idx2, output_size=e2.size())
        d2   = self.relu(self.dec2_conv2(self.relu(self.dec2_conv1(torch.cat([up2, e2], 1)))))
        up1  = self.unpool1(d2, idx1, output_size=e1.size())
        d1   = self.relu(self.dec1_conv2(self.relu(self.dec1_conv1(torch.cat([up1, e1], 1)))))
        return self.final_conv(d1)

    # ------------------------------------------------------------------

    def forward_with_state(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass returning all intermediate tensors.

        Returns (all spatial tensors on the same device as x):
          enc1_conv1_pre   (B,16,32,32) pre-act enc1_conv1
          enc1_conv2_pre   (B,16,32,32) pre-act enc1_conv2
          enc1_post        (B,16,32,32) post-act enc1_conv2  = enc1 skip
          pool1_indices    (B,16,16,16) MaxPool1 argmax
          pool1_out        (B,16,16,16) MaxPool1 output
          enc2_conv1_pre   (B,32,16,16)
          enc2_conv2_pre   (B,32,16,16)
          enc2_post        (B,32,16,16) = enc2 skip
          pool2_indices    (B,32,8,8)
          pool2_out        (B,32,8,8)   bottleneck
          dec2_in          (B,64,16,16) concat input to dec2_conv1
          dec2_conv1_pre   (B,32,16,16)
          dec2_conv2_pre   (B,16,16,16)
          dec2_post        (B,16,16,16) post-act dec2_conv2
          dec1_in          (B,32,32,32) concat input to dec1_conv1
          dec1_conv1_pre   (B,16,32,32)
          dec1_conv2_pre   (B,16,32,32)
          dec1_post        (B,16,32,32) post-act dec1_conv2
          final_pre        (B,1,32,32)  output (no activation)
        """
        s: Dict[str, torch.Tensor] = {}

        c = self.enc1_conv1(x);        s["enc1_conv1_pre"] = c
        a = self.relu(c)
        c = self.enc1_conv2(a);        s["enc1_conv2_pre"] = c
        e1 = self.relu(c);             s["enc1_post"] = e1
        p1, idx1 = self.pool1(e1);     s["pool1_indices"] = idx1;  s["pool1_out"] = p1

        c = self.enc2_conv1(p1);       s["enc2_conv1_pre"] = c
        a = self.relu(c)
        c = self.enc2_conv2(a);        s["enc2_conv2_pre"] = c
        e2 = self.relu(c);             s["enc2_post"] = e2
        p2, idx2 = self.pool2(e2);     s["pool2_indices"] = idx2;  s["pool2_out"] = p2

        up2 = self.unpool2(p2, idx2, output_size=e2.size())
        di2 = torch.cat([up2, e2], 1); s["dec2_in"] = di2
        c = self.dec2_conv1(di2);      s["dec2_conv1_pre"] = c
        a = self.relu(c)
        c = self.dec2_conv2(a);        s["dec2_conv2_pre"] = c
        d2 = self.relu(c);             s["dec2_post"] = d2

        up1 = self.unpool1(d2, idx1, output_size=e1.size())
        di1 = torch.cat([up1, e1], 1); s["dec1_in"] = di1
        c = self.dec1_conv1(di1);      s["dec1_conv1_pre"] = c
        a = self.relu(c)
        c = self.dec1_conv2(a);        s["dec1_conv2_pre"] = c
        d1 = self.relu(c);             s["dec1_post"] = d1

        out = self.final_conv(d1);     s["final_pre"] = out
        return s
