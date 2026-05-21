"""
diagnostic/runners/arch_coverage/e6_unet.py
============================================

E6 — Small BN-free U-Net on MNIST denoising. Probes H4 (dense continuous
per-pixel targets carry retraining signal) and the channel-concat skip
connection on a reconstruction task.

Task: MNIST denoising. Input = clean image + Gaussian noise (std≈0.5).
Target = clean image. Metric = MSE (pixel-level, not scaled).
Images padded to 32×32 (2px border) for two clean 2× downsamples.

Battery (run via --option):

  train     B-train. Train SmallUNet by backprop; cache weights.

  distill   B-distill (feature-level) from RANDOM init. Each conv is OLS-fit
            to reproduce the TEACHER's per-layer pre-activation, rebuilt-
            upstream forward sweep. Pool and concat are replayed in the
            student's forward pass. Tests whether concat skips break feature
            distillation (they should not — gate B1: MSE ≈ teacher).

  retrain   B-retrain-rand. Chain back-prop of the clean-image target from
            RANDOM init. Decoder concat inputs split along channel axis:
              dec2_in (64ch): 32ch from dec_up2 → encoder path;
                              32ch from enc2_post skip → second enc2 target.
              dec1_in (32ch): 16ch from dec_up1 → encoder path;
                              16ch from enc1_post skip → second enc1 target.
            Encoder skip + pool-path targets summed at each encoder feature.
            MaxPool inverted via switch-aware unpool (E3/E5 pattern);
            MaxUnpool inverted via adjoint gather operation.

Usage:
    python -m diagnostic.runners.arch_coverage.e6_unet --option train
    python -m diagnostic.runners.arch_coverage.e6_unet --option distill
    python -m diagnostic.runners.arch_coverage.e6_unet --option retrain
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.models.unet import SmallUNet
from diagnostic.multi_step import _empirical_mean_cov
from diagnostic.target_prop_retrainer import solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e6"
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"
UNET_WEIGHTS = WEIGHTS_DIR / "unet_mnist.pt"

N_IMAGES   = 512
SEED       = 42
OLS_LAMBDA = 1e-4
OLS_EPS    = 1e-4
TRAIN_EPOCHS = 20
NOISE_STD  = 0.5
IMG_SIZE   = 32      # 28×28 padded to 32×32

_LAYER_NAMES = [
    "enc1_conv1", "enc1_conv2",
    "enc2_conv1", "enc2_conv2",
    "dec2_conv1", "dec2_conv2",
    "dec1_conv1", "dec1_conv2",
    "final_conv",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _load_mnist(data_root: Path):
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        sys.exit("torchvision required for E6")
    tf = T.Compose([T.Pad(2), T.ToTensor()])    # 28→32, pixels in [0,1]
    train = torchvision.datasets.MNIST(str(data_root), train=True,  download=False, transform=tf)
    test  = torchvision.datasets.MNIST(str(data_root), train=False, download=False, transform=tf)
    return train, test


def _build_clean_tensors(dataset, n: int, seed: int = 0):
    """Extract n clean images as a tensor (n,1,32,32)."""
    loader = DataLoader(dataset, batch_size=n, shuffle=False, num_workers=0)
    imgs, _ = next(iter(loader))
    return imgs[:n].float()


@torch.no_grad()
def _eval_mse(model, noisy_t, clean_t, device, batch_size=256) -> float:
    model.eval()
    total = 0.0
    N = noisy_t.shape[0]
    for i in range(0, N, batch_size):
        xb = noisy_t[i:i+batch_size].to(device)
        yb = clean_t[i:i+batch_size].to(device)
        pred = model(xb)
        total += F.mse_loss(pred, yb, reduction="sum").item()
    return total / (N * 1 * IMG_SIZE * IMG_SIZE)   # per-pixel MSE


def _train(model, train_set, noisy_test, clean_test, *, epochs, device):
    model.to(device).train()
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng   = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        model.train()
        for clean, _ in train_loader:
            clean = clean.to(device)
            noisy = (clean + torch.randn_like(clean) * NOISE_STD).clamp(0.0, 1.0)
            opt.zero_grad()
            F.mse_loss(model(noisy), clean).backward()
            opt.step()
        sched.step()
        mse = _eval_mse(model, noisy_test, clean_test, device)
        print(f"  [train] epoch {epoch+1:2d}/{epochs}  test_mse={mse:.4f}")
    return _eval_mse(model, noisy_test, clean_test, device)


def _im2col(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """(B,C_in,H,W) → (B*H_out*W_out, C_in*kH*kW)."""
    unf = F.unfold(x, kernel_size=conv.kernel_size, dilation=conv.dilation,
                   padding=conv.padding, stride=conv.stride)
    return unf.permute(0, 2, 1).reshape(-1, unf.shape[1])


def _fold_to_spatial(ah_im2col, conv, out_hw, n_img):
    """im2col-space target (n*H*W, C*k*k) → spatial (n, C, H, W)."""
    C_in = conv.in_channels
    kH, kW = (conv.kernel_size if isinstance(conv.kernel_size, tuple)
               else (conv.kernel_size, conv.kernel_size))
    L = out_hw[0] * out_hw[1]
    fi = ah_im2col.reshape(n_img, L, C_in * kH * kW).permute(0, 2, 1).float()
    folded = F.fold(fi, output_size=out_hw, kernel_size=conv.kernel_size,
                    dilation=conv.dilation, padding=conv.padding, stride=conv.stride)
    cnt    = F.fold(torch.ones_like(fi), output_size=out_hw,
                    kernel_size=conv.kernel_size, dilation=conv.dilation,
                    padding=conv.padding, stride=conv.stride)
    return (folded / cnt.clamp(min=1)).double()


def _switch_unpool(pooled_target, indices, prepool_forward, out_hw):
    """Invert MaxPool: scatter pooled_target to argmax; forward value elsewhere."""
    pt = pooled_target.float()
    up  = F.max_unpool2d(pt, indices, kernel_size=2, stride=2, output_size=out_hw)
    msk = F.max_unpool2d(torch.ones_like(pt), indices, kernel_size=2,
                         stride=2, output_size=out_hw)
    return (up + (1.0 - msk) * prepool_forward.float()).double()


def _unpool_adjoint(t_up, indices, pool_hw):
    """Adjoint of MaxUnpool2d: gather values at argmax positions.
    Inverts the MaxUnpool step in the target-prop backward pass.
    t_up     : (B, C, H_up, W_up) — target for the unpooled tensor
    indices  : (B, C, Hp, Wp)     — argmax from MaxPool2d (same C)
    pool_hw  : (Hp, Wp)
    Returns  : (B, C, Hp, Wp)
    """
    B, C = t_up.shape[:2]
    Hp, Wp = pool_hw
    return (t_up.reshape(B, C, -1)
               .gather(2, indices.reshape(B, C, -1))
               .reshape(B, C, Hp, Wp))


def _cp2d(t):
    """(B,C,H,W) → (B*H*W, C) channel-last flatten."""
    return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])


def _invert_conv(model_layer, a_in_fwd_spatial, t_out_post, a_pre_fwd_2d,
                 activation, out_hw, n_img, method, device):
    """Invert one conv layer (backward target prop). Returns spatial target for input.

    model_layer    : the nn.Conv2d
    a_in_fwd_spatial: (B,C_in,H,W) forward input to this layer (for inversion)
    t_out_post     : (B*H*W, C_out) post-activation target for this layer's output
    a_pre_fwd_2d   : (B*H*W, C_out) forward pre-activation (for dead-unit mask)
    activation     : nn.ReLU or None (following activation)
    out_hw         : (H, W) of this layer's output (= input to next)
    """
    W = model_layer.weight.detach().reshape(model_layer.out_channels, -1).double().cpu()
    b = (model_layer.bias.detach().double().cpu() if model_layer.bias is not None else None)
    a_in_2d = _im2col(a_in_fwd_spatial.float(), model_layer).double().cpu()
    mu, S = _empirical_mean_cov(a_in_2d)
    if method == "naive":
        ah = invert_layer(W, b, t_out_post, a_pre_fwd_2d, activation, "naive", eps=OLS_EPS)
    else:
        ah = invert_layer(W, b, t_out_post, a_pre_fwd_2d, activation, "kfac_a",
                          mu_a=mu, Sigma_a=S)
    return _fold_to_spatial(ah, model_layer, out_hw, n_img)


def _fit_conv(conv, X_spatial, target_pre, device):
    """OLS-fit conv to pre-activation target. Returns residual."""
    X = _im2col(X_spatial, conv).double().cpu()
    n = min(X.shape[0], target_pre.shape[0])
    W, b = solve_ols_layer(X[:n], target_pre[:n],
                            with_bias=(conv.bias is not None), ols_lambda=OLS_LAMBDA)
    pred = X[:n] @ W.T + (b.unsqueeze(0) if b is not None else 0)
    res  = float((pred - target_pre[:n]).norm() / target_pre[:n].norm().clamp(min=1e-30))
    with torch.no_grad():
        conv.weight.copy_(W.reshape(conv.weight.shape).to(conv.weight.device, conv.weight.dtype))
        if conv.bias is not None and b is not None:
            conv.bias.copy_(b.to(conv.bias.device, conv.bias.dtype))
    return res


# ===========================================================================
# B-train
# ===========================================================================

def battery_train(train_set, noisy_test, clean_test, *, device, epochs):
    print("\n[E6] === B-train: SmallUNet MNIST denoising (32x32 padded) ===")
    torch.manual_seed(SEED)
    model = SmallUNet().to(device)
    rand_mse = _eval_mse(model, noisy_test, clean_test, device)
    print(f"  random-init MSE = {rand_mse:.4f}")

    mse = _train(model, train_set, noisy_test, clean_test, epochs=epochs, device=device)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), UNET_WEIGHTS)
    print(f"  trained MSE = {mse:.4f}  -> {UNET_WEIGHTS.name}")

    _save_json(RESULTS_DIR / "b_train.json",
               {"battery": "B-train", "exp_id": "e6", "seed": SEED,
                "epochs": epochs, "trained_mse": mse, "random_mse": rand_mse,
                "img_size": IMG_SIZE, "noise_std": NOISE_STD,
                "note": "28x28 MNIST padded to 32x32 with 2px border"})
    return model


# ===========================================================================
# B-distill — feature-level OLS, rebuilt-upstream forward sweep
# ===========================================================================

def battery_distill(teacher, images_clean, noisy_test, clean_test, *, device):
    print("\n[E6] === B-distill: feature-level OLS distillation (random init) ===")
    teacher.eval()
    teacher_mse = _eval_mse(teacher, noisy_test, clean_test, device)
    print(f"  teacher MSE = {teacher_mse:.4f}")

    g = torch.Generator().manual_seed(SEED + 10)
    noisy = (images_clean + torch.randn_like(images_clean, generator=g) * NOISE_STD).clamp(0,1)
    n_img = noisy.shape[0]

    relu = nn.ReLU()
    with torch.no_grad():
        noisy_d = noisy.to(device)
        st = teacher.forward_with_state(noisy_d)

    # All teacher pre-activations (as im2col-flattened targets)
    tgt = {
        "enc1_conv1": _cp2d(st["enc1_conv1_pre"].cpu()).double(),
        "enc1_conv2": _cp2d(st["enc1_conv2_pre"].cpu()).double(),
        "enc2_conv1": _cp2d(st["enc2_conv1_pre"].cpu()).double(),
        "enc2_conv2": _cp2d(st["enc2_conv2_pre"].cpu()).double(),
        "dec2_conv1": _cp2d(st["dec2_conv1_pre"].cpu()).double(),
        "dec2_conv2": _cp2d(st["dec2_conv2_pre"].cpu()).double(),
        "dec1_conv1": _cp2d(st["dec1_conv1_pre"].cpu()).double(),
        "dec1_conv2": _cp2d(st["dec1_conv2_pre"].cpu()).double(),
        "final_conv": _cp2d(st["final_pre"].cpu()).double(),
    }

    torch.manual_seed(SEED)
    student = SmallUNet().to(device)
    rand_mse = _eval_mse(student, noisy_test, clean_test, device)
    print(f"  random-init student MSE = {rand_mse:.4f}")

    info = {}

    # ---------- Rebuilt-upstream forward sweep ----------
    with torch.no_grad():
        x_d = noisy_d

        info["enc1_conv1"] = {"target_residual": _fit_conv(student.enc1_conv1, x_d, tgt["enc1_conv1"], device)}
        a = relu(student.enc1_conv1(x_d))
        info["enc1_conv2"] = {"target_residual": _fit_conv(student.enc1_conv2, a, tgt["enc1_conv2"], device)}
        enc1_post = relu(student.enc1_conv2(a))
        p1, idx1  = student.pool1(enc1_post)

        info["enc2_conv1"] = {"target_residual": _fit_conv(student.enc2_conv1, p1, tgt["enc2_conv1"], device)}
        a = relu(student.enc2_conv1(p1))
        info["enc2_conv2"] = {"target_residual": _fit_conv(student.enc2_conv2, a, tgt["enc2_conv2"], device)}
        enc2_post = relu(student.enc2_conv2(a))
        p2, idx2  = student.pool2(enc2_post)

        up2      = student.unpool2(p2, idx2, output_size=enc2_post.size())
        dec2_in  = torch.cat([up2, enc2_post], 1)
        info["dec2_conv1"] = {"target_residual": _fit_conv(student.dec2_conv1, dec2_in, tgt["dec2_conv1"], device)}
        a = relu(student.dec2_conv1(dec2_in))
        info["dec2_conv2"] = {"target_residual": _fit_conv(student.dec2_conv2, a, tgt["dec2_conv2"], device)}
        dec2_post = relu(student.dec2_conv2(a))

        up1      = student.unpool1(dec2_post, idx1, output_size=enc1_post.size())
        dec1_in  = torch.cat([up1, enc1_post], 1)
        info["dec1_conv1"] = {"target_residual": _fit_conv(student.dec1_conv1, dec1_in, tgt["dec1_conv1"], device)}
        a = relu(student.dec1_conv1(dec1_in))
        info["dec1_conv2"] = {"target_residual": _fit_conv(student.dec1_conv2, a, tgt["dec1_conv2"], device)}
        dec1_post = relu(student.dec1_conv2(a))

        info["final_conv"] = {"target_residual": _fit_conv(student.final_conv, dec1_post, tgt["final_conv"], device)}

    student_mse = _eval_mse(student, noisy_test, clean_test, device)
    print(f"  feature-distill student MSE = {student_mse:.4f}  "
          f"(delta vs teacher {student_mse - teacher_mse:+.4f})")
    for nm in _LAYER_NAMES:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm:<14s}: residual={r:.3e}" if r is not None else f"    {nm}: --")

    result = {
        "battery": "B-distill-feature", "exp_id": "e6", "seed": SEED,
        "teacher_mse": teacher_mse, "random_init_mse": rand_mse,
        "student_mse": student_mse, "delta_vs_teacher": student_mse - teacher_mse,
        "per_layer_residuals": {nm: info.get(nm, {}).get("target_residual") for nm in _LAYER_NAMES},
        "note": "32×32 padded MNIST, noise_std=0.5, BN-free, no moment-correction.",
    }
    _save_json(RESULTS_DIR / "b_distill.json", result)
    return result


# ===========================================================================
# B-retrain-rand — chain back-prop with concat-skip decomposition
# ===========================================================================

def battery_retrain(images_clean, noisy_test, clean_test, *, device, method="kfac_a"):
    print("\n[E6] === B-retrain-rand: chain back-prop + concat-skip decomposition ===")
    torch.manual_seed(SEED)
    model = SmallUNet().to(device)
    rand_mse = _eval_mse(model, noisy_test, clean_test, device)
    print(f"  random-init MSE = {rand_mse:.4f}")

    g = torch.Generator().manual_seed(SEED + 20)
    noisy = (images_clean + torch.randn_like(images_clean, generator=g) * NOISE_STD).clamp(0,1)
    n_img = noisy.shape[0]
    relu  = nn.ReLU()
    t0    = time.time()

    # ---------- forward pass on random model ----------
    model.eval()
    with torch.no_grad():
        noisy_d = noisy.to(device)
        st = model.forward_with_state(noisy_d)
    # Move all to CPU+double for OLS math
    fwd = {k: v.detach().cpu().double() for k, v in st.items()}

    # clean target = reconstruction target
    T_recon = images_clean.cpu().double()   # (n,1,32,32)

    # =================== BACKWARD PASS ===================
    # All invert_layer calls use the random model's weights + forward stats.
    # The result is post-activation targets for each layer's output.

    def _inv(layer, a_in_spatial, t_post_2d, a_pre_2d, act):
        """Invert one conv: return spatial target for the layer's INPUT (post-act)."""
        return _invert_conv(layer, a_in_spatial, t_post_2d, a_pre_2d,
                            act, (a_in_spatial.shape[2], a_in_spatial.shape[3]),
                            a_in_spatial.shape[0], method, device)

    # 1. final_conv (no act): target for dec1_post
    T_recon_2d = _cp2d(T_recon)                           # (n*H*W, 1)
    T_dec1_post = _inv(model.final_conv,
                       fwd["dec1_post"].float(),
                       T_recon_2d,
                       _cp2d(fwd["final_pre"]),
                       None)                               # (n,16,32,32)

    # 2. dec1_conv2 (ReLU) → target for dec1_conv1 post-act
    T_d1c1_post = _inv(model.dec1_conv2,
                       relu(fwd["dec1_conv1_pre"]).float(),
                       _cp2d(T_dec1_post),
                       _cp2d(fwd["dec1_conv2_pre"]),
                       relu)                               # (n,16,32,32)

    # 3. dec1_conv1 (ReLU) → target for dec1_in (32ch concat)
    T_dec1_in = _inv(model.dec1_conv1,
                     fwd["dec1_in"].float(),
                     _cp2d(T_d1c1_post),
                     _cp2d(fwd["dec1_conv1_pre"]),
                     relu)                                 # (n,32,32,32)
    # Split: first 16ch = dec_up1 branch; last 16ch = enc1 skip
    T_dec_up1   = T_dec1_in[:, :16, :, :]
    T_enc1_skip = T_dec1_in[:, 16:, :, :]

    # 4. Adjoint of MaxUnpool1: gather T_dec_up1 at pool1_indices
    pool1_idx = fwd["pool1_indices"].long()
    T_dec2_post = _unpool_adjoint(T_dec_up1, pool1_idx, (16, 16))   # (n,16,16,16)

    # 5. dec2_conv2 (ReLU) → target for dec2_conv1 post-act
    T_d2c1_post = _inv(model.dec2_conv2,
                       relu(fwd["dec2_conv1_pre"]).float(),
                       _cp2d(T_dec2_post),
                       _cp2d(fwd["dec2_conv2_pre"]),
                       relu)                               # (n,32,16,16)

    # 6. dec2_conv1 (ReLU) → target for dec2_in (64ch concat)
    T_dec2_in = _inv(model.dec2_conv1,
                     fwd["dec2_in"].float(),
                     _cp2d(T_d2c1_post),
                     _cp2d(fwd["dec2_conv1_pre"]),
                     relu)                                 # (n,64,16,16)
    # Split: first 32ch = dec_up2 branch; last 32ch = enc2 skip
    T_dec_up2   = T_dec2_in[:, :32, :, :]
    T_enc2_skip = T_dec2_in[:, 32:, :, :]

    # 7. Adjoint of MaxUnpool2: gather T_dec_up2 at pool2_indices
    pool2_idx = fwd["pool2_indices"].long()
    T_pool2_out = _unpool_adjoint(T_dec_up2, pool2_idx, (8, 8))     # (n,32,8,8)

    # 8. Invert pool2 (MaxPool, switch-aware) → target for enc2_post (16×16)
    T_enc2_from_pool = _switch_unpool(T_pool2_out, pool2_idx,
                                      fwd["enc2_post"], (16, 16))    # (n,32,16,16)
    T_enc2_post = T_enc2_from_pool + T_enc2_skip                     # sum at merge

    # 9. enc2_conv2 (ReLU) → target for enc2_conv1 post-act
    T_e2c1_post = _inv(model.enc2_conv2,
                       relu(fwd["enc2_conv1_pre"]).float(),
                       _cp2d(T_enc2_post),
                       _cp2d(fwd["enc2_conv2_pre"]),
                       relu)                               # (n,16,16,16)

    # 10. enc2_conv1 (ReLU) → target for pool1_out (16ch, 16×16)
    T_pool1_out = _inv(model.enc2_conv1,
                       fwd["pool1_out"].float(),
                       _cp2d(T_e2c1_post),
                       _cp2d(fwd["enc2_conv1_pre"]),
                       relu)                               # (n,16,16,16)

    # 11. Invert pool1 (switch-aware) → target for enc1_post (32×32)
    T_enc1_from_pool = _switch_unpool(T_pool1_out, pool1_idx,
                                      fwd["enc1_post"], (32, 32))    # (n,16,32,32)
    T_enc1_post = T_enc1_from_pool + T_enc1_skip                     # sum at merge

    # 12. enc1_conv2 (ReLU) → target for enc1_conv1 post-act
    T_e1c1_post = _inv(model.enc1_conv2,
                       relu(fwd["enc1_conv1_pre"]).float(),
                       _cp2d(T_enc1_post),
                       _cp2d(fwd["enc1_conv2_pre"]),
                       relu)                               # (n,1,32,32) post-activation

    # =================== FORWARD OLS SWEEP ===================
    # Shallowest first; each layer uses the rebuilt upstream output as its X.
    # invert_activation converts the post-activation target to a pre-activation target.

    info = {}

    with torch.no_grad():
        x_d = noisy_d

        # enc1_conv1 (ReLU after)
        t_pre = invert_activation(_cp2d(T_e1c1_post), _cp2d(fwd["enc1_conv1_pre"]), relu)
        info["enc1_conv1"] = {"target_residual": _fit_conv(model.enc1_conv1, x_d, t_pre, device)}
        enc1c1_new = relu(model.enc1_conv1(x_d))

        # enc1_conv2 (ReLU after); target = T_enc1_post
        t_pre = invert_activation(_cp2d(T_enc1_post), _cp2d(fwd["enc1_conv2_pre"]), relu)
        info["enc1_conv2"] = {"target_residual": _fit_conv(model.enc1_conv2, enc1c1_new, t_pre, device)}
        enc1_post_new = relu(model.enc1_conv2(enc1c1_new))
        p1_new, idx1_new = model.pool1(enc1_post_new)

        # enc2_conv1 (ReLU after)
        t_pre = invert_activation(_cp2d(T_e2c1_post), _cp2d(fwd["enc2_conv1_pre"]), relu)
        info["enc2_conv1"] = {"target_residual": _fit_conv(model.enc2_conv1, p1_new, t_pre, device)}
        enc2c1_new = relu(model.enc2_conv1(p1_new))

        # enc2_conv2 (ReLU after); target = T_enc2_post
        t_pre = invert_activation(_cp2d(T_enc2_post), _cp2d(fwd["enc2_conv2_pre"]), relu)
        info["enc2_conv2"] = {"target_residual": _fit_conv(model.enc2_conv2, enc2c1_new, t_pre, device)}
        enc2_post_new = relu(model.enc2_conv2(enc2c1_new))
        p2_new, idx2_new = model.pool2(enc2_post_new)

        up2_new   = model.unpool2(p2_new, idx2_new, output_size=enc2_post_new.size())
        dec2_in_new = torch.cat([up2_new, enc2_post_new], 1)

        # dec2_conv1 (ReLU after)
        t_pre = invert_activation(_cp2d(T_d2c1_post), _cp2d(fwd["dec2_conv1_pre"]), relu)
        info["dec2_conv1"] = {"target_residual": _fit_conv(model.dec2_conv1, dec2_in_new, t_pre, device)}
        d2c1_new = relu(model.dec2_conv1(dec2_in_new))

        # dec2_conv2 (ReLU after); target = T_dec2_post
        t_pre = invert_activation(_cp2d(T_dec2_post), _cp2d(fwd["dec2_conv2_pre"]), relu)
        info["dec2_conv2"] = {"target_residual": _fit_conv(model.dec2_conv2, d2c1_new, t_pre, device)}
        dec2_post_new = relu(model.dec2_conv2(d2c1_new))

        up1_new   = model.unpool1(dec2_post_new, idx1_new, output_size=enc1_post_new.size())
        dec1_in_new = torch.cat([up1_new, enc1_post_new], 1)

        # dec1_conv1 (ReLU after)
        t_pre = invert_activation(_cp2d(T_d1c1_post), _cp2d(fwd["dec1_conv1_pre"]), relu)
        info["dec1_conv1"] = {"target_residual": _fit_conv(model.dec1_conv1, dec1_in_new, t_pre, device)}
        d1c1_new = relu(model.dec1_conv1(dec1_in_new))

        # dec1_conv2 (ReLU after); target = T_dec1_post
        t_pre = invert_activation(_cp2d(T_dec1_post), _cp2d(fwd["dec1_conv2_pre"]), relu)
        info["dec1_conv2"] = {"target_residual": _fit_conv(model.dec1_conv2, d1c1_new, t_pre, device)}
        dec1_post_new = relu(model.dec1_conv2(d1c1_new))

        # final_conv (no activation)
        info["final_conv"] = {"target_residual": _fit_conv(model.final_conv, dec1_post_new, T_recon_2d, device)}

    retrain_mse = _eval_mse(model, noisy_test, clean_test, device)
    wall = time.time() - t0
    print(f"  retrain-rand MSE = {retrain_mse:.4f}  "
          f"(delta random {retrain_mse - rand_mse:+.4f})  ({wall:.1f}s)")
    for nm in _LAYER_NAMES:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm:<14s}: residual={r:.3e}" if r is not None else f"    {nm}: --")

    result = {
        "battery": "B-retrain-rand", "exp_id": "e6", "seed": SEED, "method": method,
        "random_mse": rand_mse, "retrained_mse": retrain_mse,
        "delta_from_random": retrain_mse - rand_mse,
        "wall_s": wall,
        "per_layer_residuals": {nm: info.get(nm, {}).get("target_residual") for nm in _LAYER_NAMES},
    }
    _save_json(RESULTS_DIR / "b_retrain.json", result)
    return result


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--option", choices=("train", "distill", "retrain", "all"), default="all")
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--method", choices=("naive", "kfac_a"), default="kfac_a")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[E6] device={device}  option={args.option}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_set, test_set = _load_mnist(Path(args.data_root))
    # Build test tensors once
    clean_test = _build_clean_tensors(test_set, min(len(test_set), 2000))
    g = torch.Generator().manual_seed(SEED + 1)
    noisy_test = (clean_test + torch.randn_like(clean_test, generator=g) * NOISE_STD).clamp(0, 1)
    # Capture images for OLS
    images_clean = _build_clean_tensors(train_set, N_IMAGES)

    def _load_trained():
        if not UNET_WEIGHTS.exists():
            print(f"[E6] no weights at {UNET_WEIGHTS}; run --option train first.")
            return None
        m = SmallUNet().to(device)
        m.load_state_dict(torch.load(UNET_WEIGHTS, map_location=device, weights_only=True))
        return m

    if args.option in ("train", "all"):
        battery_train(train_set, noisy_test, clean_test, device=device, epochs=args.epochs)

    if args.option in ("distill", "all"):
        teacher = _load_trained()
        if teacher is None:
            return 1
        battery_distill(teacher, images_clean, noisy_test, clean_test, device=device)

    if args.option in ("retrain", "all"):
        battery_retrain(images_clean, noisy_test, clean_test, device=device, method=args.method)

    print("\n[E6] === COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
