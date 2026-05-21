"""
diagnostic/runners/arch_coverage/e3_lenet.py
=============================================

E3: LeNet CNN on MNIST classification.

Architecture: conv1→relu→pool1→conv2→relu→pool2→flatten→fc1→relu→fc2→relu→fc3

Standardized battery:
  B-train          — train via backprop, cache weights
  B-distill        — single-pass TP+OLS from trained weights
  B-retrain-rand   — single-pass TP+OLS from random init (GT one-hot targets)
  B-residual-profile — per-layer residual curve (all 5 parametric layers)
  B-iter           — 10 iterations from random init
  B-subset         — layer-subset sweep (FC subsets, then FC+conv subsets)
  B-K-vs-N         — {naive, kfac_a} × {no_mom, mom_match}

Conv layers use im2col (F.unfold) via the extended RawActivationHooks.
MaxPool inversion uses nearest-neighbor upsampling.
OLS for conv layers: W reshaped from (C_out, C_in*kH*kW) → (C_out, C_in, kH, kW).

N_IMAGES activations are collected per pass (consistent across FC and conv layers so
that the TP chain targets align spatially).

Outputs: benchmark/results/diagnostic/arch_coverage/e3/{battery_item}.json
Weights: benchmark/weights/arch_coverage/lenet.pt

Usage:
    python -m diagnostic.runners.arch_coverage.e3_lenet
    python -m diagnostic.runners.arch_coverage.e3_lenet --skip-train
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, Subset

from diagnostic.capture import collect_activations
from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.models.lenet import LeNet
from diagnostic.multi_step import _empirical_mean_cov, _moment_match, _shift_mean_to
from diagnostic.runners.phase1_gt_retrain import make_gt_logit_target
from diagnostic.runners.phase1_mlp_mnist import _eval, _load_mnist
from diagnostic.target_prop_retrainer import solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = (
    REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e3"
)
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"
WEIGHTS_PATH = WEIGHTS_DIR / "lenet.pt"

# Number of full images used per TP pass (consistent across FC + conv).
# conv1 stores N_IMAGES*784 im2col patches, conv2 stores N_IMAGES*100 patches.
N_IMAGES = 512
N_SAMPLES = 16_384   # for training / standard eval
SEED = 42
GT_MARGIN = 5.0
OLS_LAMBDA = 1e-4
OLS_EPS = 1e-4
N_ITER = 10
TRAIN_EPOCHS = 15

# LeNet spatial constants (28×28 MNIST input)
CONV2_H_OUT, CONV2_W_OUT = 10, 10    # conv2 output spatial size
CONV1_H_OUT, CONV1_W_OUT = 28, 28    # conv1 output spatial size (padding=2)
POOL2_H_OUT, POOL2_W_OUT = 5, 5      # after MaxPool 2×2
POOL1_H_OUT, POOL1_W_OUT = 14, 14    # after MaxPool 2×2
FC1_IN = 400                          # 16 * 5 * 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _train(model: nn.Module, train_loader, test_loader, *,
           epochs: int, device: torch.device, lr: float = 1e-3) -> float:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F_torch.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()
        acc = _eval(model, test_loader, device)
        print(f"  [train] epoch {epoch + 1:2d}/{epochs}  "
              f"loss={running / seen:.4f}  test_acc={acc:.4f}")
    print(f"  [train] total: {time.time() - t0:.1f}s")
    return _eval(model, test_loader, device)


def _build_loader_list(dataset, n_images: int, batch_size: int = 128):
    """Return (images_list, labels_tensor) for exactly n_images samples (no shuffle)."""
    subset_size = min(len(dataset), n_images * 4)
    subset = Subset(dataset, list(range(subset_size)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, label_chunks = [], []
    for x, y in loader:
        images.append(x)
        label_chunks.append(y)
    labels = torch.cat(label_chunks, dim=0)[:n_images]
    # Truncate image list so total images == n_images exactly
    truncated, total = [], 0
    for batch in images:
        needed = n_images - total
        if needed <= 0:
            break
        truncated.append(batch[:needed] if batch.shape[0] > needed else batch)
        total += truncated[-1].shape[0]
    return truncated, labels


# ---------------------------------------------------------------------------
# Capture: separate FC and conv activations with consistent N_IMAGES
# ---------------------------------------------------------------------------

def _capture_all(model: LeNet, images_list: list, n_images: int,
                 device: torch.device):
    """Capture FC and conv activations for the same n_images images.

    FC layers: (n_images, d) tensors.
    Conv layers: all spatial positions — (n_images*H_out*W_out, C) in im2col form.
    """
    fc_layers = model.get_fc_layers()       # [fc1, fc2, fc3]
    conv_layers = model.get_conv_layers()   # [conv1, conv2]

    # FC captures: n_images FC samples
    fc_captured = collect_activations(
        model, images_list, fc_layers,
        max_samples=n_images,
        seq_subsample=n_images,
        device=device,
    )

    # Conv captures: ALL spatial positions for n_images images (no subsampling).
    # conv2: n_images * 10 * 10 = n_images * 100
    # conv1: n_images * 28 * 28 = n_images * 784
    conv_max = n_images * CONV1_H_OUT * CONV1_W_OUT  # conv1 is the larger
    conv_captured = collect_activations(
        model, images_list, conv_layers,
        max_samples=conv_max,
        seq_subsample=conv_max,
        device=device,
    )

    return fc_captured, conv_captured


# ---------------------------------------------------------------------------
# TP chain for LeNet: FC chain → flatten/pool transitions → conv chain
# ---------------------------------------------------------------------------

def _lenet_tp_retrain(
    model: LeNet,
    images_list: list,
    gt_targets: torch.Tensor,   # (n_images, 10) GT logit targets
    n_images: int,
    method: str,
    correct_target_mean: bool,
    correct_target_cov: bool,
    device: torch.device,
    *,
    target_layer_set: set = None,  # if None, retrain ALL 5 layers
) -> Dict[str, dict]:
    """Single-pass TP+OLS for full LeNet chain (FC + conv).

    Returns info dict keyed by layer name ('fc3', ..., 'conv1').
    If target_layer_set is provided, only those layers get OLS weight updates.
    """
    fc_captured, conv_captured = _capture_all(model, images_list, n_images, device)

    fc3, fc2, fc1 = model.fc3, model.fc2, model.fc1
    conv2, conv1 = model.conv2, model.conv1

    n_fc = fc_captured[fc3]["a_post"].shape[0]
    n_use = min(n_fc, gt_targets.shape[0])

    # ------------------------------------------------------------------
    # Phase 1: Back-prop chain through FC layers (fc3 → fc2 → fc1)
    # ------------------------------------------------------------------
    cur_target = gt_targets[:n_use].double().to(
        fc_captured[fc3]["a_pre"].device)

    fc_targets: Dict[nn.Linear, torch.Tensor] = {}
    for layer in [fc3, fc2, fc1]:
        cap = fc_captured[layer]
        a_pre_t = cap["a_pre"].double()[:n_use]
        activation = cap["activation"]
        a_in_t = cap["a_in"].double()[:n_use]
        mu_a, Sigma_a = _empirical_mean_cov(a_in_t)
        fc_targets[layer] = cur_target.clone()

        W = layer.weight.detach().to(device=a_pre_t.device, dtype=a_pre_t.dtype)
        b = (layer.bias.detach().to(device=a_pre_t.device, dtype=a_pre_t.dtype)
             if layer.bias is not None else None)

        if method == "naive":
            a_hat = invert_layer(W, b, cur_target, a_pre_t, activation, "naive", eps=OLS_EPS)
        else:
            a_hat = invert_layer(W, b, cur_target, a_pre_t, activation, "kfac_a",
                                 mu_a=mu_a, Sigma_a=Sigma_a)

        next_target = a_hat
        if correct_target_mean and layer is not fc1:
            tgt_mean = a_in_t.mean(dim=0)
            if correct_target_cov:
                _, tgt_Sigma = _empirical_mean_cov(a_in_t)
                next_target = _moment_match(a_hat, tgt_mean, tgt_Sigma)
            else:
                next_target = _shift_mean_to(a_hat, tgt_mean)
        cur_target = next_target

    # cur_target: (n_use, 400) — target for fc1 input = flatten(pool2 output)

    # ------------------------------------------------------------------
    # Transition: FC → conv2 (unflatten + invert pool2)
    # ------------------------------------------------------------------
    # (n_use, 400) → (n_use, 16, 5, 5) [pool2 output target]
    pool2_target = cur_target.reshape(n_use, 16, POOL2_H_OUT, POOL2_W_OUT)
    # Invert MaxPool2d(2,2): nearest-neighbor upsample × 2
    # (n_use, 16, 5, 5) → (n_use, 16, 10, 10)
    conv2_apost_spatial = F_torch.interpolate(
        pool2_target.float(), scale_factor=2, mode="nearest"
    ).double()

    # ------------------------------------------------------------------
    # Back-prop through conv2
    # ------------------------------------------------------------------
    n_conv2 = n_use * CONV2_H_OUT * CONV2_W_OUT   # n_use * 100
    # Reshape to im2col form: (n_use*100, 16)
    conv2_apost_2d = conv2_apost_spatial.permute(0, 2, 3, 1).reshape(n_conv2, 16)

    cap2 = conv_captured[conv2]
    a_pre_c2 = cap2["a_pre"].double()[:n_conv2]
    a_in_c2 = cap2["a_in"].double()[:n_conv2]
    activation_c2 = cap2["activation"]
    mu_c2, Sigma_c2 = _empirical_mean_cov(a_in_c2)

    W_c2 = conv2.weight.detach().reshape(conv2.out_channels, -1).to(
        device=a_in_c2.device, dtype=a_in_c2.dtype)  # (16, 150)
    b_c2 = (conv2.bias.detach().to(device=a_in_c2.device, dtype=a_in_c2.dtype)
            if conv2.bias is not None else None)

    if method == "naive":
        a_hat_c2 = invert_layer(W_c2, b_c2, conv2_apost_2d, a_pre_c2, activation_c2,
                                "naive", eps=OLS_EPS)
    else:
        a_hat_c2 = invert_layer(W_c2, b_c2, conv2_apost_2d, a_pre_c2, activation_c2,
                                "kfac_a", mu_a=mu_c2, Sigma_a=Sigma_c2)
    # a_hat_c2: (n_use*100, 150) — target for conv2 input in im2col form

    # Fold back to (n_use, 6, 14, 14) [pool1 output target]
    # F.fold(input=(n_use, 150, 100), output_size=(14,14), kernel_size=5)
    a_hat_c2_fold = a_hat_c2.reshape(n_use, CONV2_H_OUT * CONV2_W_OUT, 150)
    a_hat_c2_fold = a_hat_c2_fold.permute(0, 2, 1).float()    # (n_use, 150, 100)
    folded = F_torch.fold(a_hat_c2_fold, output_size=(POOL1_H_OUT, POOL1_W_OUT),
                          kernel_size=conv2.kernel_size,
                          dilation=conv2.dilation, padding=conv2.padding,
                          stride=conv2.stride)   # (n_use, 6, 14, 14)
    ones = torch.ones_like(a_hat_c2_fold)
    count = F_torch.fold(ones, output_size=(POOL1_H_OUT, POOL1_W_OUT),
                         kernel_size=conv2.kernel_size,
                         dilation=conv2.dilation, padding=conv2.padding,
                         stride=conv2.stride)
    pool1_target = (folded / count.clamp(min=1)).double()  # (n_use, 6, 14, 14)

    # ------------------------------------------------------------------
    # Transition: conv2 → conv1 (invert pool1)
    # ------------------------------------------------------------------
    # (n_use, 6, 14, 14) → (n_use, 6, 28, 28)
    conv1_apost_spatial = F_torch.interpolate(
        pool1_target.float(), scale_factor=2, mode="nearest"
    ).double()

    # ------------------------------------------------------------------
    # Back-prop through conv1 (just collect targets; no further chain)
    # ------------------------------------------------------------------
    n_conv1 = n_use * CONV1_H_OUT * CONV1_W_OUT   # n_use * 784
    # (n_use*784, 6)
    conv1_apost_2d = conv1_apost_spatial.permute(0, 2, 3, 1).reshape(n_conv1, 6)

    cap1 = conv_captured[conv1]
    a_pre_c1 = cap1["a_pre"].double()[:n_conv1]
    a_in_c1 = cap1["a_in"].double()[:n_conv1]
    activation_c1 = cap1["activation"]

    # ------------------------------------------------------------------
    # Phase 2: Forward OLS sweep (shallowest → deepest):
    #   conv1 → conv2 → fc1 → fc2 → fc3
    # ------------------------------------------------------------------
    if target_layer_set is None:
        target_layer_set = {"conv1", "conv2", "fc1", "fc2", "fc3"}

    info: Dict[str, dict] = {}

    # --- conv1 ---
    if "conv1" in target_layer_set:
        t_pre_c1 = invert_activation(conv1_apost_2d, a_pre_c1, activation_c1)
        W_c1_new, b_c1_new = solve_ols_layer(
            a_in_c1, t_pre_c1,
            with_bias=(conv1.bias is not None), ols_lambda=OLS_LAMBDA,
        )
        pred_c1 = a_in_c1 @ W_c1_new.T
        if b_c1_new is not None:
            pred_c1 = pred_c1 + b_c1_new.unsqueeze(0)
        residual_c1 = (pred_c1 - t_pre_c1).norm() / t_pre_c1.norm().clamp(min=1e-30)
        with torch.no_grad():
            conv1.weight.copy_(
                W_c1_new.reshape(conv1.weight.shape).to(conv1.weight.device, conv1.weight.dtype)
            )
            if conv1.bias is not None and b_c1_new is not None:
                conv1.bias.copy_(b_c1_new.to(conv1.bias.device, conv1.bias.dtype))
        info["conv1"] = {"target_residual": residual_c1.item()}

    # --- conv2 ---
    if "conv2" in target_layer_set:
        t_pre_c2 = invert_activation(conv2_apost_2d, a_pre_c2, activation_c2)
        W_c2_new, b_c2_new = solve_ols_layer(
            a_in_c2, t_pre_c2,
            with_bias=(conv2.bias is not None), ols_lambda=OLS_LAMBDA,
        )
        pred_c2 = a_in_c2 @ W_c2_new.T
        if b_c2_new is not None:
            pred_c2 = pred_c2 + b_c2_new.unsqueeze(0)
        residual_c2 = (pred_c2 - t_pre_c2).norm() / t_pre_c2.norm().clamp(min=1e-30)
        with torch.no_grad():
            conv2.weight.copy_(
                W_c2_new.reshape(conv2.weight.shape).to(conv2.weight.device, conv2.weight.dtype)
            )
            if conv2.bias is not None and b_c2_new is not None:
                conv2.bias.copy_(b_c2_new.to(conv2.bias.device, conv2.bias.dtype))
        info["conv2"] = {"target_residual": residual_c2.item()}

    # --- fc1, fc2, fc3 (shallowest → deepest, use captured a_in) ---
    for layer, name in [(fc1, "fc1"), (fc2, "fc2"), (fc3, "fc3")]:
        if name not in target_layer_set:
            continue
        cap = fc_captured[layer]
        a_pre_fwd = cap["a_pre"].double()[:n_use]
        act_after = cap["activation"]
        t_post = fc_targets[layer]
        t_pre = invert_activation(t_post, a_pre_fwd, act_after)
        X = cap["a_in"].double()[:n_use]
        W_new, b_new = solve_ols_layer(
            X, t_pre, with_bias=(layer.bias is not None), ols_lambda=OLS_LAMBDA
        )
        pred = X @ W_new.T
        if b_new is not None:
            pred = pred + b_new.unsqueeze(0)
        residual = (pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30)
        with torch.no_grad():
            layer.weight.copy_(W_new.to(layer.weight.device, layer.weight.dtype))
            if layer.bias is not None and b_new is not None:
                layer.bias.copy_(b_new.to(layer.bias.device, layer.bias.dtype))
        info[name] = {"target_residual": residual.item()}

    return info


# ---------------------------------------------------------------------------
# Battery items
# ---------------------------------------------------------------------------

_LAYER_NAMES_DF = ["fc3", "fc2", "fc1", "conv2", "conv1"]  # deepest-first


def b_train(train_set, test_loader, *, device: torch.device) -> dict:
    print("\n[E3] === B-train: training LeNet ===")
    model = LeNet().to(device)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    t0 = time.time()
    acc = _train(model, train_loader, test_loader, epochs=TRAIN_EPOCHS, device=device)
    wall_s = time.time() - t0
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"  [B-train] test_acc={acc:.4f}  wall={wall_s:.1f}s  weights -> {WEIGHTS_PATH}")
    result = {
        "battery": "B-train", "exp_id": "e3",
        "test_acc": acc, "epochs": TRAIN_EPOCHS, "wall_s": wall_s,
    }
    _save_json(RESULTS_DIR / "b_train.json", result)
    return result


def b_distill(trained_model: LeNet, images_list: list,
              test_loader, *, device: torch.device) -> dict:
    """Self-distillation: trained weights, deepest target = model's own logits."""
    print("\n[E3] === B-distill: self-distillation from trained model ===")
    # Collect model's own logit outputs as GT targets for fc3
    trained_model.eval()
    logit_chunks = []
    with torch.no_grad():
        for batch in images_list:
            x = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            logit_chunks.append(trained_model(x).cpu())
    distill_targets = torch.cat(logit_chunks, dim=0)[:N_IMAGES].double()

    model = copy.deepcopy(trained_model)
    t0 = time.time()
    info = _lenet_tp_retrain(
        model, images_list, distill_targets, N_IMAGES,
        method="kfac_a", correct_target_mean=True, correct_target_cov=True,
        device=device,
    )
    acc = _eval(model, test_loader, device)
    wall_s = time.time() - t0
    print(f"  [B-distill] test_acc={acc:.4f}  wall={wall_s:.1f}s")
    for name in _LAYER_NAMES_DF:
        r = info.get(name, {}).get("target_residual")
        print(f"    {name}: residual={r:.3g}" if r is not None else f"    {name}: --")
    result = {
        "battery": "B-distill", "exp_id": "e3",
        "test_acc": acc, "wall_s": wall_s,
        "per_layer_residuals": {n: info.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
    }
    _save_json(RESULTS_DIR / "b_distill.json", result)
    return result


def b_retrain_rand(train_set, test_loader, *, device: torch.device,
                   n_images: int) -> dict:
    print("\n[E3] === B-retrain-rand: random init + GT labels, single pass ===")
    torch.manual_seed(SEED)
    model = LeNet().to(device)
    rand_acc = _eval(model, test_loader, device)
    print(f"  random-init test_acc={rand_acc:.4f}")

    images_list, labels = _build_loader_list(train_set, n_images)
    gt_targets = make_gt_logit_target(labels[:n_images], n_classes=10, margin=GT_MARGIN)

    t0 = time.time()
    info = _lenet_tp_retrain(
        model, images_list, gt_targets, n_images,
        method="kfac_a", correct_target_mean=True, correct_target_cov=True,
        device=device,
    )
    acc = _eval(model, test_loader, device)
    wall_s = time.time() - t0
    print(f"  [B-retrain-rand] test_acc={acc:.4f}  delta={acc - rand_acc:+.4f}  wall={wall_s:.1f}s")
    for name in _LAYER_NAMES_DF:
        r = info.get(name, {}).get("target_residual")
        print(f"    {name}: residual={r:.3g}" if r is not None else f"    {name}: --")
    result = {
        "battery": "B-retrain-rand", "exp_id": "e3", "seed": SEED,
        "random_init_acc": rand_acc, "retrained_acc": acc,
        "delta": acc - rand_acc, "wall_s": wall_s,
        "per_layer_residuals": {n: info.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
    }
    _save_json(RESULTS_DIR / "b_retrain_rand.json", result)
    return result


def b_residual_profile(distill_result: dict, retrain_rand_result: dict) -> dict:
    print("\n[E3] === B-residual-profile ===")
    layer_names = list(reversed(_LAYER_NAMES_DF))  # display shallowest→deepest
    dr = distill_result.get("per_layer_residuals", {})
    rr = retrain_rand_result.get("per_layer_residuals", {})
    profile = {
        name: {"distill": dr.get(name), "retrain_rand": rr.get(name)}
        for name in layer_names
    }
    print(f"  {'layer':<8s}  {'distill':>10s}  {'retrain_rand':>12s}")
    for name in layer_names:
        d = profile[name]["distill"]
        r = profile[name]["retrain_rand"]
        print(f"  {name:<8s}  "
              f"{(f'{d:.3g}' if d is not None else '--'):>10s}  "
              f"{(f'{r:.3g}' if r is not None else '--'):>12s}")
    result = {
        "battery": "B-residual-profile", "exp_id": "e3",
        "layer_order": layer_names, "profile": profile,
        "distill_acc": distill_result.get("test_acc"),
        "retrain_rand_acc": retrain_rand_result.get("retrained_acc"),
    }
    _save_json(RESULTS_DIR / "b_residual_profile.json", result)
    _plot_residual_profile(profile, layer_names)
    return result


def _plot_residual_profile(profile: dict, layer_names: List[str]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    distill_vals = [profile[n]["distill"] for n in layer_names]
    retrain_vals = [profile[n]["retrain_rand"] for n in layer_names]
    x = range(len(layer_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, distill_vals, "o-", label="B-distill (trained start)")
    ax.plot(x, retrain_vals, "s--", label="B-retrain-rand (random init)")
    ax.set_xticks(list(x)); ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylabel("OLS target residual"); ax.set_title("E3 LeNet: per-layer residual profile")
    ax.legend(); ax.set_yscale("log"); fig.tight_layout()
    out = RESULTS_DIR / "b_residual_profile.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def b_iter(train_set, test_loader, *, device: torch.device,
           n_images: int) -> dict:
    print(f"\n[E3] === B-iter: {N_ITER} iterations from random init ===")
    torch.manual_seed(SEED)
    model = LeNet().to(device)
    rand_acc = _eval(model, test_loader, device)
    print(f"  random-init test_acc={rand_acc:.4f}")

    images_list, labels = _build_loader_list(train_set, n_images)
    gt_targets = make_gt_logit_target(labels[:n_images], n_classes=10, margin=GT_MARGIN)

    iter_accs = [rand_acc]
    t0 = time.time()
    for it in range(N_ITER):
        _lenet_tp_retrain(
            model, images_list, gt_targets, n_images,
            method="kfac_a", correct_target_mean=True, correct_target_cov=True,
            device=device,
        )
        acc = _eval(model, test_loader, device)
        iter_accs.append(acc)
        print(f"  iter {it + 1:2d}: acc={acc:.4f}  delta={acc - rand_acc:+.4f}")
    wall_s = time.time() - t0
    traj = " -> ".join(f"{a:.4f}" for a in iter_accs)
    print(f"  [B-iter] done in {wall_s:.1f}s\n  trajectory: {traj}")
    _plot_iter_trajectory(iter_accs)
    result = {
        "battery": "B-iter", "exp_id": "e3", "seed": SEED, "n_iterations": N_ITER,
        "random_init_acc": rand_acc, "iter_accs": iter_accs,
        "final_acc": iter_accs[-1], "wall_s": wall_s,
    }
    _save_json(RESULTS_DIR / "b_iter.json", result)
    return result


def _plot_iter_trajectory(iter_accs: List[float]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(iter_accs)), iter_accs, "o-")
    ax.set_xlabel("iteration"); ax.set_ylabel("test accuracy")
    ax.set_title("E3 LeNet: B-iter accuracy trajectory"); fig.tight_layout()
    out = RESULTS_DIR / "b_iter.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def b_subset(trained_model: LeNet, images_list: list, labels: torch.Tensor,
             test_loader, *, device: torch.device, n_images: int) -> dict:
    """Sweep subsets deepest-first: {fc3}, {fc3,fc2}, ..., {all 5 layers}."""
    print("\n[E3] === B-subset: layer-subset sweep from trained model ===")
    trained_acc = _eval(trained_model, test_loader, device)
    print(f"  trained baseline acc={trained_acc:.4f}")

    gt_targets = make_gt_logit_target(labels[:n_images], n_classes=10, margin=GT_MARGIN)
    subset_results = {}
    for k in range(1, len(_LAYER_NAMES_DF) + 1):
        subset = set(_LAYER_NAMES_DF[:k])  # deepest-first k layers
        subset_label = "+".join(_LAYER_NAMES_DF[:k])
        t0 = time.time()
        work_model = copy.deepcopy(trained_model)
        info = _lenet_tp_retrain(
            work_model, images_list, gt_targets, n_images,
            method="kfac_a", correct_target_mean=True, correct_target_cov=True,
            device=device, target_layer_set=subset,
        )
        acc = _eval(work_model, test_loader, device)
        wall_s = time.time() - t0
        label = f"deepest_{k}"
        subset_results[label] = {
            "n_retrained": k, "target_layers": list(subset),
            "acc": acc, "delta": acc - trained_acc, "wall_s": wall_s,
        }
        print(f"  {label} ({k} layers [{subset_label}]): "
              f"acc={acc:.4f}  delta={acc - trained_acc:+.4f}  ({wall_s:.1f}s)")
    result = {
        "battery": "B-subset", "exp_id": "e3",
        "trained_acc": trained_acc, "subsets": subset_results,
    }
    _save_json(RESULTS_DIR / "b_subset.json", result)
    _plot_subset_curve(subset_results, trained_acc)
    return result


def _plot_subset_curve(subset_results: dict, trained_acc: float) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = sorted(subset_results, key=lambda k: subset_results[k]["n_retrained"])
    ns = [subset_results[k]["n_retrained"] for k in labels]
    accs = [subset_results[k]["acc"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, accs, "o-", label="TP+OLS retrain (trained start)")
    ax.axhline(trained_acc, color="gray", linestyle="--",
               label=f"trained baseline ({trained_acc:.3f})")
    ax.set_xlabel("# layers retrained (deepest-first)"); ax.set_ylabel("test accuracy")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n}\n({'+'.join(_LAYER_NAMES_DF[:n])})" for n in ns],
                       fontsize=7, rotation=20, ha="right")
    ax.set_title("E3 LeNet: B-subset layer-sweep"); ax.legend(); fig.tight_layout()
    out = RESULTS_DIR / "b_subset.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def b_k_vs_n(train_set, test_loader, *, device: torch.device,
             n_images: int) -> dict:
    print("\n[E3] === B-K-vs-N: method × momentum comparison ===")
    torch.manual_seed(SEED)
    base_model = LeNet().to(device)
    rand_acc = _eval(base_model, test_loader, device)
    print(f"  random-init test_acc={rand_acc:.4f}")

    images_list, labels = _build_loader_list(train_set, n_images)
    gt_targets = make_gt_logit_target(labels[:n_images], n_classes=10, margin=GT_MARGIN)

    combos = [
        ("naive",  False, False, "naive_no_mom"),
        ("naive",  True,  True,  "naive_mom_match"),
        ("kfac_a", False, False, "kfac_a_no_mom"),
        ("kfac_a", True,  True,  "kfac_a_mom_match"),
    ]
    combo_results = {}
    for method, mean_corr, cov_corr, label in combos:
        model = copy.deepcopy(base_model)
        t0 = time.time()
        _lenet_tp_retrain(
            model, images_list, gt_targets, n_images,
            method=method, correct_target_mean=mean_corr, correct_target_cov=cov_corr,
            device=device,
        )
        acc = _eval(model, test_loader, device)
        wall_s = time.time() - t0
        combo_results[label] = {
            "method": method, "correct_target_mean": mean_corr,
            "correct_target_cov": cov_corr, "acc": acc,
            "delta": acc - rand_acc, "wall_s": wall_s,
        }
        print(f"  {label:<22s}: acc={acc:.4f}  delta={acc - rand_acc:+.4f}  ({wall_s:.1f}s)")
    result = {
        "battery": "B-K-vs-N", "exp_id": "e3", "seed": SEED,
        "random_init_acc": rand_acc, "combos": combo_results,
    }
    _save_json(RESULTS_DIR / "b_k_vs_n.json", result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-distill", action="store_true")
    ap.add_argument("--skip-retrain-rand", action="store_true")
    ap.add_argument("--skip-iter", action="store_true")
    ap.add_argument("--skip-subset", action="store_true")
    ap.add_argument("--skip-k-vs-n", action="store_true")
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[E3] LeNet  device={device}  n_images={args.n_images}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[E3] loading MNIST...")
    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    if args.skip_train and WEIGHTS_PATH.exists():
        print(f"[E3] loading cached weights from {WEIGHTS_PATH}")
        trained_model = LeNet().to(device)
        trained_model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        )
        trained_acc = _eval(trained_model, test_loader, device)
        print(f"[E3] loaded; test_acc={trained_acc:.4f}")
        train_result = {"battery": "B-train", "test_acc": trained_acc, "loaded_from_cache": True}
    else:
        train_result = b_train(train_set, test_loader, device=device)
        trained_model = LeNet().to(device)
        trained_model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        )

    images_list, labels = _build_loader_list(train_set, args.n_images)

    distill_result = {}
    if not args.skip_distill:
        distill_result = b_distill(trained_model, images_list, test_loader, device=device)

    retrain_rand_result = {}
    if not args.skip_retrain_rand:
        retrain_rand_result = b_retrain_rand(
            train_set, test_loader, device=device, n_images=args.n_images
        )

    if distill_result and retrain_rand_result:
        b_residual_profile(distill_result, retrain_rand_result)

    if not args.skip_iter:
        b_iter(train_set, test_loader, device=device, n_images=args.n_images)

    if not args.skip_subset:
        b_subset(trained_model, images_list, labels, test_loader,
                 device=device, n_images=args.n_images)

    if not args.skip_k_vs_n:
        b_k_vs_n(train_set, test_loader, device=device, n_images=args.n_images)

    print(f"\n[E3] === COMPLETE ===")
    print(f"  B-train acc:       {train_result.get('test_acc', '?')}")
    print(f"  B-distill acc:     {distill_result.get('test_acc', '?')}")
    print(f"  B-retrain-rand acc:{retrain_rand_result.get('retrained_acc', '?')}")
    print(f"  Results at: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
