"""
diagnostic/runners/arch_coverage/e3_conv_diag.py
=================================================

Conv diagnostics for the E3 LeNet failure (CNN retraining collapsed to ~random).

Three options, run in order:

  OPTION 1 — round-trip / feature-level OLS.
    For the TRAINED LeNet, OLS-refit each layer to reproduce its OWN captured
    a_pre (the layer's actual output). Targets are achievable by construction,
    so residual should be ~0 for ALL layers, including conv. If conv residual
    is ~0 here, the conv OLS + im2col capture is sound and the failure is the
    pool inversion alone. Decisive isolation test.

  OPTION 2 — switch-aware max-unpool.
    Re-run conv TP retrain from random init, but invert MaxPool using the
    stored argmax switch indices (F.max_unpool2d): scatter the target to the
    argmax position, keep the forward pre-pool value elsewhere. Replaces the
    nearest-neighbor upsampling that broadcasts the max to all 4 positions.

  OPTION 3 — avg-pool variant.
    Train a LeNetAvgPool (avg-pool instead of max-pool), then run the existing
    conv TP retrain on it. Nearest-neighbor upsampling IS the correct adjoint
    for avg-pool, so this isolates whether the conv path works once pooling is
    well-behaved.

Usage:
    python -m diagnostic.runners.arch_coverage.e3_conv_diag --option 1
    python -m diagnostic.runners.arch_coverage.e3_conv_diag --option 2
    python -m diagnostic.runners.arch_coverage.e3_conv_diag --option 3
    python -m diagnostic.runners.arch_coverage.e3_conv_diag --option all
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diagnostic.capture import collect_activations
from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.models.lenet import LeNet, LeNetAvgPool
from diagnostic.multi_step import _empirical_mean_cov, _moment_match, _shift_mean_to
from diagnostic.runners.phase1_gt_retrain import make_gt_logit_target
from diagnostic.runners.phase1_mlp_mnist import _eval, _load_mnist
from diagnostic.target_prop_retrainer import solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e3"
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"
LENET_WEIGHTS = WEIGHTS_DIR / "lenet.pt"
AVGPOOL_WEIGHTS = WEIGHTS_DIR / "lenet_avgpool.pt"

N_IMAGES = 512
SEED = 42
GT_MARGIN = 5.0
OLS_LAMBDA = 1e-4
OLS_EPS = 1e-4
TRAIN_EPOCHS = 15

# LeNet spatial constants (28x28 MNIST)
CONV1_HW = (28, 28)
POOL1_HW = (14, 14)
CONV2_HW = (10, 10)
POOL2_HW = (5, 5)

_LAYER_NAMES_DF = ["fc3", "fc2", "fc1", "conv2", "conv1"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _build_loader_list(dataset, n_images: int, batch_size: int = 128):
    subset = Subset(dataset, list(range(min(len(dataset), n_images * 4))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, label_chunks = [], []
    for x, y in loader:
        images.append(x)
        label_chunks.append(y)
    labels = torch.cat(label_chunks, dim=0)[:n_images]
    truncated, total = [], 0
    for batch in images:
        need = n_images - total
        if need <= 0:
            break
        truncated.append(batch[:need] if batch.shape[0] > need else batch)
        total += truncated[-1].shape[0]
    return truncated, labels


def _im2col(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """(B, C_in, H, W) -> (B*H_out*W_out, C_in*kH*kW)."""
    unf = F.unfold(x, kernel_size=conv.kernel_size, dilation=conv.dilation,
                   padding=conv.padding, stride=conv.stride)
    return unf.permute(0, 2, 1).reshape(-1, unf.shape[1])


def _train(model: nn.Module, train_loader, test_loader, *, epochs, device, lr=1e-3):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
        acc = _eval(model, test_loader, device)
        print(f"  [train] epoch {epoch + 1:2d}/{epochs}  test_acc={acc:.4f}")
    return _eval(model, test_loader, device)


# ===========================================================================
# OPTION 1 — round-trip / feature-level OLS
# ===========================================================================

def option1_round_trip(trained_model, images_list, test_loader, *, device) -> dict:
    print("\n[E3-diag] === OPTION 1: round-trip feature-level OLS ===")
    layers = trained_model.get_layers_deepest_first()  # [fc3,fc2,fc1,conv2,conv1]
    name_of = {trained_model.fc3: "fc3", trained_model.fc2: "fc2",
               trained_model.fc1: "fc1", trained_model.conv2: "conv2",
               trained_model.conv1: "conv1"}

    conv_max = N_IMAGES * CONV1_HW[0] * CONV1_HW[1]
    captured = collect_activations(
        trained_model, images_list, layers,
        max_samples=conv_max, seq_subsample=conv_max, device=device,
    )

    work = copy.deepcopy(trained_model)
    work_layer = {"fc3": work.fc3, "fc2": work.fc2, "fc1": work.fc1,
                  "conv2": work.conv2, "conv1": work.conv1}

    residuals = {}
    for layer in layers:
        nm = name_of[layer]
        a_in = captured[layer]["a_in"].double()
        a_pre = captured[layer]["a_pre"].double()       # the layer's OWN output
        W_new, b_new = solve_ols_layer(
            a_in, a_pre, with_bias=(layer.bias is not None), ols_lambda=OLS_LAMBDA)
        pred = a_in @ W_new.T
        if b_new is not None:
            pred = pred + b_new.unsqueeze(0)
        res = (pred - a_pre).norm() / a_pre.norm().clamp(min=1e-30)
        residuals[nm] = float(res.item())
        # write into the work model (reshape for conv)
        wl = work_layer[nm]
        with torch.no_grad():
            wl.weight.copy_(W_new.reshape(wl.weight.shape).to(wl.weight.device, wl.weight.dtype))
            if wl.bias is not None and b_new is not None:
                wl.bias.copy_(b_new.to(wl.bias.device, wl.bias.dtype))
        print(f"  {nm:<6s}: round-trip residual = {res.item():.3e}")

    acc_before = _eval(trained_model, test_loader, device)
    acc_after = _eval(work, test_loader, device)
    print(f"  trained acc           = {acc_before:.4f}")
    print(f"  all-layers-refit acc  = {acc_after:.4f}  (delta {acc_after - acc_before:+.4f})")

    result = {
        "battery": "OPT1-round-trip", "exp_id": "e3",
        "round_trip_residuals": residuals,
        "trained_acc": acc_before, "refit_acc": acc_after,
        "interpretation": (
            "Residuals ~0 (incl. conv) => conv OLS + im2col capture is sound; "
            "the E3 failure is the pool inversion, not the conv solve."
        ),
    }
    _save_json(RESULTS_DIR / "opt1_round_trip.json", result)
    return result


# ===========================================================================
# OPTION 2 — switch-aware max-unpool
# ===========================================================================

def _switch_unpool(pooled_target, indices, prepool_forward, out_hw):
    """Scatter pooled_target to argmax positions; keep prepool_forward elsewhere."""
    pooled_target = pooled_target.float()
    unpooled = F.max_unpool2d(pooled_target, indices, kernel_size=2, stride=2,
                              output_size=out_hw)                 # target @ argmax, 0 else
    mask = F.max_unpool2d(torch.ones_like(pooled_target), indices, kernel_size=2,
                          stride=2, output_size=out_hw)            # 1 @ argmax, 0 else
    return (unpooled + (1.0 - mask) * prepool_forward.float()).double()


def _switch_aware_tp_retrain(model, images_list, gt_targets, *, method, device,
                             correct_mean=True, correct_cov=True,
                             target_layers=None):
    """Conv TP retrain using switch-aware unpool. Single pass. Returns residual dict."""
    if target_layers is None:
        target_layers = {"conv1", "conv2", "fc1", "fc2", "fc3"}

    # ---- forward pass: collect pool state + per-layer activations ----
    c1_list, idx1_list, c2_list, idx2_list = [], [], [], []
    p1_list, p2_list = [], []
    fc1_in_list, fc1_pre_list = [], []
    fc2_in_list, fc2_pre_list = [], []
    fc3_in_list, fc3_pre_list = [], []
    conv1_in_list, conv1_pre_list = [], []
    conv2_in_list, conv2_pre_list = [], []

    model.eval()
    with torch.no_grad():
        for batch in images_list:
            x = batch.to(device)
            st = model.forward_with_pool_state(x)
            c1, idx1 = st["conv1_prepool"], st["pool1_indices"]
            c2, idx2 = st["conv2_prepool"], st["pool2_indices"]
            p1 = F.max_pool2d(c1, 2, 2)
            p2 = F.max_pool2d(c2, 2, 2)
            c1_list.append(c1.cpu()); idx1_list.append(idx1.cpu())
            c2_list.append(c2.cpu()); idx2_list.append(idx2.cpu())
            p1_list.append(p1.cpu()); p2_list.append(p2.cpu())
            # conv im2col
            conv1_in_list.append(_im2col(x, model.conv1).cpu())
            conv1_pre_list.append(model.conv1(x).permute(0, 2, 3, 1).reshape(-1, 6).cpu())
            conv2_in_list.append(_im2col(p1, model.conv2).cpu())
            conv2_pre_list.append(model.conv2(p1).permute(0, 2, 3, 1).reshape(-1, 16).cpu())
            # fc
            flat = model.flatten(p2)
            h1_pre = model.fc1(flat)
            h1 = model.act3(h1_pre)
            h2_pre = model.fc2(h1)
            h2 = model.act4(h2_pre)
            fc1_in_list.append(flat.cpu()); fc1_pre_list.append(h1_pre.cpu())
            fc2_in_list.append(h1.cpu()); fc2_pre_list.append(h2_pre.cpu())
            fc3_in_list.append(h2.cpu()); fc3_pre_list.append(model.fc3(h2).cpu())

    cat = lambda lst: torch.cat(lst, dim=0)
    c1 = cat(c1_list).double(); idx1 = cat(idx1_list)
    c2 = cat(c2_list).double(); idx2 = cat(idx2_list)
    n_use = min(fc3_in_list and cat(fc3_in_list).shape[0], gt_targets.shape[0])

    fc_in = {"fc1": cat(fc1_in_list).double(), "fc2": cat(fc2_in_list).double(),
             "fc3": cat(fc3_in_list).double()}
    fc_pre = {"fc1": cat(fc1_pre_list).double(), "fc2": cat(fc2_pre_list).double(),
              "fc3": cat(fc3_pre_list).double()}
    conv1_in = cat(conv1_in_list).double(); conv1_pre = cat(conv1_pre_list).double()
    conv2_in = cat(conv2_in_list).double(); conv2_pre = cat(conv2_pre_list).double()

    n_img = c1.shape[0]
    n_use_img = min(n_img, gt_targets.shape[0])

    # ---- FC chain back-prop (fc3 -> fc2 -> fc1) ----
    fc3, fc2, fc1 = model.fc3, model.fc2, model.fc1
    relu = nn.ReLU()
    cur_target = gt_targets[:n_use_img].double()
    fc_targets = {}
    for layer, nm in [(fc3, "fc3"), (fc2, "fc2"), (fc1, "fc1")]:
        a_in_t = fc_in[nm][:n_use_img]
        a_pre_t = fc_pre[nm][:n_use_img]
        activation = None if nm == "fc3" else relu
        mu_a, Sigma_a = _empirical_mean_cov(a_in_t)
        fc_targets[nm] = cur_target.clone()
        W = layer.weight.detach().double().cpu()
        b = layer.bias.detach().double().cpu() if layer.bias is not None else None
        if method == "naive":
            a_hat = invert_layer(W, b, cur_target, a_pre_t, activation, "naive", eps=OLS_EPS)
        else:
            a_hat = invert_layer(W, b, cur_target, a_pre_t, activation, "kfac_a",
                                 mu_a=mu_a, Sigma_a=Sigma_a)
        next_t = a_hat
        if correct_mean and nm != "fc1":
            tgt_mean = a_in_t.mean(dim=0)
            if correct_cov:
                _, tgt_S = _empirical_mean_cov(a_in_t)
                next_t = _moment_match(a_hat, tgt_mean, tgt_S)
            else:
                next_t = _shift_mean_to(a_hat, tgt_mean)
        cur_target = next_t
    # cur_target: (n_use_img, 400) = pool2-output target

    # ---- transition FC->conv2 : switch-aware unpool of pool2 ----
    pool2_target = cur_target.reshape(n_use_img, 16, *POOL2_HW)
    conv2_apost = _switch_unpool(pool2_target, idx2[:n_use_img],
                                 c2[:n_use_img], CONV2_HW)        # (n,16,10,10)
    conv2_apost_2d = conv2_apost.permute(0, 2, 3, 1).reshape(-1, 16)

    # ---- back-prop conv2 -> pool1 target ----
    n_c2 = n_use_img * CONV2_HW[0] * CONV2_HW[1]
    a_pre_c2 = conv2_pre[:n_c2]
    a_in_c2 = conv2_in[:n_c2]
    mu_c2, S_c2 = _empirical_mean_cov(a_in_c2)
    W_c2 = model.conv2.weight.detach().reshape(16, -1).double().cpu()
    b_c2 = model.conv2.bias.detach().double().cpu() if model.conv2.bias is not None else None
    if method == "naive":
        a_hat_c2 = invert_layer(W_c2, b_c2, conv2_apost_2d, a_pre_c2, relu, "naive", eps=OLS_EPS)
    else:
        a_hat_c2 = invert_layer(W_c2, b_c2, conv2_apost_2d, a_pre_c2, relu, "kfac_a",
                                mu_a=mu_c2, Sigma_a=S_c2)
    fold_in = a_hat_c2.reshape(n_use_img, CONV2_HW[0] * CONV2_HW[1], 150).permute(0, 2, 1).float()
    folded = F.fold(fold_in, output_size=POOL1_HW, kernel_size=model.conv2.kernel_size,
                    dilation=model.conv2.dilation, padding=model.conv2.padding,
                    stride=model.conv2.stride)
    cnt = F.fold(torch.ones_like(fold_in), output_size=POOL1_HW,
                 kernel_size=model.conv2.kernel_size, dilation=model.conv2.dilation,
                 padding=model.conv2.padding, stride=model.conv2.stride)
    pool1_target = (folded / cnt.clamp(min=1)).double()           # (n,6,14,14)

    # ---- transition conv2->conv1 : switch-aware unpool of pool1 ----
    conv1_apost = _switch_unpool(pool1_target, idx1[:n_use_img],
                                 c1[:n_use_img], CONV1_HW)         # (n,6,28,28)
    conv1_apost_2d = conv1_apost.permute(0, 2, 3, 1).reshape(-1, 6)

    n_c1 = n_use_img * CONV1_HW[0] * CONV1_HW[1]
    a_pre_c1 = conv1_pre[:n_c1]
    a_in_c1 = conv1_in[:n_c1]

    # ---- forward OLS sweep WITH REBUILT-UPSTREAM propagation ----
    # Each layer's OLS X is recomputed from the just-rebuilt upstream layers
    # (not the stale captured a_in). Targets stay fixed; the ReLU mask for
    # invert_activation still uses the captured a_pre (matches the MLP fix).
    info = {}
    x_all = torch.cat(list(images_list), dim=0)[:n_use_img].to(device)

    # --- conv1 (X = im2col of model input; input never changes) ---
    if "conv1" in target_layers:
        X = _im2col(x_all, model.conv1).double().cpu()
        t_pre = invert_activation(conv1_apost_2d, a_pre_c1, relu)
        W_new, b_new = solve_ols_layer(X, t_pre, with_bias=True, ols_lambda=OLS_LAMBDA)
        pred = X @ W_new.T + (b_new.unsqueeze(0) if b_new is not None else 0)
        info["conv1"] = {"target_residual": float((pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30))}
        with torch.no_grad():
            model.conv1.weight.copy_(W_new.reshape(model.conv1.weight.shape).to(model.conv1.weight.device, model.conv1.weight.dtype))
            if model.conv1.bias is not None:
                model.conv1.bias.copy_(b_new.to(model.conv1.bias.device, model.conv1.bias.dtype))

    # Re-propagate to get rebuilt pool1 output
    with torch.no_grad():
        new_p1 = F.max_pool2d(relu(model.conv1(x_all)), 2, 2)

    # --- conv2 (X = im2col of REBUILT pool1) ---
    if "conv2" in target_layers:
        X = _im2col(new_p1, model.conv2).double().cpu()
        t_pre = invert_activation(conv2_apost_2d, a_pre_c2, relu)
        W_new, b_new = solve_ols_layer(X, t_pre, with_bias=True, ols_lambda=OLS_LAMBDA)
        pred = X @ W_new.T + (b_new.unsqueeze(0) if b_new is not None else 0)
        info["conv2"] = {"target_residual": float((pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30))}
        with torch.no_grad():
            model.conv2.weight.copy_(W_new.reshape(model.conv2.weight.shape).to(model.conv2.weight.device, model.conv2.weight.dtype))
            if model.conv2.bias is not None:
                model.conv2.bias.copy_(b_new.to(model.conv2.bias.device, model.conv2.bias.dtype))

    # Re-propagate to get rebuilt flatten(pool2) — this is fc1's input
    with torch.no_grad():
        new_p2 = F.max_pool2d(relu(model.conv2(new_p1)), 2, 2)
        cur_X = model.flatten(new_p2).double().cpu()   # (n_use_img, 400)

    # --- fc1, fc2, fc3 (X = rebuilt upstream output, propagated each step) ---
    for layer, nm in [(fc1, "fc1"), (fc2, "fc2"), (fc3, "fc3")]:
        a_pre_fwd = fc_pre[nm][:n_use_img]
        act_after = None if nm == "fc3" else relu
        if nm in target_layers:
            t_pre = invert_activation(fc_targets[nm], a_pre_fwd, act_after)
            W_new, b_new = solve_ols_layer(cur_X, t_pre, with_bias=True, ols_lambda=OLS_LAMBDA)
            pred = cur_X @ W_new.T + (b_new.unsqueeze(0) if b_new is not None else 0)
            info[nm] = {"target_residual": float((pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30))}
            with torch.no_grad():
                layer.weight.copy_(W_new.to(layer.weight.device, layer.weight.dtype))
                if layer.bias is not None:
                    layer.bias.copy_(b_new.to(layer.bias.device, layer.bias.dtype))
        # propagate cur_X through this (possibly just-rebuilt) layer for the next
        with torch.no_grad():
            pre = layer(cur_X.float().to(device))
            cur_X = (relu(pre) if act_after is not None else pre).double().cpu()

    return info


def option2_switch_aware(train_set, test_loader, *, device) -> dict:
    print("\n[E3-diag] === OPTION 2: switch-aware max-unpool, random init ===")
    torch.manual_seed(SEED)
    model = LeNet().to(device)
    rand_acc = _eval(model, test_loader, device)
    print(f"  random-init acc = {rand_acc:.4f}")

    images_list, labels = _build_loader_list(train_set, N_IMAGES)
    gt = make_gt_logit_target(labels[:N_IMAGES], n_classes=10, margin=GT_MARGIN)

    t0 = time.time()
    info = _switch_aware_tp_retrain(model, images_list, gt, method="kfac_a", device=device)
    acc = _eval(model, test_loader, device)
    wall = time.time() - t0
    print(f"  switch-aware retrain acc = {acc:.4f}  delta={acc - rand_acc:+.4f}  ({wall:.1f}s)")
    for nm in _LAYER_NAMES_DF:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm}: residual={r:.3g}" if r is not None else f"    {nm}: --")
    result = {
        "battery": "OPT2-switch-unpool", "exp_id": "e3", "seed": SEED,
        "random_init_acc": rand_acc, "retrained_acc": acc, "delta": acc - rand_acc,
        "wall_s": wall,
        "per_layer_residuals": {n: info.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
    }
    _save_json(RESULTS_DIR / "opt2_switch_unpool.json", result)
    return result


# ===========================================================================
# OPTION 3 — avg-pool variant
# ===========================================================================

def option3_avgpool(train_set, test_loader, *, device, skip_train=False) -> dict:
    print("\n[E3-diag] === OPTION 3: avg-pool LeNet variant ===")
    from diagnostic.runners.arch_coverage.e3_lenet import _lenet_tp_retrain

    if skip_train and AVGPOOL_WEIGHTS.exists():
        model = LeNetAvgPool().to(device)
        model.load_state_dict(torch.load(AVGPOOL_WEIGHTS, map_location=device, weights_only=True))
        trained_acc = _eval(model, test_loader, device)
        print(f"  loaded avgpool weights; acc = {trained_acc:.4f}")
    else:
        print("  training LeNetAvgPool ...")
        model = LeNetAvgPool().to(device)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
        trained_acc = _train(model, train_loader, test_loader, epochs=TRAIN_EPOCHS, device=device)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), AVGPOOL_WEIGHTS)
        print(f"  trained avgpool acc = {trained_acc:.4f}  -> {AVGPOOL_WEIGHTS.name}")

    images_list, labels = _build_loader_list(train_set, N_IMAGES)

    # B-retrain-rand on avg-pool, reusing the existing nearest-neighbor inversion
    # (which IS the correct adjoint for avg-pool).
    torch.manual_seed(SEED)
    rand_model = LeNetAvgPool().to(device)
    rand_acc = _eval(rand_model, test_loader, device)
    gt = make_gt_logit_target(labels[:N_IMAGES], n_classes=10, margin=GT_MARGIN)
    t0 = time.time()
    info = _lenet_tp_retrain(
        rand_model, images_list, gt, N_IMAGES,
        method="kfac_a", correct_target_mean=True, correct_target_cov=True, device=device,
    )
    retr_acc = _eval(rand_model, test_loader, device)
    wall = time.time() - t0
    print(f"  random-init acc       = {rand_acc:.4f}")
    print(f"  retrain-rand acc      = {retr_acc:.4f}  delta={retr_acc - rand_acc:+.4f}  ({wall:.1f}s)")
    for nm in _LAYER_NAMES_DF:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm}: residual={r:.3g}" if r is not None else f"    {nm}: --")
    result = {
        "battery": "OPT3-avgpool", "exp_id": "e3", "seed": SEED,
        "trained_acc": trained_acc, "random_init_acc": rand_acc,
        "retrained_acc": retr_acc, "delta": retr_acc - rand_acc, "wall_s": wall,
        "per_layer_residuals": {n: info.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
    }
    _save_json(RESULTS_DIR / "opt3_avgpool.json", result)
    return result


# ===========================================================================
# OPTION 4 — ceiling test: switch-aware + rebuilt-upstream from TRAINED start
# ===========================================================================

def option4_ceiling(trained_model, images_list, labels, test_loader, *, device) -> dict:
    """From the TRAINED LeNet, run the fixed (switch-aware + rebuilt-upstream)
    conv TP retrain with two deepest-target choices:
      (a) distill  — target = trained model's own logits (should ~preserve acc)
      (b) gt       — target = GT one-hot (retraining-style, like MLP's 0.95)
    If (a) preserves ~0.99, the conv machinery is fully sound and the random-init
    0.52 is just from-scratch difficulty. If (a) caps lower, the conv transitions
    (fold-averaging) are still lossy."""
    print("\n[E3-diag] === OPTION 4: ceiling test from TRAINED start ===")
    trained_acc = _eval(trained_model, test_loader, device)
    print(f"  trained baseline acc = {trained_acc:.4f}")

    # (a) distillation target = trained model's own logits
    trained_model.eval()
    logit_chunks = []
    with torch.no_grad():
        for batch in images_list:
            x = batch.to(device)
            logit_chunks.append(trained_model(x).cpu())
    distill_targets = torch.cat(logit_chunks, dim=0)[:N_IMAGES].double()

    model_a = copy.deepcopy(trained_model)
    info_a = _switch_aware_tp_retrain(model_a, images_list, distill_targets,
                                      method="kfac_a", device=device)
    acc_a = _eval(model_a, test_loader, device)
    print(f"  (a) distill (own logits): acc={acc_a:.4f}  delta={acc_a - trained_acc:+.4f}")
    for nm in _LAYER_NAMES_DF:
        r = info_a.get(nm, {}).get("target_residual")
        print(f"      {nm}: residual={r:.3g}" if r is not None else f"      {nm}: --")

    # (b) GT one-hot target from trained start
    gt = make_gt_logit_target(labels[:N_IMAGES], n_classes=10, margin=GT_MARGIN)
    model_b = copy.deepcopy(trained_model)
    info_b = _switch_aware_tp_retrain(model_b, images_list, gt,
                                      method="kfac_a", device=device)
    acc_b = _eval(model_b, test_loader, device)
    print(f"  (b) GT one-hot:           acc={acc_b:.4f}  delta={acc_b - trained_acc:+.4f}")
    for nm in _LAYER_NAMES_DF:
        r = info_b.get(nm, {}).get("target_residual")
        print(f"      {nm}: residual={r:.3g}" if r is not None else f"      {nm}: --")

    result = {
        "battery": "OPT4-ceiling-trained-start", "exp_id": "e3",
        "trained_acc": trained_acc,
        "distill_acc": acc_a, "distill_delta": acc_a - trained_acc,
        "gt_acc": acc_b, "gt_delta": acc_b - trained_acc,
        "distill_residuals": {n: info_a.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
        "gt_residuals": {n: info_b.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
    }
    _save_json(RESULTS_DIR / "opt4_ceiling.json", result)
    return result


# ===========================================================================
# OPTION 5 — feature-level distillation from RANDOM init (FitNets-style)
# ===========================================================================

def option5_feature_distill(trained_model, images_list, test_loader, *, device) -> dict:
    """Random-init student; each layer OLS-fit to reproduce the TEACHER's
    captured pre-activation at that layer (achievable targets, NO chain
    back-prop, NO pooling inversion), with rebuilt-upstream propagation.

    If the student recovers ~teacher acc, the conv machinery is fully capable
    and chain-back-prop (Option 4) is the sole retraining bottleneck."""
    print("\n[E3-diag] === OPTION 5: feature-level distillation from random init ===")
    teacher = trained_model.eval()
    teacher_acc = _eval(teacher, test_loader, device)
    print(f"  teacher acc = {teacher_acc:.4f}")

    relu = nn.ReLU()
    # Capture teacher pre-activations per layer (spatially aligned, no subsample)
    tgt = {"conv1": [], "conv2": [], "fc1": [], "fc2": [], "fc3": []}
    with torch.no_grad():
        for batch in images_list:
            x = batch.to(device)
            c1 = teacher.conv1(x)
            p1 = teacher.pool1(teacher.act1(c1))
            c2 = teacher.conv2(p1)
            p2 = teacher.pool2(teacher.act2(c2))
            flat = teacher.flatten(p2)
            h1 = teacher.fc1(flat)
            h2 = teacher.fc2(teacher.act3(h1))
            lg = teacher.fc3(teacher.act4(h2))
            tgt["conv1"].append(c1.permute(0, 2, 3, 1).reshape(-1, 6).cpu())
            tgt["conv2"].append(c2.permute(0, 2, 3, 1).reshape(-1, 16).cpu())
            tgt["fc1"].append(h1.cpu())
            tgt["fc2"].append(h2.cpu())
            tgt["fc3"].append(lg.cpu())
    tgt = {k: torch.cat(v, dim=0).double() for k, v in tgt.items()}

    torch.manual_seed(SEED)
    student = LeNet().to(device)
    rand_acc = _eval(student, test_loader, device)
    print(f"  random-init student acc = {rand_acc:.4f}")

    x_all = torch.cat(list(images_list), dim=0).to(device)
    info = {}

    def _fit_conv(conv, X_spatial, target):
        X = _im2col(X_spatial, conv).double().cpu()
        n = min(X.shape[0], target.shape[0])
        W, b = solve_ols_layer(X[:n], target[:n], with_bias=True, ols_lambda=OLS_LAMBDA)
        pred = X[:n] @ W.T + (b.unsqueeze(0) if b is not None else 0)
        res = float((pred - target[:n]).norm() / target[:n].norm().clamp(min=1e-30))
        with torch.no_grad():
            conv.weight.copy_(W.reshape(conv.weight.shape).to(conv.weight.device, conv.weight.dtype))
            if conv.bias is not None:
                conv.bias.copy_(b.to(conv.bias.device, conv.bias.dtype))
        return res

    def _fit_fc(fc, X, target):
        X = X.double().cpu()
        n = min(X.shape[0], target.shape[0])
        W, b = solve_ols_layer(X[:n], target[:n], with_bias=True, ols_lambda=OLS_LAMBDA)
        pred = X[:n] @ W.T + (b.unsqueeze(0) if b is not None else 0)
        res = float((pred - target[:n]).norm() / target[:n].norm().clamp(min=1e-30))
        with torch.no_grad():
            fc.weight.copy_(W.to(fc.weight.device, fc.weight.dtype))
            if fc.bias is not None:
                fc.bias.copy_(b.to(fc.bias.device, fc.bias.dtype))
        return res

    # conv1 (input is shared with teacher -> exact recovery expected)
    info["conv1"] = {"target_residual": _fit_conv(student.conv1, x_all, tgt["conv1"])}
    with torch.no_grad():
        sp1 = student.pool1(relu(student.conv1(x_all)))
    # conv2 (X = student's rebuilt pool1)
    info["conv2"] = {"target_residual": _fit_conv(student.conv2, sp1, tgt["conv2"])}
    with torch.no_grad():
        sp2 = student.pool2(relu(student.conv2(sp1)))
        flat = student.flatten(sp2)
    # fc1, fc2, fc3
    info["fc1"] = {"target_residual": _fit_fc(student.fc1, flat, tgt["fc1"])}
    with torch.no_grad():
        r1 = relu(student.fc1(flat))
    info["fc2"] = {"target_residual": _fit_fc(student.fc2, r1, tgt["fc2"])}
    with torch.no_grad():
        r2 = relu(student.fc2(r1))
    info["fc3"] = {"target_residual": _fit_fc(student.fc3, r2, tgt["fc3"])}

    acc = _eval(student, test_loader, device)
    print(f"  feature-distill student acc = {acc:.4f}  (delta vs teacher {acc - teacher_acc:+.4f})")
    for nm in _LAYER_NAMES_DF:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm}: residual={r:.3g}" if r is not None else f"    {nm}: --")

    result = {
        "battery": "OPT5-feature-distill-random", "exp_id": "e3",
        "teacher_acc": teacher_acc, "random_init_acc": rand_acc,
        "student_acc": acc, "delta_vs_teacher": acc - teacher_acc,
        "per_layer_residuals": {n: info.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
    }
    _save_json(RESULTS_DIR / "opt5_feature_distill.json", result)
    return result


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--option", choices=("1", "2", "3", "4", "5", "all"), default="all")
    ap.add_argument("--skip-train-avgpool", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[E3-diag] device={device}  option={args.option}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    images_list, labels = _build_loader_list(train_set, N_IMAGES)

    def _load_trained():
        if not LENET_WEIGHTS.exists():
            print(f"[E3-diag] no LeNet weights at {LENET_WEIGHTS}; run e3_lenet first.")
            return None
        m = LeNet().to(device)
        m.load_state_dict(torch.load(LENET_WEIGHTS, map_location=device, weights_only=True))
        return m

    if args.option in ("1", "all"):
        trained = _load_trained()
        if trained is None:
            return 1
        option1_round_trip(trained, images_list, test_loader, device=device)

    if args.option in ("2", "all"):
        option2_switch_aware(train_set, test_loader, device=device)

    if args.option in ("3", "all"):
        option3_avgpool(train_set, test_loader, device=device,
                        skip_train=args.skip_train_avgpool)

    if args.option in ("4", "all"):
        trained = _load_trained()
        if trained is None:
            return 1
        option4_ceiling(trained, images_list, labels, test_loader, device=device)

    if args.option in ("5", "all"):
        trained = _load_trained()
        if trained is None:
            return 1
        option5_feature_distill(trained, images_list, test_loader, device=device)

    print("\n[E3-diag] === COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
