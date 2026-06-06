"""
diagnostic/runners/arch_coverage/e5_resnet.py
==============================================

E5 — small BN-free ResNet on CIFAR-10. Probes the *additive identity skip*
(Q1: decompose y = x + f(x)) and RGB conv input.

Battery (run via --option):

  train     B-train.  Train MiniResNet by backprop; cache weights.

  distill   B-distill (feature-level, FitNets-style) from RANDOM init.
            Each layer is OLS-fit to reproduce the TEACHER's captured
            pre-activation, with rebuilt-upstream forward sweep (the skip and
            pools are replayed in the student's forward). NO chain back-prop,
            NO pooling/skip inversion. Tests whether additive skips break
            feature distillation (they should not — each conv is fit
            independently to an achievable target).

  retrain   B-retrain-rand. Chain back-prop of GT one-hot logits from RANDOM
            init, with the additive-skip target decomposition:
              out = ReLU(x + f(x));  t_sum = ReLU^{-1}(t_out);  t_f = t_sum - x
            f's branch is then inverted conv-by-conv (im2col + fold), the
            block input target recurses through f's first conv, and MaxPool is
            inverted with switch-aware max-unpool (reused from E3). GAP is
            inverted by spatial broadcast (its min-norm adjoint). The forward
            OLS sweep is rebuilt-upstream (each layer fit against the output of
            the just-rebuilt previous layer). Tests whether the skip changes
            the chain-retrain ceiling vs the plain CNN (E3 ~0.52).

Usage:
    python -m diagnostic.runners.arch_coverage.e5_resnet --option train
    python -m diagnostic.runners.arch_coverage.e5_resnet --option distill
    python -m diagnostic.runners.arch_coverage.e5_resnet --option retrain
    python -m diagnostic.runners.arch_coverage.e5_resnet --option all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.models.miniresnet import MiniResNet
from diagnostic.multi_step import _empirical_mean_cov
from diagnostic.runners.phase1_gt_retrain import make_gt_logit_target
from diagnostic.target_prop_retrainer import solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e5"
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"
RESNET_WEIGHTS = WEIGHTS_DIR / "miniresnet_cifar.pt"

WIDTH = 32
N_IMAGES = 256          # images used for OLS capture (conv -> many rows each)
SEED = 42
GT_MARGIN = 5.0
OLS_LAMBDA = 1e-4
OLS_EPS = 1e-4
TRAIN_EPOCHS = 30

# spatial sizes
P0_HW = (16, 16)        # block1 operating size (pool0 output)
P1_HW = (8, 8)          # block2 operating size (pool1 output)
C0_HW = (32, 32)        # conv0 output size (pre pool0)

_LAYER_NAMES_DF = ["fc", "b2_conv2", "b2_conv1", "b1_conv2", "b1_conv1", "conv0"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _load_cifar10(data_root: Path):
    import torchvision
    import torchvision.transforms as T
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10(str(data_root), train=True,
                                         download=False, transform=train_tf)
    # separate eval-transform copy for capture (no augmentation)
    train_eval = torchvision.datasets.CIFAR10(str(data_root), train=True,
                                              download=False, transform=test_tf)
    test = torchvision.datasets.CIFAR10(str(data_root), train=False,
                                        download=False, transform=test_tf)
    return train, train_eval, test


@torch.no_grad()
def _eval(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


def _train(model, train_loader, test_loader, *, epochs, device, lr=1e-3):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
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


def _build_image_list(dataset, n_images: int, batch_size: int = 128):
    """Return (list-of-image-batches, labels) using the eval transform dataset."""
    subset = Subset(dataset, list(range(min(len(dataset), n_images))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, labels = [], []
    for x, y in loader:
        images.append(x)
        labels.append(y)
    return images, torch.cat(labels, dim=0)[:n_images]


def _im2col(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """(B, C_in, H, W) -> (B*H_out*W_out, C_in*kH*kW)."""
    unf = F.unfold(x, kernel_size=conv.kernel_size, dilation=conv.dilation,
                   padding=conv.padding, stride=conv.stride)
    return unf.permute(0, 2, 1).reshape(-1, unf.shape[1])


def _fold_to_spatial(a_hat_im2col, conv, out_hw, n_img):
    """Map an im2col-space target (n_img*H*W, C_in*k*k) back to a spatial
    post-activation target (n_img, C_in, H, W) by overlap-averaged folding."""
    C_in = conv.in_channels
    kH, kW = (conv.kernel_size if isinstance(conv.kernel_size, tuple)
              else (conv.kernel_size, conv.kernel_size))
    L = out_hw[0] * out_hw[1]
    fold_in = a_hat_im2col.reshape(n_img, L, C_in * kH * kW).permute(0, 2, 1).float()
    folded = F.fold(fold_in, output_size=out_hw, kernel_size=conv.kernel_size,
                    dilation=conv.dilation, padding=conv.padding, stride=conv.stride)
    cnt = F.fold(torch.ones_like(fold_in), output_size=out_hw,
                 kernel_size=conv.kernel_size, dilation=conv.dilation,
                 padding=conv.padding, stride=conv.stride)
    return (folded / cnt.clamp(min=1)).double()


def _switch_unpool(pooled_target, indices, prepool_forward, out_hw):
    """Scatter pooled_target to argmax positions; keep prepool_forward elsewhere."""
    pooled_target = pooled_target.float()
    unpooled = F.max_unpool2d(pooled_target, indices, kernel_size=2, stride=2,
                              output_size=out_hw)
    mask = F.max_unpool2d(torch.ones_like(pooled_target), indices, kernel_size=2,
                          stride=2, output_size=out_hw)
    return (unpooled + (1.0 - mask) * prepool_forward.float()).double()


def _conv_pre_2d(t):
    """(B,C,H,W) -> (B*H*W, C) channel-last flatten (matches im2col a_pre)."""
    return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])


# ===========================================================================
# capture teacher state (shared by distill + retrain)
# ===========================================================================

def _capture_teacher_state(model, images_list, device):
    """Run forward_with_state over all images; concat tensors on CPU."""
    keys = ["conv0_prepool", "pool0_indices", "b1_in", "b1_conv1_pre",
            "b1_conv2_pre", "b1_sum_pre", "b1_out", "pool1_indices", "b2_in",
            "b2_conv1_pre", "b2_conv2_pre", "b2_sum_pre", "b2_out", "gap_out",
            "logits"]
    acc = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for batch in images_list:
            st = model.forward_with_state(batch.to(device))
            for k in keys:
                acc[k].append(st[k].cpu())
    return {k: torch.cat(v, dim=0) for k, v in acc.items()}


# ===========================================================================
# B-train
# ===========================================================================

def battery_train(train_set, train_eval_set, test_loader, *, device, epochs):
    print("\n[E5] === B-train: MiniResNet on CIFAR-10 ===")
    torch.manual_seed(SEED)
    model = MiniResNet(width=WIDTH).to(device)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    acc = _train(model, train_loader, test_loader, epochs=epochs, device=device)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESNET_WEIGHTS)
    print(f"  trained acc = {acc:.4f}  -> {RESNET_WEIGHTS.name}")
    _save_json(RESULTS_DIR / "b_train.json",
               {"battery": "B-train", "exp_id": "e5", "seed": SEED,
                "epochs": epochs, "trained_acc": acc, "width": WIDTH})
    return model


# ===========================================================================
# B-distill — feature-level distillation from random init
# ===========================================================================

def battery_distill(teacher, images_list, test_loader, *, device):
    print("\n[E5] === B-distill: feature-level distillation (random init) ===")
    teacher_acc = _eval(teacher, test_loader, device)
    print(f"  teacher acc = {teacher_acc:.4f}")

    # Feature distillation fits each conv to its RAW pre-activation (pre-ReLU).
    # Recompute raw conv pre-activations directly from the teacher.
    relu = nn.ReLU()
    raw = {"conv0": [], "b1_conv1": [], "b1_conv2": [],
           "b2_conv1": [], "b2_conv2": [], "logits": []}
    teacher.eval()
    with torch.no_grad():
        for batch in images_list:
            x = batch.to(device)
            c0 = teacher.conv0(x)
            p0 = teacher.pool0(relu(c0))
            c1a = teacher.b1_conv1(p0)
            c1b = teacher.b1_conv2(relu(c1a))
            b1out = relu(p0 + c1b)
            p1 = teacher.pool1(b1out)
            c2a = teacher.b2_conv1(p1)
            c2b = teacher.b2_conv2(relu(c2a))
            b2out = relu(p1 + c2b)
            gap = teacher.flatten(teacher.gap(b2out))
            lg = teacher.fc(gap)
            raw["conv0"].append(_conv_pre_2d(c0).cpu())
            raw["b1_conv1"].append(_conv_pre_2d(c1a).cpu())
            raw["b1_conv2"].append(_conv_pre_2d(c1b).cpu())
            raw["b2_conv1"].append(_conv_pre_2d(c2a).cpu())
            raw["b2_conv2"].append(_conv_pre_2d(c2b).cpu())
            raw["logits"].append(lg.cpu())
    tgt = {k: torch.cat(v, dim=0).double() for k, v in raw.items()}

    torch.manual_seed(SEED)
    student = MiniResNet(width=WIDTH).to(device)
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

    # forward sweep with rebuilt-upstream propagation
    info["conv0"] = {"target_residual": _fit_conv(student.conv0, x_all, tgt["conv0"])}
    with torch.no_grad():
        sp0 = student.pool0(relu(student.conv0(x_all)))
    info["b1_conv1"] = {"target_residual": _fit_conv(student.b1_conv1, sp0, tgt["b1_conv1"])}
    with torch.no_grad():
        mid1 = relu(student.b1_conv1(sp0))
    info["b1_conv2"] = {"target_residual": _fit_conv(student.b1_conv2, mid1, tgt["b1_conv2"])}
    with torch.no_grad():
        f1 = student.b1_conv2(mid1)
        b1out = relu(sp0 + f1)
        sp1 = student.pool1(b1out)
    info["b2_conv1"] = {"target_residual": _fit_conv(student.b2_conv1, sp1, tgt["b2_conv1"])}
    with torch.no_grad():
        mid2 = relu(student.b2_conv1(sp1))
    info["b2_conv2"] = {"target_residual": _fit_conv(student.b2_conv2, mid2, tgt["b2_conv2"])}
    with torch.no_grad():
        f2 = student.b2_conv2(mid2)
        b2out = relu(sp1 + f2)
        gap = student.flatten(student.gap(b2out))
    info["fc"] = {"target_residual": _fit_fc(student.fc, gap, tgt["logits"])}

    acc = _eval(student, test_loader, device)
    print(f"  feature-distill student acc = {acc:.4f}  (delta vs teacher {acc - teacher_acc:+.4f})")
    for nm in _LAYER_NAMES_DF:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm:<9s}: residual={r:.3g}" if r is not None else f"    {nm}: --")

    result = {
        "battery": "B-distill-feature", "exp_id": "e5",
        "teacher_acc": teacher_acc, "random_init_acc": rand_acc,
        "student_acc": acc, "delta_vs_teacher": acc - teacher_acc,
        "per_layer_residuals": {n: info.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
    }
    _save_json(RESULTS_DIR / "b_distill.json", result)
    return result


# ===========================================================================
# B-retrain-rand — chain GT back-prop with additive-skip decomposition
# ===========================================================================

def battery_retrain(images_list, labels, test_loader, *, device, method="kfac_a"):
    print("\n[E5] === B-retrain-rand: chain GT back-prop + skip decomposition ===")
    torch.manual_seed(SEED)
    model = MiniResNet(width=WIDTH).to(device)
    rand_acc = _eval(model, test_loader, device)
    print(f"  random-init acc = {rand_acc:.4f}")

    n_img = min(N_IMAGES, labels.shape[0])
    gt = make_gt_logit_target(labels[:n_img], n_classes=10, margin=GT_MARGIN).double()

    st = _capture_teacher_state(model, images_list, device)  # forward state of RANDOM model
    relu = nn.ReLU()
    info = {}
    t0 = time.time()

    # ---------- BACKWARD PASS: build per-layer post-activation targets ----------
    # fc: invert (no activation) -> target for gap_out
    W_fc = model.fc.weight.detach().double().cpu()
    b_fc = model.fc.bias.detach().double().cpu()
    gap_fwd = st["gap_out"][:n_img].double()
    mu_g, S_g = _empirical_mean_cov(gap_fwd)
    if method == "naive":
        t_gap = invert_layer(W_fc, b_fc, gt, gap_fwd, None, "naive", eps=OLS_EPS)
    else:
        t_gap = invert_layer(W_fc, b_fc, gt, gap_fwd, None, "kfac_a", mu_a=mu_g, Sigma_a=S_g)

    # GAP inverse: broadcast t_gap (n,W) -> (n,W,8,8)
    t_b2out = t_gap.reshape(n_img, WIDTH, 1, 1).expand(n_img, WIDTH, *P1_HW).clone()

    # block2: out = ReLU(b2_in + f2)
    b2_sum_pre = st["b2_sum_pre"][:n_img].double()
    b2_in = st["b2_in"][:n_img].double()
    t_sum2 = invert_activation(_conv_pre_2d(t_b2out), _conv_pre_2d(b2_sum_pre), relu)
    t_sum2 = t_sum2.reshape(n_img, *P1_HW, WIDTH).permute(0, 3, 1, 2)   # back to (n,W,8,8)
    t_f2 = t_sum2 - b2_in                                                # target for f2 (raw)

    # b2_conv2: invert (no act) -> target post-act of b2_conv1
    a_in_c2b = _im2col(relu(st["b2_conv1_pre"][:n_img].double()).float().to(device), model.b2_conv2).double().cpu()
    a_pre_c2b = _conv_pre_2d(st["b2_conv2_pre"][:n_img].double())
    mu, S = _empirical_mean_cov(a_in_c2b)
    W = model.b2_conv2.weight.detach().reshape(WIDTH, -1).double().cpu()
    b = model.b2_conv2.bias.detach().double().cpu()
    tgt_c2b = _conv_pre_2d(t_f2)
    if method == "naive":
        ah = invert_layer(W, b, tgt_c2b, a_pre_c2b, None, "naive", eps=OLS_EPS)
    else:
        ah = invert_layer(W, b, tgt_c2b, a_pre_c2b, None, "kfac_a", mu_a=mu, Sigma_a=S)
    t_b2c1_post = _fold_to_spatial(ah, model.b2_conv2, P1_HW, n_img)     # (n,W,8,8)

    # b2_conv1: invert (relu after) -> target for b2_in (block2 input)
    a_in_c2a = _im2col(b2_in.float().to(device), model.b2_conv1).double().cpu()
    a_pre_c2a = _conv_pre_2d(st["b2_conv1_pre"][:n_img].double())
    mu, S = _empirical_mean_cov(a_in_c2a)
    W = model.b2_conv1.weight.detach().reshape(WIDTH, -1).double().cpu()
    b = model.b2_conv1.bias.detach().double().cpu()
    tgt_c2a = _conv_pre_2d(t_b2c1_post)
    if method == "naive":
        ah = invert_layer(W, b, tgt_c2a, a_pre_c2a, relu, "naive", eps=OLS_EPS)
    else:
        ah = invert_layer(W, b, tgt_c2a, a_pre_c2a, relu, "kfac_a", mu_a=mu, Sigma_a=S)
    t_b2in = _fold_to_spatial(ah, model.b2_conv1, P1_HW, n_img)          # (n,W,8,8)

    # invert pool1 (switch-aware) -> target for block1 output (post out-ReLU)
    t_b1out = _switch_unpool(t_b2in, st["pool1_indices"][:n_img], st["b1_out"][:n_img], P0_HW)

    # block1: out = ReLU(b1_in + f1)
    b1_sum_pre = st["b1_sum_pre"][:n_img].double()
    b1_in = st["b1_in"][:n_img].double()
    t_sum1 = invert_activation(_conv_pre_2d(t_b1out), _conv_pre_2d(b1_sum_pre), relu)
    t_sum1 = t_sum1.reshape(n_img, *P0_HW, WIDTH).permute(0, 3, 1, 2)
    t_f1 = t_sum1 - b1_in

    # b1_conv2: invert (no act) -> target post-act of b1_conv1
    a_in_c1b = _im2col(relu(st["b1_conv1_pre"][:n_img].double()).float().to(device), model.b1_conv2).double().cpu()
    a_pre_c1b = _conv_pre_2d(st["b1_conv2_pre"][:n_img].double())
    mu, S = _empirical_mean_cov(a_in_c1b)
    W = model.b1_conv2.weight.detach().reshape(WIDTH, -1).double().cpu()
    b = model.b1_conv2.bias.detach().double().cpu()
    tgt_c1b = _conv_pre_2d(t_f1)
    if method == "naive":
        ah = invert_layer(W, b, tgt_c1b, a_pre_c1b, None, "naive", eps=OLS_EPS)
    else:
        ah = invert_layer(W, b, tgt_c1b, a_pre_c1b, None, "kfac_a", mu_a=mu, Sigma_a=S)
    t_b1c1_post = _fold_to_spatial(ah, model.b1_conv2, P0_HW, n_img)

    # b1_conv1: invert (relu after) -> target for block1 input
    a_in_c1a = _im2col(b1_in.float().to(device), model.b1_conv1).double().cpu()
    a_pre_c1a = _conv_pre_2d(st["b1_conv1_pre"][:n_img].double())
    mu, S = _empirical_mean_cov(a_in_c1a)
    W = model.b1_conv1.weight.detach().reshape(WIDTH, -1).double().cpu()
    b = model.b1_conv1.bias.detach().double().cpu()
    tgt_c1a = _conv_pre_2d(t_b1c1_post)
    if method == "naive":
        ah = invert_layer(W, b, tgt_c1a, a_pre_c1a, relu, "naive", eps=OLS_EPS)
    else:
        ah = invert_layer(W, b, tgt_c1a, a_pre_c1a, relu, "kfac_a", mu_a=mu, Sigma_a=S)
    t_b1in = _fold_to_spatial(ah, model.b1_conv1, P0_HW, n_img)

    # invert pool0 (switch-aware) -> target for conv0 post-ReLU
    t_conv0_post = _switch_unpool(t_b1in, st["pool0_indices"][:n_img], st["conv0_prepool"][:n_img], C0_HW)

    # ---------- FORWARD OLS SWEEP (shallowest first, rebuilt upstream) ----------
    x_all = torch.cat(list(images_list), dim=0)[:n_img].to(device)

    def _fit_conv(conv, X_spatial, t_pre):
        X = _im2col(X_spatial, conv).double().cpu()
        Wn, bn = solve_ols_layer(X, t_pre, with_bias=True, ols_lambda=OLS_LAMBDA)
        pred = X @ Wn.T + (bn.unsqueeze(0) if bn is not None else 0)
        res = float((pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30))
        with torch.no_grad():
            conv.weight.copy_(Wn.reshape(conv.weight.shape).to(conv.weight.device, conv.weight.dtype))
            if conv.bias is not None:
                conv.bias.copy_(bn.to(conv.bias.device, conv.bias.dtype))
        return res

    # conv0
    t_pre = invert_activation(_conv_pre_2d(t_conv0_post), _conv_pre_2d(st["conv0_prepool"][:n_img].double()), relu)
    info["conv0"] = {"target_residual": _fit_conv(model.conv0, x_all, t_pre)}
    with torch.no_grad():
        sp0 = model.pool0(relu(model.conv0(x_all)))
    # b1_conv1
    t_pre = invert_activation(_conv_pre_2d(t_b1c1_post), _conv_pre_2d(st["b1_conv1_pre"][:n_img].double()), relu)
    info["b1_conv1"] = {"target_residual": _fit_conv(model.b1_conv1, sp0, t_pre)}
    with torch.no_grad():
        mid1 = relu(model.b1_conv1(sp0))
    # b1_conv2 (no activation after -> t_pre = t_f1)
    t_pre = _conv_pre_2d(t_f1)
    info["b1_conv2"] = {"target_residual": _fit_conv(model.b1_conv2, mid1, t_pre)}
    with torch.no_grad():
        f1 = model.b1_conv2(mid1)
        b1out = relu(sp0 + f1)
        sp1 = model.pool1(b1out)
    # b2_conv1
    t_pre = invert_activation(_conv_pre_2d(t_b2c1_post), _conv_pre_2d(st["b2_conv1_pre"][:n_img].double()), relu)
    info["b2_conv1"] = {"target_residual": _fit_conv(model.b2_conv1, sp1, t_pre)}
    with torch.no_grad():
        mid2 = relu(model.b2_conv1(sp1))
    # b2_conv2 (no activation after -> t_pre = t_f2)
    t_pre = _conv_pre_2d(t_f2)
    info["b2_conv2"] = {"target_residual": _fit_conv(model.b2_conv2, mid2, t_pre)}
    with torch.no_grad():
        f2 = model.b2_conv2(mid2)
        b2out = relu(sp1 + f2)
        gap = model.flatten(model.gap(b2out)).double().cpu()
    # fc (no activation -> target = GT logits directly)
    Wn, bn = solve_ols_layer(gap, gt, with_bias=True, ols_lambda=OLS_LAMBDA)
    pred = gap @ Wn.T + bn.unsqueeze(0)
    info["fc"] = {"target_residual": float((pred - gt).norm() / gt.norm().clamp(min=1e-30))}
    with torch.no_grad():
        model.fc.weight.copy_(Wn.to(model.fc.weight.device, model.fc.weight.dtype))
        model.fc.bias.copy_(bn.to(model.fc.bias.device, model.fc.bias.dtype))

    acc = _eval(model, test_loader, device)
    wall = time.time() - t0
    print(f"  retrain-rand acc = {acc:.4f}  delta={acc - rand_acc:+.4f}  ({wall:.1f}s)")
    for nm in _LAYER_NAMES_DF:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm:<9s}: residual={r:.3g}" if r is not None else f"    {nm}: --")

    result = {
        "battery": "B-retrain-rand", "exp_id": "e5", "seed": SEED, "method": method,
        "random_init_acc": rand_acc, "retrained_acc": acc, "delta": acc - rand_acc,
        "wall_s": wall,
        "per_layer_residuals": {n: info.get(n, {}).get("target_residual") for n in _LAYER_NAMES_DF},
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
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    globals()["SEED"] = args.seed

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[E5] device={device}  option={args.option}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_set, train_eval_set, test_set = _load_cifar10(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    images_list, labels = _build_image_list(train_eval_set, N_IMAGES)

    def _load_trained():
        if not RESNET_WEIGHTS.exists():
            print(f"[E5] no weights at {RESNET_WEIGHTS}; run --option train first.")
            return None
        m = MiniResNet(width=WIDTH).to(device)
        m.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device, weights_only=True))
        return m

    if args.option in ("train", "all"):
        battery_train(train_set, train_eval_set, test_loader,
                      device=device, epochs=args.epochs)

    if args.option in ("distill", "all"):
        teacher = _load_trained()
        if teacher is None:
            return 1
        battery_distill(teacher, images_list, test_loader, device=device)

    if args.option in ("retrain", "all"):
        battery_retrain(images_list, labels, test_loader, device=device, method=args.method)

    print("\n[E5] === COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
