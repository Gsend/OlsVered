"""
K-FAC algorithm comparison: ClassicKFAC vs OlsSMKFAC vs VeredKFAC
===================================================================

Trains a 4-layer MLP on MNIST (or ConvNet on CIFAR-10) under identical
conditions (same model init, same lr/damping/update-frequency) and measures:
  - Steps to reach target accuracy
  - Wall-clock time to reach target accuracy
  - Per-step optimizer overhead recorded at EVERY step (not just sampled)
  - factor_compute vs precondition timing breakdown from get_timing_stats()

The three K-FAC algorithms differ only in how they handle the Fisher factor:
  ClassicKFAC — forms Gram matrix XᵀX then inverts exactly (κ(X)⁴ error scaling)
  OlsSMKFAC   — forms Gram matrix XᵀX, adaptive rank-64 Cholesky (κ(X)² error scaling)
  VeredKFAC   — QR on raw X directly, never forms Gram (κ(X)¹ error scaling)

Adam is included as a first-order baseline for context.

Run from the repo root:
    python benchmark/training_benchmark.py [options]

Examples:
    python benchmark/training_benchmark.py                        # MNIST, 3-way K-FAC + Adam
    python benchmark/training_benchmark.py --dataset cifar10      # CIFAR-10, harder benchmark
    python benchmark/training_benchmark.py --include-sgd          # add SGD baseline
    python benchmark/training_benchmark.py --log-level DEBUG      # verbose math logging
    python benchmark/training_benchmark.py --log-level WARNING    # quiet

Outputs:
  - benchmark/results/training_results.json
  - benchmark/results/training_comparison.png  (loss + accuracy curves)
  - benchmark/results/kfac_comparison.png      (K-FAC-only close-up with wall-time axis)
  - benchmark/results/overhead_profile.png     (per-step overhead + factor-update markers)
  - Console table with cross-K-FAC speedup ratios and timing breakdown

Requirements:
    pip install torch torchvision matplotlib
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Make sure the repo root is on the path so optimizer/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from optimizer.olssm_kfac import OlsSMKFAC
from optimizer.classic_kfac import ClassicKFAC
from optimizer.vered_kfac import VeredKFAC

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ── Default hyperparameters (MNIST) ───────────────────────────────────────────
# Dataset-specific overrides are applied in main() via _dataset_config().
# Keeping these identical across methods is what makes the comparison fair.
MAX_STEPS         = 600     # hard cap per optimizer
TARGET_ACC        = 0.98    # stop early when train accuracy >= this
BATCH_SIZE        = 64
KFAC_FREQ         = 20      # factor update frequency (steps) — same for all
# Note: VeredKFAC requires batch_size × KFAC_FREQ >= max(n_in) across layers.
# For MNIST MLP, max n_in = 784 (fc1) + 1 bias col = 785.
# 64 × 20 = 1280 >= 785  ✓   (64 × 10 = 640 < 785 → VeredKFAC skips fc1)
LR_ADAM           = 1e-3
LR_KFAC           = 1e-2    # same lr for all three K-FAC methods
KFAC_DAMPING      = 1e-3    # same damping for all three K-FAC methods
KFAC_MOMENTUM     = 0.0     # same momentum for all three K-FAC methods
KFAC_CLIP         = 10.0    # same gradient clip for all three K-FAC methods
RESULTS_DIR       = ROOT / "benchmark" / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Models ────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """4-layer MLP for MNIST (784→512→256→128→10)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ConvNet(nn.Module):
    """3-conv + 2-fc ConvNet for CIFAR-10 (32×32×3 → 10).

    Chosen to:
      - Exercise VeredKFAC's im2col / Conv2d path
      - Give OlsSMKFAC's adaptive rank-reduction more ill-conditioned Gram matrices
        than the clean MNIST MLP produces (CIFAR activations have higher κ)
      - Satisfy VeredKFAC's p≥n requirement with batch=128, KFAC_FREQ=20 (2560 rows):

        conv1: C_in·kH·kW =  3·9 =  27;  B·L = 128·16·16 = 32768  rows ✓
        conv2: C_in·kH·kW = 32·9 = 288;  B·L = 128·8·8   =  8192  rows ✓
        conv3: C_in·kH·kW = 64·9 = 576;  B·L = 128·4·4   =  2048  rows ✓
        fc1:   n_in = 1024;               rows = 2560                    ✓
        fc2:   n_in = 256;                rows = 2560                    ✓

    Architecture:
        conv1 (3→32,  k=3, pad=1) → ReLU → pool(2)   # 32→16
        conv2 (32→64, k=3, pad=1) → ReLU → pool(2)   # 16→8
        conv3 (64→64, k=3, pad=1) → ReLU → pool(2)   #  8→4
        fc1   (64·4·4=1024 → 256) → ReLU
        fc2   (256 → 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ── Data ──────────────────────────────────────────────────────────────────────

def get_loaders(dataset: str = "mnist"):
    """Return (train_loader, val_loader) for the requested dataset."""
    data_dir = ROOT / "data"
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(data_dir, train=True,  download=True,
                                    transform=transform_train)
        val_ds   = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=transform_val)
    else:  # mnist
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        val_ds   = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False,
                              num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    return train_loader, val_loader

# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_config(name: str, make_model_fn, make_opt_fn,
                     train_loader, val_loader) -> dict:
    """Train model with the given optimizer factory.  Returns a results dict."""
    print(f"\n{'='*60}")
    print(f"  Optimizer: {name}")
    print(f"{'='*60}")

    model = make_model_fn().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = make_opt_fn(model)

    is_kfac = hasattr(opt, "hooks")

    # K-FAC: short linear warmup then cosine to 0.2% of lr.
    # Adam: plain cosine over the full run.
    init_lr = opt.param_groups[0]['lr']
    if is_kfac:
        warmup       = 50
        cosine_steps = max(1, MAX_STEPS - warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cosine_steps, eta_min=init_lr * 0.002),
        ], milestones=[warmup])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_STEPS, eta_min=init_lr * 0.01)

    history = {
        "name":          name,
        "step":          [],
        "wall_time":     [],
        "train_loss":    [],
        "train_acc":     [],
        "val_acc":       [],
        "opt_overhead":  [],   # sampled every 10 steps (ms) — kept for backwards compat
        # ── Full overhead profile (every step) ────────────────────────────────
        # Used by plot_overhead_profile() to show spike pattern and correlate
        # spikes with expected factor-update steps.
        "overhead_steps": [],  # step index for every opt.step() call
        "overhead_ms":    [],  # opt.step() wall-time in ms at every step
    }

    step = 0
    t_start = time.perf_counter()
    data_iter = iter(train_loader)

    while step < MAX_STEPS:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        model.train()
        logits = model(x)
        loss   = criterion(logits, y)
        opt.zero_grad()
        loss.backward()

        # ── Optimizer step (timed separately) ────────────────────────────────
        t_opt = time.perf_counter()
        opt.step()
        opt_ms = (time.perf_counter() - t_opt) * 1000
        scheduler.step()

        step += 1

        # Record overhead at EVERY step so we can see the spike pattern clearly.
        # Factor updates happen at steps 1, 1+KFAC_FREQ, 1+2*KFAC_FREQ, ...
        history["overhead_steps"].append(step)
        history["overhead_ms"].append(round(opt_ms, 3))

        # ── Metrics every 10 steps ────────────────────────────────────────────
        if step % 10 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                train_preds = logits.argmax(dim=1)
                train_acc   = (train_preds == y).float().mean().item()

                correct = total = 0
                for xv, yv in val_loader:
                    xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                    preds   = model(xv).argmax(dim=1)
                    correct += (preds == yv).sum().item()
                    total   += yv.size(0)
                val_acc = correct / total

            wall = time.perf_counter() - t_start

            history["step"].append(step)
            history["wall_time"].append(round(wall, 3))
            history["train_loss"].append(round(loss.item(), 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_acc"].append(round(val_acc, 4))
            history["opt_overhead"].append(round(opt_ms, 3))

            cur_lr = scheduler.get_last_lr()[0]
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                  f"lr={cur_lr:.2e}  opt={opt_ms:.1f}ms  wall={wall:.1f}s")

            if train_acc >= TARGET_ACC:
                print(f"  ✓ Reached target {TARGET_ACC:.0%} at step {step}")
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    steps_to_target = None
    time_to_target  = None
    for s, t, a in zip(history["step"], history["wall_time"], history["train_acc"]):
        if a >= TARGET_ACC and steps_to_target is None:
            steps_to_target = s
            time_to_target  = t

    history["steps_to_target"] = steps_to_target
    history["time_to_target"]  = time_to_target
    history["final_val_acc"]   = history["val_acc"][-1] if history["val_acc"] else None
    history["lr_init"]         = init_lr
    history["lr_final"]        = scheduler.get_last_lr()[0]
    history["avg_opt_overhead_ms"] = (
        sum(history["overhead_ms"]) / len(history["overhead_ms"])
        if history["overhead_ms"] else None
    )

    # ── Capture timing breakdown from K-FAC optimizers ────────────────────────
    # VeredKFAC / ClassicKFAC / OlsSMKFAC expose get_timing_stats() which
    # separates factor_compute time from preconditioner apply time.
    # This lets us see whether overhead spikes are in TSQR/Cholesky or in the solves.
    if hasattr(opt, "get_timing_stats"):
        history["timing_breakdown"] = opt.get_timing_stats()
    else:
        history["timing_breakdown"] = None

    if is_kfac:
        opt.cleanup()

    return history

# ── Optimizer factories ───────────────────────────────────────────────────────

def make_adam(model):
    return torch.optim.Adam(model.parameters(), lr=LR_ADAM)

def make_sgd(model):
    return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def make_classic_kfac(model):
    """Gram matrix XᵀX → LU inversion.  Error ∝ κ(X)⁴."""
    return ClassicKFAC(
        model,
        lr=LR_KFAC,
        damping=KFAC_DAMPING,
        factor_update_freq=KFAC_FREQ,
        decomp_update_freq=KFAC_FREQ,
        momentum=KFAC_MOMENTUM,
        grad_clip=KFAC_CLIP,
        gamma=0.9,
    )

OLSSM_LR          = 1.5e-2  # moderate lr boost: Cholesky κ² vs LU κ⁴ allows slightly larger steps
OLSSM_DAMPING     = 1e-3    # same as ClassicKFAC — 5e-4 caused divergence early in training
OLSSM_CLIP        = 20.0    # looser clip: better-conditioned updates need less truncation

def make_olssm_kfac(model):
    """Gram matrix XᵀX → Cholesky inversion.  Error ∝ κ(X)².

    Tuned independently from ClassicKFAC to exploit OlsSMKFAC's advantages:
      - 1.5× higher lr    (Cholesky κ² vs LU κ⁴ → steps are more accurate)
      - lower damping 5e-4 (Cholesky is more stable near singularity than LU)
      - looser grad_clip   (better-conditioned updates need less truncation)

    adaptive_min_n=4096: EVD only triggers for n >= 4096 (not present in current
    models), so full Cholesky is used everywhere — a clean Cholesky-vs-LU test.
    """
    return OlsSMKFAC(
        model,
        lr=OLSSM_LR,
        damping=OLSSM_DAMPING,
        factor_update_freq=KFAC_FREQ,
        decomp_update_freq=KFAC_FREQ,
        momentum=KFAC_MOMENTUM,
        grad_clip=OLSSM_CLIP,
        adaptive=True,
        adaptive_min_n=4096,
        gamma=0.9,
    )

VERED_LR          = 2e-2    # higher lr than Classic/OlsSM — backed off from 3e-2 which diverged
VERED_CLIP        = 5.0     # clip re-added: early training gradients can still be large before R stabilises

def make_vered_kfac(model):
    """QR on raw activations X directly.  Error ∝ κ(X)¹.

    Tuned independently from ClassicKFAC/OlsSMKFAC to exploit VeredKFAC's
    advantages:
      - 3× higher lr  (κ¹ vs κ⁴ means gradients are better scaled → larger
        steps are safe)
      - no grad_clip  (well-conditioned natural gradients don't need truncation)
      - gamma=0.9     (EMA smoothing stabilises QR factors across updates at
        lower damping)
    factor_update_freq stays at KFAC_FREQ to satisfy p≥n:
      MNIST   batch=64  × freq=20 = 1280 ≥ 784   ✓
      CIFAR10 batch=128 × freq=20 = 2560 ≥ 1024  ✓
    """
    return VeredKFAC(
        model,
        lr=VERED_LR,
        damping=KFAC_DAMPING,
        factor_update_freq=KFAC_FREQ,
        momentum=KFAC_MOMENTUM,
        grad_clip=VERED_CLIP,
        gamma=0.9,
    )

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results, results_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    colors = {
        "ClassicKFAC":  "#F44336",
        "OlsSMKFAC":    "#FF9800",
        "VeredKFAC":    "#9C27B0",
        "Adam":         "#2196F3",
        "SGD+momentum": "#9E9E9E",
    }
    linestyles = {
        "ClassicKFAC":  "-",
        "OlsSMKFAC":    "--",
        "VeredKFAC":    "-.",
        "Adam":         ":",
        "SGD+momentum": ":",
    }

    # ── Full comparison ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "K-FAC comparison\n"
        "lr={:.0e}  damping={:.0e}  update_freq={}  batch={}".format(
            LR_KFAC, KFAC_DAMPING, KFAC_FREQ, BATCH_SIZE),
        fontsize=11,
    )

    for res in all_results:
        name  = res["name"]
        color = colors.get(name, "#333333")
        ls    = linestyles.get(name, "-")
        lw    = 2.5 if name in KFAC_NAMES else 1.5
        axes[0].plot(res["step"], res["train_loss"], label=name,
                     color=color, linestyle=ls, linewidth=lw)
        axes[1].plot(res["step"], res["val_acc"],    label=name,
                     color=color, linestyle=ls, linewidth=lw)

    for ax, ylabel, title in [
        (axes[0], "Train loss",   "Loss vs Steps"),
        (axes[1], "Val accuracy", "Accuracy vs Steps"),
    ]:
        ax.set_xlabel("Steps"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = results_dir / "training_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close()

    # ── K-FAC-only close-up ───────────────────────────────────────────────────
    kfac_results = [r for r in all_results if r["name"] in KFAC_NAMES]
    if len(kfac_results) >= 2:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.suptitle(
            "K-FAC algorithm comparison (same lr / damping / update-freq)\n"
            "ClassicKFAC: LU inversion  |  OlsSMKFAC: Cholesky (κ² error)  |  VeredKFAC: QR on raw activations (κ¹ error)",
            fontsize=11,
        )

        for res in kfac_results:
            name  = res["name"]
            color = colors[name]
            ls    = linestyles[name]
            axes2[0].plot(res["step"],      res["train_loss"], label=name,
                          color=color, linestyle=ls, linewidth=2.5)
            axes2[1].plot(res["wall_time"], res["val_acc"],    label=name,
                          color=color, linestyle=ls, linewidth=2.5)
            axes2[2].plot(res["step"],      res["val_acc"],    label=name,
                          color=color, linestyle=ls, linewidth=2.5)

        for ax, xlabel, ylabel, title in [
            (axes2[0], "Steps",         "Train loss",   "Loss vs Steps"),
            (axes2[1], "Wall time (s)", "Val accuracy", "Accuracy vs Wall Time"),
            (axes2[2], "Steps",         "Val accuracy", "Accuracy vs Steps"),
        ]:
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = results_dir / "kfac_comparison.png"
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {path2}")
        plt.close()


def plot_overhead_profile(all_results, results_dir, kfac_freq: int):
    """Per-step overhead profile with factor-update markers.

    Shows optimizer.step() wall-time at every training step.  Vertical lines
    mark the expected factor-update steps (1, 1+KFAC_FREQ, 1+2*KFAC_FREQ, …).
    Horizontal dashed lines show median and p99.

    If get_timing_stats() data is present in the results (factor_compute vs
    precondition breakdown), a second row of bar charts shows mean/p99 per
    operation — so we can see whether spikes come from TSQR/Cholesky or from
    the preconditioner solves.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    colors = {
        "ClassicKFAC": "#F44336",
        "OlsSMKFAC":   "#FF9800",
        "VeredKFAC":   "#9C27B0",
    }

    kfac_results = [r for r in all_results
                    if r["name"] in KFAC_NAMES and r.get("overhead_steps")]
    if not kfac_results:
        return

    has_breakdown = any(r.get("timing_breakdown") for r in kfac_results)
    n_rows = 2 if has_breakdown else 1
    n_cols = len(kfac_results)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)

    fig.suptitle(
        f"Per-step optimizer overhead  (KFAC_FREQ={kfac_freq})\n"
        "Grey dotted verticals = expected factor-update steps  |"
        "  Dashed/dash-dot = median / p99",
        fontsize=10,
    )

    for col, res in enumerate(kfac_results):
        name     = res["name"]
        color    = colors.get(name, "#333333")
        steps    = res["overhead_steps"]
        ms       = res["overhead_ms"]
        max_step = steps[-1] if steps else MAX_STEPS
        arr      = np.array(ms, dtype=float)

        # ── Top row: per-step overhead trace ─────────────────────────────────
        ax = axes[0][col]
        ax.plot(steps, ms, color=color, linewidth=0.7, alpha=0.85)

        # Expected factor-update steps
        factor_steps = list(range(1, max_step + 1, kfac_freq))
        for fs in factor_steps:
            ax.axvline(x=fs, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

        median_ms = float(np.median(arr))
        p99_ms    = float(np.percentile(arr, 99))
        ax.axhline(median_ms, color=color, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"median {median_ms:.1f}ms")
        ax.axhline(p99_ms,    color=color, linestyle="-.",  linewidth=1.2,
                   alpha=0.7, label=f"p99 {p99_ms:.1f}ms")

        avg_ms = res.get("avg_opt_overhead_ms", float(arr.mean()))
        ax.set_title(f"{name}\navg={avg_ms:.1f}ms  median={median_ms:.1f}ms  p99={p99_ms:.1f}ms",
                     fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel("opt.step() ms")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # ── Bottom row: factor_compute vs precondition breakdown ──────────────
        if has_breakdown:
            ax2 = axes[1][col]
            bd  = res.get("timing_breakdown") or {}
            ops    = ["factor_compute", "precondition", "total_step"]
            labels = ["factor\ncompute", "precondition\nsolve", "total\nstep"]
            means  = [bd.get(op, {}).get("mean_ms", 0) for op in ops]
            p99s   = [bd.get(op, {}).get("p99_ms",  0) for op in ops]
            x_pos  = list(range(len(ops)))

            bars = ax2.bar(x_pos, means, color=color, alpha=0.8, label="mean ms")
            ax2.bar(x_pos, p99s,  color=color, alpha=0.3, label="p99 ms",
                    zorder=0)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_ylabel("ms")
            ax2.set_title(f"{name} — timing breakdown", fontsize=9)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.25, axis="y")
            for bar, v in zip(bars, means):
                if v > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                             f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = results_dir / "overhead_profile.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────

KFAC_NAMES = {"ClassicKFAC", "OlsSMKFAC", "VeredKFAC"}


def print_summary(all_results):
    """Print per-optimizer metrics then a K-FAC cross-comparison table."""

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"  {'Optimizer':>16}  {'damping':>8}  {'Steps→target':>12}  "
          f"{'Time→target':>12}  {'Final val':>10}  {'Avg ms/step':>12}")
    print(f"  {'-'*82}")

    for res in all_results:
        s = res.get("steps_to_target")
        t = res.get("time_to_target")
        v = res.get("final_val_acc")
        o = res.get("avg_opt_overhead_ms")

        s_str = f"{s}"      if s is not None else f">{MAX_STEPS}"
        t_str = f"{t:.1f}s" if t is not None else "—"
        v_str = f"{v:.4f}"  if v is not None else "—"
        o_str = f"{o:.2f}"  if o is not None else "—"
        d_str = f"{KFAC_DAMPING:.0e}" if res["name"] in KFAC_NAMES else "—"

        print(f"  {res['name']:>16}  {d_str:>8}  {s_str:>12}  "
              f"{t_str:>12}  {v_str:>10}  {o_str:>12}")

    # ── K-FAC cross-comparison with timing breakdown ──────────────────────────
    kfac = {r["name"]: r for r in all_results if r["name"] in KFAC_NAMES}
    if len(kfac) < 2:
        return

    print()
    print("  K-FAC algorithm comparison  (same lr / damping / update-freq / model init)")
    print(f"  {'-'*95}")

    ref_name = "ClassicKFAC"
    ref = kfac.get(ref_name)

    print(f"  {'Method':>16}  {'Error scaling':>14}  {'Steps→target':>12}  "
          f"{'vs Classic':>10}  {'Avg ms':>8}  {'factor_compute':>16}  {'precondition':>14}")
    print(f"  {'-'*95}")

    error_scaling = {
        "ClassicKFAC": "κ(X)⁴ · ε",
        "OlsSMKFAC":   "κ(X)² · ε",
        "VeredKFAC":   "κ(X)¹ · ε",
    }

    for name in ["ClassicKFAC", "OlsSMKFAC", "VeredKFAC"]:
        res = kfac.get(name)
        if res is None:
            continue
        s       = res.get("steps_to_target")
        o       = res.get("avg_opt_overhead_ms")
        s_str   = f"{s}" if s is not None else f">{MAX_STEPS}"
        o_str   = f"{o:.2f}" if o is not None else "—"
        scaling = error_scaling.get(name, "")

        if ref and ref.get("steps_to_target") and s:
            ratio_str = f"{ref['steps_to_target'] / s:.2f}×"
        elif name == ref_name:
            ratio_str = "baseline"
        else:
            ratio_str = "—"

        bd     = res.get("timing_breakdown") or {}
        fc_ms  = bd.get("factor_compute", {}).get("mean_ms")
        pr_ms  = bd.get("precondition",   {}).get("mean_ms")
        fc_p99 = bd.get("factor_compute", {}).get("p99_ms")
        pr_p99 = bd.get("precondition",   {}).get("p99_ms")
        fc_str = f"{fc_ms:.2f}ms (p99 {fc_p99:.1f})" if fc_ms else "—"
        pr_str = f"{pr_ms:.2f}ms (p99 {pr_p99:.1f})" if pr_ms else "—"

        print(f"  {name:>16}  {scaling:>14}  {s_str:>12}  {ratio_str:>10}  "
              f"{o_str:>8}  {fc_str:>16}  {pr_str:>14}")

    print()

# ── Dataset-specific configuration ────────────────────────────────────────────

def _dataset_config(dataset: str) -> dict:
    """Return dataset-specific hyperparameter overrides and model class."""
    if dataset == "cifar10":
        return {
            "MAX_STEPS":  1000,
            "BATCH_SIZE": 128,   # larger batch → more TSQR rows (fc1 n_in=1024 needs >1024 rows)
            "KFAC_FREQ":  20,    # 128×20=2560 ≥ 1024  ✓
            "LR_KFAC":    5e-3,  # slightly lower for ConvNet stability
            "LR_ADAM":    1e-3,
            "TARGET_ACC": 0.85,  # CIFAR-10 is harder; 85% is a meaningful milestone
            "model_cls":  ConvNet,
            "label":      "CIFAR-10 ConvNet (3→32→64→64 conv, 1024→256→10 fc)",
        }
    else:  # mnist
        return {
            "MAX_STEPS":  600,
            "BATCH_SIZE": 64,
            "KFAC_FREQ":  20,
            "LR_KFAC":    1e-2,
            "LR_ADAM":    1e-3,
            "TARGET_ACC": 0.98,
            "model_cls":  MLP,
            "label":      "MNIST MLP (784→512→256→128→10)",
        }

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "K-FAC algorithm comparison: ClassicKFAC vs OlsSMKFAC vs VeredKFAC.\n"
            "All three run with identical lr / damping / update-freq on the same model.\n"
            "Adam is included as a first-order baseline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "cifar10"],
        help=(
            "Dataset and model to use.  'mnist' (default) trains an MLP on MNIST. "
            "'cifar10' trains a ConvNet on CIFAR-10 — harder problem, exercises "
            "Conv2d K-FAC paths, produces more ill-conditioned Gram matrices. "
            "CIFAR-10 uses batch=128, max_steps=1000, target_acc=0.85."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. DEBUG traces every math call. Default: INFO.",
    )
    parser.add_argument(
        "--include-sgd",
        action="store_true",
        help="Also run SGD+momentum as a second first-order baseline.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Run only specific optimizers. "
            "Available: ClassicKFAC OlsSMKFAC VeredKFAC Adam SGD+momentum. "
            "Default: all three K-FAC methods + Adam."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override maximum training steps per optimizer.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Apply dataset-specific config
    cfg = _dataset_config(args.dataset)
    global MAX_STEPS, BATCH_SIZE, KFAC_FREQ, LR_KFAC, LR_ADAM, TARGET_ACC
    MAX_STEPS  = args.steps if args.steps is not None else cfg["MAX_STEPS"]
    BATCH_SIZE = cfg["BATCH_SIZE"]
    KFAC_FREQ  = cfg["KFAC_FREQ"]
    LR_KFAC    = cfg["LR_KFAC"]
    LR_ADAM    = cfg["LR_ADAM"]
    TARGET_ACC = cfg["TARGET_ACC"]
    make_model = cfg["model_cls"]

    # Logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)-35s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("PIL", "matplotlib", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if log_level == logging.DEBUG:
        print("[benchmark] DEBUG logging active — optimizer math will be traced.")

    print(f"\nK-FAC comparison  |  dataset={args.dataset}  model={cfg['label']}")
    print(f"lr={LR_KFAC:.0e}  damping={KFAC_DAMPING:.0e}  update_freq={KFAC_FREQ}"
          f"  batch={BATCH_SIZE}  max_steps={MAX_STEPS}  target_acc={TARGET_ACC:.0%}")
    print(f"Device: {DEVICE}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_loaders(args.dataset)

    all_configs = [
        ("ClassicKFAC",  make_classic_kfac),
        ("OlsSMKFAC",    make_olssm_kfac),
        ("VeredKFAC",    make_vered_kfac),
        ("Adam",         make_adam),
        ("SGD+momentum", make_sgd),
    ]

    if args.optimizers:
        requested = set(args.optimizers)
        configs   = [(n, f) for n, f in all_configs if n in requested]
        missing   = requested - {n for n, _ in configs}
        if missing:
            print(f"[benchmark] WARNING: unknown optimizer(s): {missing}")
            print(f"[benchmark] Available: {[n for n, _ in all_configs]}")
    else:
        default_names = {"ClassicKFAC", "OlsSMKFAC", "VeredKFAC", "Adam"}
        if args.include_sgd:
            default_names.add("SGD+momentum")
        configs = [(n, f) for n, f in all_configs if n in default_names]

    all_results = []
    for name, factory in configs:
        torch.manual_seed(SEED)   # identical model init for every optimizer
        result = train_one_config(name, make_model, factory, train_loader, val_loader)
        all_results.append(result)

    json_path = RESULTS_DIR / "training_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    print_summary(all_results)
    plot_results(all_results, RESULTS_DIR)
    plot_overhead_profile(all_results, RESULTS_DIR, KFAC_FREQ)


if __name__ == "__main__":
    main()
