"""
K-FAC algorithm comparison: ClassicKFAC vs OlsSMKFAC vs VeredKFAC
===================================================================

Trains a 4-layer MLP on MNIST under identical conditions (same model
init, same lr/damping/update-frequency) and measures:
  - Steps to reach target accuracy
  - Wall-clock time to reach target accuracy
  - Per-step optimizer overhead

The three K-FAC algorithms differ only in how they handle the Fisher factor:
  ClassicKFAC — forms Gram matrix XᵀX then inverts (κ(X)⁴ error scaling)
  OlsSMKFAC   — forms Gram matrix XᵀX then Cholesky (κ(X)² error scaling)
  VeredKFAC   — QR on raw X directly, never forms Gram (κ(X)¹ error scaling)

Adam is included as a first-order baseline for context.

Run from the repo root:
    python benchmark/training_benchmark.py [options]

Examples:
    python benchmark/training_benchmark.py                       # 3-way K-FAC + Adam
    python benchmark/training_benchmark.py --include-sgd         # add SGD baseline
    python benchmark/training_benchmark.py --log-level DEBUG     # verbose math logging
    python benchmark/training_benchmark.py --log-level WARNING   # quiet

Outputs:
  - benchmark/results/training_results.json
  - benchmark/results/training_comparison.png  (loss + accuracy curves)
  - benchmark/results/kfac_comparison.png      (K-FAC-only close-up)
  - Console table with cross-K-FAC speedup ratios

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

# ── Shared K-FAC hyperparameters (identical across all three methods) ─────────
# Keeping these the same is what makes the comparison fair — any difference
# in convergence reflects the algorithm, not the tuning.
MAX_STEPS         = 600     # hard cap per optimizer
TARGET_ACC        = 0.98    # stop early when train accuracy >= this
BATCH_SIZE        = 64
KFAC_FREQ         = 20      # factor update frequency (steps) — same for all
# Note: VeredKFAC requires batch_size × KFAC_FREQ >= max(n_in) across layers.
# For this MLP, max n_in = 784 (fc1) + 1 bias col = 785.
# 64 × 20 = 1280 >= 785  ✓   (64 × 10 = 640 < 785 → VeredKFAC skips fc1)
LR_ADAM           = 1e-3
LR_KFAC           = 1e-2    # same lr for all three K-FAC methods
KFAC_DAMPING      = 5e-3    # same damping for all three K-FAC methods
KFAC_MOMENTUM     = 0.0     # same momentum for all three K-FAC methods
KFAC_CLIP         = 10.0    # same gradient clip for all three K-FAC methods
RESULTS_DIR       = ROOT / "benchmark" / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """4-layer MLP matching the architecture used in numerical benchmarks."""
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

# ── Data ──────────────────────────────────────────────────────────────────────

def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_dir = ROOT / "data"
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    val_ds   = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    return train_loader, val_loader

# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_config(name: str, make_opt_fn, train_loader, val_loader) -> dict:
    """Train model with the given optimizer factory.  Returns a results dict."""
    print(f"\n{'='*60}")
    print(f"  Optimizer: {name}")
    print(f"{'='*60}")

    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = make_opt_fn(model)

    is_kfac = hasattr(opt, "hooks")  # both KFAC optimizers have hooks

    # K-FAC: short linear warmup then monotonic cosine to 0.2% of lr.
    # Warmup avoids large steps before Gram matrices are populated.
    # T_max MUST equal the number of cosine phase steps — setting it shorter
    # causes LR to bounce back to lr_init at 2*T_max (V-shape anti-pattern).
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
        "opt_overhead":  [],   # ms — time spent inside opt.step() only
    }

    step = 0
    t_start = time.perf_counter()
    data_iter = iter(train_loader)

    while step < MAX_STEPS:
        # Refill iterator when exhausted
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        # ── Forward + backward ────────────────────────────────────────────────
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

        # ── Metrics every 10 steps ────────────────────────────────────────────
        if step % 10 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy on current batch (fast proxy)
                train_preds = logits.argmax(dim=1)
                train_acc   = (train_preds == y).float().mean().item()

                # Validation accuracy on full val set
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
    for i, (s, t, a) in enumerate(zip(
            history["step"], history["wall_time"], history["train_acc"])):
        if a >= TARGET_ACC and steps_to_target is None:
            steps_to_target = s
            time_to_target  = t

    history["steps_to_target"] = steps_to_target
    history["time_to_target"]  = time_to_target
    history["final_val_acc"]   = history["val_acc"][-1] if history["val_acc"] else None
    history["lr_init"]         = init_lr
    history["lr_final"]        = scheduler.get_last_lr()[0]
    history["avg_opt_overhead_ms"] = (
        sum(history["opt_overhead"]) / len(history["opt_overhead"])
        if history["opt_overhead"] else None
    )

    if is_kfac:
        opt.cleanup()

    return history

# ── Optimizer factories ───────────────────────────────────────────────────────
# All three K-FAC factories use identical hyperparameters.
# The only difference is the algorithm each implements.

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
    )

def make_olssm_kfac(model):
    """Gram matrix XᵀX → Cholesky factorisation.  Error ∝ κ(X)²."""
    return OlsSMKFAC(
        model,
        lr=LR_KFAC,
        damping=KFAC_DAMPING,
        factor_update_freq=KFAC_FREQ,
        decomp_update_freq=KFAC_FREQ,
        momentum=KFAC_MOMENTUM,
        grad_clip=KFAC_CLIP,
        # No rank reduction, no adaptive approximation — full Cholesky
        # so we isolate the algorithm difference from any low-rank effect.
    )

def make_vered_kfac(model):
    """QR on raw activations X directly.  Error ∝ κ(X)¹."""
    return VeredKFAC(
        model,
        lr=LR_KFAC,
        damping=KFAC_DAMPING,
        factor_update_freq=KFAC_FREQ,
        momentum=KFAC_MOMENTUM,
        grad_clip=KFAC_CLIP,
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

    # Color scheme: K-FAC trio uses a consistent ramp; baselines are muted
    colors = {
        "ClassicKFAC":  "#F44336",   # red   — forms Gram + LU inversion
        "OlsSMKFAC":    "#FF9800",   # orange — forms Gram + Cholesky
        "VeredKFAC":    "#9C27B0",   # purple — QR on raw activations
        "Adam":         "#2196F3",   # blue  — first-order baseline
        "SGD+momentum": "#9E9E9E",   # gray  — first-order baseline
    }
    linestyles = {
        "ClassicKFAC":  "-",
        "OlsSMKFAC":    "--",
        "VeredKFAC":    "-.",
        "Adam":         ":",
        "SGD+momentum": ":",
    }
    # ── Full comparison (all optimizers) ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "K-FAC comparison — MNIST MLP (784→512→256→128→10)\n"
        "lr={:.0e}  damping={:.0e}  update_freq={}  batch={}".format(
            LR_KFAC, KFAC_DAMPING, KFAC_FREQ, BATCH_SIZE),
        fontsize=11,
    )

    for res in all_results:
        name  = res["name"]
        color = colors.get(name, "#333333")
        ls    = linestyles.get(name, "-")
        lw    = 2.5 if name in KFAC_NAMES else 1.5
        axes[0].plot(res["step"],      res["train_loss"], label=name,
                     color=color, linestyle=ls, linewidth=lw)
        axes[1].plot(res["step"],      res["val_acc"],    label=name,
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
            "ClassicKFAC: κ⁴ error  |  OlsSMKFAC: κ² error  |  VeredKFAC: κ¹ error",
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
            (axes2[0], "Steps",        "Train loss",    "Loss vs Steps"),
            (axes2[1], "Wall time (s)","Val accuracy",  "Accuracy vs Wall Time"),
            (axes2[2], "Steps",        "Val accuracy",  "Accuracy vs Steps"),
        ]:
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = results_dir / "kfac_comparison.png"
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {path2}")
        plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────

KFAC_NAMES = {"ClassicKFAC", "OlsSMKFAC", "VeredKFAC"}


def print_summary(all_results):
    """Print per-optimizer metrics then a K-FAC cross-comparison table."""

    # ── Per-optimizer metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("RESULTS")
    print("=" * 88)
    print(f"  {'Optimizer':>16}  {'damping':>8}  {'Steps→98%':>10}  "
          f"{'Time→98%':>10}  {'Final val':>10}  {'Opt ms/step':>12}")
    print(f"  {'-'*82}")

    for res in all_results:
        s = res.get("steps_to_target")
        t = res.get("time_to_target")
        v = res.get("final_val_acc")
        o = res.get("avg_opt_overhead_ms")

        s_str = f"{s}"    if s else f">{MAX_STEPS}"
        t_str = f"{t:.1f}s" if t else "—"
        v_str = f"{v:.4f}" if v is not None else "—"
        o_str = f"{o:.2f}" if o is not None else "—"

        # show damping for K-FAC methods, "—" for first-order
        d_str = f"{KFAC_DAMPING:.0e}" if res["name"] in KFAC_NAMES else "—"

        print(f"  {res['name']:>16}  {d_str:>8}  {s_str:>10}  "
              f"{t_str:>10}  {v_str:>10}  {o_str:>12}")

    # ── K-FAC cross-comparison ────────────────────────────────────────────────
    kfac = {r["name"]: r for r in all_results if r["name"] in KFAC_NAMES}
    if len(kfac) < 2:
        return

    print()
    print("  K-FAC algorithm comparison  (same lr / damping / update-freq / model init)")
    print(f"  {'-'*82}")

    # Use ClassicKFAC as the reference baseline for cross-ratios
    ref_name = "ClassicKFAC"
    ref = kfac.get(ref_name)

    header = f"  {'Method':>16}  {'Error scaling':>14}  {'Steps→98%':>10}  {'vs Classic':>10}  {'Opt ms/step':>12}"
    print(header)
    print(f"  {'-'*70}")

    error_scaling = {
        "ClassicKFAC": "κ(X)⁴ · ε",
        "OlsSMKFAC":   "κ(X)² · ε",
        "VeredKFAC":   "κ(X)¹ · ε",
    }

    for name in ["ClassicKFAC", "OlsSMKFAC", "VeredKFAC"]:
        res = kfac.get(name)
        if res is None:
            continue
        s = res.get("steps_to_target")
        o = res.get("avg_opt_overhead_ms")
        s_str = f"{s}" if s else f">{MAX_STEPS}"
        o_str = f"{o:.2f}" if o is not None else "—"
        scaling = error_scaling.get(name, "")

        if ref and ref.get("steps_to_target") and s:
            ratio = ref["steps_to_target"] / s
            ratio_str = f"{ratio:.2f}×"
        elif name == ref_name:
            ratio_str = "baseline"
        else:
            ratio_str = "—"

        print(f"  {name:>16}  {scaling:>14}  {s_str:>10}  {ratio_str:>10}  {o_str:>12}")

    print()

# ── Main ──────────────────────────────────────────────────────────────────────

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
            "Run only specific optimizers by name. "
            "Available: ClassicKFAC OlsSMKFAC VeredKFAC Adam SGD+momentum. "
            "Default: ClassicKFAC OlsSMKFAC VeredKFAC Adam."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum training steps per optimizer. Default: {MAX_STEPS}.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Allow --steps to override the module-level constant
    global MAX_STEPS
    MAX_STEPS = args.steps

    # ── Logging ───────────────────────────────────────────────────────────────
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

    # ── Print run configuration ───────────────────────────────────────────────
    print(f"\nK-FAC comparison  |  lr={LR_KFAC:.0e}  damping={KFAC_DAMPING:.0e}"
          f"  update_freq={KFAC_FREQ}  batch={BATCH_SIZE}  max_steps={MAX_STEPS}")
    print(f"Device: {DEVICE}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_loaders()

    # ── Default run order: K-FAC trio first, then Adam baseline ───────────────
    all_configs = [
        ("ClassicKFAC",  make_classic_kfac),   # κ⁴ — Gram + LU
        ("OlsSMKFAC",    make_olssm_kfac),     # κ² — Gram + Cholesky
        ("VeredKFAC",    make_vered_kfac),     # κ¹ — QR on raw X
        ("Adam",         make_adam),            # first-order baseline
        ("SGD+momentum", make_sgd),            # first-order baseline
    ]

    if args.optimizers:
        # User explicitly selected a subset
        requested = set(args.optimizers)
        configs = [(n, f) for n, f in all_configs if n in requested]
        missing = requested - {n for n, _ in configs}
        if missing:
            print(f"[benchmark] WARNING: unknown optimizer(s): {missing}")
            print(f"[benchmark] Available: {[n for n, _ in all_configs]}")
    else:
        # Default: K-FAC trio + Adam; optionally add SGD
        default_names = {"ClassicKFAC", "OlsSMKFAC", "VeredKFAC", "Adam"}
        if args.include_sgd:
            default_names.add("SGD+momentum")
        configs = [(n, f) for n, f in all_configs if n in default_names]

    # ── Run ───────────────────────────────────────────────────────────────────
    all_results = []
    for name, factory in configs:
        torch.manual_seed(SEED)   # identical model init for every optimizer
        result = train_one_config(name, factory, train_loader, val_loader)
        all_results.append(result)

    # ── Save & report ─────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "training_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    print_summary(all_results)
    plot_results(all_results, RESULTS_DIR)

if __name__ == "__main__":
    main()
