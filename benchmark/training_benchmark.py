"""
Training benchmark: Adam vs ClassicKFAC vs OlsveredKFAC
========================================================

Trains a 4-layer MLP on MNIST and compares:
  - Steps to reach target accuracy
  - Wall-clock time to reach target accuracy
  - Per-step optimizer overhead

Run from the repo root:
    python benchmark/training_benchmark.py

Outputs:
  - benchmark/results/training_results.json
  - benchmark/results/loss_vs_steps.png
  - benchmark/results/loss_vs_time.png
  - benchmark/results/accuracy_vs_steps.png
  - Console table summarising all optimizers

Requirements:
    pip install torch torchvision matplotlib
"""

import json
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

from optimizer.olsvered_kfac import OlsveredKFAC
from optimizer.classic_kfac import ClassicKFAC

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_STEPS    = 500          # hard cap per optimizer
TARGET_ACC   = 0.98         # stop early when train accuracy >= this
BATCH_SIZE   = 64
KFAC_FREQ    = 10           # K-FAC update frequency (steps)
LR_ADAM      = 1e-3
LR_KFAC      = 1e-2         # K-FAC typically uses a larger lr
DAMPING      = 1e-2
RESULTS_DIR  = ROOT / "benchmark" / "results"

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

            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                  f"opt={opt_ms:.1f}ms  wall={wall:.1f}s")

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
    history["avg_opt_overhead_ms"] = (
        sum(history["opt_overhead"]) / len(history["opt_overhead"])
        if history["opt_overhead"] else None
    )

    if is_kfac:
        opt.cleanup()

    return history


# ── Optimizer factories ───────────────────────────────────────────────────────

def make_adam(model):
    return torch.optim.Adam(model.parameters(), lr=LR_ADAM)

def make_sgd(model):
    return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def make_classic_kfac(model):
    return ClassicKFAC(
        model,
        lr=LR_KFAC,
        damping=DAMPING,
        factor_update_freq=KFAC_FREQ,
        inv_update_freq=KFAC_FREQ,
    )

def make_olsvered_adaptive(model):
    return OlsveredKFAC(
        model,
        lr=LR_KFAC,
        damping=DAMPING,
        factor_update_freq=KFAC_FREQ,
        inv_update_freq=KFAC_FREQ,
        adaptive=True,
        adaptive_min_n=128,
        adaptive_rank_budget=64,
    )

def make_olsvered_rank32(model):
    return OlsveredKFAC(
        model,
        lr=LR_KFAC,
        damping=DAMPING,
        factor_update_freq=KFAC_FREQ,
        inv_update_freq=KFAC_FREQ,
        rank=32,
        randomized=True,
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
        "Adam":              "#2196F3",
        "SGD+momentum":      "#9E9E9E",
        "ClassicKFAC":       "#F44336",
        "OlsveredKFAC-adaptive": "#4CAF50",
        "OlsveredKFAC-rank32":   "#FF9800",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Optimizer Comparison — MNIST MLP (784→512→256→128→10)", fontsize=13)

    for res in all_results:
        name  = res["name"]
        color = colors.get(name, "#333333")
        steps = res["step"]
        times = res["wall_time"]
        loss  = res["train_loss"]
        vacc  = res["val_acc"]

        axes[0].plot(steps, loss, label=name, color=color, linewidth=2)
        axes[1].plot(times, vacc, label=name, color=color, linewidth=2)
        axes[2].plot(steps, vacc, label=name, color=color, linewidth=2)

    for ax, xlabel, ylabel, title in [
        (axes[0], "Steps",       "Train loss",      "Loss vs Steps"),
        (axes[1], "Wall time (s)","Val accuracy",   "Accuracy vs Wall Time"),
        (axes[2], "Steps",       "Val accuracy",    "Accuracy vs Steps"),
    ]:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = results_dir / "training_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {path}")
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Optimizer':>24}  {'Steps→target':>12}  {'Time→target':>12}  "
          f"{'Final val acc':>14}  {'Opt ms/step':>11}")
    print(f"  {'-'*75}")

    adam_time = next(
        (r["time_to_target"] for r in all_results if r["name"] == "Adam"), None)

    for res in all_results:
        s = res.get("steps_to_target")
        t = res.get("time_to_target")
        v = res.get("final_val_acc")
        o = res.get("avg_opt_overhead_ms")

        s_str = f"{s}" if s else ">500"
        t_str = f"{t:.1f}s" if t else "—"
        v_str = f"{v:.4f}" if v else "—"
        o_str = f"{o:.1f}" if o else "—"

        # Speedup vs Adam
        if t and adam_time:
            ratio = adam_time / t
            ratio_str = f" ({ratio:.2f}× vs Adam)"
        else:
            ratio_str = ""

        print(f"  {res['name']:>24}  {s_str:>12}  {t_str + ratio_str:>25}  "
              f"{v_str:>14}  {o_str:>11}")

    print()

    # Explain convergence ratio if K-FAC reached target
    adam_steps = next(
        (r["steps_to_target"] for r in all_results if r["name"] == "Adam"), None)
    for res in all_results:
        if "KFAC" in res["name"] and res.get("steps_to_target") and adam_steps:
            ratio = adam_steps / res["steps_to_target"]
            print(f"  {res['name']} converged in {ratio:.1f}× fewer steps than Adam")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_loaders()

    configs = [
        ("Adam",                    make_adam),
        ("SGD+momentum",            make_sgd),
        ("ClassicKFAC",             make_classic_kfac),
        ("OlsveredKFAC-adaptive",   make_olsvered_adaptive),
        ("OlsveredKFAC-rank32",     make_olsvered_rank32),
    ]

    all_results = []
    for name, factory in configs:
        torch.manual_seed(SEED)   # same init for every optimizer
        result = train_one_config(name, factory, train_loader, val_loader)
        all_results.append(result)

    # Save JSON
    json_path = RESULTS_DIR / "training_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    print_summary(all_results)
    plot_results(all_results, RESULTS_DIR)


if __name__ == "__main__":
    main()
