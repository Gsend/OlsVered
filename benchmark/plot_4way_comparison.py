"""
benchmark/plot_4way_comparison.py

Two figures from the 4-way comparison sweep:
  - Figure A (2×2 grid): training-loss curves for transformer + cnn, fp32 + bf16
  - Figure B (2×2 grid): wall-time bar charts (mean per method, with seed range)

Output: benchmark/results/comparison_loss.png
        benchmark/results/comparison_wall.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RES = Path(__file__).resolve().parent / "results"

METHODS = [
    ("adamw",      "AdamW",         "#888888"),
    ("classic",    "Classic K-FAC", "#cc4444"),
    ("vered",      "Vered K-FAC",   "#2266aa"),
    ("vered_wgso", "Vered + WGSO",  "#1144cc"),
    ("singd",      "SINGD-Dense",   "#44aa66"),
]
ARCHS = [("transformer", "SmallGPT-medium (22M)"),
         ("cnn",         "ResNet-34 CIFAR-10 (21M)")]
PRECISIONS = ["fp32", "bf16"]
SEEDS = [42, 43, 44]
SMOOTH = 20


def load_run(arch, precision, method, seed):
    p = RES / f"per_step_4way_{arch}_{precision}_{method}_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def smooth(x, w):
    if len(x) < w or w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / float(w)


# ---- Figure A: loss curves -----------------------------------------------

def plot_loss():
    fig, axes = plt.subplots(len(ARCHS), len(PRECISIONS),
                              figsize=(11.5, 7), sharex=True)
    for ai, (arch, arch_title) in enumerate(ARCHS):
        for pi, precision in enumerate(PRECISIONS):
            ax = axes[ai, pi]
            for method, label, color in METHODS:
                curves = []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    losses = np.array([r["loss"] for r in d["per_step"]])
                    if len(losses) >= 100:
                        curves.append(losses)
                if not curves:
                    continue
                min_len = min(len(c) for c in curves)
                stacked = np.stack([c[:min_len] for c in curves])
                mean = stacked.mean(axis=0)
                std = stacked.std(axis=0)
                if SMOOTH > 1 and min_len >= SMOOTH:
                    mean = smooth(mean, SMOOTH)
                    std = smooth(std, SMOOTH)
                    steps = np.arange(SMOOTH, min_len + 1)
                else:
                    steps = np.arange(1, min_len + 1)
                ax.plot(steps, mean, color=color, linewidth=1.8,
                        label=f"{label} (n={len(curves)})")
                ax.fill_between(steps, mean - std, mean + std,
                                color=color, alpha=0.15)
            ax.set_title(f"{arch_title} — {precision}", fontsize=10)
            ax.grid(alpha=0.3)
            if ai == len(ARCHS) - 1:
                ax.set_xlabel("training step")
            if pi == 0:
                ax.set_ylabel("loss (cross-entropy)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlim(0, 1000)
    fig.suptitle(f"Training-loss curves: 4-way comparison "
                 f"(mean ± std over {len(SEEDS)} seeds, "
                 f"{SMOOTH}-step rolling smoothing)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out = RES / "comparison_loss.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---- Figure B: wall-time bars --------------------------------------------

def plot_wall():
    fig, axes = plt.subplots(len(ARCHS), len(PRECISIONS),
                              figsize=(10.5, 6.5), sharey=False)
    for ai, (arch, arch_title) in enumerate(ARCHS):
        for pi, precision in enumerate(PRECISIONS):
            ax = axes[ai, pi]
            labels, means, stds, colors = [], [], [], []
            for method, label, color in METHODS:
                walls = []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    walls.append(d.get("wall_s", 0) / 60)
                if not walls: continue
                labels.append(label)
                means.append(np.mean(walls))
                stds.append(np.std(walls))
                colors.append(color)
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                   edgecolor="black", linewidth=0.5)
            for xi, m in zip(x, means):
                ax.text(xi, m, f"{m:.1f}m", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
            ax.set_title(f"{arch_title} — {precision}", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            if pi == 0:
                ax.set_ylabel("wall time (min / 1000 steps)")
    fig.suptitle("Wall-time per 1000 training steps  (mean ± std, 3 seeds)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out = RES / "comparison_wall.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    plot_loss()
    plot_wall()

    # text summary
    print("\n=== final metric (ppl for transformer, acc%% for cnn) and wall time ===")
    print(f"  {'arch':>12}  {'prec':>5}  {'method':>8}  {'metric':>10}  {'wall(m)':>8}")
    for arch, _ in ARCHS:
        for precision in PRECISIONS:
            for method, _, _ in METHODS:
                mvals, walls = [], []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    k = "final_ppl" if arch == "transformer" else "final_acc"
                    v = d.get(k)
                    if v is not None:
                        if k == "final_acc": v *= 100
                        mvals.append(v)
                    walls.append(d.get("wall_s", 0) / 60)
                if not mvals: continue
                mu_m = np.mean(mvals); mu_w = np.mean(walls)
                print(f"  {arch:>12}  {precision:>5}  {method:>8}  {mu_m:>10.2f}  {mu_w:>8.2f}")


if __name__ == "__main__":
    main()
