"""
benchmark/plot_loss_curves.py

Loss-curve comparison plot: Classic K-FAC vs Vered K-FAC vs SINGD vs AdamW
on SmallGPT/WikiText-2 (bf16), small and medium arch side-by-side.

Per method: plot the seed-averaged loss curve with a shaded band for ± std.

Output: benchmark/results/loss_curves_4way.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RES = Path(__file__).resolve().parent / "results"

# (label_in_filename, display_label, color)
METHODS = [
    ("classic",  "Classic K-FAC",     "#cc4444"),
    ("vered",    "Vered K-FAC",       "#2266aa"),
    ("singd",    "SINGD-Dense",       "#44aa66"),
    ("adamw",    "AdamW",             "#888888"),
]

ARCHS = ["small", "medium"]
SEEDS = [42, 43, 44, 45, 46]
SMOOTH = 20   # rolling mean window for noise reduction


def load_loss(arch, label, seed):
    p = RES / f"per_step_bf16_{arch}_{label}_seed{seed}_s1000.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    losses = np.array([r["loss"] for r in d.get("per_step", []) if "loss" in r])
    return losses if len(losses) > 0 else None


def smooth(x, w):
    if len(x) < w or w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / float(w)


def main():
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(13, 4.8), sharey=True)

    for ax, arch in zip(axes, ARCHS):
        for label, display, color in METHODS:
            curves = []
            for s in SEEDS:
                losses = load_loss(arch, label, s)
                if losses is not None:
                    curves.append(losses)
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            stacked = np.stack([c[:min_len] for c in curves])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            steps = np.arange(1, min_len + 1)
            # smooth
            if SMOOTH > 1 and min_len >= SMOOTH:
                mean_s = smooth(mean, SMOOTH)
                std_s = smooth(std, SMOOTH)
                steps_s = steps[SMOOTH-1:]
            else:
                mean_s, std_s, steps_s = mean, std, steps
            ax.plot(steps_s, mean_s, color=color, linewidth=1.8,
                    label=f"{display} (n={len(curves)})")
            ax.fill_between(steps_s, mean_s - std_s, mean_s + std_s,
                            color=color, alpha=0.15)

        ax.set_xlabel("training step")
        ax.set_title(f"SmallGPT-{arch}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(0, 1000)

    axes[0].set_ylabel("training loss (cross-entropy)")
    fig.suptitle(f"Training-loss curves at bf16: 4-way comparison "
                 f"(mean ± std over {len(SEEDS)} seeds, "
                 f"{SMOOTH}-step rolling smoothing)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = RES / "loss_curves_4way.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Also print a quick text summary
    print("\n=== Final loss per method × arch ===")
    print(f"  {'arch':>8}  {'method':>15}  {'final_loss':>11}  {'n_seeds':>7}")
    for arch in ARCHS:
        for label, display, _ in METHODS:
            finals = []
            for s in SEEDS:
                losses = load_loss(arch, label, s)
                if losses is not None:
                    finals.append(float(losses[-1]))
            if finals:
                mean = np.mean(finals); std = np.std(finals)
                print(f"  {arch:>8}  {display:>15}  {mean:>6.3f}±{std:>4.3f}  {len(finals):>7}")


if __name__ == "__main__":
    main()
