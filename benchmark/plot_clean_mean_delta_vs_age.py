"""
benchmark/plot_clean_mean_delta_vs_age.py

Mean delta_loss vs age (step within K-FAC window) for the gamma=0
mom=0 high-LR runs. Grid: variant x lr; freqs overlaid per panel.

Skips windows that start before step 200 (warmup).

Output: benchmark/results/clean_mean_delta_vs_age.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

RES = Path(__file__).resolve().parent.parent / "benchmark" / "results"
START_STEP  = 200
LRS         = [1.0, 0.5, 0.1]
FREQS       = [20]   # only freq=20 has enough windows post-warmup to be meaningful
VARIANTS    = ["VeredKFAC", "ClassicKFAC"]


def main():
    nrow, ncol = len(VARIANTS), len(LRS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow))
    for i, variant in enumerate(VARIANTS):
        for j, lr in enumerate(LRS):
            ax = axes[i, j] if nrow > 1 else axes[j]
            for k, freq in enumerate(FREQS):
                lr_tag = f"lr{lr:.0e}"
                p = RES / f"per_step_clean_{variant}_{lr_tag}_f{freq}_s1000.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                dl = np.array([r["delta_loss"] for r in d.get("per_step") or []])
                if len(dl) < 2 * freq:
                    continue
                n_win = len(dl) // freq
                folded = dl[:n_win * freq].reshape(n_win, freq)
                # keep only windows whose start step >= START_STEP
                window_starts = np.arange(n_win) * freq
                keep = window_starts >= START_STEP
                if keep.sum() < 3:   # need >=3 windows to be a meaningful mean
                    continue
                mean = folded[keep].mean(0)
                color = cm.viridis(k / max(1, len(FREQS) - 1))
                final = d.get("final_ppl")
                ax.plot(np.arange(freq), mean, color=color, linewidth=1.4,
                        label=f"freq={freq}  ppl={final:.0f}  n_win={keep.sum()}"
                              if final else f"freq={freq}  n_win={keep.sum()}")
            ax.axhline(0, color='black', linewidth=0.4, alpha=0.4)
            ax.set_title(f"{variant.replace('KFAC','')}  lr={lr}", fontsize=10)
            ax.set_xlabel('age (step within K-FAC window;  0 = refresh)')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=7)
            if j == 0:
                ax.set_ylabel('mean delta_loss')

    fig.suptitle(f"gamma=0, mom=0  —  mean delta_loss vs age  (windows starting at step >= {START_STEP})",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = RES / "clean_mean_delta_vs_age.png"
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
