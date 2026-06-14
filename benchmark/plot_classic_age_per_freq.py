"""
benchmark/plot_classic_age_per_freq.py

For each Classic freq run at the champion cell (lr=2e-3, damping=1e-4,
mom=0.7, gamma=0.9), plot mean delta_loss vs age (step-within-window)
in a separate subplot.

Sources:
    freq=20:  classic_2dscreen_matched_m0.70_lr2e-03_s1000_const.json
    freq=50,100,200,500: classic_freq{freq}_d1e-04_lr2e-03_s1000_const.json

Output: benchmark/results/classic_delta_vs_age_per_freq.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "benchmark" / "results"


def load_losses(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    # per_step shape (per_step_* JSONs from per_step_loss_compare.py)
    per_step = d.get("per_step")
    if per_step:
        losses = np.array([r["loss"] for r in per_step])
        final = d.get("final_ppl")
        return (losses, final) if len(losses) > 0 else None
    # run_probe shape (matched-screen, freq sweeps)
    r = d.get("result") or {}
    losses = np.array(r.get("train_losses") or [])
    final = r.get("final_ppl")
    return (losses, final) if len(losses) > 0 else None


def main():
    cells = [
        (20,  RES / "per_step_ClassicKFAC_champion_s1000.json"),  # matches the variant-comparison delta plot
        (50,  RES / "classic_freq50_d1e-04_lr2e-03_s1000_const.json"),
        (100, RES / "classic_freq100_d1e-04_lr2e-03_s1000_const.json"),
        (200, RES / "classic_freq200_d1e-04_lr2e-03_s1000_const.json"),
        (500, RES / "classic_freq500_d1e-04_lr2e-03_s1000_const.json"),
    ]
    available = []
    for freq, p in cells:
        r = load_losses(p)
        if r is not None and len(r[0]) >= 2 * freq:
            available.append((freq, r[0], r[1]))
        else:
            print(f"  [skip] freq={freq}: {p.name} missing or too short")

    if not available:
        print("No Classic freq runs available.")
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (freq, losses, final) in zip(axes, available):
        dl = np.concatenate([[0.0], np.diff(losses)])
        n_win = len(dl) // freq
        folded = dl[:n_win * freq].reshape(n_win, freq)
        mean = folded.mean(0)
        std = folded.std(0)
        positions = np.arange(freq)
        ax.plot(positions, mean, color='tab:orange', linewidth=2.0,
                label=f"mean of {n_win} windows")
        ax.fill_between(positions, mean - std, mean + std,
                        color='tab:orange', alpha=0.18, label='±1σ')
        ax.axhline(0, color='black', linewidth=0.4, alpha=0.4)
        ax.set_title(f"Classic  freq={freq}   final_ppl={final:.0f}" if final
                     else f"Classic  freq={freq}")
        ax.set_xlabel('age (step within K-FAC window;  0 = refresh)')
        ax.set_ylabel('mean delta_loss')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out = RES / "classic_delta_vs_age_per_freq.png"
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out.name}  ({n} freq panels)")


if __name__ == "__main__":
    main()
