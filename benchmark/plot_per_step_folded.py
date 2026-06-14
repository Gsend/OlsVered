"""
benchmark/plot_per_step_folded.py

Reads the per_step_VeredKFAC and per_step_ClassicKFAC champion JSONs and
emits two plots, each overlaying both variants:

    vered_classic_loss_folded_by_kfac_window.png  -- raw loss
    vered_classic_delta_folded_by_kfac_window.png -- delta loss

Top panel: all K-FAC windows overlaid (freq=20).
Bottom panel: late training only (windows 30..49) for clearer steady-state shape.

Usage: python benchmark/plot_per_step_folded.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmark" / "results"
FREQ = 20

VARIANTS = [
    ("VeredKFAC",   "Vered",   "tab:blue",   "-",  2.4),
    ("ClassicKFAC", "Classic", "tab:orange", "--", 1.6),
]


def load(variant: str):
    p = RES / f"per_step_{variant}_champion_s1000.json"
    if not p.exists():
        raise FileNotFoundError(p)
    d = json.loads(p.read_text())
    per_step = d.get("per_step", [])
    losses = np.array([r["loss"] for r in per_step])
    deltas = np.array([r["delta_loss"] for r in per_step])
    final = d.get("final_ppl")
    return losses, deltas, final


def fold(arr: np.ndarray) -> np.ndarray:
    n = (len(arr) // FREQ) * FREQ
    return arr[:n].reshape(-1, FREQ)


def make_plot(kind: str, get_series):
    """kind: 'loss' or 'delta'."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    positions = np.arange(FREQ)
    for variant, label, color, style, lw in VARIANTS:
        try:
            series, final = get_series(variant)
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
            continue
        folded = fold(series)
        n_win = folded.shape[0]
        # Top: all-windows mean + a shaded band for std
        mean = folded.mean(0)
        std = folded.std(0)
        ax1.plot(positions, mean, color=color, linewidth=lw, linestyle=style,
                 label=f"{label}  final_ppl={final:.0f}  (mean of {n_win} windows)")
        ax1.fill_between(positions, mean - std, mean + std,
                         color=color, alpha=0.12)
        # Bottom: late windows mean + std
        late = folded[30:]
        mean_l = late.mean(0)
        std_l = late.std(0)
        ax2.plot(positions, mean_l, color=color, linewidth=lw, linestyle=style,
                 label=f"{label}  (late mean, windows 30-49)")
        ax2.fill_between(positions, mean_l - std_l, mean_l + std_l,
                         color=color, alpha=0.12)

    ax1.set_xlabel("step within K-FAC window  (0 = refresh step)")
    ax1.set_xticks([0, 5, 10, 15, 19])
    ax1.grid(alpha=0.3, which='both')
    ax1.legend(loc='upper right')
    ax2.set_xlabel("step within K-FAC window")
    ax2.set_xticks([0, 5, 10, 15, 19])
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')

    if kind == 'loss':
        ax1.set_ylabel("train loss")
        ax1.set_yscale('log')
        ax2.set_ylabel("train loss")
        ax1.set_title("Raw loss folded by K-FAC window (freq=20) — champion cell, no WGSO")
        ax2.set_title("Late training only (windows 30-49)")
        out = RES / "vered_classic_loss_folded_by_kfac_window.png"
    else:
        ax1.axhline(0, color='black', linewidth=0.5, alpha=0.4)
        ax2.axhline(0, color='black', linewidth=0.5, alpha=0.4)
        ax1.set_ylabel("delta loss")
        ax2.set_ylabel("delta loss")
        ax1.set_title("Delta loss folded by K-FAC window (freq=20) — champion cell, no WGSO")
        ax2.set_title("Late training only (windows 30-49)")
        out = RES / "vered_classic_delta_folded_by_kfac_window.png"

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out.name}")


def main():
    # Raw loss
    make_plot('loss',
              lambda v: (load(v)[0], load(v)[2]))
    # Delta loss
    make_plot('delta',
              lambda v: (load(v)[1], load(v)[2]))


if __name__ == "__main__":
    main()
