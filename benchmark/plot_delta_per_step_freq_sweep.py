"""
benchmark/plot_delta_per_step_freq_sweep.py

Plots per-step delta_loss (smoothed) across the freq sweeps so the
effect of refresh cadence is visible. One panel per variant / regime.

Looks at:
    vered_freq*_d*_lr*_s1000_const.json       (Vered, no WGSO)
    vered_wgso_freq*_d*_lr*_s1000_const.json  (Vered, WGSO)
    classic_freq*_d*_lr*_s1000_const.json     (Classic)

Output: benchmark/results/delta_per_step_freq_sweep.png
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmark" / "results"

SMOOTH_WIN = 25  # rolling mean window over steps


def rolling_mean(x, w):
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    pad = np.full(w - 1, out[0])
    return np.concatenate([pad, out])


def load_run(path: Path):
    d = json.loads(path.read_text())
    cfg = d.get("config") or {}
    res = d.get("result") or {}
    losses = np.array(res.get("train_losses") or [])
    if len(losses) < 50:
        return None
    return {
        "freq":  cfg.get("factor_update_freq", 20),
        "lr":    cfg.get("kfac_lr"),
        "damp":  cfg.get("damping"),
        "final": res.get("final_ppl"),
        "delta": np.concatenate([[0.0], np.diff(losses)]),
    }


GROUPS = [
    ("Vered (no WGSO)",  "vered_freq*_d*_lr*_s1000_const.json",       None),
    ("Vered (WGSO)",     "vered_wgso_freq*_d*_lr*_s1000_const.json",  None),
    ("Classic",          "classic_freq*_d*_lr*_s1000_const.json",     None),
]

# Include matched-screen freq=20 baseline anchors where appropriate.
ANCHORS = {
    "Vered (no WGSO)":  ("vered_2dscreen_m0.70_lr2e-03_s1000_const.json", "freq=20 baseline (ppl=921)"),
    "Vered (WGSO)":     ("vered_wgso_lr_d1e-06_lr2e-03_s1000_const.json", "freq=20 WGSO best (ppl=923)"),
    "Classic":          ("classic_2dscreen_matched_m0.70_lr2e-03_s1000_const.json", "freq=20 baseline (ppl=922)"),
}


def main():
    groups_with_data = []
    for label, pat, _ in GROUPS:
        files = sorted(RES.glob(pat))
        runs = [load_run(f) for f in files]
        runs = [r for r in runs if r]
        anchor_path, anchor_label = ANCHORS.get(label, (None, None))
        if anchor_path:
            ap = RES / anchor_path
            if ap.exists():
                a = load_run(ap)
                if a is not None:
                    a["_anchor_label"] = anchor_label
                    runs.insert(0, a)
        if runs:
            groups_with_data.append((label, runs))

    if not groups_with_data:
        print("No freq-sweep runs found yet.")
        return

    n = len(groups_with_data)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, runs) in zip(axes, groups_with_data):
        # Sort by freq for consistent legend order.
        runs.sort(key=lambda r: r["freq"])
        for i, r in enumerate(runs):
            smoothed = rolling_mean(r["delta"], SMOOTH_WIN)
            steps = np.arange(len(smoothed))
            color = cm.viridis(i / max(1, len(runs) - 1))
            tag = r.get("_anchor_label") or \
                  f"freq={r['freq']}  ppl={r['final']:.0f}" if r['final'] else f"freq={r['freq']}"
            ax.plot(steps, smoothed, color=color, linewidth=1.2, alpha=0.85,
                    label=tag)
        ax.axhline(0, color='black', linewidth=0.4, alpha=0.4)
        ax.set_title(f"{label}: per-step delta_loss (rolling mean, window={SMOOTH_WIN}) by factor_update_freq")
        ax.set_ylabel("delta_loss (smoothed)")
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    out = RES / "delta_per_step_freq_sweep.png"
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out.name}  ({sum(len(rs) for _, rs in groups_with_data)} runs total)")


if __name__ == "__main__":
    main()
