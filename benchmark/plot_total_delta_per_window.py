"""
benchmark/plot_total_delta_per_window.py

For every run we have with per-step loss data, fold into K-FAC windows
(size = each run's factor_update_freq) and plot the SUM of delta_loss
within each window vs window index.  Each window's total delta =
loss[end] - loss[start] = how much loss progress that K-FAC cycle made.

Sources:
    benchmark/results/per_step_VeredKFAC_champion_s1000.json     (per_step list)
    benchmark/results/per_step_ClassicKFAC_champion_s1000.json   (per_step list)
    benchmark/results/per_step_*.json                            (any other per_step runs)
    benchmark/results/vered_wgso_perstep_d*_s1000_const.json     (per_step list)
    benchmark/results/vered_wgso_lr_d*_lr*_s1000_const.json      (train_losses)
    benchmark/results/vered_wgso_freq*_d*_lr*_s1000_const.json   (train_losses)
    benchmark/results/vered_freq*_d*_lr*_s1000_const.json        (train_losses)

Output: benchmark/results/total_delta_per_window.png
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmark" / "results"


def gather_losses(jpath: Path):
    """Return (per_step_losses np.ndarray, freq int, label str) or None."""
    try:
        d = json.loads(jpath.read_text())
    except Exception:
        return None
    # Two shapes: per_step list-of-dicts (per_step_* and vered_wgso_perstep_*),
    # or run_probe with train_losses (wgso_lr, freq sweeps, etc.)
    per_step = d.get("per_step")
    if per_step:
        losses = np.array([r["loss"] for r in per_step])
        cfg = d.get("config") or {}
        freq = cfg.get("factor_update_freq") or 20
        damping = d.get("damping", cfg.get("damping"))
        variant = d.get("variant", cfg.get("variant", "?"))
        wgso = d.get("wgso", cfg.get("wgso", False))
        label = f"{variant}{' WGSO' if wgso else ''}  d={damping:.0e}  freq={freq}"
        final = d.get("final_ppl")
        return losses, freq, label, final
    # run_probe shape
    res = d.get("result", {})
    losses = res.get("train_losses") or []
    if not losses:
        return None
    losses = np.array(losses)
    cfg = d.get("config") or {}
    freq = cfg.get("factor_update_freq") or 20
    damping = cfg.get("damping")
    lr = cfg.get("kfac_lr")
    wgso = cfg.get("wgso", False)
    final = res.get("final_ppl")
    label = f"Vered{' WGSO' if wgso else ''}  d={damping:.0e}  lr={lr:.0e}  freq={freq}"
    return losses, freq, label, final


def main():
    pats = [
        "per_step_*.json",
        "vered_wgso_perstep_d*_s1000_const.json",
        "vered_wgso_lr_d*_lr*_s1000_const.json",
        "vered_wgso_freq*_d*_lr*_s1000_const.json",
        "vered_freq*_d*_lr*_s1000_const.json",
    ]
    seen = set()
    runs = []
    for pat in pats:
        for f in sorted(RES.glob(pat)):
            if f in seen:
                continue
            seen.add(f)
            r = gather_losses(f)
            if r is not None and len(r[0]) >= 2 * r[1]:
                runs.append(r)

    if not runs:
        print("No runs with per-step loss data found.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    for i, (losses, freq, label, final) in enumerate(runs):
        n_win = len(losses) // freq
        # total delta per window = loss[end_of_window] - loss[start_of_window]
        # where start = window i * freq, end = (i+1) * freq - 1
        starts = losses[np.arange(n_win) * freq]
        ends   = losses[np.arange(n_win) * freq + freq - 1]
        totals = ends - starts
        win_idx = np.arange(n_win)
        ax.plot(win_idx, totals, marker='.', markersize=3, linewidth=1.0,
                color=cmap(i % 10),
                label=f"{label}  final_ppl={final:.0f}" if final else label)

    ax.axhline(0, color='black', linewidth=0.5, alpha=0.4)
    ax.set_xlabel("K-FAC window index")
    ax.set_ylabel("total delta_loss per window  (end - start)")
    ax.set_title("Per-K-FAC-window training progress across runs (negative = loss dropped that cycle)")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=1)
    fig.tight_layout()
    out = RES / "total_delta_per_window.png"
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out.name}  ({len(runs)} runs plotted)")


if __name__ == "__main__":
    main()
