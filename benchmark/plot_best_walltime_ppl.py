"""
benchmark/plot_best_walltime_ppl.py

Plot training-PPL vs cumulative wall time for the BEST configuration of each
method, comparing AdamW (best: lr=3e-3, wd=0.1, beta2=0.95) against
Classic K-FAC and Vered K-FAC at their best wd values (wd=0.1 for both).

Loads from:
  - benchmark/results/adamw_tune_transformer_fp32_lr3e-03_wd0.1.json (AdamW)
  - benchmark/results/kfac_wd_classickfac_wd0.1.json                 (Classic)
  - benchmark/results/kfac_wd_veredkfac_wd0.1.json                   (Vered)

Handles JSON files that may contain two concatenated objects (recovers the
first one).

Output: benchmark/results/best_walltime_vs_ppl.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RES = Path(__file__).resolve().parent / "results"


def safe_load(p: Path):
    """Load JSON, recovering the first object if the file has trailing junk."""
    text = p.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj


def smooth(x, w=20):
    if len(x) < w or w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / float(w)


# (display_label, file_path, color)
CONFIGS = [
    ("AdamW (lr=3e-3, wd=0.1, β₂=0.95)",
     RES / "adamw_tune_transformer_fp32_lr3e-03_wd0.1.json",
     "#888888"),
    ("Classic K-FAC (wd=0.1)",
     RES / "kfac_wd_classickfac_wd0.1.json",
     "#cc4444"),
    ("Vered K-FAC (wd=0.1)",
     RES / "kfac_wd_veredkfac_wd0.1.json",
     "#2266aa"),
]


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    summary = []
    for label, path, color in CONFIGS:
        if not path.exists():
            print(f"missing: {path}")
            continue
        d = safe_load(path)
        recs = d.get("per_step", [])
        if not recs:
            print(f"no per_step records in {path.name}")
            continue
        losses = np.array([r["loss"] for r in recs])
        ppls = np.exp(np.minimum(losses, 30))
        wall_total = d.get("wall_s", 0.0)
        n = len(losses)
        # Per-step wall time (we only have total; linear interpolation)
        per_step_wall = wall_total / max(n, 1)
        cum_wall_min = np.arange(1, n + 1) * per_step_wall / 60.0
        ppl_smooth = smooth(ppls, w=20)
        wall_smooth = cum_wall_min[19:] if len(cum_wall_min) >= 20 else cum_wall_min
        final_ppl = d.get("final_ppl") or float(ppls[-1])
        ax.plot(wall_smooth, ppl_smooth, color=color, linewidth=2.0,
                label=f"{label}  →  final ppl {final_ppl:.0f}, wall {wall_total/60:.1f} min")
        # Mark the final point with a star
        ax.scatter([cum_wall_min[-1]], [final_ppl], marker="*", s=160,
                   color=color, edgecolor="black", linewidth=0.8, zorder=5)
        summary.append((label, final_ppl, wall_total / 60))

    # Annotate the AdamW horizontal reference
    adamw_ppl = next((p for label, p, _ in summary if "AdamW" in label), None)
    if adamw_ppl is not None:
        ax.axhline(adamw_ppl, color="#444444", linestyle="--",
                   linewidth=0.8, alpha=0.5)
        ax.text(0.5, adamw_ppl * 1.03, f"  AdamW final ppl ≈ {adamw_ppl:.0f}",
                fontsize=8, va="bottom", color="#444")

    ax.set_xlabel("cumulative wall time (minutes)")
    ax.set_ylabel("training perplexity (smoothed, lower is better)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Best configuration of each method: training PPL vs wall time\n"
                 "(SmallGPT-medium, WikiText-2, fp32, seed 42, 1000 steps)",
                 fontsize=10)
    fig.tight_layout()
    out = RES / "best_walltime_vs_ppl.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")

    print("\n=== final summary ===")
    print(f"  {'method':>40}  {'final_ppl':>10}  {'wall_min':>9}")
    for label, ppl, wall in summary:
        print(f"  {label:>40}  {ppl:>10.0f}  {wall:>9.1f}")


if __name__ == "__main__":
    main()
