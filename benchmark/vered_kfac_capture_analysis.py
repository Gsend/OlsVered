"""
benchmark/vered_kfac_capture_analysis.py

Offline matrix-level analysis of matched Vered/Classic captures.

For each captured step, reconstructs each variant's cached factor state in
float64, computes a high-precision reference natural gradient via direct
linear solves, then produces a set of element-wise DIFFERENCE matrices:

    A_diff       = A_v - A_c                  (in x in)
    G_diff       = G_v - G_c                  (out x out)
    ref_diff     = natgrad_v_f64 - natgrad_c_f64  (out x in)
    ng32_diff    = natgrad_v_f32 - natgrad_c_f32  (out x in)
    err_vered    = natgrad_v_f32 - natgrad_v_f64  (out x in)
    err_classic  = natgrad_c_f32 - natgrad_c_f64  (out x in)

For each matrix we report min / max / abs_max / mean / std / abs percentiles
and the top-k outlier locations.  All matrices are also saved as PNG
heatmaps under benchmark/results/captures/heatmaps/.

Heatmap interpretation cheat-sheet:
    - uniform color    -> systematic bias / scale difference
    - banded rows      -> specific output neurons drift between variants
    - banded columns   -> specific input directions drift
    - high-contrast pixels in a sea of zeros -> per-element outliers,
      probably finite-precision instabilities near zero pivots

Usage:
    python benchmark/vered_kfac_capture_analysis.py ^
        --vered   capture_VeredKFAC_m0.90_lr8e-03_layer_<name>.pt ^
        --classic capture_ClassicKFAC_m0.90_lr8e-03_layer_<name>.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAP_OUT      = ROOT / "benchmark" / "results" / "captures"
HEATMAP_OUT  = CAP_OUT / "heatmaps"
HEATMAP_OUT.mkdir(parents=True, exist_ok=True)


# ---------- matrix helpers --------------------------------------------------

def reconstruct_A_from_R(R: torch.Tensor) -> torch.Tensor:
    R64 = R.to(torch.float64)
    return R64.T @ R64


def reconstruct_A_from_Ainv(A_inv: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(A_inv.to(torch.float64))


def reference_natgrad(dW: torch.Tensor, A_plus_lam: torch.Tensor,
                       G_plus_lam: torch.Tensor) -> torch.Tensor:
    """Compute G^{-1} dW A^{-1} in float64 via direct solves."""
    dW64 = dW.to(torch.float64)
    tmp = torch.linalg.solve(G_plus_lam, dW64)
    return torch.linalg.solve(A_plus_lam, tmp.T).T


def diff_stats(M: torch.Tensor) -> Dict[str, float]:
    """Outlier-focused stats on a difference matrix."""
    arr = M.detach().cpu().to(torch.float64).numpy().ravel()
    abs_arr = np.abs(arr)
    std = float(arr.std())
    return {
        "n":            int(arr.size),
        "min":          float(arr.min()),
        "max":          float(arr.max()),
        "abs_max":      float(abs_arr.max()),
        "mean":         float(arr.mean()),
        "std":          std,
        "p50_abs":      float(np.percentile(abs_arr, 50)),
        "p95_abs":      float(np.percentile(abs_arr, 95)),
        "p99_abs":      float(np.percentile(abs_arr, 99)),
        "p999_abs":     float(np.percentile(abs_arr, 99.9)),
        "frac_gt_3std": float((abs_arr > 3 * std).mean()) if std > 0 else 0.0,
        "frac_gt_5std": float((abs_arr > 5 * std).mean()) if std > 0 else 0.0,
    }


def top_outliers(M: torch.Tensor, k: int = 8) -> List[Tuple[int, int, float]]:
    """Top-k entries of M by |value|; returns list of (row, col, signed_value)."""
    M_cpu = M.detach().cpu().to(torch.float64)
    flat_abs = torch.abs(M_cpu).flatten()
    k = min(k, flat_abs.numel())
    _, idx = torch.topk(flat_abs, k=k)
    rows = (idx // M_cpu.shape[-1]).tolist()
    cols = (idx %  M_cpu.shape[-1]).tolist()
    vals = [float(M_cpu[r, c].item()) for r, c in zip(rows, cols)]
    return list(zip(rows, cols, vals))


# ---------- plotting --------------------------------------------------------

def plot_one(ax, M: torch.Tensor, title: str, clip_pct: float = 99.5):
    arr = M.detach().cpu().to(torch.float64).numpy()
    abs_arr = np.abs(arr)
    if abs_arr.max() == 0:
        vmax = 1e-30
    else:
        vmax = float(np.percentile(abs_arr, clip_pct))
        if vmax == 0:
            vmax = abs_arr.max()
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_step_heatmaps(step: int, mats: Dict[str, torch.Tensor],
                       out_path: Path, suptitle: str):
    """Save a multi-subplot PNG of all diff matrices for one step."""
    items = [(label, M) for label, M in mats.items() if M is not None]
    n = len(items)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flatten() if rows > 1 or cols > 1 else [axes]
    for ax, (label, M) in zip(axes_flat, items):
        plot_one(ax, M, label)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- per-variant analysis -------------------------------------------

def analyse_one(cap_struct: Dict, variant: str, label: str) -> Dict:
    print(f"\n=== {label} ({variant}) ===")
    print(f"  Layer: {cap_struct['layer_name']}  "
          f"in={cap_struct['in_features']}, out={cap_struct['out_features']}")
    print(f"  val_ppl trajectory: " +
          ", ".join(f"(step={s},ppl={p:.0f})"
                    for s, p in cap_struct["val_ppl_history"][:8]))

    out: Dict = {"by_step": {}, "variant": variant, "layer": cap_struct["layer_name"]}
    for cap in cap_struct["captures"]:
        step = cap["step"]
        dW = cap["dW"]
        ng32 = cap["natgrad"]

        # Pick the factor snapshot that was actually used in this step's apply.
        # _update_factors() runs when step_count % factor_update_freq == 1.
        # factor_update_freq=20 (the screen default) -> refresh at step 1,21,...
        use_post = (step - 1) % 20 == 0
        factors = cap["factors_post"] if use_post else cap["factors_pre"]
        if factors is None or len(factors) == 0:
            print(f"  step={step}: NO CACHED FACTORS, skipping")
            continue

        if variant == "VeredKFAC":
            A_plus = reconstruct_A_from_R(factors["R_X"])
            G_plus = reconstruct_A_from_R(factors["R_G"])
        else:
            A_plus = reconstruct_A_from_Ainv(factors["A_inv"])
            G_plus = reconstruct_A_from_Ainv(factors["G_inv"])

        ng64 = reference_natgrad(dW, A_plus, G_plus)
        err = ng32.to(torch.float64) - ng64

        out["by_step"][step] = {
            "A_plus_lam_f64": A_plus,
            "G_plus_lam_f64": G_plus,
            "natgrad_f32":    ng32,
            "natgrad_f64":    ng64,
            "self_err":       err,
            "dW":             dW,
            "loss":           cap["loss"],
            "factor_src":     "post" if use_post else "pre",
        }
        A_eigs = torch.linalg.eigvalsh(A_plus)
        G_eigs = torch.linalg.eigvalsh(G_plus)
        out["by_step"][step]["kappa_A"] = float(A_eigs[-1] / A_eigs[0].clamp(min=1e-30))
        out["by_step"][step]["kappa_G"] = float(G_eigs[-1] / G_eigs[0].clamp(min=1e-30))

        s = diff_stats(err)
        print(f"  step={step:4d}  loss={cap['loss']:.3f}  "
              f"||dW||_F={torch.linalg.norm(dW):.3e}  "
              f"k(A)={out['by_step'][step]['kappa_A']:.2e}  "
              f"k(G)={out['by_step'][step]['kappa_G']:.2e}")
        print(f"           self-err (f32 vs own f64) abs_max={s['abs_max']:.3e} "
              f"std={s['std']:.3e} p99={s['p99_abs']:.3e} "
              f"frac>5std={s['frac_gt_5std']:.2e}")

    return out


# ---------- pairwise comparison --------------------------------------------

def print_stat_table(label: str, stats: Dict[str, float], outliers: List[Tuple[int,int,float]]):
    print(f"  {label}")
    print(f"    n={stats['n']:>9d}   min={stats['min']:>+11.3e}   max={stats['max']:>+11.3e}   abs_max={stats['abs_max']:.3e}")
    print(f"    mean={stats['mean']:>+11.3e}  std={stats['std']:.3e}   "
          f"p50_abs={stats['p50_abs']:.3e}   p95_abs={stats['p95_abs']:.3e}   "
          f"p99_abs={stats['p99_abs']:.3e}   p99.9_abs={stats['p999_abs']:.3e}")
    print(f"    frac >3*std = {stats['frac_gt_3std']:.3e}   "
          f"frac >5*std = {stats['frac_gt_5std']:.3e}")
    if outliers:
        print(f"    top outliers (row, col, value): " +
              ", ".join(f"({r},{c},{v:+.2e})" for r,c,v in outliers[:6]))


def compare_pair(vered_res: Dict, classic_res: Dict, mom: float, lr: float):
    print("\n" + "=" * 78)
    print(f"=== Pairwise matrix analysis: Vered vs Classic  "
          f"(mom={mom}, lr={lr:g}, layer={vered_res['layer']}) ===")
    print("=" * 78)

    common_steps = sorted(set(vered_res["by_step"]) & set(classic_res["by_step"]))
    if not common_steps:
        print("No matched steps.")
        return

    for step in common_steps:
        v = vered_res["by_step"][step]
        c = classic_res["by_step"][step]
        Av, Ac = v["A_plus_lam_f64"], c["A_plus_lam_f64"]
        Gv, Gc = v["G_plus_lam_f64"], c["G_plus_lam_f64"]

        A_diff      = Av - Ac
        G_diff      = Gv - Gc
        ref_diff    = v["natgrad_f64"] - c["natgrad_f64"]
        ng32_diff   = v["natgrad_f32"].to(torch.float64) - c["natgrad_f32"].to(torch.float64)
        err_vered   = v["self_err"]
        err_classic = c["self_err"]

        print(f"\n--- step {step}  "
              f"(Vered loss={v['loss']:.3f}, Classic loss={c['loss']:.3f}, "
              f"k(A)_v={v['kappa_A']:.2e}, k(A)_c={c['kappa_A']:.2e}) ---")

        print_stat_table("A_diff      (V minus C, cached input factor)",
                         diff_stats(A_diff), top_outliers(A_diff))
        print_stat_table("G_diff      (V minus C, cached output factor)",
                         diff_stats(G_diff), top_outliers(G_diff))
        print_stat_table("ref_diff    (V f64 ref minus C f64 ref, math gap)",
                         diff_stats(ref_diff), top_outliers(ref_diff))
        print_stat_table("ng32_diff   (V f32 minus C f32, applied gap)",
                         diff_stats(ng32_diff), top_outliers(ng32_diff))
        print_stat_table("err_vered   (V f32 minus V f64, V's own quant error)",
                         diff_stats(err_vered), top_outliers(err_vered))
        print_stat_table("err_classic (C f32 minus C f64, C's own quant error)",
                         diff_stats(err_classic), top_outliers(err_classic))

        # Save heatmaps for this step
        layer_slug = vered_res["layer"].replace(".", "-")
        out_path = HEATMAP_OUT / (
            f"diff_heatmap_m{mom:.2f}_lr{lr:.0e}_{layer_slug}_step{step}.png"
        )
        save_step_heatmaps(step, {
            "A_diff (in x in)":               A_diff,
            "G_diff (out x out)":             G_diff,
            "ref_diff (out x in)":            ref_diff,
            "ng32_diff (out x in)":           ng32_diff,
            "err_vered (out x in)":           err_vered,
            "err_classic (out x in)":         err_classic,
        }, out_path,
           suptitle=f"step {step}  mom={mom}  lr={lr:g}  layer={vered_res['layer']}")
        print(f"  heatmap -> {out_path.relative_to(ROOT)}")


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--vered",   type=str, default="")
    ap.add_argument("--classic", type=str, default="")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and not path.exists():
            path = CAP_OUT / p
        return path

    vered_res: Optional[Dict] = None
    classic_res: Optional[Dict] = None
    mom = None; lr = None

    if args.vered:
        v = torch.load(resolve(args.vered), weights_only=False)
        vered_res = analyse_one(v, v["args"]["variant"], "Vered")
        mom = v["args"]["mom"]; lr = v["args"]["lr"]

    if args.classic:
        c = torch.load(resolve(args.classic), weights_only=False)
        classic_res = analyse_one(c, c["args"]["variant"], "Classic")
        if mom is None: mom = c["args"]["mom"]; lr = c["args"]["lr"]

    if vered_res and classic_res:
        compare_pair(vered_res, classic_res, mom, lr)


if __name__ == "__main__":
    main()
