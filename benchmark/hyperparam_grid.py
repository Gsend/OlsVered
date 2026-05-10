"""
benchmark/hyperparam_grid.py

Two-stage hyperparameter grid search for ClassicKFAC and VeredKFAC on the
SmallGPT / WikiText-2 task.

  Stage 1 (screen):  Phase-1-style probes on the full grid, 1000 steps each.
  Stage 2 (focus):   Phase-2-style 5000-step convergence runs on the top K
                      configs per variant from Stage 1.

The grid varies LR, damping, and (for Vered) gamma.  Other parameters
(momentum, grad_clip, factor_update_freq) stay fixed at their post-fix
deployment values for fair comparison.

This benchmark exists to find post-EMA-fix optima for VeredKFAC: the prior
tuning was done under the buggy implementation and may no longer be optimal.
ClassicKFAC is included to confirm its post-_SEQ_SUBSAMPLE-bump baseline.

Usage:
    python benchmark/hyperparam_grid.py --stage all
    python benchmark/hyperparam_grid.py --stage 1   # screen only
    python benchmark/hyperparam_grid.py --stage 2   # focus only (needs stage 1)
    python benchmark/hyperparam_grid.py --variants VeredKFAC  # only run Vered

Output (under benchmark/results/):
    hpgrid_stage1_screen.json   per-config Stage-1 results
    hpgrid_stage2_focus.json    Stage-2 convergence runs on top configs
    hpgrid_summary.csv          flat table of all configs + their final_ppl
    hpgrid_topk.png             trajectory plot of top configs
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import (
    build_data, run_probe, OUT, PPL_TARGETS, time_to_ppl,
)
from benchmark.gpu_benchmark import get_device, get_hardware_info

# ============================================================================
#  Grid definitions
# ============================================================================

# Fixed across all configs (for fair comparison)
MOMENTUM    = 0.9
GRAD_CLIP   = 60.0     # all variants use the same clip; isolates LR/damp/gamma
KFAC_LR_REF = 8e-3     # not used directly; just the documented "current" LR

# Per-variant grids.  Lists are searched as Cartesian products.
GRIDS: Dict[str, Dict[str, List[float]]] = {
    "ClassicKFAC": {
        "lr":       [8e-3],
        "damping":  [1e-6, 1e-5, 1e-4, 1e-3],
        "gamma":    [0.5],
        "momentum": [0.9],                      # Classic's classical config
    },
    "VeredKFAC": {
        # The post-fix loss vs pre-fix is likely driven by total-averaging
        # being too heavy: EMA(gamma=0.7) + momentum(0.9) compounds.
        # Pre-fix's buggy EMA had effective gamma~0.49, which compensated.
        # Post-fix needs either lower gamma OR lower momentum (or both) to
        # match the buggy version's effective smoothing budget.
        # Vered's per-step direction is ~140x cleaner per the synthetic test,
        # so it shouldn't need momentum's noise-averaging at all.
        "lr":       [8e-3],
        "damping":  [1e-7, 1e-6, 1e-5],
        "gamma":    [0.3, 0.5, 0.7],
        "momentum": [0.0, 0.5, 0.9],            # 0 = trust per-step direction
    },
}

STAGE1_STEPS  = 1000   # screening probe length
STAGE2_STEPS  = 5000   # focus convergence length
PHASE2_TOP_K  = 2       # top configs per variant promoted to Stage 2


# ============================================================================
#  Grid expansion + probe runner
# ============================================================================

def expand_grid(variant: str) -> List[Dict[str, float]]:
    """All Cartesian-product configs for a variant."""
    grid = GRIDS[variant]
    keys = list(grid.keys())
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        cfg["variant"] = variant
        cfg["clip"]    = GRAD_CLIP
        # Use grid's momentum if specified; otherwise the global default
        if "momentum" not in cfg:
            cfg["momentum"] = MOMENTUM
        out.append(cfg)
    return out


def config_id(cfg: Dict) -> str:
    """Stable per-config identifier for resume / dedup."""
    return (f"{cfg['variant']}_lr{cfg['lr']:.0e}_d{cfg['damping']:.0e}_"
            f"g{cfg['gamma']:.2f}_m{cfg['momentum']:.1f}_c{cfg['clip']:.0f}")


def run_one_config(
    cfg: Dict,
    n_steps: int,
    train_loader_factory: Callable,
    val_loader,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    record_natgrad: bool,
    record_condition: bool,
) -> Dict:
    """Run a single grid-cell probe for the given config."""
    print(f"\n  Running {config_id(cfg)} ({n_steps} steps) ...")
    res = run_probe(
        variant=cfg["variant"],
        kfac_lr=cfg["lr"],
        damping=cfg["damping"],
        momentum=cfg["momentum"],
        max_steps=n_steps,
        vocab_size=vocab_size,
        train_loader_factory=train_loader_factory,
        val_loader=val_loader,
        pad_id=pad_id,
        device=device,
        seed=42,
        record_natgrad=record_natgrad,
        record_condition=record_condition,
        condition_log_every=200,
        print_progress=(n_steps >= 5000),   # only verbose on long runs
        grad_clip=cfg["clip"],
        gamma=cfg["gamma"],
    )
    res["config_id"] = config_id(cfg)
    res["gamma"]     = cfg["gamma"]   # also record in result for analysis
    return res


# ============================================================================
#  Stage 1 - grid screen
# ============================================================================

def stage1_screen(
    variants: List[str],
    train_loader_factory: Callable,
    val_loader,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    hw: dict,
) -> Dict:
    out_path = OUT / "hpgrid_stage1_screen.json"

    # Resume support
    completed: Dict[str, Dict] = {}      # config_id -> result
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            completed = {r["config_id"]: r for r in (prior.get("results") or [])}
            print(f"  Resuming Stage 1 from {out_path.name}: "
                  f"{len(completed)} configs already done")
        except (json.JSONDecodeError, KeyError):
            print(f"  WARN: could not parse {out_path.name}; starting fresh")

    all_configs: List[Dict] = []
    for v in variants:
        all_configs.extend(expand_grid(v))

    n_total = len(all_configs)
    print("\n" + "=" * 70)
    print(f"  STAGE 1 SCREEN: {n_total} configs x {STAGE1_STEPS} steps")
    print("=" * 70)
    for v in variants:
        n = len(expand_grid(v))
        print(f"    {v:<14}  {n} configs")

    results: List[Dict] = list(completed.values())

    def _save():
        out = {
            "hw":          hw,
            "stage":       1,
            "stage1_steps": STAGE1_STEPS,
            "fixed":       {"momentum": MOMENTUM, "grad_clip": GRAD_CLIP},
            "grids":       GRIDS,
            "results":     results,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=str))
        os.replace(tmp, out_path)

    for i, cfg in enumerate(all_configs):
        cid = config_id(cfg)
        if cid in completed:
            print(f"  [{i+1}/{n_total}] {cid}: skipping (cached)")
            continue
        print(f"  [{i+1}/{n_total}] {cid}")
        res = run_one_config(
            cfg, STAGE1_STEPS, train_loader_factory, val_loader,
            vocab_size, pad_id, device,
            record_natgrad=False, record_condition=False,
        )
        results.append(res)
        completed[cid] = res
        # Drop heavy curves from in-memory dict (kept in JSON via results list)
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl else f"DIVERGED@{res.get('diverge_step')}"
        ms = res.get("median_step_ms")
        ms_str = f"{ms:.0f}ms" if ms else "n/a"
        print(f"      -> ppl={ppl_str}  step={ms_str}  "
              f"wall={res['wall_s']/60:.1f}m  status={res['status']}")
        _save()

    # Print top-k summary per variant
    print("\n" + "=" * 70)
    print("  Stage 1 results — top configs per variant:")
    print("=" * 70)
    by_variant: Dict[str, List[Dict]] = {}
    for r in results:
        if r["status"] != "stable" or r.get("final_ppl") is None:
            continue
        by_variant.setdefault(r["variant"], []).append(r)
    for v in variants:
        runs = by_variant.get(v, [])
        runs.sort(key=lambda r: r["final_ppl"])
        print(f"\n  {v}:  ({len(runs)} stable configs out of "
              f"{len(expand_grid(v))} total)")
        for r in runs[:5]:
            print(f"    ppl={r['final_ppl']:>7.0f}  {r['config_id']}")

    return {"results": results, "by_variant": by_variant}


# ============================================================================
#  Stage 2 - focus on top configs
# ============================================================================

def stage2_focus(
    stage1: Dict,
    train_loader_factory: Callable,
    val_loader,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    hw: dict,
) -> Dict:
    out_path = OUT / "hpgrid_stage2_focus.json"

    by_variant = stage1["by_variant"]

    # Pick top K stable configs per variant by Stage 1 final_ppl
    top_configs: List[Dict] = []
    for variant, runs in by_variant.items():
        runs_sorted = sorted(runs, key=lambda r: r["final_ppl"])
        top_runs = runs_sorted[:PHASE2_TOP_K]
        for r in top_runs:
            top_configs.append({
                "variant": variant,
                "lr":      r["kfac_lr"],
                "damping": r["damping"],
                "gamma":   r.get("gamma", 0.5),
                "clip":    r.get("grad_clip", GRAD_CLIP),
                "momentum": r.get("momentum", MOMENTUM),
                "stage1_ppl": r["final_ppl"],
            })

    print("\n" + "=" * 70)
    print(f"  STAGE 2 FOCUS: {len(top_configs)} configs x {STAGE2_STEPS} steps")
    print("=" * 70)
    for cfg in top_configs:
        print(f"    {config_id(cfg):<60}  stage1_ppl={cfg['stage1_ppl']:.0f}")

    completed: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            completed = {r["config_id"]: r
                          for r in (prior.get("results") or [])}
            print(f"  Resuming Stage 2 from {out_path.name}: "
                  f"{len(completed)} configs already done")
        except (json.JSONDecodeError, KeyError):
            pass

    results: List[Dict] = list(completed.values())

    def _save():
        out = {
            "hw":           hw,
            "stage":        2,
            "stage2_steps": STAGE2_STEPS,
            "ppl_targets":  PPL_TARGETS,
            "results":      results,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=str))
        os.replace(tmp, out_path)

    for i, cfg in enumerate(top_configs):
        cid = config_id(cfg)
        if cid in completed:
            print(f"\n  [{i+1}/{len(top_configs)}] {cid}: skipping (cached)")
            continue
        print(f"\n  [{i+1}/{len(top_configs)}] {cid}")
        res = run_one_config(
            cfg, STAGE2_STEPS, train_loader_factory, val_loader,
            vocab_size, pad_id, device,
            record_natgrad=True, record_condition=True,
        )
        ttt = {f"ppl<={int(t)}": time_to_ppl(res["val_ppls"], res["val_times"], t)
               for t in PPL_TARGETS}
        res["time_to_target"] = ttt
        res["stage1_ppl"] = cfg["stage1_ppl"]
        results.append(res)
        completed[cid] = res
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl else "DIVERGED"
        print(f"  -> {cid}: stage2_ppl={ppl_str}  "
              f"wall={res['wall_s']/60:.1f}m")
        _save()

    return {"results": results, "top_configs": top_configs}


# ============================================================================
#  Reporting
# ============================================================================

def write_summary_csv(stage1: Dict, stage2: Optional[Dict]):
    csv_path = OUT / "hpgrid_summary.csv"
    lines = ["stage,variant,lr,damping,gamma,clip,status,final_ppl,"
             "wall_s,median_step_ms,steps,running_min_loss,config_id"]
    if stage1:
        for r in stage1.get("results", []):
            ppl = r.get("final_ppl") or ""
            ms  = r.get("median_step_ms") or ""
            lines.append(",".join(str(x) for x in [
                "1", r["variant"], r["kfac_lr"], r["damping"],
                r.get("gamma", ""), r.get("grad_clip", ""),
                r["status"], ppl, f"{r['wall_s']:.1f}", ms,
                r["steps_completed"], r.get("running_min_loss") or "",
                r["config_id"],
            ]))
    if stage2:
        for r in stage2.get("results", []):
            ppl = r.get("final_ppl") or ""
            ms  = r.get("median_step_ms") or ""
            lines.append(",".join(str(x) for x in [
                "2", r["variant"], r["kfac_lr"], r["damping"],
                r.get("gamma", ""), r.get("grad_clip", ""),
                r["status"], ppl, f"{r['wall_s']:.1f}", ms,
                r["steps_completed"], r.get("running_min_loss") or "",
                r["config_id"],
            ]))
    csv_path.write_text("\n".join(lines))
    print(f"  Saved -> {csv_path}")


def make_topk_plot(stage2: Optional[Dict]):
    if not stage2 or not stage2.get("results"):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    COLORS = {"ClassicKFAC": "tab:red",
              "OlsSMKFAC":   "tab:blue",
              "VeredKFAC":   "tab:green"}

    fig, (ax_steps, ax_wall) = plt.subplots(1, 2, figsize=(13, 5))
    for r in stage2["results"]:
        if r.get("status") != "stable":
            continue
        label = (f"{r['variant']} lr={r['kfac_lr']:.0e} "
                 f"λ={r['damping']:.0e} γ={r.get('gamma', 0.5)}")
        c = COLORS.get(r["variant"], "gray")
        ax_steps.plot(r["val_samples"], r["val_ppls"], "-",
                      color=c, label=label, alpha=0.8)
        ax_wall.plot(r["val_times"], r["val_ppls"], "-",
                     color=c, label=label, alpha=0.8)
    for ax, xl in [(ax_steps, "Samples seen"), (ax_wall, "Wall-clock seconds")]:
        ax.set_yscale("log")
        ax.set_ylabel("Validation perplexity")
        ax.set_xlabel(xl)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    ax_steps.set_title("Stage 2 top configs — convergence by samples")
    ax_wall.set_title("Stage 2 top configs — convergence by wall time")
    fig.suptitle("Hyperparameter grid: top-K configs per variant",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT / "hpgrid_topk.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    parser.add_argument("--variants", default=",".join(GRIDS.keys()),
                        help="Comma-separated variant names")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in variants if v not in GRIDS]
    if invalid:
        print(f"  ERROR: unknown variants {invalid}")
        sys.exit(1)

    print("\nHyperparameter Grid Benchmark — K-FAC variants")
    print("=" * 70)
    device = get_device()
    hw = get_hardware_info()
    print(f"  Hardware: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")
    print(f"  Variants: {variants}")
    n_total = sum(len(expand_grid(v)) for v in variants)
    print(f"  Stage 1: {n_total} configs x {STAGE1_STEPS} steps")
    print(f"  Stage 2: top {PHASE2_TOP_K} per variant x {STAGE2_STEPS} steps")
    print(f"           = {len(variants) * PHASE2_TOP_K} long runs\n")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    stage1_out = None
    stage2_out = None

    if args.stage in ("1", "all"):
        stage1_out = stage1_screen(variants, train_loader_factory, val_loader,
                                    vocab_size, pad_id, device, hw)
    else:
        s1_path = OUT / "hpgrid_stage1_screen.json"
        if not s1_path.exists():
            print(f"  ERROR: --stage 2 requires {s1_path}; run --stage 1 first")
            sys.exit(2)
        stage1_loaded = json.loads(s1_path.read_text())
        results = stage1_loaded.get("results", [])
        # Re-build by_variant for stage 2
        by_variant: Dict[str, List[Dict]] = {}
        for r in results:
            if r["status"] == "stable" and r.get("final_ppl") is not None:
                by_variant.setdefault(r["variant"], []).append(r)
        stage1_out = {"results": results, "by_variant": by_variant}
        print(f"  Loaded Stage 1 from {s1_path.name}: "
              f"{len(results)} configs total, "
              f"{sum(len(v) for v in by_variant.values())} stable")

    if args.stage in ("2", "all"):
        stage2_out = stage2_focus(stage1_out, train_loader_factory, val_loader,
                                   vocab_size, pad_id, device, hw)

    print("\n" + "=" * 70)
    print("  Writing summary + plots")
    print("=" * 70)
    write_summary_csv(stage1_out, stage2_out)
    make_topk_plot(stage2_out)
    print("\nDone.")


if __name__ == "__main__":
    main()
