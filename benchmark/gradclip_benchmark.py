"""
benchmark/gradclip_benchmark.py

Numerical-stability comparison via gradient-clip frontier instead of LR sweep.

Fixed configuration:
    LR        = 8e-3      (the deployment optimum from prior LR-sweep result
                           for the SmallGPT/WikiText-2 transformer task)
    momentum  = 0.9       (deployment regime)
    damping   = 1e-3      (matches gpu_benchmark.py transformer config)

Sole varying axis:
    grad_clip in {0.3, 1, 3, 10, 30, 100, 300, 1000}    (8 log-spaced values)

The hypothesis being tested:
    A variant with cleaner natural-gradient updates (smaller per-step
    inversion noise) should tolerate a HIGHER grad_clip before the unclipped
    natural-gradient norm causes training to diverge.  Equivalently: at any
    fixed clip value, the variant with the cleanest updates has the most
    "headroom" before the clip starts limiting it.

    Per the math doc:
        ClassicKFAC : nat-grad noise ~ kappa(X)^4 * eps   -> diverges at lowest clip
        OlsSMKFAC   : nat-grad noise ~ kappa(X)^2 * eps   -> medium
        VeredKFAC   : nat-grad noise ~ kappa(X)^1 * eps   -> tolerates highest clip

Two phases:

  Phase 1 - Grad-clip frontier
    For each (variant, grad_clip) pair, run PROBE_STEPS=1000 steps at the
    fixed config above.  Classify as stable or diverged.  Records the highest
    grad_clip that still trains successfully per variant.

  Phase 2 - Convergence at max stable clip
    For each variant, run PHASE2_STEPS=5000 steps at its highest stable
    grad_clip.  Records validation perplexity curve, time-to-target ppl,
    median per-step natural-gradient norm, and per-layer condition numbers.
    Compares convergence quality at each variant's "best operating point".

Usage:
    python benchmark/gradclip_benchmark.py --phase all
    python benchmark/gradclip_benchmark.py --phase 1
    python benchmark/gradclip_benchmark.py --phase 2

Output (under benchmark/results/):
    gradclip_phase1_sweep.json
    gradclip_phase2_runs.json
    gradclip_summary.csv
    gradclip_frontier.png
    gradclip_convergence.png
"""
from __future__ import annotations

import argparse
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

# Reuse all infrastructure from the LR-sweep benchmark
from benchmark.stability_benchmark import (
    build_data,
    extract_factors,
    KFACConditionTracker,
    OUT,
    PPL_TARGETS,
    run_probe,
    time_to_ppl,
)
from benchmark.gpu_benchmark import get_device, get_hardware_info

# ============================================================================
#  Experimental constants
# ============================================================================

# Fixed at the prior deployment optimum
KFAC_LR    = 8e-3        # back to sane training LR.  Was bumped to 8e-2 to
                          # stress-test stability differences but absolute
                          # results were uniformly bad (ppl ~9999) at that LR;
                          # back to deployment-grade so Vered's stability
                          # advantage can show up as actual convergence wins.
MOMENTUM   = 0.90

# Per-variant damping. The math doc predicts OlsSM/Vered tolerate lower
# damping than Classic because of better stability scaling
# (kappa(X)^4 vs kappa(X)^2 vs kappa(X)^1).  Setting each variant at its
# "natural" damping makes the benchmark a head-to-head at each variant's
# best operating point - rather than artificially constraining OlsSM/Vered
# to the conservative damping Classic needs.
#
# To use a single uniform damping instead, set all three to the same value.
VARIANT_DAMPINGS: Dict[str, float] = {
    "ClassicKFAC": 1.5e-4,    # conservative - kappa^4 noise floor demands it
    "OlsSMKFAC":   1.5e-4,    # 5x lower - bumped from 1e-4 which was bistable
                            # in clip range 30-300 (NaN at 30, OK at 100, NaN at 300)
    "VeredKFAC":   1e-5,    # bumped from 1e-5: was numerically stable but
                            # converging worse than Classic.  Sample-noise
                            # amplification at 1e-5 dominated stability win.
                            # 5e-5 keeps Vered 4x lower than OlsSM (still tests
                            # kappa^1 advantage) but raises noise floor.
}

# Backward-compat: code still references DAMPING in JSON output for the
# uniform case; computed as min for warning purposes.
DAMPING = min(VARIANT_DAMPINGS.values())

# Grad-clip grid - log-spaced.
#
# DEFAULT (high-clip, sane LR): focuses on the convergence regime around
# clip=30-60 where the prior Phase 2 found Vered's 19% win.
GRADCLIP_GRID: List[float] = [30.0, 40.0, 60.0]

# ALTERNATIVE: "direction quality" regime - low clip + high LR.
# Tests whether Vered's kappa^1 stability gives a measurable convergence
# win when clip dominates magnitude and only DIRECTION quality matters.
# Use by setting BENCHMARK_REGIME = "direction_quality" below.
GRADCLIP_GRID_DIRECTION_QUALITY: List[float] = [0.3, 1.0, 3.0]
KFAC_LR_DIRECTION_QUALITY: float = 5e-2

# Switch between regimes.  "default" = the convergence-regime sweep.
# "direction_quality" = low-clip + high-LR (LR=0.05, clip in {0.3,1.0,3.0}).
# Override at CLI via --regime.
BENCHMARK_REGIME: str = "default"

VARIANTS: List[str] = ["VeredKFAC","ClassicKFAC",  "OlsSMKFAC"]

# Phase lengths
PROBE_STEPS  = 1000
PHASE2_STEPS = 5000

# ============================================================================
#  Phase 1 - Grad-clip frontier
# ============================================================================

def phase1_sweep(
    train_loader_factory: Callable,
    val_loader,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    hw: dict,
) -> Dict:
    """For each (variant, grad_clip) pair, run a probe and classify stability.

    Saves incrementally after each probe so a crash mid-sweep is recoverable
    via the resume logic at the top of this function.
    """
    n_probes = len(VARIANTS) * len(GRADCLIP_GRID)
    print("\n" + "=" * 70)
    print(f"  PHASE 1 - Grad-clip frontier sweep "
          f"({n_probes} probes x {PROBE_STEPS} steps)")
    print(f"  Fixed: LR={KFAC_LR}  momentum={MOMENTUM}")
    print(f"  Per-variant damping:")
    for v in VARIANTS:
        print(f"    {v:<14}  damping={VARIANT_DAMPINGS[v]:.0e}")
    print(f"  Sweep: grad_clip in {GRADCLIP_GRID}")
    print("=" * 70)

    out_path = OUT / "gradclip_phase1_sweep.json"

    # Resume support: per-variant.  We keep already-completed probes for
    # variants whose damping is UNCHANGED; discard only the variants whose
    # damping value differs from the current VARIANT_DAMPINGS setting.
    # This avoids redoing 8 ClassicKFAC probes when only OlsSMKFAC's damping
    # got bumped.
    all_results: Dict[str, List[Dict]] = {v: [] for v in VARIANTS}
    completed: Set[Tuple[str, float]] = set()
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            if (prior.get("kfac_lr") != KFAC_LR
                    or prior.get("momentum") != MOMENTUM):
                print(f"  WARN: prior {out_path.name} has different "
                      f"LR/momentum; starting fresh")
            else:
                prior_dampings = prior.get("variant_dampings") or {}
                kept = 0
                discarded_variants: List[str] = []
                for v, runs in (prior.get("runs") or {}).items():
                    if v not in all_results:
                        continue
                    # Per-variant damping check
                    prior_damp = prior_dampings.get(v)
                    cur_damp   = VARIANT_DAMPINGS[v]
                    if prior_damp is None or float(prior_damp) != cur_damp:
                        discarded_variants.append(v)
                        continue
                    for r in runs:
                        all_results[v].append(r)
                        completed.add((v, float(r["grad_clip"])))
                        kept += 1
                if discarded_variants:
                    print(f"  Damping changed for: {discarded_variants}  "
                          f"-> their prior probes discarded")
                if kept:
                    print(f"  Resuming from {out_path.name}: "
                          f"keeping {kept} completed probes "
                          f"(variants with unchanged damping)")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  WARN: could not parse {out_path.name} ({e}); starting fresh")
            all_results = {v: [] for v in VARIANTS}
            completed = set()

    def _max_stable() -> Dict[str, Optional[float]]:
        ms: Dict[str, Optional[float]] = {}
        for variant in VARIANTS:
            stable_clips = [r["grad_clip"]
                            for r in all_results[variant]
                            if r["status"] == "stable"]
            ms[variant] = max(stable_clips) if stable_clips else None
        return ms

    def _save():
        out = {
            "hw":               hw,
            "kfac_lr":          KFAC_LR,
            "momentum":         MOMENTUM,
            "damping":          DAMPING,    # min for backward-compat
            "variant_dampings": {v: VARIANT_DAMPINGS[v] for v in VARIANTS},
            "gradclip_grid":    GRADCLIP_GRID,
            "probe_steps":      PROBE_STEPS,
            "max_stable":       _max_stable(),
            "runs":             all_results,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=str))
        os.replace(tmp, out_path)
        return out

    for variant in VARIANTS:
        damp = VARIANT_DAMPINGS[variant]
        print(f"\n  -- {variant}  (damping={damp:.0e}) --")
        for clip in GRADCLIP_GRID:
            if (variant, clip) in completed:
                print(f"    probe grad_clip={clip:.1f}  (already done, skipping)")
                continue
            print(f"    probe grad_clip={clip:.1f} ...")
            res = run_probe(
                variant=variant, kfac_lr=KFAC_LR, damping=damp,
                momentum=MOMENTUM,
                max_steps=PROBE_STEPS, vocab_size=vocab_size,
                train_loader_factory=train_loader_factory,
                val_loader=val_loader, pad_id=pad_id, device=device,
                seed=42, record_natgrad=False, print_progress=False,
                grad_clip=clip,
            )
            summary = {k: v for k, v in res.items()
                       if k not in ("train_losses", "natgrad_norms",
                                    "condition_history")}
            all_results[variant].append(summary)

            tag = "stable  " if res["status"] == "stable" else "DIVERGED"
            if res["final_ppl"] is not None:
                ppl_str = f"final_ppl={res['final_ppl']:.0f}"
            else:
                dl = res.get("diverge_loss")
                ppl_str = (f"@step {res['diverge_step']} loss={dl:.2f}"
                           if dl is not None
                           else f"@step {res['diverge_step']}")
            ms = res.get("median_step_ms")
            ms_str = f"{ms:.1f}" if ms is not None else "n/a"
            print(f"      -> {tag}  {ppl_str}  step_ms={ms_str}")
            _save()

    out = _save()
    max_stable = out["max_stable"]

    print(f"\n  Phase 1 frontier - max stable grad_clip per variant "
          f"(higher = cleaner natural-gradient noise):")
    print(f"    {'variant':<14}  {'max stable clip':>16}")
    for v in VARIANTS:
        c = max_stable[v]
        s = f"{c:.1f}" if c is not None else "NONE"
        print(f"    {v:<14}  {s:>16}")
    print(f"\n  Saved -> {out_path}")
    return out


# ============================================================================
#  Phase 2 - Convergence at max stable grad_clip
# ============================================================================

def phase2_convergence(
    max_stable: Dict[str, Optional[float]],
    train_loader_factory: Callable,
    val_loader,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    hw: dict,
    clip_overrides: Optional[Dict[str, float]] = None,
    output_tag: str = "",
) -> Dict:
    """Full PHASE2_STEPS run at each variant's grad_clip.

    By default uses the variant's max-stable clip from Phase 1.
    `clip_overrides` (e.g. {"ClassicKFAC": 60, "VeredKFAC": 120}) overrides
    per-variant.  `output_tag` appends to the output filename so multiple
    Phase 2 configurations can coexist (e.g. _matched, _clip120_long).
    """
    overrides = clip_overrides or {}
    suffix = f"_{output_tag}" if output_tag else ""

    print("\n" + "=" * 70)
    print(f"  PHASE 2 - Convergence runs   "
          f"({PHASE2_STEPS} steps, LR={KFAC_LR}, mom={MOMENTUM}"
          f"{', tag=' + output_tag if output_tag else ''})")
    if overrides:
        print(f"  Clip overrides: {overrides}")
    print("=" * 70)

    out_path = OUT / f"gradclip_phase2_runs{suffix}.json"

    runs: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            if (prior.get("kfac_lr") == KFAC_LR
                    and prior.get("momentum") == MOMENTUM
                    and prior.get("phase2_steps") == PHASE2_STEPS):
                for v, r in (prior.get("runs") or {}).items():
                    if (v in VARIANTS
                            and r.get("status") not in (None, "no_stable_clip")
                            and r.get("steps_completed", 0) >= PHASE2_STEPS):
                        runs[v] = r
                if runs:
                    print(f"  Resuming: {len(runs)} variant(s) already done: "
                          f"{list(runs.keys())}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  WARN: could not parse {out_path.name} ({e}); fresh run")

    def _save():
        o = {
            "hw":               hw,
            "kfac_lr":          KFAC_LR,
            "momentum":         MOMENTUM,
            "damping":          DAMPING,
            "variant_dampings": {v: VARIANT_DAMPINGS[v] for v in VARIANTS},
            "phase2_steps":     PHASE2_STEPS,
            "max_stable":       max_stable,
            "ppl_targets":      PPL_TARGETS,
            "runs":             runs,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(o, indent=2, default=str))
        os.replace(tmp, out_path)
        return o

    for variant in VARIANTS:
        if variant in runs:
            r = runs[variant]
            print(f"\n  -- {variant}: already complete "
                  f"(final_ppl={r.get('final_ppl', 'n/a')}), skipping --")
            continue
        # Per-variant override takes precedence over Phase 1 max_stable
        if variant in overrides:
            clip = float(overrides[variant])
        else:
            clip = max_stable.get(variant)
        if clip is None:
            print(f"\n  -- {variant}: SKIPPED (no stable clip found in Phase 1 "
                  f"and no override provided) --")
            runs[variant] = {"variant": variant, "status": "no_stable_clip",
                             "grad_clip": None}
            continue
        damp = VARIANT_DAMPINGS[variant]
        print(f"\n  -- {variant} @ grad_clip={clip:.1f}, damping={damp:.0e} --")
        res = run_probe(
            variant=variant, kfac_lr=KFAC_LR, damping=damp,
            momentum=MOMENTUM,
            max_steps=PHASE2_STEPS, vocab_size=vocab_size,
            train_loader_factory=train_loader_factory,
            val_loader=val_loader, pad_id=pad_id, device=device,
            seed=42,
            record_natgrad=True,
            record_condition=True,
            condition_log_every=100,
            print_progress=True,
            grad_clip=clip,
        )
        ttt = {f"ppl<={int(t)}": time_to_ppl(res["val_ppls"], res["val_times"], t)
               for t in PPL_TARGETS}
        res["time_to_target"] = ttt
        runs[variant] = res

        cs = res.get("condition_summary") or {}
        if cs:
            avg_ka = float(np.mean([v["mean_kappa_A"] for v in cs.values()]))
            avg_kg = float(np.mean([v["mean_kappa_G"] for v in cs.values()]))
            kappa_str = f"  avg_kappa(A)={avg_ka:.1e}  avg_kappa(G)={avg_kg:.1e}"
        else:
            kappa_str = ""
        print(f"    final_ppl={res['final_ppl']:.1f}  "
              f"wall={res['wall_s']/60:.1f}m  "
              f"natgrad_norm={res.get('median_natgrad_norm')}{kappa_str}")
        _save()

    out = _save()
    print(f"\n  Saved -> {out_path}")
    return out


# ============================================================================
#  Plotting
# ============================================================================

def make_plots(phase1: Optional[Dict], phase2: Optional[Dict]):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots.")
        return

    COLORS = {"ClassicKFAC": "tab:red",
              "OlsSMKFAC":   "tab:blue",
              "VeredKFAC":   "tab:green"}

    # Phase 1: ppl vs grad_clip per variant
    if phase1 is not None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for variant, runs in phase1["runs"].items():
            xs_stable, ys_stable = [], []
            xs_div,    ys_div    = [], []
            for r in runs:
                clip = r["grad_clip"]
                if r["status"] == "stable":
                    ppl = r["final_ppl"] if r["final_ppl"] else 9999.0
                    xs_stable.append(clip); ys_stable.append(ppl)
                else:
                    xs_div.append(clip); ys_div.append(9999.0)
            c = COLORS.get(variant, "gray")
            ax.plot(xs_stable, ys_stable, "o-", color=c,
                    label=f"{variant} stable", linewidth=2, markersize=8)
            ax.plot(xs_div, ys_div, "x", color=c, markersize=12,
                    label=f"{variant} diverged")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("grad_clip (log)")
        ax.set_ylabel(f"Validation perplexity at step {PROBE_STEPS} "
                      f"(9999 = diverged)")
        ax.set_title(f"Phase 1: grad_clip frontier  "
                     f"(LR={KFAC_LR}, mom={MOMENTUM}, lower ppl = better)")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out = OUT / "gradclip_frontier.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  Saved -> {out}")

    # Phase 2: convergence trajectories
    if phase2 is not None and phase2.get("runs"):
        fig, (ax_steps, ax_wall) = plt.subplots(1, 2, figsize=(13, 5))
        for variant, r in phase2["runs"].items():
            if r.get("status") == "no_stable_clip":
                continue
            label = f"{variant} (clip={r['grad_clip']:.1f})"
            ax_steps.plot(r["val_samples"], r["val_ppls"], "o-",
                          color=COLORS.get(variant, "gray"),
                          label=label, linewidth=2)
            ax_wall.plot(r["val_times"], r["val_ppls"], "o-",
                         color=COLORS.get(variant, "gray"),
                         label=label, linewidth=2)
        for ax, xl in [(ax_steps, "Samples seen"),
                       (ax_wall, "Wall-clock seconds")]:
            ax.set_yscale("log")
            ax.set_ylabel("Validation perplexity")
            ax.set_xlabel(xl)
            ax.legend(fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
        ax_steps.set_title("Phase 2: convergence by samples")
        ax_wall.set_title("Phase 2: convergence by wall time")
        fig.suptitle(f"Convergence at each variant's max stable grad_clip "
                     f"(LR={KFAC_LR}, mom={MOMENTUM})", fontsize=11, y=1.02)
        fig.tight_layout()
        out = OUT / "gradclip_convergence.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")


def write_summary_csv(phase1: Optional[Dict], phase2: Optional[Dict]):
    csv_path = OUT / "gradclip_summary.csv"
    lines = ["phase,variant,grad_clip,kfac_lr,momentum,damping,status,"
             "steps,final_ppl,wall_s,median_step_ms,min_train_loss,"
             "median_natgrad_norm"]
    if phase1:
        for variant, runs in phase1["runs"].items():
            for r in runs:
                lines.append(",".join(str(x) for x in [
                    "1", variant, r.get("grad_clip"), r["kfac_lr"],
                    r["momentum"], r["damping"], r["status"],
                    r["steps_completed"], r.get("final_ppl") or "",
                    f"{r['wall_s']:.1f}", r.get("median_step_ms") or "",
                    r.get("running_min_loss") or "",
                    r.get("median_natgrad_norm") or "",
                ]))
    if phase2:
        for variant, r in (phase2.get("runs") or {}).items():
            if r.get("status") == "no_stable_clip":
                continue
            lines.append(",".join(str(x) for x in [
                "2", variant, r.get("grad_clip"), r["kfac_lr"],
                r["momentum"], r["damping"], r["status"],
                r["steps_completed"], r.get("final_ppl") or "",
                f"{r['wall_s']:.1f}", r.get("median_step_ms") or "",
                r.get("running_min_loss") or "",
                r.get("median_natgrad_norm") or "",
            ]))
    csv_path.write_text("\n".join(lines))
    print(f"  Saved -> {csv_path}")


# ============================================================================
#  Main
# ============================================================================

def main():
    # Must declare globals BEFORE argparse reads them as default values
    global PROBE_STEPS, PHASE2_STEPS, VARIANTS, KFAC_LR, GRADCLIP_GRID

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--probe-steps", type=int, default=PROBE_STEPS)
    parser.add_argument("--phase2-steps", type=int, default=PHASE2_STEPS)
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help="Comma-separated variants to include")
    parser.add_argument("--phase2-clips", default="",
                        help="Per-variant clip overrides for Phase 2, format: "
                             "'ClassicKFAC=60,OlsSMKFAC=60,VeredKFAC=120'. "
                             "Variants not listed fall back to Phase 1 max_stable.")
    parser.add_argument("--phase2-tag", default="",
                        help="Suffix appended to the Phase 2 output JSON filename "
                             "(e.g. 'matched' -> gradclip_phase2_runs_matched.json). "
                             "Lets multiple Phase 2 configs coexist.")
    parser.add_argument("--regime",
                        choices=["default", "direction_quality"],
                        default=BENCHMARK_REGIME,
                        help="default: high-clip / sane-LR convergence regime "
                             "(LR=0.008, clip~30-60). "
                             "direction_quality: low-clip / high-LR regime "
                             "(LR=0.05, clip~0.3-3) where clip dominates magnitude "
                             "and only direction quality decides convergence - "
                             "the cleanest test of Vered's kappa^1 advantage.")
    args = parser.parse_args()

    # Apply regime selection
    if args.regime == "direction_quality":
        KFAC_LR = KFAC_LR_DIRECTION_QUALITY
        GRADCLIP_GRID = GRADCLIP_GRID_DIRECTION_QUALITY
        print(f"  Regime: direction_quality (LR={KFAC_LR}, clips={GRADCLIP_GRID})")
    else:
        print(f"  Regime: default (LR={KFAC_LR}, clips={GRADCLIP_GRID})")

    PROBE_STEPS = args.probe_steps
    PHASE2_STEPS = args.phase2_steps
    VARIANTS = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in VARIANTS
               if v not in ("ClassicKFAC", "OlsSMKFAC", "VeredKFAC")]
    if invalid:
        print(f"  ERROR: unknown variants {invalid}")
        sys.exit(1)

    print("\nGrad-Clip Stability Benchmark - K-FAC variants")
    print("=" * 70)
    device = get_device()
    hw = get_hardware_info()
    print(f"  Hardware: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    phase1_out = None
    phase2_out = None

    if args.phase in ("1", "all"):
        phase1_out = phase1_sweep(train_loader_factory, val_loader,
                                   vocab_size, pad_id, device, hw)
        max_stable = phase1_out["max_stable"]
    else:
        p1_path = OUT / "gradclip_phase1_sweep.json"
        if not p1_path.exists():
            print(f"  ERROR: --phase 2 requires {p1_path}; run --phase 1 first")
            sys.exit(2)
        phase1_out = json.loads(p1_path.read_text())
        max_stable = {k: (float(v) if v is not None else None)
                      for k, v in phase1_out["max_stable"].items()}
        print(f"  Loaded Phase 1 from {p1_path}")
        print(f"  Max stable clips: {max_stable}")

    # Parse per-variant clip overrides for Phase 2
    clip_overrides: Dict[str, float] = {}
    if args.phase2_clips:
        for entry in args.phase2_clips.split(","):
            entry = entry.strip()
            if not entry:
                continue
            try:
                v, c = entry.split("=")
                clip_overrides[v.strip()] = float(c.strip())
            except (ValueError, KeyError):
                print(f"  ERROR: bad --phase2-clips entry '{entry}'; "
                      f"expected 'Variant=clip'")
                sys.exit(1)
        unknown = [v for v in clip_overrides
                   if v not in ("ClassicKFAC", "OlsSMKFAC", "VeredKFAC")]
        if unknown:
            print(f"  ERROR: unknown variant(s) in --phase2-clips: {unknown}")
            sys.exit(1)

    if args.phase in ("2", "all"):
        phase2_out = phase2_convergence(max_stable, train_loader_factory,
                                         val_loader, vocab_size, pad_id,
                                         device, hw,
                                         clip_overrides=clip_overrides,
                                         output_tag=args.phase2_tag)

    print("\n" + "=" * 70)
    print("  Writing summary + plots")
    print("=" * 70)
    write_summary_csv(phase1_out, phase2_out)
    make_plots(phase1_out, phase2_out)
    print("\nDone.")


if __name__ == "__main__":
    main()
