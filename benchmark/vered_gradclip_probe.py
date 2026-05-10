"""
benchmark/vered_gradclip_probe.py

Targeted probe of grad_clip at the current Vered winner config.

Background:
    The (gamma, lambda) coupling probe (vered_gamma_lambda_probe.py) found
    that the full grid's gamma=0.3 winner was a local optimum, not the
    global one.  At lambda=1e-4 + mom=0.3 (the right damping/momentum),
    the gamma optimum moves to gamma=0.9 with ppl ~670:

        Vered new winner: gamma=0.9, mom=0.3, lambda=1e-4, grad_clip=60.0
                          -> ~670 ppl (still descending at step 5000)

    The grad_clip value of 60 was inherited from an earlier project default
    and has not been optimized at this operating point.

    Prior gradclip benchmark (Phase 1, at 1000 steps, mom=0.9, gamma=0.7)
    found the optimum at grad_clip=30 across all three K-FAC variants.
    At the new winner (gamma=0.9, mom=0.3), per-step curvature variance
    is LOWER than at the old (gamma=0.3) operating point because high
    gamma smooths factor noise.  Lower per-step variance might mean the
    optimal clip is LOOSER, not tighter.  Conversely, lower momentum
    (0.3 vs 0.9 in the Phase 1 sweep) means less step-magnitude smoothing,
    which pushes the other way.  Net effect is empirically ambiguous;
    this probe measures it.

Sweep (asymmetric around the reference, going both looser and tighter):
    grad_clip = {200, 100, 30, 15, 8}   # 60 already done -> ~670 ppl, reused

    The looser values (100, 200) test the hypothesis that high gamma +
    high lambda already do most of the noise-bounding work, so the
    optimum at the new config may be LOOSER than Phase 1's clip=30.
    The tighter values (30, 15, 8) test the converse hypothesis that
    lower momentum (0.3 vs Phase 1's 0.9) raises per-step magnitude
    variance and tightens the optimum.

Decision rule:
    grad_clip=100 or 200 wins by >=10 ppl  -> looser is better; high gamma+
                                              lambda already handle noise.
                                              Probe even looser values if 200
                                              is best (300, 500).
    grad_clip=30 wins by >=10 ppl          -> Phase 1 prediction holds; adopt.
    grad_clip=15 or 8 wins by >=10         -> tighter than expected; explore lower.
    All within 10 ppl                      -> grad_clip is insensitive here.
    grad_clip=8 diverges                   -> lower bound is between 8-15.
    grad_clip=200 diverges                 -> upper bound is between 100-200.

Wall time: ~6.25 hours (5 runs x 75 min).

Usage:
    python benchmark/vered_gradclip_probe.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data, run_probe, OUT, time_to_ppl
from benchmark.gpu_benchmark import get_device, get_hardware_info


# ---- Configuration ---------------------------------------------------------

# Fixed across the probe (the (gamma, lambda)-probe winner)
FIXED = {
    "variant":   "VeredKFAC",
    "kfac_lr":   8e-3,
    "gamma":     0.9,
    "momentum":  0.3,
    "damping":   1e-4,
    "max_steps": 5000,
}

# grad_clip values to test, ordered for most-informative-first execution.
# 100 first (tests "is looser better at high gamma + lambda?"), 30 second
# (Phase 1 control), then extremes 200, 15, 8.  60 is the reused reference.
CLIPS = [100.0, 30.0, 200.0, 15.0, 8.0]

# Reference value (already done as part of the (gamma, lambda) probe)
REFERENCE_CLIP = 60.0
REFERENCE_PATH = OUT / "vered_grid_g0.90_m0.30_l1e-04.json"


# ---- Helpers ---------------------------------------------------------------

def out_path(clip: float) -> Path:
    """Naming convention: vered_clip_g{gamma}_c{clip}.json.
    Includes gamma in the filename so probes at different (gamma, mom, lambda)
    operating points don't collide with each other."""
    return OUT / f"vered_clip_g{FIXED['gamma']:.2f}_c{clip:g}.json"


def reuse_reference() -> Optional[Dict]:
    """Pull in the grad_clip=60 result from the full-grid winner file
    if the canonical clip-probe file doesn't already exist."""
    canonical = out_path(REFERENCE_CLIP)
    if canonical.exists():
        try:
            return json.loads(canonical.read_text())
        except Exception:
            pass
    if not REFERENCE_PATH.exists():
        return None
    shutil.copy2(REFERENCE_PATH, canonical)
    print(f"  [reuse] {REFERENCE_PATH.name} -> {canonical.name}  "
          f"(grad_clip=60 reference)")
    try:
        return json.loads(canonical.read_text())
    except Exception:
        return None


def run_one(clip: float, train_loader_factory, val_loader, vocab_size: int,
            pad_id: int, device, hw: Dict) -> Optional[Dict]:
    p = out_path(clip)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists. final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(FIXED, grad_clip=clip)
    print()
    print("=" * 72)
    print(f"  Vered gradclip probe: clip={clip:g}  "
          f"(at gamma={FIXED['gamma']}, mom={FIXED['momentum']}, "
          f"lambda={FIXED['damping']:.0e})")
    print("=" * 72)
    for k, v in cfg.items():
        print(f"  {k:<10s} = {v}")
    print("=" * 72)

    t0 = time.perf_counter()
    res = run_probe(
        variant=cfg["variant"],
        kfac_lr=cfg["kfac_lr"],
        damping=cfg["damping"],
        momentum=cfg["momentum"],
        max_steps=cfg["max_steps"],
        vocab_size=vocab_size,
        train_loader_factory=train_loader_factory,
        val_loader=val_loader,
        pad_id=pad_id,
        device=device,
        seed=42,
        record_natgrad=True,
        record_condition=True,
        condition_log_every=200,
        print_progress=True,
        grad_clip=cfg["grad_clip"],
        gamma=cfg["gamma"],
    )
    wall = time.perf_counter() - t0

    saved = {
        "config":   cfg,
        "hw":       hw,
        "wall_s":   wall,
        "result":   res,
        "time_to_target": {
            f"ppl<={int(t)}": time_to_ppl(res.get("val_ppls", []),
                                          res.get("val_times", []), t)
            for t in [1000.0, 800.0, 750.0, 700.0, 680.0, 660.0]
        },
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  Saved -> {p}")
    return saved


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered grad_clip probe at full-grid winner config")
    print(f"  Fixed:    gamma={FIXED['gamma']}, mom={FIXED['momentum']}, "
          f"lambda={FIXED['damping']:.0e}, lr={FIXED['kfac_lr']:.0e}")
    print(f"  Sweeping: grad_clip in {CLIPS}  "
          f"(grad_clip={REFERENCE_CLIP:g} reused -> ~670 ppl)")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs: List[Dict] = []

    # Reference (clip=60)
    ref = reuse_reference()
    if ref is not None:
        runs.append(ref)
    else:
        print(f"\n  [warn] reference file at {REFERENCE_PATH} not found; "
              f"summary will not include grad_clip=60")

    # Sweep new clips
    for c in CLIPS:
        saved = run_one(c, train_loader_factory, val_loader, vocab_size,
                        pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # Summary
    print()
    print("=" * 72)
    print("  Probe summary: grad_clip sweep at "
          f"gamma={FIXED['gamma']}, mom={FIXED['momentum']}, "
          f"lambda={FIXED['damping']:.0e}")
    print("=" * 72)
    print(f"  {'grad_clip':>10}  {'final_ppl':>10}  {'wall (min)':>11}  "
          f"{'status':>10}")
    print("  " + "-" * 50)
    runs_sorted = sorted(runs, key=lambda r: r["config"]["grad_clip"])
    for run in runs_sorted:
        cfg = run["config"]
        res = run["result"]
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
        wall = run["wall_s"] / 60
        print(f"  {cfg['grad_clip']:>10.1f}  {ppl_str:>10}  "
              f"{wall:>11.1f}  {res['status']:>10}")

    valid = [r for r in runs if r["result"].get("final_ppl") is not None]
    if not valid:
        print("\n  All runs diverged at every clip value tested.")
        return

    best = min(valid, key=lambda r: r["result"]["final_ppl"])
    worst = max(valid, key=lambda r: r["result"]["final_ppl"])
    spread = worst["result"]["final_ppl"] - best["result"]["final_ppl"]
    bclip = best["config"]["grad_clip"]
    bppl = best["result"]["final_ppl"]
    ref_ppl = next((r["result"]["final_ppl"] for r in runs
                    if r["config"]["grad_clip"] == REFERENCE_CLIP), None)

    print()
    print(f"  Best:  grad_clip={bclip:g}  ppl={bppl:.0f}")
    print(f"  Spread across clip range: {spread:.0f} ppl")

    # Diverged-clip detection
    diverged = [r for r in runs if r["result"].get("final_ppl") is None
                or r["result"]["final_ppl"] > 5000]
    if diverged:
        for d in diverged:
            print(f"  [!] grad_clip={d['config']['grad_clip']:g} diverged "
                  f"({d['result'].get('status', 'unknown')})")

    print()
    print("  Reference points (all at grad_clip=60):")
    print("    Vered new winner (gamma=0.9, mom=0.3, lambda=1e-4):  ~670 ppl")
    print("    Vered prior winner (gamma=0.3, mom=0.3, lambda=1e-4): 681 ppl")
    print("    Vered pre-fix (buggy EMA, mom=0.9):                   756 ppl")
    print("    Classic untuned (mom=0.9, lambda=1e-4):               776 ppl")

    print()
    if ref_ppl is not None and bppl < ref_ppl - 10:
        gain = ref_ppl - bppl
        print(f"  -> grad_clip={bclip:g} beats grad_clip={REFERENCE_CLIP:g} "
              f"by {gain:.0f} ppl.")
        if bclip > REFERENCE_CLIP:
            print(f"     LOOSER clip is better; high gamma + lambda already")
            print(f"     handle most noise.  Probe even looser values if 200")
            print(f"     is the winner (300, 500, no-clip).")
        elif bclip == 30.0:
            print("     Phase 1 prediction (clip=30 was the optimum) confirmed")
            print("     at the new (gamma, mom, lambda) operating point.")
        elif bclip < 30.0:
            print(f"     TIGHTER clip than Phase 1; the lower momentum and")
            print(f"     fresh-curvature variance dominate.  Probe {{3, 5}} next.")
        new_total = bppl
        delta_to_buggy = 756 - new_total
        delta_to_classic = 776 - new_total
        print(f"     New Vered ppl: {new_total:.0f}")
        print(f"       vs buggy 756:  -{delta_to_buggy:.0f} ppl")
        print(f"       vs Classic 776: -{delta_to_classic:.0f} ppl")
    elif spread < 10:
        print(f"  -> All clip values within {spread:.0f} ppl: grad_clip is")
        print(f"     insensitive at this config.  The 60 chosen for stability")
        print(f"     buffer was harmless; no gain available from this axis.")
    elif ref_ppl is not None and bppl < ref_ppl:
        gain = ref_ppl - bppl
        print(f"  -> Modest improvement: grad_clip={bclip:g} gives "
              f"{gain:.0f} ppl over clip=60.")
        print("     Worth adopting but doesn't substantially shift the headline.")
    else:
        print(f"  -> grad_clip=60 (reference) stays best.  "
              f"No gain from tightening or loosening at this config.")

    print("=" * 72)


if __name__ == "__main__":
    main()
