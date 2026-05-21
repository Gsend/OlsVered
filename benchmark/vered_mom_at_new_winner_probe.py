"""
benchmark/vered_mom_at_new_winner_probe.py

Vered mom-axis re-sweep at the current best operating point.

Background:
    Stage B of vered_full_grid.py swept mom at (gamma=0.3, lambda=1e-5) and
    found all values gave 798 ppl, suggesting mom was irrelevant for Vered.
    That conclusion was applied at the new operating point (gamma=0.9,
    lambda=1e-4) without re-verification — the subsequent (gamma, lambda)
    probe held mom=0.3 fixed.

    The Classic grid then showed that Classic at gamma=0.7, lambda=1e-4
    benefits dramatically from mom=0.9 -> 0.7 (703 -> 594 ppl, -109 ppl).
    This suggests the mom-axis insensitivity finding was specific to
    (gamma=0.3, lambda=1e-5) and may not hold at the Vered winner config.

    If Vered also benefits from mom=0.7 at (gamma=0.9, lambda=1e-4,
    clip=300), the actual Vered best may be 560-590 instead of 618 — and
    the fair Vered vs Classic comparison could swing.

Sweep:
    momentum = {0.0, 0.5, 0.7, 0.9}    # mom=0.3 already done -> 618 ppl, reused

    Rationale per cell:
        mom=0.0:  baseline; no momentum amplification, pure per-step direction
        mom=0.5:  intermediate; mirrors Vered Stage B's 0.5 cell
        mom=0.7:  Classic's winner at analogous (gamma, lambda)
        mom=0.9:  literature default; sanity check vs over-smoothing

Decision rule:
    mom=0.7 wins by >=10 ppl  ->  matches Classic's pattern; Vered's
                                  acceptable region tightens at high gamma
    mom=0.5 wins                  -> like Vered, less momentum needed than
                                  Classic at the same gamma
    mom=0.3 (reused) stays best   -> the original choice was correct;
                                  Vered's 618 is the actual peak
    mom=0.9 wins (unexpected)     -> momentum-amplification helps when
                                  curvature is well-smoothed

Wall time: ~5 hours (4 runs x 75 min).

Usage:
    python benchmark/vered_mom_at_new_winner_probe.py
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

# Fixed across the probe (the current Vered winner config)
FIXED = {
    "variant":   "VeredKFAC",
    "kfac_lr":   8e-3,
    "gamma":     0.9,
    "damping":   1e-4,
    "grad_clip": 300.0,
    "max_steps": 5000,
}

# Momentum values to sweep
MOMS = [0.7, 0.5, 0.0, 0.9]   # mom=0.3 reused (618 ppl from gradclip probe winner)

REFERENCE_MOM  = 0.3
REFERENCE_PATH = OUT / "vered_clip_g0.90_c300.json"   # contains the 618 result


# ---- Helpers ---------------------------------------------------------------

def make_tag(mom: float) -> str:
    return (f"g{FIXED['gamma']:.2f}_m{mom:.2f}"
            f"_l{FIXED['damping']:.0e}_c{int(FIXED['grad_clip'])}")


def out_path(mom: float) -> Path:
    return OUT / f"vered_mom_{make_tag(mom)}.json"


def reuse_reference() -> Optional[Dict]:
    """Pull in the mom=0.3 result (from gradclip probe winner) into the
    canonical mom-probe filename."""
    canonical = out_path(REFERENCE_MOM)
    if canonical.exists():
        try:
            return json.loads(canonical.read_text())
        except Exception:
            pass
    if not REFERENCE_PATH.exists():
        return None
    shutil.copy2(REFERENCE_PATH, canonical)
    print(f"  [reuse] {REFERENCE_PATH.name} -> {canonical.name}  "
          f"(mom=0.3 reference)")
    try:
        return json.loads(canonical.read_text())
    except Exception:
        return None


def run_one(mom: float, train_loader_factory, val_loader, vocab_size: int,
            pad_id: int, device, hw: Dict) -> Optional[Dict]:
    p = out_path(mom)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists. final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(FIXED, momentum=mom)
    print()
    print("=" * 72)
    print(f"  Vered mom probe: mom={mom:g}  (at gamma={FIXED['gamma']}, "
          f"lambda={FIXED['damping']:.0e}, clip={FIXED['grad_clip']:g})")
    print("=" * 72)
    for k, v in cfg.items():
        print(f"  {k:<10s} = {v}")
    print("=" * 72)

    t0 = time.perf_counter()
    try:
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
    except Exception as e:
        crash_msg = f"{type(e).__name__}: {e}"
        print()
        print(f"  [!] CRASH during run_probe: {crash_msg}")
        print(f"      Treating as DIVERGED.  Saving sentinel and continuing.")
        res = {
            "status":          "diverged_crash",
            "error":           crash_msg,
            "final_ppl":       None,
            "val_ppls":        [],
            "val_times":       [],
            "median_step_ms":  None,
        }
    wall = time.perf_counter() - t0

    try:
        ttt = {
            f"ppl<={int(t)}": time_to_ppl(res.get("val_ppls", []),
                                          res.get("val_times", []), t)
            for t in [1000.0, 800.0, 700.0, 650.0, 600.0]
        }
    except Exception:
        ttt = {}

    saved = {
        "config":         cfg,
        "hw":             hw,
        "wall_s":         wall,
        "result":         res,
        "time_to_target": ttt,
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  Saved -> {p}")
    return saved


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered mom-axis re-sweep at the current best operating point")
    print(f"  Fixed:    gamma={FIXED['gamma']}, lambda={FIXED['damping']:.0e}, "
          f"clip={FIXED['grad_clip']:g}, lr={FIXED['kfac_lr']:.0e}")
    print(f"  Sweeping: mom in {MOMS}  (mom={REFERENCE_MOM} reused -> 618 ppl)")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs: List[Dict] = []

    ref = reuse_reference()
    if ref is not None:
        runs.append(ref)
    else:
        print(f"\n  [warn] reference file at {REFERENCE_PATH} not found; "
              f"summary will not include mom=0.3")

    for m in MOMS:
        saved = run_one(m, train_loader_factory, val_loader, vocab_size,
                        pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # Summary
    print()
    print("=" * 72)
    print("  Probe summary: mom-axis sweep at "
          f"gamma={FIXED['gamma']}, lambda={FIXED['damping']:.0e}, "
          f"clip={FIXED['grad_clip']:g}")
    print("=" * 72)
    print(f"  {'mom':>6}  {'final_ppl':>10}  {'wall (min)':>11}  "
          f"{'status':>14}")
    print("  " + "-" * 50)
    runs_sorted = sorted(runs, key=lambda r: r["config"]["momentum"])
    for run in runs_sorted:
        cfg = run["config"]
        res = run["result"]
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
        wall = run["wall_s"] / 60
        print(f"  {cfg['momentum']:>6.2f}  {ppl_str:>10}  "
              f"{wall:>11.1f}  {res['status']:>14}")

    valid = [r for r in runs if r["result"].get("final_ppl") is not None
             and r["result"]["final_ppl"] < 5000]
    if not valid:
        print("\n  All runs diverged.")
        return

    best = min(valid, key=lambda r: r["result"]["final_ppl"])
    bmom = best["config"]["momentum"]
    bppl = best["result"]["final_ppl"]
    ref_ppl = next((r["result"]["final_ppl"] for r in runs
                    if r["config"]["momentum"] == REFERENCE_MOM), None)

    print()
    print(f"  Best:  mom={bmom:g}  ppl={bppl:.0f}")
    print()
    print("  Reference points:")
    print("    Vered prior best (mom=0.3, this config):              618 ppl")
    print("    Classic tuned best (gamma=0.7, mom=0.7, lambda=1e-3): 592 ppl")
    print("    Vered pre-fix (buggy EMA):                            756 ppl")

    print()
    if ref_ppl is not None and bppl < ref_ppl - 10:
        gain = ref_ppl - bppl
        print(f"  -> mom={bmom:g} beats mom={REFERENCE_MOM:g} by {gain:.0f} ppl.")
        if bppl < 592:
            print(f"     Vered overtakes Classic.  New headline: {bppl:.0f} ppl.")
            print(f"     Gap to Classic: -{592 - bppl:.0f} ppl.")
        elif bppl < 618:
            print(f"     Vered narrows the gap to Classic.")
            print(f"     New gap to Classic: +{bppl - 592:.0f} ppl (still behind).")
    elif ref_ppl is not None and abs(bppl - ref_ppl) <= 10:
        print(f"  -> mom-axis is insensitive at this operating point.")
        print(f"     The original mom=0.3 choice was already near-optimal.")
        print(f"     Vered headline stays at ~618 ppl.")
        print(f"     Final gap to Classic: +{618 - 592} ppl.")
    else:
        print(f"  -> All swept mom values are worse than mom=0.3.")
        print(f"     Vered headline stays at ~618 ppl.")

    print("=" * 72)


if __name__ == "__main__":
    main()
