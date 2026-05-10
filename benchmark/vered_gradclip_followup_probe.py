"""
benchmark/vered_gradclip_followup_probe.py

Follow-up to the gradclip probe.  The first probe (vered_gradclip_probe.py)
swept {200, 100, 60, 30, 15, 8} at the (gamma=0.9, mom=0.3, lambda=1e-4)
winner config and found a strictly monotonic improvement with looser clip:
        clip=8  -> 959 ppl
        clip=15 -> 830 ppl
        clip=30 -> 731 ppl
        clip=60 -> 670 ppl   (reference)
        clip=100 -> 652 ppl
        clip=200 -> 631 ppl   <-- best, no U-curve found yet

The right edge of the U-curve hasn't been observed.  This probe extends
the sweep upward to find it (or confirm there isn't one in the practical
range).

Sweep:
    grad_clip = {300, 500, 1000, 1e9}   # 200 already done -> 631 ppl, reused
                                          # 1e9 is effectively no-clip

Theoretical hypotheses:
    (A) The optimum is somewhere in 300-1000.  Best ppl 615-625.
        Implication: Vered+tuning has a real but narrow optimal clip range.
    (B) No-clip (1e9) is stable and best.  Best ppl 620-630.
        Implication: at this (gamma, mom, lambda), Vered's clean per-step
        natural-gradient direction is good enough that no magnitude
        bounding is needed.  Clean story for the math doc.
    (C) Some intermediate value diverges.  Best ppl is at 200 or just
        slightly looser.  Implication: noise tolerance has a real upper
        bound; we found it.

Risks:
    - Phase 1 (at OLD config: gamma=0.7, mom=0.9, mid-tier damping) saw
      Classic NaN and OlsSM diverged at clip=1000.  At the new config
      (gamma=0.9 + lambda=1e-4 doing aggressive noise reduction), the
      noise-tolerance headroom is much higher, but clip=1000 and 1e9 are
      still the most likely cells to crash.
    - Order is least-risky-first: {300, 500, 1000, 1e9}.  If 1000
      diverges, you can Ctrl-C before 1e9 wastes a slot.

Wall time: ~5 hours (4 runs x 75 min).

Usage:
    python benchmark/vered_gradclip_followup_probe.py
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

# Sweep upper-end clip values, ordered least-risky-first.
# 1e9 is effectively no-clip (model gradients shouldn't reach this magnitude).
CLIPS = [300.0, 500.0, 1000.0, 1e9]

# Reference from the prior probe (the new prior best at clip=200)
REFERENCE_CLIP = 200.0
REFERENCE_PATH = OUT / "vered_clip_g0.90_c200.json"


# ---- Helpers ---------------------------------------------------------------

def out_path(clip: float) -> Path:
    """Same naming convention as the prior gradclip probe."""
    return OUT / f"vered_clip_g{FIXED['gamma']:.2f}_c{clip:g}.json"


def reuse_reference() -> Optional[Dict]:
    canonical = out_path(REFERENCE_CLIP)
    if canonical.exists():
        try:
            return json.loads(canonical.read_text())
        except Exception:
            pass
    if not REFERENCE_PATH.exists():
        return None
    if canonical != REFERENCE_PATH:
        shutil.copy2(REFERENCE_PATH, canonical)
        print(f"  [reuse] {REFERENCE_PATH.name} -> {canonical.name}")
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
    clip_label = "no-clip (1e9)" if clip >= 1e8 else f"clip={clip:g}"
    print(f"  Vered gradclip follow-up: {clip_label}  "
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
            for t in [1000.0, 800.0, 750.0, 700.0, 650.0, 620.0]
        },
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  Saved -> {p}")

    # Early-stop hint: if this run diverged, alert user
    final_ppl = res.get("final_ppl")
    if final_ppl is None or final_ppl > 5000 or res.get("status") != "stable":
        print(f"  [!] DIVERGENCE DETECTED at clip={clip:g}.  Subsequent")
        print(f"      looser-clip runs are likely to diverge too.  Consider")
        print(f"      Ctrl-C and refining the upper bound between this clip")
        print(f"      and the previous stable clip.")

    return saved


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered grad_clip FOLLOW-UP probe (upper end of U-curve)")
    print(f"  Fixed:    gamma={FIXED['gamma']}, mom={FIXED['momentum']}, "
          f"lambda={FIXED['damping']:.0e}, lr={FIXED['kfac_lr']:.0e}")
    print(f"  Sweeping: grad_clip in {CLIPS}  "
          f"(grad_clip={REFERENCE_CLIP:g} reused -> 631 ppl)")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs: List[Dict] = []

    # Reference (clip=200)
    ref = reuse_reference()
    if ref is not None:
        runs.append(ref)
    else:
        print(f"\n  [warn] reference file at {REFERENCE_PATH} not found; "
              f"summary will not include grad_clip=200")

    # Pull in any other prior gradclip probe results that exist, so the final
    # summary table has the full picture (clip=8, 15, 30, 60, 100, 200, plus
    # the new ones).
    for c in [8.0, 15.0, 30.0, 60.0, 100.0]:
        prior = out_path(c)
        if prior.exists():
            try:
                runs.append(json.loads(prior.read_text()))
            except Exception:
                pass

    # Sweep new clips
    for c in CLIPS:
        saved = run_one(c, train_loader_factory, val_loader, vocab_size,
                        pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # Summary
    print()
    print("=" * 72)
    print("  Full grad_clip sweep summary at "
          f"gamma={FIXED['gamma']}, mom={FIXED['momentum']}, "
          f"lambda={FIXED['damping']:.0e}")
    print("=" * 72)
    print(f"  {'grad_clip':>12}  {'final_ppl':>10}  {'wall (min)':>11}  "
          f"{'status':>10}")
    print("  " + "-" * 52)
    runs_sorted = sorted(runs, key=lambda r: r["config"]["grad_clip"])
    for run in runs_sorted:
        cfg = run["config"]
        res = run["result"]
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
        wall = run["wall_s"] / 60
        clip_v = cfg["grad_clip"]
        clip_label = "no-clip (1e9)" if clip_v >= 1e8 else f"{clip_v:.1f}"
        print(f"  {clip_label:>12}  {ppl_str:>10}  "
              f"{wall:>11.1f}  {res['status']:>10}")

    valid = [r for r in runs if r["result"].get("final_ppl") is not None
             and r["result"]["final_ppl"] < 5000]
    if not valid:
        print("\n  All runs diverged.  Should not happen for clip in [8, 200].")
        return

    best = min(valid, key=lambda r: r["result"]["final_ppl"])
    bclip = best["config"]["grad_clip"]
    bppl = best["result"]["final_ppl"]
    bclip_label = "no-clip (1e9)" if bclip >= 1e8 else f"{bclip:g}"

    diverged = [r for r in runs
                if (r["result"].get("final_ppl") is None
                    or r["result"]["final_ppl"] >= 5000
                    or r["result"].get("status") != "stable")]

    print()
    print(f"  Best:  grad_clip={bclip_label}  ppl={bppl:.0f}")
    if diverged:
        for d in diverged:
            dc = d["config"]["grad_clip"]
            label = "no-clip (1e9)" if dc >= 1e8 else f"{dc:g}"
            print(f"  [!] grad_clip={label} DIVERGED ({d['result'].get('status')})")

    print()
    print("  Reference points:")
    print("    Vered tuned, clip=60  (start of clip optimization):  670 ppl")
    print("    Vered tuned, clip=200 (prior probe winner):           631 ppl")
    print("    Vered prior config, clip=60 (gamma=0.3 winner):       681 ppl")
    print("    Vered pre-fix (buggy EMA):                            756 ppl")
    print("    Classic untuned:                                      776 ppl")

    print()
    if bppl < 631 - 5:
        gain = 631 - bppl
        print(f"  -> grad_clip={bclip_label} beats clip=200 by {gain:.0f} ppl.")
        if bclip >= 1e8:
            print("     NO-CLIP IS BEST.  At this (gamma=0.9, mom=0.3, lambda=1e-4)")
            print("     operating point, Vered's clean per-step natural-gradient")
            print("     direction is good enough that magnitude bounding is")
            print("     unnecessary.  Clean story for the math doc.")
        elif bclip == 1000.0:
            print("     1000 is a sweet spot.  Worth probing 700, 1500 to refine.")
        elif bclip == 500.0:
            print("     500 is the sweet spot.  Probe 400, 700 to refine.")
        elif bclip == 300.0:
            print("     300 is the sweet spot; further loosening hurts.")
        new_total = bppl
        print(f"     New Vered headline: {new_total:.0f} ppl")
        print(f"       vs buggy 756:   -{756 - new_total:.0f} ppl")
        print(f"       vs Classic 776: -{776 - new_total:.0f} ppl")
    elif diverged:
        first_div = min((r["config"]["grad_clip"] for r in diverged), default=None)
        print(f"  -> Upper bound of stable clip is between 200 and {first_div:g}.")
        print("     Worth refining to find the exact divergence threshold.")
    else:
        print(f"  -> All looser values within ~5 ppl of clip=200.  The plateau")
        print(f"     has been reached at the right edge of the U-curve.")
        print(f"     Vered headline stays at ~631 ppl.")
    print("=" * 72)


if __name__ == "__main__":
    main()
