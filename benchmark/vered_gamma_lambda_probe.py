"""
benchmark/vered_gamma_lambda_probe.py

Targeted probe of the (gamma, lambda) coupling in VeredKFAC post-fix,
following the full-grid result that lambda=1e-4 dominates at gamma=0.3.

The full grid (benchmark/vered_full_grid.py) found:
    Best so far:      gamma=0.3, mom=0.3, lambda=1e-4  -> 681 ppl
    Stage C spread:   damping varied 1e-7 to 1e-4, monotonic improvement
                      with higher damping at gamma=0.3.

What the grid did NOT test: whether lambda=1e-4 is also optimal at higher
gamma. All Stage C runs fixed gamma=0.3. So we don't know whether:

    (A) lambda=1e-4 is universally optimal -> then gamma is a free parameter,
        and the simpler recipe "VeredKFAC with lambda=1e-4" wins at any gamma.

    (B) gamma=0.3 + lambda=1e-4 is a coupled optimum -> different gamma
        values have different optimal damping; the (gamma, lambda) landscape
        is non-separable.

This probe runs three new configs at (mom=0.3, lambda=1e-4):
    gamma = 0.5
    gamma = 0.7
    gamma = 0.9

The existing gamma=0.3 result (681 ppl) is reused for comparison.

Decision rule:
    If best stays at gamma=0.3:        case (B) confirmed; the coupling is
                                       real, gamma=0.3 is the right choice
                                       at lambda=1e-4.
    If best moves to higher gamma:     case (A) partially supported; the
                                       grid's gamma=0.3 was an artifact of
                                       only sweeping lambda there.  Worth
                                       re-running the grid with lambda=1e-4
                                       fixed instead of 1e-5.
    If all are within ~10 ppl:         damping dominates, gamma is largely
                                       free above some floor; settle on the
                                       cheapest-to-train gamma.

Wall time: ~3.75 hours (3 runs x 75 min).

Usage:
    python benchmark/vered_gamma_lambda_probe.py
"""
from __future__ import annotations

import json
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

# Fixed across the probe
FIXED = {
    "variant":   "VeredKFAC",
    "kfac_lr":   8e-3,
    "momentum":  0.3,        # winner from full grid
    "damping":   1e-4,       # winner from full grid
    "grad_clip": 60.0,
    "max_steps": 5000,
}

# Gamma values to sweep
GAMMAS = [0.5, 0.7, 0.9]   # gamma=0.3 already done -> 681 ppl


# ---- Helpers ---------------------------------------------------------------
# Use the same filename convention as vered_full_grid.py so that the existing
# gamma=0.3 result (vered_grid_g0.30_m0.30_l1e-04.json) is found and reused.

def make_tag(gamma: float, momentum: float, damping: float) -> str:
    return f"g{gamma:.2f}_m{momentum:.2f}_l{damping:.0e}"


def out_path(gamma: float, momentum: float, damping: float) -> Path:
    return OUT / f"vered_grid_{make_tag(gamma, momentum, damping)}.json"


def run_one(gamma: float, train_loader_factory, val_loader, vocab_size: int,
            pad_id: int, device, hw: Dict) -> Optional[Dict]:
    momentum = FIXED["momentum"]
    damping  = FIXED["damping"]
    p = out_path(gamma, momentum, damping)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists. final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(FIXED, gamma=gamma)
    print()
    print("=" * 72)
    print(f"  Vered probe: gamma={gamma:g}  mom={momentum:g}  "
          f"lambda={damping:.0e}  lr={cfg['kfac_lr']:.0e}")
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
            for t in [1000.0, 800.0, 750.0, 700.0, 680.0]
        },
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  Saved -> {p}")
    return saved


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered (gamma, lambda) coupling probe")
    print(f"  Fixed:    mom={FIXED['momentum']}, "
          f"lambda={FIXED['damping']:.0e}, lr={FIXED['kfac_lr']:.0e}")
    print(f"  Sweeping: gamma in {GAMMAS}  (gamma=0.3 reused -> 681 ppl)")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    # Pull in the existing gamma=0.3 winner for context
    runs: List[Dict] = []
    g03_path = out_path(0.3, FIXED["momentum"], FIXED["damping"])
    if g03_path.exists():
        try:
            runs.append(json.loads(g03_path.read_text()))
            print(f"\n  [reuse] gamma=0.3 reference loaded from {g03_path.name}")
        except Exception as e:
            print(f"\n  [warn] could not load gamma=0.3 reference: {e}")
    else:
        print(f"\n  [warn] gamma=0.3 reference file not found at {g03_path};")
        print(f"         summary will not include it.  Run full grid first.")

    # Sweep new gammas
    for g in GAMMAS:
        saved = run_one(g, train_loader_factory, val_loader, vocab_size,
                        pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # Summary
    print()
    print("=" * 72)
    print("  Probe summary: gamma sweep at mom=0.3, lambda=1e-4")
    print("=" * 72)
    print(f"  {'gamma':>6} {'mom':>5} {'lambda':>10}  {'final_ppl':>10}  "
          f"{'wall (min)':>11}  {'status':>10}")
    print("  " + "-" * 65)
    runs_sorted = sorted(runs, key=lambda r: r["config"]["gamma"])
    for run in runs_sorted:
        cfg = run["config"]
        res = run["result"]
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
        wall = run["wall_s"] / 60
        print(f"  {cfg['gamma']:>6.2f} {cfg['momentum']:>5.2f} "
              f"{cfg['damping']:>10.0e}  {ppl_str:>10}  "
              f"{wall:>11.1f}  {res['status']:>10}")

    valid = [r for r in runs if r["result"].get("final_ppl") is not None]
    if not valid:
        print("\n  All runs diverged.  Damping floor reached at lambda=1e-4 + this gamma range.")
        return

    best = min(valid, key=lambda r: r["result"]["final_ppl"])
    worst = max(valid, key=lambda r: r["result"]["final_ppl"])
    spread = worst["result"]["final_ppl"] - best["result"]["final_ppl"]
    bg = best["config"]["gamma"]
    bppl = best["result"]["final_ppl"]

    print()
    print(f"  Best:  gamma={bg:g}  ppl={bppl:.0f}")
    print(f"  Spread across gamma range: {spread:.0f} ppl")

    print()
    print("  Reference points:")
    print("    Vered post-fix (gamma=0.3, mom=0.3, lambda=1e-4):   681 ppl  <-- prior best")
    print("    Vered pre-fix  (buggy, gamma=0.7, mom=0.9):         756 ppl")
    print("    Classic        (mom=0.9, lambda=1e-4):              776 ppl")

    print()
    if bg == 0.3:
        print("  -> gamma=0.3 stays best.  (gamma, lambda) ARE coupled:")
        print("     gamma=0.3 + lambda=1e-4 is a non-separable optimum.")
        print("     The full-grid winner is genuine and not an artifact.")
        if spread < 20:
            print("     But spread is small; gamma is fairly insensitive at this lambda.")
    elif bppl < 681 - 10:
        print(f"  -> WINNER MOVED to gamma={bg:g}, beating gamma=0.3 by "
              f"{681 - bppl:.0f} ppl.")
        print("     The full-grid was anchored on lambda=1e-5 in Stages A/B,")
        print("     which masked the true gamma optimum at lambda=1e-4.")
        print("     Worth re-running the full grid with lambda=1e-4 fixed.")
    elif spread < 15:
        print(f"  -> All gammas within {spread:.0f} ppl: damping dominates,")
        print("     gamma is largely free above floor.  Pick cheapest-to-train.")
    else:
        print(f"  -> Best is gamma={bg:g} ({bppl:.0f} ppl), beating gamma=0.3 by")
        print(f"     {681 - bppl:.0f} ppl.  Modest improvement; tradeoff is")
        print("     gamma's effect on convergence speed vs. final ppl.")

    print("=" * 72)


if __name__ == "__main__":
    main()
