"""
benchmark/vered_2d_mom_lr_screen.py

2D screening grid: momentum x learning rate, at the current best
operating point.  Short (1000-step) runs to identify the top configs;
a follow-up 5000-step refinement grid will then drill into the winners.

CRITICAL CONTEXT:
    This probe runs AFTER the discovery that stability_benchmark.py:378
    hardcoded `momentum=0.0` for VeredKFAC.  Every prior Vered run was
    effectively at mom=0.  The fix is now in place (line 378 changed to
    `momentum=momentum`).

    This grid is the first valid test of how Vered actually responds to
    momentum and how mom interacts with lr.  The hypothesis under test:
    is mom=0 with elevated LR strictly better than mom>0 (the strong
    version of the lag hypothesis), or do mom and lr have separate
    contributions?

Grid (32 cells at 1000 steps each; cells with existing JSON are skipped):
    momentum: {0.0, 0.3, 0.7, 0.9}
    lr:       {5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 2e-2, 4e-2}

Theoretical iso-magnitude diagonals (effective step ~lr / (1-mom)):
    (mom=0.7, lr=2e-3) ~ (mom=0, lr=6e-3)   # screen-champion control
    (mom=0.7, lr=8e-3) ~ (mom=0, lr=2.7e-2)
    (mom=0.9, lr=8e-3) ~ (mom=0, lr=8e-2)
    If mom and lr are strictly interchangeable, these pairs should
    perform similarly.  If mom does more (variance reduction, etc.),
    the mom>0 cells should be better.

Wall time: ~4 hours.
Resume support: cells with existing JSON are skipped.

Output JSONs: vered_2dscreen_m{mom}_lr{lr}_s1000.json

Usage:
    python benchmark/vered_2d_mom_lr_screen.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data, run_probe, OUT, time_to_ppl
from benchmark.gpu_benchmark import get_device, get_hardware_info


# ---- Configuration ---------------------------------------------------------

FIXED = {
    "variant":     "VeredKFAC",
    "gamma":       0.9,
    "damping":     1e-4,
    "grad_clip":   300.0,
    "max_steps":   1000,            # screening run; will follow up at 5000
    "lr_schedule": "constant_warmup",  # see note below
}
# lr_schedule note:
#   Switched from "cosine_max_steps" (run_probe's default) to "constant_warmup"
#   on 2026-05-12.  Reason: under cosine the per-cell trajectory depends on
#   max_steps -- the 1000-step screen decays ~6x faster than a 5000-step
#   refinement run would, so a cell labeled lr=8e-3 trains very differently
#   here vs in a follow-up.  With constant_warmup, "lr=X" means "trains at
#   LR=X after 200-step warmup" regardless of run length.  Old JSONs from
#   the cosine sweep live under filenames WITHOUT the _const suffix; new
#   ones include it so the two regimes don't get mixed.

MOMS = [0.0, 0.3, 0.7, 0.9]
# LR sweep: K-FAC's "lr" is a multiplier on an already-sized natural-gradient
# step (different from SGD's role).  Cleaner gradients (Vered's kappa^1) help
# with direction noise, not step-size calibration.  Higher LR risks
# overshooting the local quadratic approximation, especially when combined
# with momentum's steady-state amplification.  So we test BOTH below and
# above the current 8e-3 baseline.
LRS  = [5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 2e-2, 4e-2]
# Extension rationale (added after first screen at 5 LRs):
#   - 5e-4, 1e-3:  top-5 from the first pass all sat at lr=2e-3; the trend
#                  (lr=2e-3 beats lr=4e-3 at every momentum) points smaller.
#                  Probe whether mom>0 cells keep improving below 2e-3.
#   - 6e-3:        effective-LR control for the screen champion
#                  (mom=0.7, lr=2e-3) -> effective ~6.7e-3 ~ (mom=0, lr=6e-3).
#                  Tests the strong lag hypothesis directly at the winner.

# Cells to skip — was previously {(0.0, 8e-3)} on the assumption that the
# historical 5000-step winner (nominally mom=0.3, lr=8e-3, but effectively
# mom=0.0 due to the pre-fix momentum bug) didn't need re-measuring.
#
# Cleared on 2026-05-12 after noticing the screen's best 1000-step ppl (~1848)
# is ~2x worse than the historical baseline at step 1000 (val_ppls in
# vered_clip_g0.90_c300.json show ~1150 @ step 938, ~1014 @ step 1094).
# Need a post-fix (0.0, 8e-3) measurement to know whether the screen
# reproduces the historical baseline or whether something else regressed.
SKIP_CELLS: set = set()

# Historical 5000-step reference (kept for context; no longer used to short-
# circuit the table now that the cell is being re-measured).
REFERENCE_PPL_5000: dict = {}


# ---- Helpers ---------------------------------------------------------------

def make_tag(mom: float, lr: float) -> str:
    # _const suffix marks results produced under lr_schedule="constant_warmup";
    # cosine-schedule JSONs from earlier runs do not carry it, so the two
    # regimes never collide and resume support won't mis-skip a stale cell.
    suffix = "_const" if FIXED.get("lr_schedule") == "constant_warmup" else ""
    return f"m{mom:.2f}_lr{lr:.0e}_s{FIXED['max_steps']}{suffix}"


def out_path(mom: float, lr: float) -> Path:
    return OUT / f"vered_2dscreen_{make_tag(mom, lr)}.json"


def run_one(mom: float, lr: float,
            train_loader_factory, val_loader, vocab_size: int, pad_id: int,
            device, hw: Dict) -> Optional[Dict]:
    p = out_path(mom, lr)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists. final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(FIXED, momentum=mom, kfac_lr=lr)
    print()
    print("=" * 72)
    print(f"  Vered 2D screen: mom={mom:g}  lr={lr:.0e}  "
          f"(at gamma={FIXED['gamma']}, lambda={FIXED['damping']:.0e}, "
          f"clip={FIXED['grad_clip']:g}, steps={FIXED['max_steps']})")
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
            lr_schedule=cfg.get("lr_schedule", "cosine_max_steps"),
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

    # Compute slope over the last ~200 steps for screening signal
    val_ppls = res.get("val_ppls", []) or []
    slope_per_100 = None
    if len(val_ppls) >= 3:
        recent = val_ppls[-3:]
        if all(v is not None and v < 5000 for v in recent):
            slope_per_100 = (recent[0] - recent[-1]) / 2.0
            # interpret: ppl drop per ~100 steps near end

    try:
        ttt = {
            f"ppl<={int(t)}": time_to_ppl(res.get("val_ppls", []),
                                          res.get("val_times", []), t)
            for t in [3000.0, 2000.0, 1500.0, 1200.0, 1000.0]
        }
    except Exception:
        ttt = {}

    saved = {
        "config":         cfg,
        "hw":             hw,
        "wall_s":         wall,
        "result":         res,
        "slope_per_100":  slope_per_100,
        "time_to_target": ttt,
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  Saved -> {p}   "
          f"final_ppl={res.get('final_ppl')}   "
          f"slope/100={slope_per_100}")
    return saved


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered 2D (mom x lr) screening grid")
    print(f"  Fixed:  gamma={FIXED['gamma']}, lambda={FIXED['damping']:.0e}, "
          f"clip={FIXED['grad_clip']:g}, max_steps={FIXED['max_steps']}")
    print(f"  Sweep:  mom in {MOMS}, lr in {LRS}  ({len(MOMS) * len(LRS)} cells)")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs: List[Dict] = []
    for mom, lr in product(MOMS, LRS):
        # Skip cells whose 5000-step result is already known.
        cell_key = (round(mom, 4), float(f"{lr:.4g}"))
        # Use approximate matching since float comparisons can be flaky.
        matched_skip = any(abs(mom - sm) < 1e-9 and abs(lr - sl) / max(sl, 1e-12) < 1e-3
                           for (sm, sl) in SKIP_CELLS)
        if matched_skip:
            ref = next((v for (sm, sl), v in REFERENCE_PPL_5000.items()
                        if abs(mom - sm) < 1e-9 and abs(lr - sl) / max(sl, 1e-12) < 1e-3),
                       None)
            print(f"\n  [skip-known] mom={mom:g}, lr={lr:.0e}: known reference, "
                  f"5000-step ppl = {ref:.0f}" if ref is not None else
                  f"\n  [skip-known] mom={mom:g}, lr={lr:.0e}: known reference")
            continue
        saved = run_one(mom, lr, train_loader_factory, val_loader,
                        vocab_size, pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # Summary: 2D table of final_ppl
    print()
    print("=" * 72)
    print("  Screening result table  (final_ppl at step 1000)")
    print("=" * 72)
    # Header row of LR values
    header = "  mom \\ lr |" + "".join(f"  {lr:>8.0e}" for lr in LRS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mom in MOMS:
        row = f"  {mom:>8.2f} |"
        for lr in LRS:
            # Reference (skipped) cell -- show known 5000-step ppl in brackets
            ref = next((v for (sm, sl), v in REFERENCE_PPL_5000.items()
                        if abs(mom - sm) < 1e-9
                        and abs(lr - sl) / max(sl, 1e-12) < 1e-3),
                       None)
            if ref is not None:
                row += f"  {f'[{ref:.0f}]':>8}"  # bracketed = 5000-step reference
                continue
            r = next((rr for rr in runs
                      if abs(rr["config"]["momentum"] - mom) < 1e-9
                      and abs(rr["config"]["kfac_lr"] - lr) < 1e-12), None)
            if r is None:
                row += f"  {'(missing)':>8}"
                continue
            ppl = r["result"].get("final_ppl")
            if ppl is None or ppl >= 5000:
                row += f"  {'DIV':>8}"
            else:
                row += f"  {ppl:>8.0f}"
        print(row)

    # Also print slope table -- helps predict 5000-step ranking
    print()
    print("  Slope (ppl decrease per ~100 steps near end -- higher = more headroom):")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mom in MOMS:
        row = f"  {mom:>8.2f} |"
        for lr in LRS:
            r = next((rr for rr in runs
                      if abs(rr["config"]["momentum"] - mom) < 1e-9
                      and abs(rr["config"]["kfac_lr"] - lr) < 1e-12), None)
            if r is None:
                row += f"  {'(miss)':>8}"
                continue
            s = r.get("slope_per_100")
            if s is None:
                row += f"  {'n/a':>8}"
            else:
                row += f"  {s:>8.1f}"
        print(row)

    # Identify top 5 cells by final_ppl
    valid = [r for r in runs
             if r["result"].get("final_ppl") is not None
             and r["result"]["final_ppl"] < 5000]
    print()
    if valid:
        valid_sorted = sorted(valid, key=lambda r: r["result"]["final_ppl"])
        print("=" * 72)
        print("  Top 5 cells by final_ppl at step 1000:")
        print("=" * 72)
        print(f"  {'rank':>4}  {'mom':>5}  {'lr':>9}  {'final_ppl':>10}  "
              f"{'slope/100':>10}")
        for rank, r in enumerate(valid_sorted[:5], 1):
            cfg = r["config"]
            ppl = r["result"]["final_ppl"]
            s = r.get("slope_per_100")
            s_str = f"{s:.1f}" if s is not None else "n/a"
            print(f"  {rank:>4}  {cfg['momentum']:>5.2f}  "
                  f"{cfg['kfac_lr']:>9.0e}  {ppl:>10.0f}  {s_str:>10}")
        print()
        print("  These 5 cells are candidates for the 5000-step refinement grid.")
        print("  Add cells with high slope (still descending fast) that are")
        print("  close in ppl to the top -- they may overtake by step 5000.")
    else:
        print("  All cells diverged or produced no valid result.")
    print("=" * 72)


if __name__ == "__main__":
    main()
