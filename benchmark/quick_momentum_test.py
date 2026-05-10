"""
benchmark/quick_momentum_test.py

Single-shot test: Vered post-fix at momentum=0.0 to test the lag hypothesis.

Hypothesis under test:
    VeredKFAC's per-step natural gradient is ~140x cleaner than Classic's
    (per the synthetic stability test).  At momentum=0.9, the optimizer is
    averaging across 10 steps' worth of natural gradients, but each step's
    direction has already changed (weights updated).  So momentum trades
    noise reduction (which Vered doesn't need) for direction lag (which it
    doesn't want).

    Predict: Vered post-fix at momentum=0.0 should converge to a lower
    perplexity than Vered post-fix at momentum=0.9 (which got 908 ppl), and
    perhaps even beat the buggy-EMA pre-fix run (756 ppl).

Reference points:
    Vered post-fix mom=0.9 lambda=1e-5:    908 ppl  (our recent baseline)
    Vered pre-fix  mom=0.9 lambda=1e-5:    756 ppl  (buggy EMA, "lucky")
    Classic       mom=0.9 lambda=1e-4:    776 ppl  (best Classic so far)

Decision rule for what to do next based on this run:
    final_ppl < 800 ppl :  lag hypothesis confirmed; launch full grid
    final_ppl 800-900 ppl: partial confirmation; launch full grid
    final_ppl 900+ ppl :  hypothesis falsified; momentum=0 is not the answer
                          and momentum's noise reduction is load-bearing for
                          Vered after all.

Usage:
    python benchmark/quick_momentum_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data, run_probe, OUT, time_to_ppl
from benchmark.gpu_benchmark import get_device, get_hardware_info


# ---- Configuration ---------------------------------------------------------

# Without momentum=0.9, the effective steady-state step size loses its 1/(1-0.9)
# = 10x amplification.  To compensate, sweep LR upward.  Three LR values:
#   8e-3   = original (no compensation; effectively 1/10 of the prior step size)
#   2.4e-2 = 3x compensation (mild)
#   8e-2   = 10x compensation (full match for the lost momentum amplification)
#
# The 75-minute wall time is per-LR; total ~3.75 hours for all three.  You can
# interrupt early if you see a clear winner emerge.

BASE_CONFIG = {
    "variant":   "VeredKFAC",
    "damping":   1e-5,
    "momentum":  0.0,             # <-- the key change (was 0.9)
    "gamma":     0.7,             # (matches prior run for direct comparison)
    "grad_clip": 60.0,
    "max_steps": 5000,
}

LR_SWEEP = [8e-3, 2.4e-2, 8e-2]
OUTPUT_TAG_PREFIX = "mom00"


# ---- Run -------------------------------------------------------------------

def run_one(cfg: Dict, lr: float, train_loader_factory, val_loader,
            vocab_size: int, pad_id: int, device, hw: Dict) -> Dict:
    """One full Phase-2-style run at the given LR."""
    print()
    print("=" * 72)
    print(f"  Vered post-fix at momentum=0.0, lr={lr:.0e}")
    print("=" * 72)
    full_cfg = dict(cfg)
    full_cfg["kfac_lr"] = lr
    for k, v in full_cfg.items():
        print(f"  {k:<10s} = {v}")
    print("=" * 72)

    t0 = time.perf_counter()
    res = run_probe(
        variant=cfg["variant"],
        kfac_lr=lr,
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

    out_path = OUT / f"quick_momentum_test_{OUTPUT_TAG_PREFIX}_lr{lr:.0e}.json"
    saved = {
        "config":   {**full_cfg},
        "hw":       hw,
        "wall_s":   wall,
        "result":   res,
        "time_to_target": {
            f"ppl<={int(t)}": time_to_ppl(res.get("val_ppls", []),
                                          res.get("val_times", []), t)
            for t in [1000.0, 800.0, 700.0]
        },
    }
    out_path.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  Saved -> {out_path}")
    return saved


def main():
    print("=" * 72)
    print("  Quick momentum test: Vered post-fix at momentum=0.0")
    print(f"  Sweeping LR over {LR_SWEEP}")
    print("  (without momentum=0.9, effective step size loses its 10x")
    print("   amplification; LR sweep compensates)")
    print("=" * 72)

    device = get_device()
    hw     = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs = []
    for lr in LR_SWEEP:
        # Skip if the file already exists (resume support)
        out_path = OUT / f"quick_momentum_test_{OUTPUT_TAG_PREFIX}_lr{lr:.0e}.json"
        if out_path.exists():
            print(f"\n  {out_path.name} already exists; skipping lr={lr:.0e}")
            try:
                runs.append(json.loads(out_path.read_text()))
            except Exception:
                pass
            continue
        saved = run_one(BASE_CONFIG, lr, train_loader_factory, val_loader,
                        vocab_size, pad_id, device, hw)
        runs.append(saved)

    # Final summary table across all LRs
    print()
    print("=" * 72)
    print("  Summary across LR sweep")
    print("=" * 72)
    print(f"  {'LR':>10}  {'final_ppl':>10}  {'wall (min)':>11}  "
          f"{'step (ms)':>10}  {'status':>10}")
    print("  " + "-" * 60)
    for run in runs:
        cfg = run["config"]
        res = run["result"]
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
        wall = run["wall_s"] / 60
        step_ms = res.get("median_step_ms")
        ms_str = f"{step_ms:.0f}" if step_ms is not None else "n/a"
        print(f"  {cfg['kfac_lr']:>10.0e}  {ppl_str:>10}  {wall:>11.1f}  "
              f"{ms_str:>10}  {res['status']:>10}")

    print()
    print("  Reference points:")
    print("    Vered pre-fix  (buggy EMA, mom=0.9, lr=8e-3):  756 ppl")
    print("    Classic        (mom=0.9, lr=8e-3, λ=1e-4):     776 ppl")
    print("    Vered post-fix (mom=0.9, lr=8e-3, λ=1e-5):     908 ppl  <-- to beat")
    print()

    # Decision rule based on best LR
    valid = [r for r in runs if r["result"].get("final_ppl") is not None]
    if not valid:
        print("  -> All LRs DIVERGED. Need an intermediate config:")
        print("     try momentum=0.3 or 0.5 instead of 0.0.")
    else:
        best = min(valid, key=lambda r: r["result"]["final_ppl"])
        best_ppl = best["result"]["final_ppl"]
        best_lr  = best["config"]["kfac_lr"]
        print(f"  Best result: ppl={best_ppl:.0f} at lr={best_lr:.0e}")
        if best_ppl < 800:
            print(f"  -> beats baseline 908 by {(908-best_ppl)/908*100:.1f}%")
            print("     LAG HYPOTHESIS CONFIRMED.  Launch the full grid.")
        elif best_ppl < 900:
            print(f"  -> partial improvement over 908.")
            print("     Lag hypothesis partly confirmed; full grid is still")
            print("     worth running to find the (gamma, momentum, damping) optimum.")
        else:
            print(f"  -> at-or-worse than baseline 908 even with LR-compensation.")
            print("     Lag hypothesis FALSIFIED.  Momentum's role isn't just")
            print("     averaging; investigate the cross-term-as-noise idea")
            print("     before launching the full grid.")
    print("=" * 72)


if __name__ == "__main__":
    main()
