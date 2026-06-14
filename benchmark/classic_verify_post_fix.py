"""
benchmark/classic_verify_post_fix.py

Re-run ClassicKFAC at 3 matched-screen cells post-fix to verify the bug
fix didn't change Classic's behavior.  All Classic baselines we've been
citing for variant comparisons (the matched-screen JSONs) were collected
BEFORE the damping bug fix.  This script empirically checks whether
those numbers are still valid by re-measuring three cells that span the
regimes where the Vered-vs-Classic story has different shapes:

    tied                 (mom=0.7, lr=2e-3)  matched 922   -- champion
    classic_wins         (mom=0.9, lr=8e-3)  matched 1667  -- Classic +142 vs Vered
    classic_dominates    (mom=0.9, lr=4e-2)  matched 2891  -- Classic +716 vs Vered

All runs use the same fixed config as the matched screen: gamma=0.9,
damping=1e-4, grad_clip=300, constant_warmup, max_steps=1000, seed=42.

Verdict (printed at end):
    * All 3 within +/- 5 ppl of matched-screen baselines
      -> bug fix did not change Classic; baselines are valid.
    * Any cell drifts by >5 ppl
      -> Classic's behavior changed; baselines need correction and the
         variant comparison should be re-derived from these new numbers.

Wall time: ~45 minutes total (3 cells x ~15 min).

Usage:
    python benchmark/classic_verify_post_fix.py
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


# Fixed config — exact match to matched-screen Classic runs.
FIXED = {
    "variant":     "ClassicKFAC",
    "gamma":       0.9,
    "damping":     1e-4,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "lr_schedule": "constant_warmup",
}

# (label, mom, kfac_lr, matched_screen_baseline_ppl)
CELLS = [
    ("tied",              0.7, 2e-3,  922.0),
    ("classic_wins",      0.9, 8e-3, 1667.0),
    ("classic_dominates", 0.9, 4e-2, 2891.0),
]


def out_path(mom: float, lr: float) -> Path:
    tag = f"m{mom:.2f}_lr{lr:.0e}_d1e-04_s1000_const"
    return OUT / f"classic_verify_{tag}.json"


def run_one(label: str, mom: float, lr: float, baseline: float,
            train_loader_factory, val_loader, vocab_size: int, pad_id: int,
            device, hw: Dict) -> Optional[Dict]:
    p = out_path(mom, lr)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists.  final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(FIXED, momentum=mom, kfac_lr=lr, verify_label=label,
               matched_screen_baseline=baseline)
    print()
    print("=" * 72)
    print(f"  Classic verify [{label}]: mom={mom:g}  lr={lr:.0e}  damping=1e-04")
    print(f"  Matched-screen baseline at this cell: {baseline:.0f} ppl")
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
            lr_schedule=cfg["lr_schedule"],
        )
    except Exception as e:
        crash_msg = f"{type(e).__name__}: {e}"
        print(f"\n  [!] CRASH during run_probe: {crash_msg}")
        res = {
            "status":         "diverged_crash",
            "error":          crash_msg,
            "final_ppl":      None,
            "val_ppls":       [],
            "val_times":      [],
            "median_step_ms": None,
        }
    wall = time.perf_counter() - t0

    val_ppls = res.get("val_ppls") or []
    slope: Optional[float] = None
    if len(val_ppls) >= 3:
        recent = val_ppls[-3:]
        if all(v is not None and v < 5000 for v in recent):
            slope = (recent[0] - recent[-1]) / 2.0

    saved = {
        "config":         cfg,
        "hw":             hw,
        "wall_s":         wall,
        "result":         res,
        "slope_per_100":  slope,
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    fin = res.get("final_ppl")
    fin_str = f"{fin:.0f}" if fin is not None else "DIV"
    sl_str = f"{slope:.1f}" if slope is not None else "n/a"
    delta = (fin - baseline) if fin is not None else None
    delta_str = f"{delta:+.0f}" if delta is not None else "--"
    print(f"  Saved -> {p.name}   final_ppl={fin_str}   "
          f"baseline={baseline:.0f}   delta={delta_str}   slope/100={sl_str}")
    return saved


def main():
    print("=" * 72)
    print("  ClassicKFAC post-fix verification (3 cells, ~45 min)")
    print("=" * 72)
    print(f"  Fixed:  variant={FIXED['variant']}, gamma={FIXED['gamma']}, "
          f"damping={FIXED['damping']:.0e}, clip={FIXED['grad_clip']:g}")
    print(f"          schedule={FIXED['lr_schedule']}, "
          f"max_steps={FIXED['max_steps']}, seed=42")
    print(f"  Cells (with matched-screen baselines):")
    for label, mom, lr, baseline in CELLS:
        print(f"    {label:>18}: (mom={mom:g}, lr={lr:.0e})  baseline = {baseline:.0f}")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs: List[Dict] = []
    for label, mom, lr, baseline in CELLS:
        saved = run_one(label, mom, lr, baseline,
                        train_loader_factory, val_loader,
                        vocab_size, pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # ----- Summary + verdict ---------------------------------------------
    print()
    print("=" * 72)
    print("  Classic post-fix verification summary")
    print("=" * 72)
    print(f"  {'label':>18}  {'mom':>5}  {'lr':>9}  "
          f"{'baseline':>10}  {'post_fix':>10}  {'delta':>+7}")
    print("  " + "-" * 70)
    deltas: List[Optional[float]] = []
    for label, mom, lr, baseline in CELLS:
        r = next((rr for rr in runs
                  if rr.get("config", {}).get("verify_label") == label), None)
        if r is None:
            print(f"  {label:>18}  {mom:>5.2f}  {lr:>9.0e}  "
                  f"{baseline:>10.0f}  {'(missing)':>10}  {'--':>+7}")
            deltas.append(None)
            continue
        fin = r["result"].get("final_ppl")
        if fin is None:
            print(f"  {label:>18}  {mom:>5.2f}  {lr:>9.0e}  "
                  f"{baseline:>10.0f}  {'DIV':>10}  {'--':>+7}")
            deltas.append(None)
            continue
        delta = fin - baseline
        print(f"  {label:>18}  {mom:>5.2f}  {lr:>9.0e}  "
              f"{baseline:>10.0f}  {fin:>10.0f}  {delta:>+7.0f}")
        deltas.append(delta)

    print()
    print("=" * 72)
    print("  Verdict")
    print("=" * 72)
    valid_deltas = [d for d in deltas if d is not None]
    if len(valid_deltas) < 3:
        print(f"  Incomplete data ({len(valid_deltas)}/3 cells valid); "
              f"verdict deferred.")
        return
    max_abs = max(abs(d) for d in valid_deltas)
    if max_abs < 5:
        print(f"  ALL 3 CELLS WITHIN +/- 5 ppl OF BASELINE  (max |delta| = {max_abs:.0f}).")
        print(f"  -> Bug fix did not change ClassicKFAC behavior.")
        print(f"     The matched-screen baselines (922 / 1667 / 2891) remain valid.")
        print(f"     The variant-comparison story stands as previously characterized:")
        print(f"     Vered ~ Classic at champion; Classic substantially better at")
        print(f"     high effective LR.")
    elif max_abs < 50:
        print(f"  MINOR DRIFT  (max |delta| = {max_abs:.0f} ppl).")
        print(f"  -> Bug fix made a small difference to Classic.  The matched-screen")
        print(f"     comparison story is qualitatively unchanged, but quantitative")
        print(f"     headlines should be re-derived from these new numbers.")
    else:
        print(f"  MATERIAL DRIFT  (max |delta| = {max_abs:.0f} ppl).")
        print(f"  -> Bug fix substantively changed Classic's behavior at some cells.")
        print(f"     Re-run the Classic matched screen before drawing any further")
        print(f"     variant-comparison conclusions.")
    print("=" * 72)


if __name__ == "__main__":
    main()
