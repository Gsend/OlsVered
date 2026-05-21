"""
benchmark/classic_2d_mom_lr_screen.py

2D screening grid for ClassicKFAC: momentum x learning rate.  Built as
the Classic counterpart to vered_2d_mom_lr_screen.py so the two
variants can be compared apples-to-apples.

Two operating-point configurations are supported via --config:

    matched   gamma=0.9, damp=1e-4, clip=300  (matches Vered's screen
              exactly; isolates the inversion method as the only
              variable.  Methodologically cleanest for kappa-scaling
              claims.)

    own       gamma=0.7, damp=1e-3, clip=60   (Classic's empirically
              best operating point per existing classic_grid_*.json
              data.  Gives Classic its best face for the "which
              variant wins in practice" comparison.)

Same (mom, lr) grid as the Vered screen so the cells line up directly:
    MOMS = [0.0, 0.3, 0.7, 0.9]
    LRS  = [5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 2e-2, 4e-2]

Schedule: lr_schedule="constant_warmup" (matches the Vered screen).
Length:   max_steps=1000 (matches the Vered screen).

Output JSONs (config_tag is "matched" or "own"):
    benchmark/results/classic_2dscreen_{config_tag}_m{mom}_lr{lr}_s1000_const.json

Resume support: cells with existing JSON for the chosen config are skipped.

Usage:
    python benchmark/classic_2d_mom_lr_screen.py --config matched
    python benchmark/classic_2d_mom_lr_screen.py --config own

Wall time: ~8 hr per config (32 cells x ~15 min/cell on RTX 3080 Laptop).
"""
from __future__ import annotations

import argparse
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


# ---- Configuration --------------------------------------------------------

CONFIGS = {
    "matched": {
        # Vered-matched: gamma=0.9, damp=1e-4, clip=300.  Isolates inversion
        # method.  Comparable cell-by-cell to vered_2dscreen_*_const.json.
        "variant":     "ClassicKFAC",
        "gamma":       0.9,
        "damping":     1e-4,
        "grad_clip":   300.0,
        "max_steps":   1000,
        "lr_schedule": "constant_warmup",
    },
    "own": {
        # Classic-preferred: gamma=0.7, damp=1e-3, clip=60.  Classic's
        # empirical best from classic_grid_*.json (gamma=0.7, mom=0.7,
        # damp=1e-3 reached 593 ppl @ 5000 steps -- the best Classic number
        # on this task).  Gives Classic its best face.
        "variant":     "ClassicKFAC",
        "gamma":       0.7,
        "damping":     1e-3,
        "grad_clip":   60.0,
        "max_steps":   1000,
        "lr_schedule": "constant_warmup",
    },
}

MOMS = [0.0, 0.3, 0.7, 0.9]
LRS  = [5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 2e-2, 4e-2]


# ---- Helpers --------------------------------------------------------------

def make_tag(config_tag: str, mom: float, lr: float, max_steps: int) -> str:
    return f"{config_tag}_m{mom:.2f}_lr{lr:.0e}_s{max_steps}_const"


def out_path(config_tag: str, mom: float, lr: float, max_steps: int) -> Path:
    return OUT / f"classic_2dscreen_{make_tag(config_tag, mom, lr, max_steps)}.json"


def run_one(config_tag: str, FIXED: Dict, mom: float, lr: float,
            train_loader_factory, val_loader, vocab_size: int, pad_id: int,
            device, hw: Dict) -> Optional[Dict]:
    p = out_path(config_tag, mom, lr, FIXED["max_steps"])

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
    print(f"  Classic 2D screen [{config_tag}]: mom={mom:g}  lr={lr:.0e}")
    print(f"  Fixed: gamma={FIXED['gamma']}, damp={FIXED['damping']:.0e}, "
          f"clip={FIXED['grad_clip']:g}, steps={FIXED['max_steps']}, "
          f"schedule={FIXED['lr_schedule']}")
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
    slope_per_100: Optional[float] = None
    if len(val_ppls) >= 3:
        recent = val_ppls[-3:]
        if all(v is not None and v < 5000 for v in recent):
            slope_per_100 = (recent[0] - recent[-1]) / 2.0

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
        "config_tag":     config_tag,
        "hw":             hw,
        "wall_s":         wall,
        "result":         res,
        "slope_per_100":  slope_per_100,
        "time_to_target": ttt,
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    final = res.get("final_ppl")
    final_str = f"{final:.0f}" if final is not None else "DIV"
    slope_str = f"{slope_per_100:.1f}" if slope_per_100 is not None else "n/a"
    print(f"  Saved -> {p.name}   final_ppl={final_str}   slope/100={slope_str}")
    return saved


# ---- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", choices=list(CONFIGS.keys()), required=True,
                    help="Which fixed-config to run.")
    args = ap.parse_args()

    config_tag = args.config
    FIXED = CONFIGS[config_tag]

    print("=" * 72)
    print(f"  Classic 2D (mom x lr) screen  [config={config_tag}]")
    print(f"  Fixed:  variant={FIXED['variant']}, gamma={FIXED['gamma']}, "
          f"damp={FIXED['damping']:.0e}, clip={FIXED['grad_clip']:g}")
    print(f"          max_steps={FIXED['max_steps']}, "
          f"schedule={FIXED['lr_schedule']}")
    print(f"  Sweep:  mom in {MOMS}, lr in {LRS}  "
          f"({len(MOMS) * len(LRS)} cells)")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs: List[Dict] = []
    for mom, lr in product(MOMS, LRS):
        saved = run_one(config_tag, FIXED, mom, lr,
                        train_loader_factory, val_loader,
                        vocab_size, pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # ----- Summary tables --------------------------------------------------
    print()
    print("=" * 72)
    print(f"  Final-ppl table  [config={config_tag}]")
    print("=" * 72)
    header = "  mom \\ lr |" + "".join(f"  {lr:>8.0e}" for lr in LRS)
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
            ppl = r["result"].get("final_ppl")
            if ppl is None or ppl >= 5000:
                row += f"  {'DIV':>8}"
            else:
                row += f"  {ppl:>8.0f}"
        print(row)

    print()
    print("  Slope/100 (late-run descent rate; higher = more headroom):")
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

    # ----- Top 5 -----------------------------------------------------------
    valid = [r for r in runs
             if r["result"].get("final_ppl") is not None
             and r["result"]["final_ppl"] < 5000]
    print()
    if valid:
        valid_sorted = sorted(valid, key=lambda r: r["result"]["final_ppl"])
        print("=" * 72)
        print(f"  Top 5 cells by final_ppl at step 1000  [config={config_tag}]")
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
    else:
        print("  All cells diverged or produced no valid result.")
    print("=" * 72)


if __name__ == "__main__":
    main()
