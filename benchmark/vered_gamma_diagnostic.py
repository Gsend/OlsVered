"""
benchmark/vered_gamma_diagnostic.py

Tests whether Vered's optimum gamma matches Classic's.

The 2D screen fixed gamma=0.9.  Classic's existing grid data
(classic_grid_g*_m*_l*.json) strongly suggests gamma=0.7 is the
sweet spot for Classic on this task:

    Classic best @ ~1000 steps:  gamma=0.7, mom=0.7, damp=1e-3 -> 942 ppl
    Classic best @ 5000 steps:                                 -> 593 ppl

Vered's current champion (from the constant_warmup screen):
    Vered  @ 1000 steps:  gamma=0.9, mom=0.7, lr=2e-3 -> 921 ppl

If Vered shares Classic's gamma preference, dropping gamma to 0.7
should improve on 921 ppl.  If gamma=0.7 ties or loses, the gamma
dependence of the two optimizers differs and the Vered refinement
grid can stay at gamma=0.9.

Cells tested (~15 min each, ~30 min total):
    (gamma=0.7, mom=0.7, lr=2e-3)   -- Classic-mom, Vered-lr (matched champion)
    (gamma=0.7, mom=0.7, lr=4e-3)   -- bracket above

Same other params as the screen: damp=1e-4, grad_clip=300, max_steps=1000,
lr_schedule=constant_warmup.

Output JSONs land at:
    benchmark/results/vered_gamma_diag_g0.70_m0.70_lr{lr}_s1000_const.json
(distinct from screen filenames so resume support won't mix them.)

Usage:
    python benchmark/vered_gamma_diagnostic.py
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

from benchmark.stability_benchmark import build_data, run_probe, OUT
from benchmark.gpu_benchmark import get_device, get_hardware_info


FIXED = {
    "variant":     "VeredKFAC",
    "damping":     1e-4,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "lr_schedule": "constant_warmup",
}

CELLS = [
    {"gamma": 0.7, "momentum": 0.7, "kfac_lr": 2e-3},
    {"gamma": 0.7, "momentum": 0.7, "kfac_lr": 4e-3},
]

# Reference points for the summary verdict.
CURRENT_CHAMPION = {
    "label": "gamma=0.9, mom=0.7, lr=2e-3",
    "ppl":   921.0,
}
CLASSIC_BEST = {
    "label": "gamma=0.7, mom=0.7, damp=1e-3, lr=8e-3",
    "ppl":   942.0,
}


def out_path(gamma: float, mom: float, lr: float) -> Path:
    return OUT / (
        f"vered_gamma_diag_g{gamma:.2f}_m{mom:.2f}_lr{lr:.0e}"
        f"_s{FIXED['max_steps']}_const.json"
    )


def run_cell(cell: Dict, train_loader_factory, val_loader,
             vocab_size: int, pad_id: int, device, hw: Dict) -> Optional[Dict]:
    gamma = cell["gamma"]; mom = cell["momentum"]; lr = cell["kfac_lr"]
    p = out_path(gamma, mom, lr)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists.  final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(FIXED, **cell)
    print()
    print("=" * 72)
    print(f"  Gamma=0.7 diagnostic: gamma={gamma:g}  mom={mom:g}  lr={lr:.0e}")
    print(f"  Comparing against current Vered champion: "
          f"{CURRENT_CHAMPION['ppl']:.0f} ppl")
    print(f"  (at {CURRENT_CHAMPION['label']})")
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
            "status":     "diverged_crash",
            "error":      crash_msg,
            "final_ppl":  None,
            "val_ppls":   [],
            "val_times":  [],
        }
    wall = time.perf_counter() - t0

    val_ppls = res.get("val_ppls") or []
    slope: Optional[float] = None
    if len(val_ppls) >= 3:
        recent = val_ppls[-3:]
        if all(v is not None and v < 5000 for v in recent):
            slope = (recent[0] - recent[-1]) / 2.0

    saved = {
        "config":        cfg,
        "hw":            hw,
        "wall_s":        wall,
        "result":        res,
        "slope_per_100": slope,
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    final = res.get("final_ppl")
    final_str = f"{final:.0f}" if final is not None else "DIVERGED"
    slope_str = f"{slope:.1f}" if slope is not None else "n/a"
    print(f"  Saved -> {p.name}   final_ppl={final_str}   slope/100={slope_str}")
    return saved


def main():
    print("=" * 72)
    print("  Vered gamma=0.7 diagnostic")
    print("=" * 72)
    print(f"  Hypothesis: Classic's grid shows gamma=0.7 beats gamma=0.9.")
    print(f"  If Vered shares this preference, ppl should drop below "
          f"{CURRENT_CHAMPION['ppl']:.0f}.")
    print(f"  Reference - Classic best at step ~1000: "
          f"{CLASSIC_BEST['ppl']:.0f} ppl  ({CLASSIC_BEST['label']})")
    print(f"  Wall-time budget: ~{15*len(CELLS)} min for {len(CELLS)} cells")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    results: List[Dict] = []
    for cell in CELLS:
        saved = run_cell(cell, train_loader_factory, val_loader,
                         vocab_size, pad_id, device, hw)
        if saved is not None:
            results.append(saved)

    # ----- Summary table ---------------------------------------------------
    print()
    print("=" * 72)
    print("  Gamma=0.7 diagnostic summary")
    print("=" * 72)
    print(f"  {'gamma':>6} {'mom':>5} {'lr':>10} {'final_ppl':>10} "
          f"{'vs champ':>10} {'slope/100':>10}")
    print("  " + "-" * 60)
    best_ppl = None
    best_cfg = None
    for r in results:
        c = r["config"]
        ppl = r["result"].get("final_ppl")
        slope = r.get("slope_per_100")
        ppl_str = f"{ppl:.0f}" if ppl is not None else "DIV"
        diff_str = (f"{ppl - CURRENT_CHAMPION['ppl']:+.0f}"
                    if ppl is not None else "--")
        slope_str = f"{slope:.1f}" if slope is not None else "--"
        print(f"  {c['gamma']:>6.2f} {c['momentum']:>5.2f} {c['kfac_lr']:>10.0e} "
              f"{ppl_str:>10} {diff_str:>10} {slope_str:>10}")
        if ppl is not None and (best_ppl is None or ppl < best_ppl):
            best_ppl = ppl
            best_cfg = c

    # ----- Verdict ---------------------------------------------------------
    print()
    if best_ppl is None:
        print("  Verdict: ALL CELLS DIVERGED.  Vered may not be stable at gamma=0.7,")
        print("           or this region needs different damping/clip.  Stick with")
        print("           gamma=0.9 for the refinement grid.")
    elif best_ppl < CURRENT_CHAMPION["ppl"] - 5:   # 5-ppl noise tolerance
        delta = CURRENT_CHAMPION["ppl"] - best_ppl
        print(f"  Verdict: gamma=0.7 BEATS gamma=0.9 by {delta:.0f} ppl at step 1000.")
        print(f"           Best cell: gamma={best_cfg['gamma']:.2f}, "
              f"mom={best_cfg['momentum']:.2f}, lr={best_cfg['kfac_lr']:.0e}")
        print(f"           -> Add gamma=0.7 cells to the 5000-step refinement grid.")
        print(f"           -> Consider extending the screen with a gamma sweep.")
    elif best_ppl < CURRENT_CHAMPION["ppl"] + 5:
        print(f"  Verdict: gamma=0.7 TIES gamma=0.9 (within ~5 ppl noise).")
        print(f"           Vered is less gamma-sensitive than Classic.  Keep")
        print(f"           gamma=0.9 for refinement; no need to extend screen.")
    else:
        delta = best_ppl - CURRENT_CHAMPION["ppl"]
        print(f"  Verdict: gamma=0.7 is WORSE than gamma=0.9 by {delta:.0f} ppl.")
        print(f"           Vered's gamma preference differs from Classic.  Keep")
        print(f"           gamma=0.9 for refinement.  (Worth noting: this is")
        print(f"           interesting in its own right -- the optimizers respond")
        print(f"           differently to the EMA gamma despite same K-FAC structure.)")
    print("=" * 72)


if __name__ == "__main__":
    main()
