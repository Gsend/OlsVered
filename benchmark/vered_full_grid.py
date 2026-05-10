"""
benchmark/vered_full_grid.py

Three-stage coordinate-descent grid for VeredKFAC post-fix.

Following the quick_momentum_test result that mom=0 at lr=8e-3 wins
(821 ppl, beating mom=0.9's 908), this script searches the
(gamma, momentum, damping) space stage-by-stage.

Stage order (gamma first, damping last):
    Stage A: gamma   sweep at (mom=0.0, lambda=1e-5, lr=8e-3)
    Stage B: momentum sweep at (best gamma from A, lambda=1e-5, lr=8e-3)
    Stage C: damping  sweep at (best gamma from A, best mom from B, lr=8e-3)

Each run = 5000 steps, ~75 minutes wall time.
Total new runs: ~10 (depending on which results we reuse from prior tests).
Estimated total wall time: 12-14 hours.

Resume support: skips any (gamma, momentum, damping) cell whose JSON exists.
The single existing result from quick_momentum_test_mom00_lr8e-03.json
(gamma=0.7, mom=0.0, lambda=1e-5 -> 821 ppl) is reused automatically
without re-running.

References (all at lr=8e-3, 5000 steps):
    Vered post-fix (mom=0.9, lambda=1e-5, gamma=0.7):  908 ppl
    Vered post-fix (mom=0.0, lambda=1e-5, gamma=0.7):  821 ppl  <-- prior best
    Classic        (mom=0.9, lambda=1e-4, gamma=0.7):  776 ppl
    Vered pre-fix  (buggy EMA, mom=0.9, gamma=0.7):    756 ppl  <-- target

Usage:
    python benchmark/vered_full_grid.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data, run_probe, OUT, time_to_ppl
from benchmark.gpu_benchmark import get_device, get_hardware_info


# ---- Configuration ---------------------------------------------------------

# Fixed across the entire grid
BASE = {
    "variant":   "VeredKFAC",
    "kfac_lr":   8e-3,
    "grad_clip": 60.0,
    "max_steps": 5000,
}

# Stage A: gamma sweep at (mom=0, lambda=1e-5)
#   Hypothesis: buggy-EMA pre-fix's gamma_eff was 0.7^2 = 0.49. That may be
#   a happier operating point for Vered's clean-but-unsmoothed natural
#   gradient than the post-fix's gamma=0.7. Test 0.5 first; 0.3 explores
#   even less smoothing; 0.9 is a sanity-check upper bound.
STAGE_A_GAMMAS  = [0.5, 0.3, 0.7, 0.9]   # gamma=0.7 reused from prior
STAGE_A_MOM     = 0.0
STAGE_A_DAMPING = 1e-5

# Stage B: momentum sweep at (best gamma from A, lambda=1e-5)
#   Hypothesis: a small amount of momentum may damp per-step direction
#   noise without inducing the lag that killed mom=0.9. Sweet spot is
#   probably mom in [0.2, 0.5]; mom=0.7 is a sanity check.
STAGE_B_MOMS    = [0.3, 0.5, 0.0, 0.7]   # mom=0.0 reused if best gamma=0.7
STAGE_B_DAMPING = 1e-5

# Stage C: damping sweep at (best gamma, best mom)
#   Vered's kappa^1 forward-error scaling means it should tolerate very
#   low damping; that's part of its theoretical advantage. Test that.
#   1e-4 is included as an upper-bound sanity check (Classic's regime).
STAGE_C_LAMBDAS = [1e-6, 1e-7, 1e-4, 1e-5]   # lambda=1e-5 reused as carry-over


# Legacy result reuse: maps (gamma, mom, damping) -> existing JSON path
LEGACY_RESULT_FILES = {
    (0.7, 0.0, 1e-5): OUT / "quick_momentum_test_mom00_lr8e-03.json",
}


# ---- Helpers ---------------------------------------------------------------

def make_tag(gamma: float, momentum: float, damping: float) -> str:
    return f"g{gamma:.2f}_m{momentum:.2f}_l{damping:.0e}"


def out_path(gamma: float, momentum: float, damping: float) -> Path:
    return OUT / f"vered_grid_{make_tag(gamma, momentum, damping)}.json"


def reuse_legacy(gamma: float, momentum: float, damping: float) -> bool:
    """Copy a legacy result to the canonical filename if applicable."""
    key = (round(gamma, 4), round(momentum, 4), float(damping))
    src = LEGACY_RESULT_FILES.get(key)
    if src is None or not src.exists():
        return False
    dst = out_path(gamma, momentum, damping)
    if dst.exists():
        return True
    shutil.copy2(src, dst)
    print(f"  [reuse] {src.name} -> {dst.name}")
    return True


def run_one(gamma: float, momentum: float, damping: float,
            train_loader_factory, val_loader, vocab_size: int, pad_id: int,
            device, hw: Dict) -> Optional[Dict]:
    p = out_path(gamma, momentum, damping)

    # Pull in legacy result if available
    if not p.exists():
        reuse_legacy(gamma, momentum, damping)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists. final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(BASE, gamma=gamma, momentum=momentum, damping=damping)
    print()
    print("=" * 72)
    print(f"  Vered grid run: gamma={gamma:g}  mom={momentum:g}  "
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
            for t in [1000.0, 800.0, 750.0, 700.0]
        },
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  Saved -> {p}")
    return saved


def find_best(runs: List[Optional[Dict]]) -> Optional[Dict]:
    valid = [r for r in runs if r and r["result"].get("final_ppl") is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: r["result"]["final_ppl"])


def print_stage_summary(stage: str, runs: List[Optional[Dict]]):
    print()
    print("=" * 72)
    print(f"  Stage {stage} summary")
    print("=" * 72)
    print(f"  {'gamma':>6} {'mom':>5} {'lambda':>10}  {'final_ppl':>10}  "
          f"{'wall (min)':>11}  {'status':>10}")
    print("  " + "-" * 65)
    for run in runs:
        if not run:
            continue
        cfg = run["config"]
        res = run["result"]
        ppl = res.get("final_ppl")
        ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
        wall = run["wall_s"] / 60
        print(f"  {cfg['gamma']:>6.2f} {cfg['momentum']:>5.2f} "
              f"{cfg['damping']:>10.0e}  {ppl_str:>10}  "
              f"{wall:>11.1f}  {res['status']:>10}")
    best = find_best(runs)
    if best:
        bc = best["config"]
        print(f"\n  Stage {stage} best: ppl={best['result']['final_ppl']:.0f} "
              f"at gamma={bc['gamma']:g}, mom={bc['momentum']:g}, "
              f"lambda={bc['damping']:.0e}")


# ---- Stages ----------------------------------------------------------------

def stage_a(loaders, vocab_size, pad_id, device, hw) -> List[Optional[Dict]]:
    print()
    print("#" * 72)
    print("# STAGE A: gamma sweep")
    print(f"#   gammas = {STAGE_A_GAMMAS}")
    print(f"#   fixed: mom={STAGE_A_MOM}, lambda={STAGE_A_DAMPING:.0e}, "
          f"lr={BASE['kfac_lr']:.0e}")
    print("#" * 72)
    runs = []
    for g in STAGE_A_GAMMAS:
        runs.append(run_one(g, STAGE_A_MOM, STAGE_A_DAMPING,
                            loaders[0], loaders[1], vocab_size, pad_id, device, hw))
    print_stage_summary("A", runs)
    return runs


def stage_b(best_a: Optional[Dict], loaders, vocab_size, pad_id, device, hw
            ) -> List[Optional[Dict]]:
    if best_a is None:
        print("\n[!] Stage A has no valid runs; skipping Stage B")
        return []
    g = best_a["config"]["gamma"]
    print()
    print("#" * 72)
    print("# STAGE B: momentum sweep")
    print(f"#   moms = {STAGE_B_MOMS}")
    print(f"#   fixed: gamma={g} (best from Stage A), "
          f"lambda={STAGE_B_DAMPING:.0e}, lr={BASE['kfac_lr']:.0e}")
    print("#" * 72)
    runs = []
    for m in STAGE_B_MOMS:
        runs.append(run_one(g, m, STAGE_B_DAMPING,
                            loaders[0], loaders[1], vocab_size, pad_id, device, hw))
    print_stage_summary("B", runs)
    return runs


def stage_c(best_a: Optional[Dict], best_b: Optional[Dict],
            loaders, vocab_size, pad_id, device, hw) -> List[Optional[Dict]]:
    if best_a is None:
        print("\n[!] No Stage A result; cannot run Stage C")
        return []
    g = best_a["config"]["gamma"]

    # Use best mom from B if it improved over A's best, else fall back to mom=0.
    if (best_b
            and best_b["result"].get("final_ppl") is not None
            and best_b["result"]["final_ppl"] <= best_a["result"]["final_ppl"]):
        m = best_b["config"]["momentum"]
        src = "Stage B"
    else:
        m = STAGE_A_MOM
        src = "Stage A (mom=0 unchanged)"

    print()
    print("#" * 72)
    print("# STAGE C: damping (Tikhonov) sweep")
    print(f"#   lambdas = {STAGE_C_LAMBDAS}")
    print(f"#   fixed: gamma={g}, mom={m} (from {src}), "
          f"lr={BASE['kfac_lr']:.0e}")
    print("#" * 72)
    runs = []
    for l in STAGE_C_LAMBDAS:
        runs.append(run_one(g, m, l,
                            loaders[0], loaders[1], vocab_size, pad_id, device, hw))
    print_stage_summary("C", runs)
    return runs


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered post-fix: full coordinate-descent grid")
    print("    Stage A: gamma sweep")
    print("    Stage B: momentum sweep at best gamma")
    print("    Stage C: damping sweep at best (gamma, mom)")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1
    loaders = (train_loader_factory, val_loader)

    runs_a = stage_a(loaders, vocab_size, pad_id, device, hw)
    best_a = find_best(runs_a)

    runs_b = stage_b(best_a, loaders, vocab_size, pad_id, device, hw)
    best_b = find_best(runs_b)

    runs_c = stage_c(best_a, best_b, loaders, vocab_size, pad_id, device, hw)

    # Final summary
    all_runs: List[Optional[Dict]] = []
    all_runs.extend(runs_a or [])
    all_runs.extend(runs_b or [])
    all_runs.extend(runs_c or [])

    print()
    print("=" * 72)
    print("  Final grid summary across all stages")
    print("=" * 72)
    print_stage_summary("ALL", all_runs)

    print()
    print("  Reference points (all 5000 steps):")
    print("    Vered post-fix (gamma=0.7, mom=0.9, lambda=1e-5):  908 ppl")
    print("    Vered post-fix (gamma=0.7, mom=0.0, lambda=1e-5):  821 ppl")
    print("    Classic        (mom=0.9, lambda=1e-4):             776 ppl")
    print("    Vered pre-fix  (buggy EMA, mom=0.9):               756 ppl  <-- target")

    best = find_best(all_runs)
    if best:
        ppl = best["result"]["final_ppl"]
        bc = best["config"]
        if ppl < 700:
            verdict = "BREAKTHROUGH: clear win over Classic and pre-fix"
        elif ppl < 750:
            verdict = "WIN: matches or beats pre-fix; Vered post-fix is now state of the art"
        elif ppl < 820:
            verdict = "Closes most of the gap to pre-fix; useful but not breakthrough"
        else:
            verdict = "Gap to 756 is structural; investigate per-step direction quality"
        print()
        print(f"  Overall best: {ppl:.0f} ppl at gamma={bc['gamma']:g}, "
              f"mom={bc['momentum']:g}, lambda={bc['damping']:.0e}")
        print(f"  -> {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
