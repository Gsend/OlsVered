"""
benchmark/classic_full_grid.py

Three-stage coordinate-descent grid for ClassicKFAC, mirroring
benchmark/vered_full_grid.py but with Classic-anchored sweep ranges.

Purpose:
    The Vered grid found a winner at gamma=0.3, mom=0.3, lambda=1e-4 (681 ppl).
    The current Classic comparison point (776 ppl) was at literature-default
    settings only — it never received a coordinate-descent grid.  This script
    closes the search-budget asymmetry so the Vered-vs-Classic comparison
    becomes methodologically symmetric.

Stage order (mirrors Vered grid: gamma -> momentum -> damping):
    Stage A: gamma sweep    at (mom=0.9, lambda=1e-4)   -- Classic's natural regime
    Stage B: momentum sweep at (best gamma from A, lambda=1e-4)
    Stage C: damping sweep  at (best gamma from A, best mom from B)

Theoretical predictions to be tested:
    - Classic should perform best at high gamma + moderate damping
      (gamma in {0.7, 0.9, 0.95}, lambda in {1e-3, 1e-4}).
    - Classic at (gamma=0.3, lambda=1e-4) should DIVERGE or strongly
      underperform: the kappa^4 forward-error scaling means low gamma
      (noisy curvature) + low lambda (under-regularized inverse) is
      Classic's worst regime.  Including this cell is the most
      theoretically-loaded data point of the entire investigation.

Each run = 5000 steps, ~75 minutes wall time (Classic is slightly faster
than Vered but use the same conservative estimate).
Total new runs: ~12 (no legacy reuse since prior Classic 776-ppl result
came from an earlier benchmark with different filename convention).

Resume support: skips any (gamma, momentum, damping) cell whose JSON exists.

Reference points:
    Vered post-fix (best, gamma=0.3, mom=0.3, lambda=1e-4):  681 ppl
    Classic        (mom=0.9, lambda=1e-4, default gamma):    776 ppl   <-- to beat
    Vered pre-fix  (buggy EMA, mom=0.9):                     756 ppl

Decision rule:
    Classic best < 700 ppl:   gap to Vered <= 20 ppl; reframe headline
                              as "marginal win with theoretical justification"
    Classic best 700-770 ppl: expected outcome; Vered still wins by 20-90 ppl
    Classic best >= 776:      grid found nothing better; strongest case for Vered
    (gamma=0.3) cell of A diverges: kappa^4 prediction empirically confirmed;
                                    additional headline result for the math doc

Usage:
    python benchmark/classic_full_grid.py
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

# Fixed across the entire grid
BASE = {
    "variant":   "ClassicKFAC",
    "kfac_lr":   8e-3,        # same as Vered grid for fair comparison
    "grad_clip": 60.0,
    "max_steps": 5000,
}

# Stage A: gamma sweep at (mom=0.9, lambda=1e-4) -- Classic's natural regime
#   gamma=0.7 is the published-default starting point.
#   gamma=0.5, 0.9, 0.95 explore around it.
#   gamma=0.3 is the kappa^4-stability test: Classic should struggle here
#   (Vered won at gamma=0.3, but Vered has kappa^1 stability).
STAGE_A_GAMMAS  = [0.7, 0.9, 0.5, 0.95, 0.3]
STAGE_A_MOM     = 0.9
STAGE_A_DAMPING = 1e-4

# Stage B: momentum sweep at (best gamma from A, lambda=1e-4)
#   Classic literature uses momentum=0.9 almost universally.  Sweep around
#   that to confirm.  mom=0.95 is the upper end of stable momentum;
#   mom=0.5 explores the lower-momentum regime that helped Vered.
STAGE_B_MOMS    = [0.9, 0.7, 0.5, 0.95]
STAGE_B_DAMPING = 1e-4

# Stage C: damping sweep at (best gamma, best mom)
#   Classic's typical damping is 1e-3 to 1e-4.  Test 1e-5 and 1e-6 to
#   probe the kappa^4-instability frontier: how low can Classic go
#   before the inverse blows up small-eigenvalue noise into the step?
STAGE_C_LAMBDAS = [1e-4, 1e-3, 1e-5, 1e-6]


# ---- Helpers ---------------------------------------------------------------

def make_tag(gamma: float, momentum: float, damping: float) -> str:
    return f"g{gamma:.2f}_m{momentum:.2f}_l{damping:.0e}"


def out_path(gamma: float, momentum: float, damping: float) -> Path:
    return OUT / f"classic_grid_{make_tag(gamma, momentum, damping)}.json"


def run_one(gamma: float, momentum: float, damping: float,
            train_loader_factory, val_loader, vocab_size: int, pad_id: int,
            device, hw: Dict) -> Optional[Dict]:
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

    cfg = dict(BASE, gamma=gamma, momentum=momentum, damping=damping)
    print()
    print("=" * 72)
    print(f"  Classic grid run: gamma={gamma:g}  mom={momentum:g}  "
          f"lambda={damping:.0e}  lr={cfg['kfac_lr']:.0e}")
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
            for t in [1000.0, 800.0, 750.0, 700.0]
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


def find_best(runs: List[Optional[Dict]]) -> Optional[Dict]:
    valid = [r for r in runs if r and r["result"].get("final_ppl") is not None
             and r["result"]["final_ppl"] < 5000]   # filter divergence
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
        # Flag suspected divergence (very high ppl but stable status)
        flag = ""
        if ppl is not None and ppl > 5000:
            flag = "  [DIVERGED]"
        print(f"  {cfg['gamma']:>6.2f} {cfg['momentum']:>5.2f} "
              f"{cfg['damping']:>10.0e}  {ppl_str:>10}  "
              f"{wall:>11.1f}  {res['status']:>10}{flag}")
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
    print("# STAGE A: gamma sweep (Classic-anchored)")
    print(f"#   gammas = {STAGE_A_GAMMAS}")
    print(f"#   fixed: mom={STAGE_A_MOM}, lambda={STAGE_A_DAMPING:.0e}, "
          f"lr={BASE['kfac_lr']:.0e}")
    print("#   note: gamma=0.3 cell is theoretically loaded "
          "(may diverge -- kappa^4 test)")
    print("#" * 72)
    runs = []
    for g in STAGE_A_GAMMAS:
        runs.append(run_one(g, STAGE_A_MOM, STAGE_A_DAMPING,
                            loaders[0], loaders[1], vocab_size, pad_id,
                            device, hw))
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
                            loaders[0], loaders[1], vocab_size, pad_id,
                            device, hw))
    print_stage_summary("B", runs)
    return runs


def stage_c(best_a: Optional[Dict], best_b: Optional[Dict],
            loaders, vocab_size, pad_id, device, hw) -> List[Optional[Dict]]:
    if best_a is None:
        print("\n[!] No Stage A result; cannot run Stage C")
        return []
    g = best_a["config"]["gamma"]
    if (best_b
            and best_b["result"].get("final_ppl") is not None
            and best_b["result"]["final_ppl"] <= best_a["result"]["final_ppl"]):
        m = best_b["config"]["momentum"]
        src = "Stage B"
    else:
        m = STAGE_A_MOM
        src = "Stage A (mom=0.9 unchanged)"

    print()
    print("#" * 72)
    print("# STAGE C: damping (Tikhonov) sweep")
    print(f"#   lambdas = {STAGE_C_LAMBDAS}")
    print(f"#   fixed: gamma={g}, mom={m} (from {src}), "
          f"lr={BASE['kfac_lr']:.0e}")
    print("#   note: lambda=1e-5, 1e-6 cells probe Classic's kappa^4 "
          "instability frontier")
    print("#" * 72)
    runs = []
    for l in STAGE_C_LAMBDAS:
        runs.append(run_one(g, m, l,
                            loaders[0], loaders[1], vocab_size, pad_id,
                            device, hw))
    print_stage_summary("C", runs)
    return runs


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  ClassicKFAC: full coordinate-descent grid")
    print("    Stage A: gamma sweep (Classic-anchored) at mom=0.9, lambda=1e-4")
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
    print("  Final Classic grid summary across all stages")
    print("=" * 72)
    print_stage_summary("ALL", all_runs)

    print()
    print("  Reference points (all 5000 steps):")
    print("    Vered post-fix winner:                      681 ppl")
    print("    Vered pre-fix (buggy EMA):                  756 ppl")
    print("    Classic untuned (mom=0.9, lambda=1e-4):     776 ppl  <-- to beat")

    best = find_best(all_runs)
    if best:
        ppl = best["result"]["final_ppl"]
        bc = best["config"]
        if ppl < 681:
            verdict = ("CLASSIC BEATS VERED at its tuned optimum -- the gap "
                       "is reversed; investigate")
        elif ppl < 700:
            verdict = ("Gap to Vered <= 20 ppl; reframe headline as "
                       "marginal win with theoretical justification")
        elif ppl < 750:
            verdict = ("Gap to Vered ~50 ppl; expected outcome; Vered's win "
                       "is real but smaller than untuned-classic suggested")
        elif ppl < 776:
            verdict = ("Modest tuning gain over untuned Classic; Vered's "
                       "~95 ppl margin holds")
        else:
            verdict = ("No tuning gain over default; strongest case for Vered")
        print()
        print(f"  Classic overall best: {ppl:.0f} ppl at gamma={bc['gamma']:g}, "
              f"mom={bc['momentum']:g}, lambda={bc['damping']:.0e}")
        print(f"  Vered vs Classic gap: {ppl - 681:.0f} ppl")
        print(f"  -> {verdict}")

    # Check for divergence at the gamma=0.3 stability test
    g03_run = next((r for r in (runs_a or [])
                    if r and r["config"]["gamma"] == 0.3
                    and abs(r["config"]["momentum"] - STAGE_A_MOM) < 1e-9
                    and abs(r["config"]["damping"] - STAGE_A_DAMPING) < 1e-12), None)
    if g03_run:
        ppl = g03_run["result"].get("final_ppl")
        print()
        if ppl is None or ppl > 5000:
            print("  [STABILITY] Classic at gamma=0.3 DIVERGED.")
            print("              kappa^4 forward-error prediction empirically")
            print("              confirmed.  This is a headline result for the math doc.")
        elif ppl > 1000:
            print(f"  [STABILITY] Classic at gamma=0.3 reached {ppl:.0f} ppl")
            print("              (well behind Vered's 798 at the same setting).")
            print("              Partial confirmation of kappa^4 disadvantage at low gamma.")
        else:
            print(f"  [STABILITY] Classic at gamma=0.3 reached {ppl:.0f} ppl")
            print("              (similar to or better than Vered's 798).")
            print("              kappa^4 prediction NOT supported empirically;")
            print("              Classic tolerates low gamma better than expected.")
    print("=" * 72)


if __name__ == "__main__":
    main()
