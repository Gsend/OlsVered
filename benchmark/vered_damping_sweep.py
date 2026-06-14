"""
benchmark/vered_damping_sweep.py

Empirical damping sweep for Vered K-FAC.

Tests the question the bidirectional-ladder policy in
diagnostic/damping.py was designed around: does Vered actually benefit
from damping smaller than the prior 1e-6 / 1e-4 range, or is the
kappa^1 numerical freedom irrelevant to training because the statistical
optimum of damping is unchanged by the solver?

Method:
    For each of two cells from the matched Vered screen, run a short
    (1000-step) probe with constant_warmup at each of several damping
    values.  Cells chosen so the two together cover the regime where
    damping is suspected to matter least (well-conditioned A, the
    optimum cell) and most (high effective LR, where matched-screen
    Vered/Classic deltas were biggest).

Cells:
    A. (gamma=0.9, mom=0.7, lr=2e-3) — matched-screen champion (Vered
        ≈ Classic ≈ 921 ppl at the pre-fix damping=1e-6 effective).
    B. (gamma=0.9, mom=0.9, lr=8e-3) — high eff_lr (8e-2); matched
        screen showed |delta|=142 there, the regime where damping
        differences are most likely to bite.

Damping values (covers the proposed Vered ladder + the conventional
Classic floor):
    1e-8, 1e-7, 1e-6, 1e-5, 1e-4

Total: 2 cells x 5 dampings = 10 runs x ~15 min = ~2.5 hours.

Outputs:
    benchmark/results/vered_damp_m{mom}_lr{lr}_d{damp}_s1000_const.json

Interpretation guide (printed at the end):
    * Final ppl roughly monotonic with damping
      -> the proposed direction of the bidirectional ladder
        (smaller first) is supported -- ppl drops as damping shrinks.
    * Final ppl roughly monotonic the other way -> bidirectional should
      flip to up-only; smaller damping hurts.
    * U-shaped with the optimum near 1e-5 to 1e-6 -> the conventional
      "centre of the ladder is the right answer" view holds and
      bidirectional is overkill.
    * The two cells disagree -> damping optimum depends on regime;
      either a single ladder is the wrong abstraction or the policy
      needs per-eff_lr branches.

Usage:
    python benchmark/vered_damping_sweep.py
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


# ---- Configuration --------------------------------------------------------

FIXED = {
    "variant":     "VeredKFAC",
    "gamma":       0.9,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "lr_schedule": "constant_warmup",
}

# (label, mom, kfac_lr)
CELLS = [
    ("A_champion", 0.7, 2e-3),
    ("B_high_eff_lr", 0.9, 8e-3),
]

DAMPINGS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 2e-4, 5e-4, 1e-3]
# Upper extension added after the first pass showed monotonic improvement
# toward larger damping with 1e-4 as the best of the original range.  The
# new values bracket the upper edge at half-decade resolution to find the
# actual optimum and detect any post-peak decline.


# ---- Helpers --------------------------------------------------------------

def out_path(mom: float, lr: float, damp: float) -> Path:
    tag = f"m{mom:.2f}_lr{lr:.0e}_d{damp:.0e}_s{FIXED['max_steps']}_const"
    return OUT / f"vered_damp_{tag}.json"


def run_one(label: str, mom: float, lr: float, damping: float,
            train_loader_factory, val_loader, vocab_size: int, pad_id: int,
            device, hw: Dict) -> Optional[Dict]:
    p = out_path(mom, lr, damping)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists. final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    cfg = dict(FIXED, momentum=mom, kfac_lr=lr, damping=damping, cell_label=label)
    print()
    print("=" * 72)
    print(f"  Vered damping sweep [{label}]: mom={mom:g}  lr={lr:.0e}  "
          f"damping={damping:.0e}")
    print(f"  Fixed: gamma={FIXED['gamma']}, clip={FIXED['grad_clip']:g}, "
          f"steps={FIXED['max_steps']}, schedule={FIXED['lr_schedule']}")
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
        "slope_per_100":  slope,
        "time_to_target": ttt,
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    fin = res.get("final_ppl")
    fin_str = f"{fin:.0f}" if fin is not None else "DIV"
    sl_str = f"{slope:.1f}" if slope is not None else "n/a"
    print(f"  Saved -> {p.name}   final_ppl={fin_str}   slope/100={sl_str}")
    return saved


# ---- Main -----------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered damping sweep — bidirectional-ladder validation")
    print("=" * 72)
    print(f"  Fixed:  variant={FIXED['variant']}, gamma={FIXED['gamma']}, "
          f"clip={FIXED['grad_clip']:g}, schedule={FIXED['lr_schedule']}")
    print(f"  Cells:  {[(l,m,lr) for (l,m,lr) in CELLS]}")
    print(f"  Damping ladder:  {DAMPINGS}")
    print(f"  Wall time est:   {len(CELLS) * len(DAMPINGS) * 15} min")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    runs: List[Dict] = []
    for (label, mom, lr), damp in product(CELLS, DAMPINGS):
        saved = run_one(label, mom, lr, damp,
                        train_loader_factory, val_loader,
                        vocab_size, pad_id, device, hw)
        if saved is not None:
            runs.append(saved)

    # ----- Summary table per cell -----------------------------------------
    print()
    print("=" * 72)
    print("  Damping sweep summary")
    print("=" * 72)
    for label, mom, lr in CELLS:
        print()
        print(f"  Cell [{label}]: mom={mom:g}, lr={lr:.0e}")
        print(f"    {'damping':>10}  {'final_ppl':>10}  {'slope/100':>10}  "
              f"{'status':>10}")
        print("    " + "-" * 50)
        for damp in DAMPINGS:
            r = next((rr for rr in runs
                      if abs(rr["config"]["momentum"] - mom) < 1e-9
                      and abs(rr["config"]["kfac_lr"] - lr) < 1e-12
                      and abs(rr["config"]["damping"] - damp) < 1e-30), None)
            if r is None:
                print(f"    {damp:>10.0e}  {'(missing)':>10}")
                continue
            res = r["result"]
            fin = res.get("final_ppl")
            sl = r.get("slope_per_100")
            fin_str = f"{fin:.0f}" if fin is not None else "DIV"
            sl_str = f"{sl:.1f}" if sl is not None else "n/a"
            status = res.get("status", "?")
            print(f"    {damp:>10.0e}  {fin_str:>10}  {sl_str:>10}  "
                  f"{status:>10}")

    # ----- Cross-cell verdict ---------------------------------------------
    print()
    print("=" * 72)
    print("  Verdict")
    print("=" * 72)
    for label, mom, lr in CELLS:
        finals = []
        for damp in DAMPINGS:
            r = next((rr for rr in runs
                      if abs(rr["config"]["momentum"] - mom) < 1e-9
                      and abs(rr["config"]["kfac_lr"] - lr) < 1e-12
                      and abs(rr["config"]["damping"] - damp) < 1e-30), None)
            if r is None or r["result"].get("final_ppl") is None:
                finals.append((damp, None))
            else:
                finals.append((damp, r["result"]["final_ppl"]))

        valid = [(d, f) for d, f in finals if f is not None]
        if len(valid) < 3:
            print(f"  [{label}]: too few valid runs ({len(valid)}/{len(DAMPINGS)})"
                  f" -- inconclusive")
            continue

        best_damp, best_ppl = min(valid, key=lambda x: x[1])
        worst_damp, worst_ppl = max(valid, key=lambda x: x[1])
        delta = worst_ppl - best_ppl
        rng_str = f"{best_ppl:.0f}@{best_damp:.0e} ... {worst_ppl:.0f}@{worst_damp:.0e}"

        # Direction: does decreasing damping decrease ppl?
        d_low, d_mid, d_high = valid[0], valid[len(valid)//2], valid[-1]
        going_smaller_helps = d_low[1] < d_high[1]
        going_larger_helps  = d_low[1] > d_high[1]
        u_shape = d_mid[1] < min(d_low[1], d_high[1])

        if best_damp == valid[0][0] and going_smaller_helps:
            direction = "SMALLER IS BETTER -- bidirectional ladder direction supported"
        elif best_damp == valid[-1][0] and going_larger_helps:
            direction = "LARGER IS BETTER -- bidirectional should flip to up-only"
        elif u_shape:
            direction = "U-SHAPED -- centre-of-ladder is optimum; bidirectional overkill"
        else:
            direction = "mixed/non-monotonic -- regime-dependent"

        print(f"  [{label}] mom={mom:g} lr={lr:.0e}:  "
              f"range {rng_str}  (delta={delta:.0f})")
        print(f"    -> {direction}")

    print("=" * 72)


if __name__ == "__main__":
    main()
