"""
benchmark/pinn_burgers_tune_and_bench.py

End-to-end tune-and-benchmark script for the PINN Burgers benchmark.

Workflow (resumable per cell — every JSON is keyed by phase + hyperparams):

  Phase 1: AdamW fp32 lr screen        4 cells × 500 steps
  Phase 2: AdamW bf16 lr screen        4 cells × 500 steps
  Phase 3: Classic K-FAC fp32 screen   3 lr × 3 damping = 9 cells × 500 steps
  Phase 4: Vered K-FAC fp32 screen     3 lr × 3 damping = 9 cells × 500 steps
  Phase 5: Multi-seed benchmark with the per-method winners
           3 methods × 2 precisions × 3 seeds = 18 cells × 5000 steps

After each phase the script prints a summary table and the picked winner.
After Phase 4 the bench config is fully populated; Phase 5 runs the real
comparison numbers we report in the paper.

Output: benchmark/results/pinn_screen_*.json   (phases 1-4)
        benchmark/results/pinn_bench_*.json    (phase 5)

Run:    python benchmark\pinn_burgers_tune_and_bench.py
Smoke:  python benchmark\pinn_burgers_tune_and_bench.py --smoke
                                         (skips phase 5, only screens)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from benchmark.pinn_burgers import (
    TanhMLP, pinn_loss, pinn_train_step,
    cole_hopf_reference, relative_l2,
    engage_bf16, disengage_bf16, need_autocast,
    KFAC_MOMENTUM, KFAC_GAMMA, KFAC_FREQ, GRAD_CLIP, KFAC_MAX_DIM,
    ADAMW_WD, ADAMW_BETA2,
)


# ---- Phase configuration ----------------------------------------------------

SCREEN_STEPS = 500       # short enough to be cheap, long enough to differentiate
BENCH_STEPS  = 5000
BENCH_SEEDS  = [42, 43, 44]
SCREEN_SEED  = 42
WARMUP       = 100
CONST_PHASE  = 500

ADAMW_LR_GRID    = [3e-4, 1e-3, 3e-3, 1e-2]
ADAMW_WD_DEFAULT = ADAMW_WD
ADAMW_B2_DEFAULT = ADAMW_BETA2

KFAC_LR_GRID      = [3e-4, 1e-3, 3e-3]
KFAC_DAMPING_GRID = [1e-3, 1e-2, 1e-1]

RESULTS = ROOT / "benchmark" / "results"


# ---- Shared training loop --------------------------------------------------

def _train_one(opt_builder, max_steps, precision, seed, ctx, device):
    """Run one training trial.  opt_builder is a no-arg callable returning
    (optimizer, model).  Returns dict with final_rel_l2, completed_steps, etc."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    opt, model = opt_builder()

    decay_steps = max(max_steps - WARMUP - CONST_PHASE, 1)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP),
            torch.optim.lr_scheduler.ConstantLR(
                opt, factor=1.0, total_iters=CONST_PHASE),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=decay_steps,
                eta_min=opt.param_groups[0]["lr"] * 1e-3),
        ],
        milestones=[WARMUP, WARMUP + CONST_PHASE],
    )

    use_amp = (precision == "bf16" and isinstance(opt, torch.optim.AdamW))
    recs = []
    t0 = time.perf_counter()
    aborted_step = None
    for step in range(1, max_steps + 1):
        model.train()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, comps = pinn_train_step(model, opt, device, gen)
        else:
            loss, comps = pinn_train_step(model, opt, device, gen)
        lv = float(loss.item())
        if not math.isfinite(lv):
            aborted_step = step
            break
        sched.step()
        recs.append({"step": step, "loss": lv, **comps})
        if step % 1000 == 0:
            rel = relative_l2(model, ctx["XT"], ctx["U_true"])
            print(f"    step {step:>5d}  loss={lv:.3e}  relL2={rel:.3e}")

    wall = time.perf_counter() - t0
    final_rel = None
    try:
        final_rel = relative_l2(model, ctx["XT"], ctx["U_true"])
    except Exception:
        pass

    out = {
        "wall_s": wall, "final_rel_l2": final_rel,
        "aborted_step": aborted_step, "completed_steps": len(recs),
        "per_step": recs,
    }

    # Cleanup
    try:
        if hasattr(opt, "cleanup"):
            opt.cleanup()
    except Exception:
        pass
    del opt, model
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---- Optimizer builders for each phase -------------------------------------

def _adamw_builder(lr, device):
    def _build():
        model = TanhMLP().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=ADAMW_WD_DEFAULT,
                                 betas=(0.9, ADAMW_B2_DEFAULT))
        return opt, model
    return _build


def _classic_builder(lr, damping, device):
    def _build():
        from optimizer.classic_kfac import ClassicKFAC
        model = TanhMLP().to(device)
        opt = ClassicKFAC(
            model, lr=lr, damping=damping,
            factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
            weight_decay=0.0, momentum=KFAC_MOMENTUM,
            grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
            max_gram_dim=KFAC_MAX_DIM,
        )
        return opt, model
    return _build


def _vered_builder(lr, damping, device):
    def _build():
        from optimizer.vered_kfac import VeredKFAC
        model = TanhMLP().to(device)
        opt = VeredKFAC(
            model, lr=lr, damping=damping,
            factor_update_freq=KFAC_FREQ, weight_decay=0.0,
            momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
            deferred_qr=True,
        )
        return opt, model
    return _build


# ---- Screen phases ----------------------------------------------------------

def screen_path(name):
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / f"pinn_screen_{name}.json"


def _run_or_skip(name, runfn):
    p = screen_path(name)
    if p.exists():
        return json.loads(p.read_text())
    print(f"  > {name}")
    out = runfn()
    p.write_text(json.dumps(out, indent=2, default=str))
    rel = out.get("final_rel_l2")
    aborted = out.get("aborted_step")
    msg = f"relL2={rel:.3e}" if rel is not None and math.isfinite(rel) else "DIV"
    if aborted:
        msg += f" (aborted at step {aborted})"
    print(f"     -> {msg}  wall={out['wall_s']/60:.1f}m")
    return out


def phase_adamw_screen(precision, ctx, device):
    results = {}
    print(f"\n=== Phase: AdamW {precision} lr screen "
          f"({len(ADAMW_LR_GRID)} cells × {SCREEN_STEPS} steps) ===")
    for lr in ADAMW_LR_GRID:
        if precision == "bf16":
            engage_bf16("adamw")
        try:
            name = f"adamw_{precision}_lr{lr:g}"
            out = _run_or_skip(name, lambda lr=lr: _train_one(
                _adamw_builder(lr, device), SCREEN_STEPS, precision,
                SCREEN_SEED, ctx, device))
            results[lr] = out
        finally:
            if precision == "bf16":
                disengage_bf16("adamw")
    # Pick the lowest finite relL2 with no abort (or, if all abort, the longest survivor)
    best = _pick_winner(results)
    print(f"  winner: lr={best['lr']:g}  relL2={best['final_rel_l2']:.3e}")
    return best


def phase_kfac_screen(method, ctx, device):
    results = {}
    print(f"\n=== Phase: {method} K-FAC fp32 screen "
          f"({len(KFAC_LR_GRID) * len(KFAC_DAMPING_GRID)} cells × {SCREEN_STEPS} steps) ===")
    for lr in KFAC_LR_GRID:
        for dmp in KFAC_DAMPING_GRID:
            name = f"{method}_fp32_lr{lr:g}_dmp{dmp:g}"
            builder = (_classic_builder(lr, dmp, device) if method == "classic"
                       else _vered_builder(lr, dmp, device))
            out = _run_or_skip(name, lambda b=builder: _train_one(
                b, SCREEN_STEPS, "fp32", SCREEN_SEED, ctx, device))
            results[(lr, dmp)] = out
    # 2D grid; pick the lowest finite relL2 with no abort
    best = _pick_winner(results, keys=("lr", "damping"))
    print(f"  winner: lr={best['lr']:g} damping={best['damping']:g}  "
          f"relL2={best['final_rel_l2']:.3e}")
    return best


def _pick_winner(results, keys=("lr",)):
    """results is a dict: key (scalar or tuple) -> result_dict.
    Returns dict with the winning hp(s) and result fields.  Prefers
    non-aborted, then lowest finite final_rel_l2."""
    candidates = []
    for k, v in results.items():
        rel = v.get("final_rel_l2")
        if rel is None or not math.isfinite(rel):
            continue
        candidates.append((v.get("aborted_step") is not None, rel, k, v))
    if not candidates:
        # Fall back to whichever ran the longest
        for k, v in results.items():
            candidates.append((True, float("inf"), k, v))
    candidates.sort(key=lambda x: (x[0], x[1]))   # non-aborted first, then lowest relL2
    aborted, rel, k, v = candidates[0]
    out = {"final_rel_l2": rel, **v}
    if len(keys) == 1:
        out[keys[0]] = k
    else:
        for i, kn in enumerate(keys):
            out[kn] = k[i]
    return out


# ---- Bench phase ------------------------------------------------------------

def bench_path(precision, method, seed):
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / f"pinn_bench_{precision}_{method}_seed{seed}.json"


def phase_bench(winners, ctx, device, hw):
    print(f"\n=== Phase: Multi-seed benchmark ({BENCH_STEPS} steps each) ===")
    total = 2 * 3 * len(BENCH_SEEDS)
    done  = 0
    for precision in ("fp32", "bf16"):
        for method in ("adamw", "classic", "vered"):
            for seed in BENCH_SEEDS:
                done += 1
                p = bench_path(precision, method, seed)
                if p.exists():
                    print(f"[{done}/{total}] [skip] {p.name}")
                    continue
                print(f"\n[{done}/{total}] pinn_bench/{precision}/{method}/seed{seed}")

                if method == "adamw":
                    lr = (winners["adamw_bf16"]["lr"] if precision == "bf16"
                          else winners["adamw_fp32"]["lr"])
                    builder = _adamw_builder(lr, device)
                elif method == "classic":
                    lr  = winners["classic"]["lr"]
                    dmp = winners["classic"]["damping"]
                    builder = _classic_builder(lr, dmp, device)
                else:  # vered
                    lr  = winners["vered"]["lr"]
                    dmp = winners["vered"]["damping"]
                    builder = _vered_builder(lr, dmp, device)

                if precision == "bf16":
                    engage_bf16(method)
                try:
                    out = _train_one(builder, BENCH_STEPS, precision,
                                      seed, ctx, device)
                finally:
                    if precision == "bf16":
                        disengage_bf16(method)
                out.update({"benchmark": "pinn_burgers_bench",
                            "precision": precision, "method": method,
                            "seed": seed, "hw": hw})
                p.write_text(json.dumps(out, indent=2, default=str))
                rel = out.get("final_rel_l2")
                aborted = out.get("aborted_step")
                msg = f"relL2={rel:.3e}" if (rel is not None and math.isfinite(rel)) else "DIV"
                if aborted:
                    msg += f" (aborted at step {aborted})"
                print(f"  -> {msg}  wall={out['wall_s']/60:.1f}m")


def print_bench_summary():
    import statistics
    print("\n=== pinn_burgers benchmark summary "
          "(relative L2 vs Cole-Hopf, lower is better) ===")
    print(f"  {'precision':>9}  {'method':>8}  {'mean_relL2':>12}  "
          f"{'std':>10}  {'wall_min':>9}")
    for precision in ("fp32", "bf16"):
        for method in ("adamw", "classic", "vered"):
            vals, walls = [], []
            for seed in BENCH_SEEDS:
                p = bench_path(precision, method, seed)
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                v = d.get("final_rel_l2")
                if v is not None and math.isfinite(v):
                    vals.append(v)
                walls.append(d.get("wall_s", 0) / 60)
            if not vals:
                continue
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) >= 2 else 0.0
            w    = statistics.mean(walls) if walls else 0.0
            print(f"  {precision:>9}  {method:>8}  {mean:>12.3e}  "
                  f"{std:>10.3e}  {w:>9.2f}")


# ---- Main -------------------------------------------------------------------

def main(smoke=False):
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    print("[setup] computing Cole-Hopf reference (256×100)...")
    t0 = time.perf_counter()
    XT, U_true = cole_hopf_reference(device=device)
    print(f"  reference ready in {time.perf_counter()-t0:.1f}s")
    ctx = {"XT": XT, "U_true": U_true}

    winners = {}
    winners["adamw_fp32"]   = phase_adamw_screen("fp32", ctx, device)
    winners["adamw_bf16"]   = phase_adamw_screen("bf16", ctx, device)
    winners["classic"]      = phase_kfac_screen("classic", ctx, device)
    winners["vered"]        = phase_kfac_screen("vered",   ctx, device)

    print("\n=== Picked hyperparameters ===")
    print(f"  AdamW fp32     : lr = {winners['adamw_fp32']['lr']:g}")
    print(f"  AdamW bf16     : lr = {winners['adamw_bf16']['lr']:g}")
    print(f"  Classic K-FAC  : lr = {winners['classic']['lr']:g}  "
          f"damping = {winners['classic']['damping']:g}")
    print(f"  Vered K-FAC    : lr = {winners['vered']['lr']:g}  "
          f"damping = {winners['vered']['damping']:g}")

    # Persist winners for downstream scripts (plotting, paper tables)
    winners_path = RESULTS / "pinn_winners.json"
    winners_serial = {k: {kk: vv for kk, vv in v.items()
                          if kk in ("lr", "damping", "final_rel_l2", "wall_s")}
                       for k, v in winners.items()}
    winners_path.write_text(json.dumps(winners_serial, indent=2, default=str))
    print(f"\nwrote {winners_path.name}")

    if smoke:
        print("\n[smoke] skipping Phase 5 (bench)")
        return

    phase_bench(winners, ctx, device, hw)
    print_bench_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                         help="run only the screen phases (no multi-seed bench)")
    args = parser.parse_args()
    main(smoke=args.smoke)
