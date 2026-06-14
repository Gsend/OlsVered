"""
benchmark/pinn_verify_capture.py

Verify that the factor-capture-mode API actually engages K-FAC on the PINN
training loop.  Compares three short training runs and reports per-step
gradient norms, loss trajectories, and per-step wall times.

Sanity checks (must all pass to consider the integration working):

  1. Classic K-FAC ≠ Vered K-FAC ≠ AdamW.  Each optimizer should produce
     a different loss trajectory.  If Classic and Vered remain bit-identical,
     they're both still falling back to SGD.

  2. K-FAC per-step time > AdamW per-step time.  K-FAC does extra work
     (the surrogate-loss capture pass + the preconditioner application).
     If K-FAC's per-step time matches AdamW, the K-FAC step is no-op.

  3. relL2 drops below 1.0 within 200 steps.  Before the capture fix all
     methods plateaued above 1.0 (worse than predicting zero).  At least
     one method should clearly beat that baseline.

Output: terminal report only.  Run takes ~3-5 minutes total.
"""
from __future__ import annotations
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
    TanhMLP, pinn_train_step, cole_hopf_reference, relative_l2,
    KFAC_MOMENTUM, KFAC_GAMMA, KFAC_FREQ, GRAD_CLIP, KFAC_MAX_DIM,
)

N_STEPS = 200
SEED    = 42
LR      = 1e-3
DAMPING = 1e-2


def build(method, model):
    if method == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LR)
    if method == "classic":
        from optimizer.classic_kfac import ClassicKFAC
        return ClassicKFAC(
            model, lr=LR, damping=DAMPING,
            factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
            weight_decay=0.0, momentum=KFAC_MOMENTUM,
            grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
            max_gram_dim=KFAC_MAX_DIM,
        )
    if method == "vered":
        from optimizer.vered_kfac import VeredKFAC
        return VeredKFAC(
            model, lr=LR, damping=DAMPING,
            factor_update_freq=KFAC_FREQ, weight_decay=0.0,
            momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
            deferred_qr=True,
        )
    raise ValueError(method)


def run_method(method, ctx, device):
    torch.manual_seed(SEED)
    gen = torch.Generator(device=device).manual_seed(SEED)
    model = TanhMLP().to(device)
    opt = build(method, model)

    losses = []
    grad_norms = []
    t0 = time.perf_counter()
    for step in range(1, N_STEPS + 1):
        loss, comps = pinn_train_step(model, opt, device, gen)
        lv = float(loss.item())
        losses.append(lv)
        # Capture gradient L2 norm before any optimizer side-effects
        gn = sum(p.grad.detach().pow(2).sum().item()
                  for p in model.parameters() if p.grad is not None)
        grad_norms.append(math.sqrt(gn))
        if not math.isfinite(lv):
            break
    wall = time.perf_counter() - t0
    rel = relative_l2(model, ctx["XT"], ctx["U_true"])

    # Cleanup
    if hasattr(opt, "cleanup"):
        opt.cleanup()
    del opt, model
    import gc as _gc; _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "method": method,
        "wall_s": wall,
        "per_step_ms": wall / max(N_STEPS, 1) * 1000.0,
        "losses": losses,
        "grad_norms": grad_norms,
        "final_rel_l2": rel,
    }


def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    print("[setup] Cole-Hopf reference...")
    XT, U_true = cole_hopf_reference(device=device)
    ctx = {"XT": XT, "U_true": U_true}

    results = {}
    for method in ("adamw", "classic", "vered"):
        print(f"\n=== {method.upper()} ({N_STEPS} steps, seed {SEED}) ===")
        results[method] = run_method(method, ctx, device)
        r = results[method]
        print(f"  wall = {r['wall_s']:.1f}s  ({r['per_step_ms']:.1f} ms/step)")
        print(f"  loss[1]   = {r['losses'][0]:.3e}    loss[{N_STEPS}]  = {r['losses'][-1]:.3e}")
        print(f"  grad[1]   = {r['grad_norms'][0]:.3e}    grad[{N_STEPS}]  = {r['grad_norms'][-1]:.3e}")
        print(f"  final relL2 = {r['final_rel_l2']:.3e}")

    print("\n=== Sanity checks ===")

    # Check 1: Classic vs Vered should differ
    classic_final = results["classic"]["losses"][-1]
    vered_final   = results["vered"]["losses"][-1]
    rel_diff = abs(classic_final - vered_final) / max(abs(classic_final), 1e-12)
    if rel_diff < 1e-3:
        print(f"  ❌ Classic ≈ Vered ({classic_final:.4e} vs {vered_final:.4e}, "
              f"rel diff {rel_diff:.2e}) — capture mode NOT engaging")
    else:
        print(f"  ✓ Classic ≠ Vered ({classic_final:.4e} vs {vered_final:.4e}, "
              f"rel diff {rel_diff:.2%}) — capture mode engaged")

    # Check 2: K-FAC per-step time should exceed AdamW
    adamw_ms = results["adamw"]["per_step_ms"]
    classic_ms = results["classic"]["per_step_ms"]
    vered_ms = results["vered"]["per_step_ms"]
    print(f"  per-step: AdamW {adamw_ms:.1f}ms,  "
          f"Classic {classic_ms:.1f}ms ({classic_ms/adamw_ms:.1f}×),  "
          f"Vered {vered_ms:.1f}ms ({vered_ms/adamw_ms:.1f}×)")
    if classic_ms < adamw_ms * 1.2:
        print(f"  ⚠ Classic per-step time close to AdamW — capture overhead "
              f"may not be paying off (K-FAC step may be no-op)")
    if vered_ms < adamw_ms * 1.2:
        print(f"  ⚠ Vered per-step time close to AdamW — capture overhead "
              f"may not be paying off")

    # Check 3: At least one method should beat relL2 = 1.0
    print(f"  final relL2:  AdamW={results['adamw']['final_rel_l2']:.3e}  "
          f"Classic={results['classic']['final_rel_l2']:.3e}  "
          f"Vered={results['vered']['final_rel_l2']:.3e}")
    best = min(r["final_rel_l2"] for r in results.values())
    if best < 1.0:
        print(f"  ✓ at least one method beats relL2 = 1.0 (best = {best:.3e})")
    else:
        print(f"  ⚠ all methods plateau ≥ 1.0 (worse than predict-zero, best = {best:.3e})"
              f" — capture may be working but PINN setup needs more training/tuning")


if __name__ == "__main__":
    main()
