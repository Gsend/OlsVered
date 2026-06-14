"""
benchmark/pinn_burgers.py

Physics-informed neural network on the 1D viscous Burgers' equation —
the canonical Raissi et al. 2019 testbed where K-FAC has historically
beaten Adam by 2-3 orders of magnitude on final relative L2 error.

PDE:           u_t + u * u_x = nu * u_xx,    nu = 0.01 / pi
Domain:        x in [-1, 1],  t in [0, 1]
Initial cond.: u(x, 0) = -sin(pi*x)
Boundary cond: u(-1, t) = u(1, t) = 0
Reference:    Cole-Hopf solution sampled on 256 x 100 (x, t) grid.

Network: MLP 2 -> [100, 100, 100, 100, 100] -> 1, tanh activations.
Loss   : L_pde + L_ic + L_bc, mean-squared residuals.
Metric : relative L2 error on the (x, t) test grid.

Sweep grid:
    methods   : adamw, classic, vered  (skip WGSO/SINGD — broken on dense MLPs)
    precisions: fp32, bf16
    seeds     : 42, 43, 44
Total: 18 cells, resumable.

NOTE on K-FAC + PINN: the loss requires autograd.grad to compute u_x and
u_xx, which traverses the MLP a second (and third) time and triggers the
K-FAC forward hooks again.  The captured activation Gram matrices therefore
mix prediction-pass activations with derivative-evaluation activations.
This is a known interaction; we accept it for the headline comparison and
note it in the paper.
"""
from __future__ import annotations
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


# ---- Config -----------------------------------------------------------------

METHODS    = ["adamw", "classic", "vered"]
PRECISIONS = ["fp32", "bf16"]
SEEDS      = [42, 43, 44]

# Physical constants
NU         = 0.01 / math.pi

# Network
HIDDEN     = [100, 100, 100, 100, 100]   # Raissi 2019 used 9x20 but 5x100 is standard

# Training
MAX_STEPS  = 10000
WARMUP     = 200
CONST_PHASE = 2000        # matches AE schedule pattern
BATCH_PDE  = 10000        # collocation points per step
BATCH_BC   = 200          # boundary points
BATCH_IC   = 200          # initial-condition points

# Tuned defaults (placeholders — to be set by a screen, similar to the AE flow)
ADAMW_LR_FP32        = 1e-3
ADAMW_LR_BF16        = 3e-4
ADAMW_WD             = 0.0
ADAMW_BETA2          = 0.999

KFAC_LR              = 1e-3
KFAC_DAMPING         = 3e-3
KFAC_MOMENTUM        = 0.9
KFAC_GAMMA           = 0.9
KFAC_FREQ            = 10
GRAD_CLIP            = 100.0
KFAC_MAX_DIM         = 4096


# ---- Architecture -----------------------------------------------------------

class TanhMLP(nn.Module):
    """MLP with tanh activations — needed for smooth higher-order derivatives."""

    def __init__(self, in_dim=2, hidden=HIDDEN, out_dim=1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, xt):
        # xt: (B, 2)
        out = xt
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = torch.tanh(out)
        return out  # (B, 1)


# ---- PDE + loss -------------------------------------------------------------

def pinn_loss(model, n_pde=BATCH_PDE, n_bc=BATCH_BC, n_ic=BATCH_IC,
              device=None, generator=None):
    """Compute the Burgers' equation PINN loss with autograd derivatives.

    Returns (loss_total, dict_of_components).
    """
    # PDE interior collocation: x in [-1, 1], t in [0, 1]
    x_pde = (torch.rand(n_pde, 1, device=device, generator=generator) * 2.0 - 1.0)
    t_pde = torch.rand(n_pde, 1, device=device, generator=generator)
    x_pde.requires_grad_(True)
    t_pde.requires_grad_(True)
    xt_pde = torch.cat([x_pde, t_pde], dim=1)
    u_pde  = model(xt_pde)

    # u_x  = du/dx
    u_x = torch.autograd.grad(
        u_pde, x_pde, grad_outputs=torch.ones_like(u_pde),
        create_graph=True, retain_graph=True,
    )[0]
    # u_t  = du/dt
    u_t = torch.autograd.grad(
        u_pde, t_pde, grad_outputs=torch.ones_like(u_pde),
        create_graph=True, retain_graph=True,
    )[0]
    # u_xx = d/dx(du/dx)
    u_xx = torch.autograd.grad(
        u_x, x_pde, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True,
    )[0]

    pde_residual = u_t + u_pde * u_x - NU * u_xx
    L_pde = (pde_residual ** 2).mean()

    # Initial condition: u(x, 0) = -sin(pi*x)
    x_ic = (torch.rand(n_ic, 1, device=device, generator=generator) * 2.0 - 1.0)
    t_ic = torch.zeros(n_ic, 1, device=device)
    u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
    u_ic_true = -torch.sin(math.pi * x_ic)
    L_ic = ((u_ic_pred - u_ic_true) ** 2).mean()

    # Boundary condition: u(+/-1, t) = 0
    t_bc  = torch.rand(n_bc, 1, device=device, generator=generator)
    x_bc_left  = -torch.ones_like(t_bc)
    x_bc_right =  torch.ones_like(t_bc)
    u_bc_left  = model(torch.cat([x_bc_left,  t_bc], dim=1))
    u_bc_right = model(torch.cat([x_bc_right, t_bc], dim=1))
    L_bc = ((u_bc_left ** 2).mean() + (u_bc_right ** 2).mean()) * 0.5

    L_total = L_pde + L_ic + L_bc
    return L_total, {"pde": float(L_pde.item()),
                     "ic":  float(L_ic.item()),
                     "bc":  float(L_bc.item())}


# ---- K-FAC factor-capture surrogate ----------------------------------------
#
# K-FAC needs a clean (X, δ) pair per layer to estimate the Kronecker-factored
# Fisher.  The PINN loss is hostile to that — its derivatives go through
# autograd.grad multiple times and contaminate the activation buffers.
#
# Standard workaround: use a surrogate regression Fisher.  Do one designated
# forward+backward through the network with the loss "½‖u‖²" — that gradient
# propagates as δ = u, which gives a well-conditioned regression-Fisher
# estimate without ever touching autograd.grad.  This is the Fisher proxy
# that K-FAC papers use for sequence models with cross-entropy losses too
# (e.g. Martens & Grosse 2015, §6).  The result of this dedicated pass is
# discarded; only the captured (X, δ) goes into the K-FAC factors.

def _kfac_capture_step(model, kfac, xt_capture):
    """Designate ONE clean forward+backward as the K-FAC factor-capture pass.

    Runs in the optimizer's :meth:`capture` context.  All other forward
    passes and autograd.grad calls in the training step must happen with
    capture paused (the optimizer's default after :meth:`enable`).

    Parameters
    ----------
    model      : the K-FAC-instrumented network
    kfac       : optimizer with a ``capture()`` context manager
    xt_capture : (B, 2) input tensor for the capture pass.  Should be on the
                 same distribution as the bulk PDE batch (random (x, t) in
                 [-1, 1] × [0, 1]).
    """
    # `with kfac.capture()` flips hooks._enabled = True for the block.
    # The fwd hook records X̃ from xt_capture; the bwd hook records δ̃ from
    # the surrogate-loss gradient.  Both buffers are then folded into the
    # running Kronecker factor accumulators by the K-FAC layer in optimizer.step().
    with kfac.capture():
        u = model(xt_capture)
        # Surrogate loss: 1/2 * ||u||² — gradient w.r.t. u is u itself, which
        # propagates back through the model as a clean δ at each layer.
        surrogate = 0.5 * (u ** 2).mean()
        # Use autograd.grad (don't touch params.grad) so we don't pollute the
        # gradient buffer that the real PINN loss will populate next.
        torch.autograd.grad(
            surrogate, list(model.parameters()),
            create_graph=False, retain_graph=False, allow_unused=True,
        )
    # Pause hooks again before the next phase of the training step.
    kfac.pause_capture()


def pinn_train_step(model, opt, device, generator,
                     n_pde=None, n_bc=None, n_ic=None):
    """One PINN training step that integrates correctly with K-FAC.

    The phases:
      1. (K-FAC only) Factor-capture pass via :func:`_kfac_capture_step` —
         one clean forward+backward through the network using the surrogate
         loss ½‖u‖².  Hooks active for this pass only.
      2. PINN loss computation with hooks paused — multi-pass autograd for
         u_x, u_t, u_xx, plus the IC and BC forwards.  Hooks fire but
         no-op, so nothing contaminates the factor estimate.
      3. ``L_total.backward()`` with hooks paused — populates ``p.grad``
         with the real PINN gradient.
      4. ``opt.step()`` — K-FAC applies its preconditioner to the gradient.

    For AdamW the capture phase is skipped (it's a no-op for non-K-FAC).
    Returns (loss_total, dict_of_components).
    """
    n_pde = n_pde or BATCH_PDE
    n_bc  = n_bc  or BATCH_BC
    n_ic  = n_ic  or BATCH_IC

    is_kfac = hasattr(opt, "capture") and callable(opt.capture)

    if is_kfac:
        # Phase 1: factor capture (hooks active for this scope only)
        xt_capture = torch.cat([
            torch.rand(n_pde, 1, device=device, generator=generator) * 2.0 - 1.0,
            torch.rand(n_pde, 1, device=device, generator=generator),
        ], dim=1)
        _kfac_capture_step(model, opt, xt_capture)
        # Hooks now paused — the rest of the step is contamination-free.

    # Phase 2: compute the full PINN loss (hooks paused for K-FAC)
    loss, comps = pinn_loss(model, n_pde, n_bc, n_ic, device, generator)

    # Phase 3: real-gradient backward (also hooks paused for K-FAC)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Phase 4: optimizer step.  K-FAC applies the preconditioner built from
    # the (X, δ) captured in phase 1; AdamW just does its usual step.
    opt.step()

    return loss, comps


# ---- Reference solution (Cole-Hopf) ----------------------------------------

def cole_hopf_reference(n_x=256, n_t=100, device=None):
    """Compute the exact Burgers' solution on an (n_x, n_t) test grid using the
    Cole-Hopf transformation and trapezoidal quadrature.

    Returns (X (n_x*n_t, 2), U (n_x*n_t, 1)) tensors on `device`.
    """
    x = torch.linspace(-1.0, 1.0, n_x, device=device)
    t = torch.linspace( 0.001, 1.0, n_t, device=device)  # avoid t=0 singularity

    # Quadrature points eta in [-1, 1] for the Cole-Hopf integral
    eta = torch.linspace(-1.0, 1.0, 512, device=device)
    d_eta = eta[1] - eta[0]

    # phi_0(eta) = exp(-cos(pi*eta) / (2*pi*nu))   (from u(x,0) = -sin(pi*x))
    log_phi0 = -torch.cos(math.pi * eta) / (2.0 * math.pi * NU)

    U = torch.empty(n_x, n_t, device=device)
    for j, tj in enumerate(t):
        # exponent of the kernel: log_phi0(eta) - (x-eta)^2 / (4 nu t)
        # we'll subtract max for numerical stability
        for i, xi in enumerate(x):
            arg = log_phi0 - (xi - eta) ** 2 / (4.0 * NU * tj)
            arg_max = arg.max()
            w = torch.exp(arg - arg_max)
            num = ((xi - eta) / tj * w).sum()
            den = w.sum()
            U[i, j] = -num / den

    # Reshape into (n_x*n_t, 2) and (n_x*n_t, 1)
    X, T = torch.meshgrid(x, t, indexing="ij")
    XT = torch.stack([X.flatten(), T.flatten()], dim=1)
    U_flat = U.flatten().unsqueeze(1)
    return XT, U_flat


def relative_l2(model, XT, U_true):
    model.eval()
    with torch.no_grad():
        U_pred = model(XT)
        num = (U_pred - U_true).norm()
        den = U_true.norm().clamp_min(1e-12)
        return float((num / den).item())


# ---- Optimizer factory ------------------------------------------------------

def build_optimizers(method, model, precision="fp32"):
    if method == "adamw":
        adamw_lr = ADAMW_LR_BF16 if precision == "bf16" else ADAMW_LR_FP32
        return torch.optim.AdamW(
            model.parameters(),
            lr=adamw_lr, weight_decay=ADAMW_WD, betas=(0.9, ADAMW_BETA2),
        ), None

    if method == "classic":
        from optimizer.classic_kfac import ClassicKFAC
        return ClassicKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING,
            factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
            weight_decay=0.0, momentum=KFAC_MOMENTUM,
            grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
            max_gram_dim=KFAC_MAX_DIM,
        ), None

    if method == "vered":
        from optimizer.vered_kfac import VeredKFAC
        return VeredKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING,
            factor_update_freq=KFAC_FREQ, weight_decay=0.0,
            momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
            deferred_qr=True,
        ), None

    raise ValueError(method)


# ---- bf16 handling (reuses AE plumbing) ------------------------------------

def engage_bf16(method):
    if method == "vered":
        from benchmark.kfac_bf16_compare import enable_bf16
        enable_bf16(wgso=False)
    elif method == "classic":
        from benchmark.kfac_bf16_compare import enable_classic_bf16
        enable_classic_bf16()


def disengage_bf16(method):
    if method == "vered":
        from benchmark.kfac_bf16_compare import disable_bf16
        disable_bf16()
    elif method == "classic":
        from benchmark.kfac_bf16_compare import disable_classic_bf16
        disable_classic_bf16()


def need_autocast(method, precision):
    return precision == "bf16" and method == "adamw"


# ---- Per-run executor -------------------------------------------------------

def out_path(precision, method, seed):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"pinn_burgers_{precision}_{method}_seed{seed}.json"


def run_one(precision, method, seed, ctx, device, hw):
    p = out_path(precision, method, seed)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    if precision == "bf16":
        engage_bf16(method)

    opt_primary = model = None
    try:
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        model = TanhMLP().to(device)
        opt_primary, _ = build_optimizers(method, model, precision)

        decay_steps = MAX_STEPS - WARMUP - CONST_PHASE
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt_primary, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt_primary, start_factor=0.1, end_factor=1.0,
                    total_iters=WARMUP),
                torch.optim.lr_scheduler.ConstantLR(
                    opt_primary, factor=1.0, total_iters=CONST_PHASE),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_primary, T_max=decay_steps,
                    eta_min=opt_primary.param_groups[0]["lr"] * 1e-3),
            ],
            milestones=[WARMUP, WARMUP + CONST_PHASE],
        )

        use_amp = need_autocast(method, precision)
        recs = []
        print(f"\n=== pinn_burgers/{precision}/{method}/seed{seed} ===")
        t0 = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            model.train()
            # pinn_train_step handles K-FAC's factor-capture phase internally
            # (designated forward+backward with a surrogate loss) and runs the
            # real PINN loss + backward + opt.step() with hooks paused.
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, comps = pinn_train_step(model, opt_primary, device, gen)
            else:
                loss, comps = pinn_train_step(model, opt_primary, device, gen)

            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step} NaN/Inf, aborting")
                break
            sched.step()
            recs.append({"step": step, "loss": lv, **comps})

            if step % 1000 == 0:
                rel = relative_l2(model, ctx["XT"], ctx["U_true"])
                print(f"  step {step:>5d}  loss={lv:.4e}  relL2={rel:.4e}")

        wall = time.perf_counter() - t0
        final_rel = None
        try:
            final_rel = relative_l2(model, ctx["XT"], ctx["U_true"])
        except Exception as e:
            print(f"  eval failed: {e}")

        out = {
            "benchmark": "pinn_burgers",
            "precision": precision, "method": method, "seed": seed,
            "wall_s": wall, "hw": hw,
            "final_rel_l2": final_rel,
            "completed_steps": len(recs),
            "per_step": recs,
        }
        p.write_text(json.dumps(out, indent=2, default=str))
        msg = f"relL2={final_rel:.3e}  " if final_rel is not None else "DIV  "
        print(f"  -> {msg}wall={wall/60:.1f}m")
    finally:
        if precision == "bf16":
            disengage_bf16(method)
        try:
            if opt_primary is not None and hasattr(opt_primary, "cleanup"):
                opt_primary.cleanup()
        except Exception:
            pass
        try:
            del opt_primary, model
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---- Smoke test -------------------------------------------------------------

def smoke_test():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    print("[smoke] computing Cole-Hopf reference (256x100 grid)...")
    t0 = time.perf_counter()
    XT, U_true = cole_hopf_reference(device=device)
    print(f"  reference built in {time.perf_counter() - t0:.1f}s, "
          f"|XT|={tuple(XT.shape)}, |U|={tuple(U_true.shape)}")

    torch.manual_seed(42)
    gen = torch.Generator(device=device).manual_seed(42)
    model = TanhMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("[smoke] training 200 steps...")
    for step in range(1, 201):
        loss, comps = pinn_train_step(model, opt, device, gen)
        if not math.isfinite(loss.item()):
            print(f"  step {step}: loss diverged")
            break
        if step in (1, 10, 50, 100, 200):
            rel = relative_l2(model, XT, U_true)
            print(f"  step {step:>3d}  loss={loss.item():.4e}  "
                  f"pde={comps['pde']:.2e} ic={comps['ic']:.2e} "
                  f"bc={comps['bc']:.2e}  relL2={rel:.3e}")
    print("[smoke] OK")


# ---- Main -------------------------------------------------------------------

def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    print("[setup] computing Cole-Hopf reference (256x100)...")
    XT, U_true = cole_hopf_reference(device=device)
    ctx = {"XT": XT, "U_true": U_true}

    total = len(PRECISIONS) * len(METHODS) * len(SEEDS)
    done = 0
    for precision in PRECISIONS:
        for method in METHODS:
            for seed in SEEDS:
                done += 1
                print(f"\n[{done}/{total}]")
                run_one(precision, method, seed, ctx, device, hw)

    print("\n=== pinn_burgers summary (relative L2 vs Cole-Hopf, lower is better) ===")
    print(f"  {'precision':>9}  {'method':>8}  {'mean_relL2':>12}  {'std':>10}  {'wall_min':>9}")
    import statistics
    for precision in PRECISIONS:
        for method in METHODS:
            vals, walls = [], []
            for seed in SEEDS:
                p = out_path(precision, method, seed)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                         help="quick AdamW sanity check (~1 min)")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        main()
