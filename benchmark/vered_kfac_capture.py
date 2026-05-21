"""
benchmark/vered_kfac_capture.py

Capture per-step layer-level data for offline Vered-vs-Classic comparison.

At each requested step, captures for one chosen Linear layer:
    - X         (forward input activations)
    - Y         (forward output activations)
    - dY        (backward gradient w.r.t. layer output)
    - dX        (backward gradient w.r.t. layer input)
    - dW        (weight gradient, == kfac's input)
    - factors   (variant-specific cached K-FAC state for this layer)
    - natgrad   (the natural-gradient produced for this layer's dW)
    - val_ppl, train_loss  (trajectory context)
    - weight, bias  (layer params at this step, for forward replay)

Saves one .pt file per run.  Pair two runs with identical (mom, lr, seed)
but different --variant to produce matched captures.

Usage:
    python benchmark/vered_kfac_capture.py --variant VeredKFAC ^
        --mom 0.9 --lr 8e-3 --capture-steps 201,300,500,800 ^
        --layer-name "blocks.2.mlp.fc1"
    python benchmark/vered_kfac_capture.py --variant ClassicKFAC ^
        --mom 0.9 --lr 8e-3 --capture-steps 201,300,500,800 ^
        --layer-name "blocks.2.mlp.fc1"

Cell choice (mom=0.9, lr=8e-3) is at eff_lr=8e-2 -- well above the
1e-2 noise-onset threshold from the screen.  Pick any cell with
known nonzero delta to characterize the noise; pick a cell with
near-zero delta as a control.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import (
    build_data, make_optimizers, evaluate_ppl,
    EMB_LR, KFAC_MAX_DIM, WARMUP_STEPS_BEFORE_CHECK,
)
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info

CAP_OUT = ROOT / "benchmark" / "results" / "captures"
CAP_OUT.mkdir(parents=True, exist_ok=True)


def get_variant_factors(kfac_opt, module: nn.Module, variant: str) -> Dict[str, torch.Tensor]:
    """Extract the variant's cached K-FAC state for one module.

    Returned dict keys are variant-specific:
        Vered:   "R_X", "R_G"
        Classic: "A_inv", "G_inv"
        OlsSM:   "A_inv", "G_inv"  (same keys; computed differently)
    """
    out: Dict[str, torch.Tensor] = {}
    if variant == "VeredKFAC":
        factors = getattr(kfac_opt, "_factors", {})
        if module in factors:
            R_X, R_G = factors[module]
            out["R_X"] = R_X.detach().cpu().clone()
            out["R_G"] = R_G.detach().cpu().clone()
    elif variant in ("ClassicKFAC", "OlsSMKFAC"):
        invs = getattr(kfac_opt, "_inverses", {})
        if module in invs:
            A_inv, G_inv = invs[module]
            out["A_inv"] = A_inv.detach().cpu().clone()
            out["G_inv"] = G_inv.detach().cpu().clone()
    return out


def find_module(model: nn.Module, name: str) -> nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise ValueError(
        f"Module {name!r} not found.  Available Linear layers:\n  " +
        "\n  ".join(n for n, m in model.named_modules()
                    if isinstance(m, nn.Linear))
    )


def compute_natgrad_for_capture(kfac_opt, module: nn.Module,
                                 variant: str, dW: torch.Tensor) -> torch.Tensor:
    """Re-run the variant's natgrad computation for a saved dW.

    Uses the variant's currently cached factors (mirrors what step() did).
    Returned tensor is the variant's natural gradient for this dW, on CPU.
    """
    if variant == "VeredKFAC":
        from optimizer.vered_kfac import apply_vered
        factors = kfac_opt._factors
        if module not in factors:
            return torch.zeros_like(dW).cpu()
        R_X, R_G = factors[module]
        dW_dev = dW.to(R_X.device)
        ng = apply_vered(dW_dev, R_X, R_G)
        return ng.detach().cpu().clone()
    elif variant in ("ClassicKFAC", "OlsSMKFAC"):
        invs = kfac_opt._inverses
        if module not in invs:
            return torch.zeros_like(dW).cpu()
        A_inv, G_inv = invs[module]
        dW_dev = dW.to(A_inv.device)
        ng = G_inv @ dW_dev @ A_inv
        return ng.detach().cpu().clone()
    return torch.zeros_like(dW).cpu()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", required=True,
                    choices=["VeredKFAC", "ClassicKFAC", "OlsSMKFAC"])
    ap.add_argument("--mom", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--damping", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--grad-clip", type=float, default=300.0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer-name", type=str, default="",
                    help="Module name to capture.  Empty -> list available Linear layers.")
    ap.add_argument("--capture-steps", type=str, default="201,300,500,800",
                    help="Comma-separated step indices to capture.")
    args = ap.parse_args()

    capture_steps = sorted({int(s) for s in args.capture_steps.split(",")})

    device = get_device()
    hw = get_hardware_info()
    print(f"[capture] variant={args.variant}  mom={args.mom}  lr={args.lr:g}  "
          f"steps={capture_steps}")
    print(f"[capture] GPU: {hw.get('gpu_name')}")

    # Build data + model (same setup as run_probe)
    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    torch.manual_seed(args.seed)
    model = SmallGPT(vocab_size=vocab_size).to(device)

    # List layers if no name given
    if not args.layer_name:
        print("\nAvailable Linear layers (pick one with --layer-name):")
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                print(f"  {n:50s} in={m.in_features:4d} out={m.out_features:4d}")
        return

    target = find_module(model, args.layer_name)
    print(f"[capture] Target layer: {args.layer_name}  "
          f"(in={target.in_features}, out={target.out_features})")

    kfac_opt, emb_opt, _ = make_optimizers(
        args.variant, model, args.lr, args.damping, args.mom,
        grad_clip=args.grad_clip, gamma=args.gamma,
    )

    # Schedulers (constant_warmup -- match the screen)
    warmup = 200
    scheduler = torch.optim.lr_scheduler.SequentialLR(kfac_opt, schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            kfac_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
        torch.optim.lr_scheduler.ConstantLR(
            kfac_opt, factor=1.0, total_iters=args.max_steps),
    ], milestones=[warmup])
    emb_scheduler = torch.optim.lr_scheduler.ConstantLR(
        emb_opt, factor=1.0, total_iters=args.max_steps)

    # ---- Hooks on the target layer to capture X, Y, dY, dX --------------
    capture_buf: Dict[str, torch.Tensor] = {}

    def fwd_hook(_module, inputs, output):
        # inputs is a tuple; for Linear it's (X,)
        capture_buf["X"] = inputs[0].detach().cpu().clone()
        capture_buf["Y"] = output.detach().cpu().clone()

    def bwd_hook(_module, grad_input, grad_output):
        # grad_input is (dX,) for Linear; grad_output is (dY,)
        if grad_input[0] is not None:
            capture_buf["dX"] = grad_input[0].detach().cpu().clone()
        capture_buf["dY"] = grad_output[0].detach().cpu().clone()

    fwd_handle = target.register_forward_hook(fwd_hook)
    bwd_handle = target.register_full_backward_hook(bwd_hook)

    # ---- Training loop ---------------------------------------------------
    train_loader = train_loader_factory()
    data_iter = iter(train_loader)
    captures: List[Dict] = []
    val_ppl_history: List[float] = []
    capture_idx = 0

    arming_set = set(capture_steps)
    armed = False

    t0 = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # Arm capture for this step BEFORE forward, so hooks save tensors
        armed = step in arming_set
        if armed:
            capture_buf.clear()

        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=pad_id)
        loss_val = loss.item()

        if not math.isfinite(loss_val):
            print(f"[capture] step={step} loss=NaN; aborting")
            break

        model.zero_grad()
        loss.backward()

        # If this step is a capture target, dW = weight.grad RIGHT NOW
        if armed:
            dW = target.weight.grad.detach().cpu().clone()
            db = (target.bias.grad.detach().cpu().clone()
                  if target.bias is not None and target.bias.grad is not None
                  else None)

        # Read variant's cached factors BEFORE step (factors may refresh on this step)
        factors_pre = get_variant_factors(kfac_opt, target, args.variant) if armed else None

        kfac_opt.step()

        # Read variant's cached factors AFTER step (factors after this step's refresh)
        factors_post = get_variant_factors(kfac_opt, target, args.variant) if armed else None

        if armed:
            # Compute what the variant produced as natural gradient for this dW,
            # using the factors that were live during step() (which are the
            # PRE-step factors for the first 19 of every 20 steps, and the
            # POST-step factors only on factor-refresh steps).
            # The variant's step() actually called _update_factors() first
            # when (step_count % factor_update_freq == 1), so the factors
            # used during this step's preconditioner apply are factors_post
            # on a refresh step, and factors_pre otherwise.
            # We capture both and let the analysis decide.
            natgrad_used = compute_natgrad_for_capture(
                kfac_opt, target, args.variant, dW)

            cap = {
                "step":           step,
                "loss":           loss_val,
                "X":              capture_buf.get("X"),
                "Y":              capture_buf.get("Y"),
                "dX":             capture_buf.get("dX"),
                "dY":             capture_buf.get("dY"),
                "dW":             dW,
                "db":             db,
                "weight":         target.weight.detach().cpu().clone(),
                "bias":           (target.bias.detach().cpu().clone()
                                   if target.bias is not None else None),
                "factors_pre":    factors_pre,
                "factors_post":   factors_post,
                "natgrad":        natgrad_used,
            }
            captures.append(cap)
            print(f"[capture] step={step}  loss={loss_val:.3f}  "
                  f"dW.shape={tuple(dW.shape)}  "
                  f"X.shape={tuple(capture_buf.get('X').shape) if capture_buf.get('X') is not None else 'NA'}")

        emb_opt.step()
        scheduler.step()
        emb_scheduler.step()

        # Periodic val eval (every ~10k samples)
        if step % 157 == 0 or step == args.max_steps:
            ppl = evaluate_ppl(model, val_loader, device, pad_id)
            val_ppl_history.append((step, ppl))
            print(f"[capture] step={step}  loss={loss_val:.3f}  val_ppl={ppl:.0f}  "
                  f"wall={(time.perf_counter()-t0)/60:.1f}m")

    fwd_handle.remove()
    bwd_handle.remove()

    # ---- Save -----------------------------------------------------------
    out_path = CAP_OUT / (
        f"capture_{args.variant}_m{args.mom:.2f}_lr{args.lr:.0e}_"
        f"layer_{args.layer_name.replace('.', '-')}.pt"
    )
    torch.save({
        "args":            vars(args),
        "hw":              hw,
        "layer_name":      args.layer_name,
        "in_features":     target.in_features,
        "out_features":    target.out_features,
        "has_bias":        target.bias is not None,
        "val_ppl_history": val_ppl_history,
        "captures":        captures,
    }, out_path)
    print(f"\n[capture] Saved {len(captures)} step captures -> {out_path}")


if __name__ == "__main__":
    main()
