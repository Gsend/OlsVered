"""
benchmark/vered_wgso_per_step.py

WGSO Vered sweep at the champion cell with per-step loss printout +
refresh markers, mirroring per_step_loss_compare.py's per-step record.

Cell:     (mom=0.7, lr=2e-3, gamma=0.9, clip=300, freq=20, schedule=constant_warmup)
Dampings: 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7
WGSO:     w_i = 1/(|x_i|^2 + median(|x|^2))
Outputs:  benchmark/results/vered_wgso_perstep_d{damp}_s1000_const.json
"""
from __future__ import annotations
import json, sys, time, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.stability_benchmark import (
    build_data, make_optimizers, evaluate_ppl, EMB_LR, OUT,
)
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info
import optimizer.raw_activation_hooks as rah


# ----- WGSO monkey-patch ---------------------------------------------------

WGSO_EPS_FRAC = 1.0


def _wgso_weight_rows(M: torch.Tensor) -> torch.Tensor:
    sq = (M * M).sum(dim=1)
    med = sq.median()
    w = 1.0 / (sq + WGSO_EPS_FRAC * med + 1e-30)
    return M * w.sqrt().unsqueeze(1)


_orig_fwd = rah.RawActivationHooks._forward_hook
_orig_bwd = rah.RawActivationHooks._backward_hook


def _wrap(method):
    def wrapped(self, *args):
        orig = rah.streaming_tsqr_update
        def weighted(running, x):
            return orig(running, _wgso_weight_rows(x))
        rah.streaming_tsqr_update = weighted
        try:
            method(self, *args)
        finally:
            rah.streaming_tsqr_update = orig
    return wrapped


def enable_wgso():
    rah.RawActivationHooks._forward_hook = _wrap(_orig_fwd)
    rah.RawActivationHooks._backward_hook = _wrap(_orig_bwd)


def disable_wgso():
    rah.RawActivationHooks._forward_hook = _orig_fwd
    rah.RawActivationHooks._backward_hook = _orig_bwd


# ----- Config --------------------------------------------------------------

CFG = {
    "gamma":              0.9,
    "momentum":           0.7,
    "kfac_lr":            2e-3,
    "grad_clip":          300.0,
    "max_steps":          1000,
    "factor_update_freq": 20,
    "warmup_steps":       200,
    "seed":               42,
}

DAMPINGS = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7]


# ----- One run with per-step printout --------------------------------------

def run_one(damping, train_loader_factory, val_loader, vocab_size, pad_id, device, hw):
    out_path = OUT / f"vered_wgso_perstep_d{damping:.0e}_s1000_const.json"
    if out_path.exists():
        print(f"[skip] {out_path.name} exists.")
        return json.loads(out_path.read_text())

    torch.manual_seed(CFG["seed"])
    model = SmallGPT(vocab_size=vocab_size).to(device)
    kfac_opt, emb_opt, _ = make_optimizers(
        "VeredKFAC", model, CFG["kfac_lr"], damping, CFG["momentum"],
        grad_clip=CFG["grad_clip"], gamma=CFG["gamma"],
        factor_update_freq=CFG["factor_update_freq"],
    )

    warmup = CFG["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.SequentialLR(kfac_opt, schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            kfac_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
        torch.optim.lr_scheduler.ConstantLR(
            kfac_opt, factor=1.0, total_iters=CFG["max_steps"]),
    ], milestones=[warmup])
    emb_scheduler = torch.optim.lr_scheduler.ConstantLR(
        emb_opt, factor=1.0, total_iters=CFG["max_steps"])

    train_loader = train_loader_factory()
    data_iter = iter(train_loader)

    freq = CFG["factor_update_freq"]
    records = []
    val_evals = []
    samples_seen = 0
    eval_every = 10_000
    next_eval = eval_every
    prev_loss = None

    print(f"\n=== WGSO Vered  damping={damping:.0e} ===")
    t0 = time.perf_counter()
    for step in range(1, CFG["max_steps"] + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=pad_id)
        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            print(f"  step={step}  loss=NaN, aborting")
            break

        model.zero_grad()
        loss.backward()

        refreshed = ((step - 1) % freq == 0) or (freq == 1)
        steps_since_refresh = (step - 1) % freq

        kfac_opt.step()
        emb_opt.step()
        scheduler.step()
        emb_scheduler.step()

        samples_seen += x.size(0)
        val_ppl = None
        if step % 40 == 0 and step != CFG["max_steps"]:
            val_ppl = evaluate_ppl(model, val_loader, device, pad_id)
            val_evals.append({"step": step, "samples": samples_seen, "val_ppl": val_ppl})

        delta_loss = (loss_val - prev_loss) if prev_loss is not None else 0.0
        prev_loss = loss_val

        records.append({
            "step": step, "loss": loss_val, "delta_loss": delta_loss,
            "refreshed": refreshed, "steps_since_refresh": steps_since_refresh,
            "val_ppl": val_ppl,
        })

        flag = "*" if refreshed else " "
        ppl_str = f"  val={val_ppl:.0f}" if val_ppl is not None else ""
        print(f"  step={step:4d} {flag}  age={steps_since_refresh:2d}  "
              f"loss={loss_val:.4f}  dL={delta_loss:+.4f}{ppl_str}")

    final_ppl = evaluate_ppl(model, val_loader, device, pad_id)
    wall = time.perf_counter() - t0
    val_evals.append({"step": CFG["max_steps"], "samples": samples_seen,
                      "val_ppl": final_ppl})

    out = {
        "variant":      "VeredKFAC",
        "wgso":         True,
        "wgso_eps":     WGSO_EPS_FRAC,
        "damping":      damping,
        "config":       CFG,
        "hw":           hw,
        "wall_s":       wall,
        "final_ppl":    final_ppl,
        "samples_seen": samples_seen,
        "val_evals":    val_evals,
        "per_step":     records,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> {out_path.name}  final_ppl={final_ppl:.0f}  wall={wall/60:.1f}m")
    return out


def main():
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    print(f"WGSO: w_i = 1/(|x_i|^2 + {WGSO_EPS_FRAC}*median(|x|^2))")
    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    enable_wgso()
    try:
        for d in DAMPINGS:
            run_one(d, train_loader_factory, val_loader, vocab_size, pad_id, device, hw)
    finally:
        disable_wgso()


if __name__ == "__main__":
    main()
