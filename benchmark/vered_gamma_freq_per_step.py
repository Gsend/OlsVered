"""
benchmark/vered_gamma_freq_per_step.py

Vered (no WGSO) per-step loss sweep across (gamma, factor_update_freq).
Tests whether the EMA smoothing (gamma) is masking K-FAC factor staleness
and any potential solver-noise differences.

Grid:
    gamma in {0.0, 0.5, 0.9}  x  freq in {20, 100, 500}   = 9 cells

Each cell runs 1000 steps and records per-step:
    step, loss, delta_loss, steps_since_refresh, refreshed (bool)

Fixed: lr=2e-3, mom=0.7, damping=1e-4, clip=300, constant_warmup, seed=42.

Outputs: benchmark/results/per_step_VeredKFAC_g{gamma}_f{freq}_s1000.json
~9 x 15 min = ~2.5 hr total.
"""
from __future__ import annotations
import json, math, sys, time
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


CFG = {
    "momentum":    0.7,
    "kfac_lr":     2e-3,
    "damping":     1e-4,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "warmup_steps": 200,
    "seed":        42,
}

# (gamma, factor_update_freq)
GRID = [
    (0.0,  20), (0.0, 100), (0.0, 500),
    (0.5,  20), (0.5, 100), (0.5, 500),
    (0.9,  20), (0.9, 100), (0.9, 500),
]


def out_path(gamma, freq):
    return OUT / f"per_step_VeredKFAC_g{gamma:.1f}_f{freq}_s1000.json"


def run_one(gamma, freq, train_loader_factory, val_loader, vocab_size, pad_id, device, hw):
    p = out_path(gamma, freq)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    torch.manual_seed(CFG["seed"])
    model = SmallGPT(vocab_size=vocab_size).to(device)
    kfac_opt, emb_opt, _ = make_optimizers(
        "VeredKFAC", model, CFG["kfac_lr"], CFG["damping"], CFG["momentum"],
        grad_clip=CFG["grad_clip"], gamma=gamma,
        factor_update_freq=freq,
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
    records = []
    prev_loss = None

    print(f"\n=== Vered  gamma={gamma}  freq={freq} ===")
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

        delta_loss = (loss_val - prev_loss) if prev_loss is not None else 0.0
        prev_loss = loss_val

        records.append({
            "step": step, "loss": loss_val, "delta_loss": delta_loss,
            "refreshed": refreshed, "steps_since_refresh": steps_since_refresh,
        })

    final_ppl = evaluate_ppl(model, val_loader, device, pad_id)
    wall = time.perf_counter() - t0
    out = {
        "variant": "VeredKFAC",
        "gamma": gamma,
        "factor_update_freq": freq,
        "config": CFG,
        "hw": hw,
        "wall_s": wall,
        "final_ppl": final_ppl,
        "per_step": records,
    }
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> {p.name}  final_ppl={final_ppl:.0f}  wall={wall/60:.1f}m")
    return out


def main():
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    results = []
    for gamma, freq in GRID:
        results.append((gamma, freq, run_one(gamma, freq, train_loader_factory,
                                              val_loader, vocab_size, pad_id, device, hw)))

    print("\n=== summary ===")
    print(f"  {'gamma':>6}  {'freq':>6}  {'final_ppl':>10}")
    for gamma, freq, r in results:
        fp = r.get('final_ppl')
        print(f"  {gamma:>6.1f}  {freq:>6}  {(f'{fp:.0f}' if fp else 'DIV/--'):>10}")


if __name__ == "__main__":
    main()
