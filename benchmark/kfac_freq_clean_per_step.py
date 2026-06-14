"""
benchmark/kfac_freq_clean_per_step.py

Raw K-FAC freq sweep with all smoothing knobs off, so per-step solver
noise and factor staleness are visible:

    gamma = 0   (no EMA on factor)
    mom   = 0   (no momentum buffer)
    freq  in {1, 5, 20, 100, 500}
    variant in {VeredKFAC, ClassicKFAC}

Per-step records: step, loss, delta_loss, refreshed, steps_since_refresh.

Fixed: lr=2e-3, damping=1e-4, clip=300, constant_warmup, seed=42, 1000 steps.

Outputs:
    benchmark/results/per_step_clean_{variant}_f{freq}_s1000.json

10 cells; rough ~2.5 hr (Vered slower).
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
    "gamma":        0.0,
    "momentum":     0.0,
    "damping":      1e-4,
    "grad_clip":    300.0,
    "max_steps":    1000,
    "warmup_steps": 200,
    "seed":         42,
}

VARIANTS = ["VeredKFAC", "ClassicKFAC"]
FREQS    = [20, 100, 500]
LRS      = [1.0, 0.5, 0.1]   # sweep large LRs; earlier runs at 2e-3 and 2e-2 still on disk


def out_path(variant, lr, freq):
    lr_tag = f"lr{lr:.0e}"
    return OUT / f"per_step_clean_{variant}_{lr_tag}_f{freq}_s1000.json"


def run_one(variant, lr, freq, tlf, vl, vocab, pad, device, hw):
    p = out_path(variant, lr, freq)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    torch.manual_seed(CFG["seed"])
    model = SmallGPT(vocab_size=vocab).to(device)
    kfac_opt, emb_opt, _ = make_optimizers(
        variant, model, lr, CFG["damping"], CFG["momentum"],
        grad_clip=CFG["grad_clip"], gamma=CFG["gamma"],
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

    train_loader = tlf()
    data_iter = iter(train_loader)
    records = []
    prev_loss = None

    print(f"\n=== {variant}  lr={lr}  freq={freq} ===")
    t0 = time.perf_counter()
    for step in range(1, CFG["max_steps"] + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader); x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=pad)
        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            print(f"  step={step}  loss=NaN, aborting"); break

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

    final_ppl = evaluate_ppl(model, vl, device, pad)
    wall = time.perf_counter() - t0
    out = {
        "variant": variant, "gamma": CFG["gamma"], "momentum": CFG["momentum"],
        "kfac_lr": lr, "factor_update_freq": freq, "config": CFG, "hw": hw,
        "wall_s": wall, "final_ppl": final_ppl, "per_step": records,
    }
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> {p.name}  final_ppl={final_ppl:.0f}  wall={wall/60:.1f}m")
    return out


def main():
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    rows = []
    for lr in LRS:
        for v in VARIANTS:
            for f in FREQS:
                rows.append((v, lr, f, run_one(v, lr, f, tlf, vl, vocab, pad, device, hw)))

    print("\n=== summary (gamma=0, mom=0) ===")
    print(f"  {'variant':>12}  {'lr':>6}  {'freq':>5}  {'final_ppl':>10}")
    for v, lr, f, r in rows:
        fp = r.get('final_ppl')
        print(f"  {v:>12}  {lr:>6}  {f:>5}  {(f'{fp:.0f}' if fp else 'DIV/--'):>10}")


if __name__ == "__main__":
    main()
