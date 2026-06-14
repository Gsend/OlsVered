"""
benchmark/kfac_bf16_classic_damping.py

Classic K-FAC in bf16 swept across damping levels to test the
'just raise damping' rescue hypothesis.

In bf16 Classic at damping=1e-4 gave ppl=1997 vs fp32 922 (+1075).
Hypothesis: kappa(A+lambda*I) drops with lambda, so kappa^4 * eps_bf16
should shrink fast as lambda grows. If Classic can match its fp32 ppl
at some lambda in bf16, the 'Classic survives bf16 too, just retune'
counter-argument lands. If even at the best lambda Classic stays
hundreds of ppl above fp32, the kappa-scaling story holds.

Cells (all at champion mom=0.7, lr=2e-3, gamma=0.9, freq=20):
    classic_bf16_d1e-4   1e-4   (existing baseline; will be skipped)
    classic_bf16_d3e-4   3e-4
    classic_bf16_d1e-3   1e-3
    classic_bf16_d3e-3   3e-3
    classic_bf16_d1e-2   1e-2
    classic_bf16_d3e-2   3e-2
    classic_bf16_d1e-1   1e-1

Output: benchmark/results/per_step_bf16_classic_d{damping}_s1000.json
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark.stability_benchmark import build_data, make_optimizers, evaluate_ppl, OUT
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info
from benchmark.kfac_bf16_compare import enable_classic_bf16, disable_classic_bf16


CFG = dict(gamma=0.9, momentum=0.7, kfac_lr=2e-3, grad_clip=300.0,
           max_steps=1000, warmup_steps=200, seed=42, factor_update_freq=20)

DAMPINGS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# fp32 Classic at champion damping=1e-4 was 922 ppl.
FP32_BASELINE = 922


def out_path(d):
    return OUT / f"per_step_bf16_classic_d{d:.0e}_s1000.json"


def run_one(damping, tlf, vl, vocab, pad, device, hw):
    p = out_path(damping)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    enable_classic_bf16()
    try:
        torch.manual_seed(CFG["seed"])
        model = SmallGPT(vocab_size=vocab).to(device)
        kfac, emb, _ = make_optimizers(
            "ClassicKFAC", model, CFG["kfac_lr"], damping, CFG["momentum"],
            grad_clip=CFG["grad_clip"], gamma=CFG["gamma"],
            factor_update_freq=CFG["factor_update_freq"])
        w = CFG["warmup_steps"]
        sched = torch.optim.lr_scheduler.SequentialLR(kfac, schedulers=[
            torch.optim.lr_scheduler.LinearLR(kfac, start_factor=0.1, end_factor=1.0, total_iters=w),
            torch.optim.lr_scheduler.ConstantLR(kfac, factor=1.0, total_iters=CFG["max_steps"]),
        ], milestones=[w])
        esched = torch.optim.lr_scheduler.ConstantLR(emb, factor=1.0, total_iters=CFG["max_steps"])

        it = iter(tlf())
        recs, prev = [], None
        print(f"\n=== bf16 classic  damp={damping:.0e} ===")
        t0 = time.perf_counter()
        for step in range(1, CFG["max_steps"]+1):
            try: x, y = next(it)
            except StopIteration: it = iter(tlf()); x, y = next(it)
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1), ignore_index=pad)
            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step}  NaN, aborting"); break
            model.zero_grad(); loss.backward()
            kfac.step(); emb.step(); sched.step(); esched.step()
            dl = (lv - prev) if prev is not None else 0.0; prev = lv
            recs.append({"step": step, "loss": lv, "delta_loss": dl,
                         "refreshed": (step-1) % CFG["factor_update_freq"] == 0,
                         "steps_since_refresh": (step-1) % CFG["factor_update_freq"]})

        fp = evaluate_ppl(model, vl, device, pad)
        wall = time.perf_counter() - t0
        out = {"variant": "ClassicKFAC", "damping": damping,
               "precision": "bf16_kfac_only", "config": CFG, "hw": hw,
               "wall_s": wall, "final_ppl": fp, "per_step": recs}
        p.write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> ppl={fp:.0f}  wall={wall/60:.1f}m")
    finally:
        disable_classic_bf16()

    return out


def main():
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device); pad = vocab - 1

    # The existing bf16 baseline at damp=1e-4 lives at
    # per_step_bf16_classic_s1000.json. Copy/link it under the swept name
    # so run_one() can skip it.
    legacy = OUT / "per_step_bf16_classic_s1000.json"
    swept = out_path(1e-4)
    if legacy.exists() and not swept.exists():
        swept.write_text(legacy.read_text())
        print(f"[reuse] {legacy.name} -> {swept.name}")

    results = []
    for d in DAMPINGS:
        results.append(run_one(d, tlf, vl, vocab, pad, device, hw))

    print("\n=== bf16 Classic damping sweep ===")
    print(f"  fp32 baseline at damp=1e-4: {FP32_BASELINE} ppl")
    print(f"  {'damping':>10}  {'bf16 ppl':>10}  {'delta vs fp32':>14}")
    for d, r in zip(DAMPINGS, results):
        fp = r.get("final_ppl")
        diff = fp - FP32_BASELINE if fp else None
        diff_s = f"{diff:+.0f}" if diff is not None else "--"
        print(f"  {d:>10.0e}  {(f'{fp:.0f}' if fp else 'DIV'):>10}  {diff_s:>14}")


if __name__ == "__main__":
    main()
