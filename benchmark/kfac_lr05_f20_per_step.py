"""
Run Classic and Vered at lr=0.5, freq=20 (gamma=0, mom=0).
Vered is likely already on disk; the script skips existing.
Compute autocorrelation of mean delta_loss vs age (after warmup).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark.stability_benchmark import build_data, make_optimizers, evaluate_ppl, OUT
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info


CFG = dict(gamma=0.0, momentum=0.0, damping=1e-4, grad_clip=300.0,
           max_steps=1000, warmup_steps=200, seed=42)
LR, FREQ = 0.5, 20
VARIANTS = ["VeredKFAC", "ClassicKFAC"]


def out_path(v):
    return OUT / f"per_step_clean_{v}_lr{LR:.0e}_f{FREQ}_s1000.json"


def run(variant, tlf, vl, vocab, pad, device, hw):
    p = out_path(variant)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())
    torch.manual_seed(CFG["seed"])
    model = SmallGPT(vocab_size=vocab).to(device)
    kfac, emb, _ = make_optimizers(variant, model, LR, CFG["damping"], CFG["momentum"],
                                    grad_clip=CFG["grad_clip"], gamma=CFG["gamma"],
                                    factor_update_freq=FREQ)
    w = CFG["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(kfac, schedulers=[
        torch.optim.lr_scheduler.LinearLR(kfac, start_factor=0.1, end_factor=1.0, total_iters=w),
        torch.optim.lr_scheduler.ConstantLR(kfac, factor=1.0, total_iters=CFG["max_steps"]),
    ], milestones=[w])
    esched = torch.optim.lr_scheduler.ConstantLR(emb, factor=1.0, total_iters=CFG["max_steps"])
    it = iter(tlf())
    recs, prev = [], None
    print(f"\n=== {variant}  lr={LR}  freq={FREQ} ===")
    for step in range(1, CFG["max_steps"]+1):
        try: x, y = next(it)
        except StopIteration: it = iter(tlf()); x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)
        lv = float(loss.item())
        if not math.isfinite(lv): break
        model.zero_grad(); loss.backward()
        kfac.step(); emb.step(); sched.step(); esched.step()
        dl = (lv - prev) if prev is not None else 0.0; prev = lv
        recs.append({"step": step, "loss": lv, "delta_loss": dl,
                     "refreshed": (step-1) % FREQ == 0, "steps_since_refresh": (step-1) % FREQ})
    fp = evaluate_ppl(model, vl, device, pad)
    out = {"variant": variant, "kfac_lr": LR, "factor_update_freq": FREQ,
           "config": CFG, "hw": hw, "final_ppl": fp, "per_step": recs}
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> ppl={fp:.0f}")
    return out


def autocorr(x, max_lag=15):
    x = x - x.mean()
    v = (x*x).mean()
    return [(lag, (x[:-lag] * x[lag:]).mean() / (v + 1e-30)) for lag in range(1, max_lag+1)]


def main():
    device = get_device(); hw = get_hardware_info()
    tlf, vl, vocab = build_data(device); pad = vocab - 1
    results = {v: run(v, tlf, vl, vocab, pad, device, hw) for v in VARIANTS}

    print("\n=== mean delta_loss vs age (warmup excluded) ===")
    for v, d in results.items():
        dl = np.array([r["delta_loss"] for r in d["per_step"]])
        folded = dl[:1000].reshape(50, FREQ)
        kept = folded[10:]   # windows starting at step 200+
        mean = kept.mean(0)
        print(f"\n{v} (ppl={d['final_ppl']:.0f}):")
        print(f"  mean: {mean.round(4)}")
        print(f"  autocorr (lag, value):")
        for lag, ac in autocorr(mean, max_lag=10):
            bar = "█" * max(0, int(abs(ac)*20))
            sign = '+' if ac >= 0 else '-'
            print(f"    lag={lag:>2}: {ac:+.3f}  {sign}{bar}")


if __name__ == "__main__":
    main()
