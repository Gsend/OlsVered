"""
benchmark/kfac_lr05_f20_frozen.py

Re-run Vered and Classic at lr=0.5, freq=20 (gamma=0, mom=0) with
non-K-FAC parameters FROZEN (embeddings, LayerNorm, LM head not updated).
Isolates pure K-FAC dynamics in delta_loss vs age.

Output: benchmark/results/per_step_clean_frozen_{variant}_lr5e-01_f20_s1000.json
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
    return OUT / f"per_step_clean_frozen_{v}_lr{LR:.0e}_f{FREQ}_s1000.json"


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
    # NOTE: emb_opt exists but we never call its .step(), so the
    # non-K-FAC params (embeddings, LayerNorm, LM head) stay frozen
    # at their initial random values. The model is still trainable via
    # the K-FAC-covered Linear layers only.

    w = CFG["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(kfac, schedulers=[
        torch.optim.lr_scheduler.LinearLR(kfac, start_factor=0.1, end_factor=1.0, total_iters=w),
        torch.optim.lr_scheduler.ConstantLR(kfac, factor=1.0, total_iters=CFG["max_steps"]),
    ], milestones=[w])

    it = iter(tlf())
    recs, prev = [], None
    print(f"\n=== {variant}  FROZEN  lr={LR}  freq={FREQ} ===")
    t0 = time.perf_counter()
    for step in range(1, CFG["max_steps"]+1):
        try: x, y = next(it)
        except StopIteration: it = iter(tlf()); x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)
        lv = float(loss.item())
        if not math.isfinite(lv):
            print(f"  step={step} NaN, aborting"); break
        model.zero_grad(); loss.backward()
        kfac.step()
        sched.step()
        # emb.step() intentionally skipped -> non-K-FAC params frozen
        dl = (lv - prev) if prev is not None else 0.0; prev = lv
        recs.append({"step": step, "loss": lv, "delta_loss": dl,
                     "refreshed": (step-1) % FREQ == 0,
                     "steps_since_refresh": (step-1) % FREQ})

    fp = evaluate_ppl(model, vl, device, pad)
    wall = time.perf_counter() - t0
    out = {"variant": variant, "kfac_lr": LR, "factor_update_freq": FREQ,
           "frozen_non_kfac": True, "config": CFG, "hw": hw,
           "wall_s": wall, "final_ppl": fp, "per_step": recs}
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> ppl={fp:.0f}  wall={wall/60:.1f}m")
    return out


def autocorr(x, max_lag=15):
    x = x - x.mean(); v = (x*x).mean()
    return [(lag, (x[:-lag] * x[lag:]).mean() / (v + 1e-30))
            for lag in range(1, max_lag+1)]


def main():
    device = get_device(); hw = get_hardware_info()
    tlf, vl, vocab = build_data(device); pad = vocab - 1
    results = {v: run(v, tlf, vl, vocab, pad, device, hw) for v in VARIANTS}

    print("\n=== mean delta_loss vs age, FROZEN non-K-FAC (lr=0.5, freq=20) ===")
    for v, d in results.items():
        dl = np.array([r["delta_loss"] for r in d["per_step"]])
        folded = dl[:1000].reshape(50, FREQ)
        kept = folded[10:]
        mean = kept.mean(0)
        print(f"\n{v}  ppl={d.get('final_ppl'):.0f}  n_win={len(kept)}")
        print(f"  mean: {mean.round(4)}")
        print(f"  autocorr:")
        for lag, ac in autocorr(mean, max_lag=10):
            bar = "█" * max(0, int(abs(ac)*20))
            sign = '+' if ac >= 0 else '-'
            print(f"    lag={lag:>2}: {ac:+.3f}  {sign}{bar}")


if __name__ == "__main__":
    main()
