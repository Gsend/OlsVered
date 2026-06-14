"""
benchmark/kfac_bf16_damping_sweep.py

Damping-robustness sweep at bf16: Vered and Vered+WGSO across many damping
values.  Pairs with the existing kfac_bf16_classic_damping.py (Classic U-curve)
to give the §7.5 hyperparameter-robustness figure in the paper.

Cells (champion params: mom=0.7, lr=2e-3, gamma=0.9, freq=20):
    Vered:        damping ∈ {1e-7 .. 3e-2}
    Vered+WGSO:   same range
    (Classic damping sweep already collected by kfac_bf16_classic_damping.py)

Output: benchmark/results/per_step_bf16_{vered|vered_wgso}_d{damp}_s1000.json
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
from benchmark.kfac_bf16_compare import enable_bf16, disable_bf16


CFG_BASE = dict(gamma=0.9, momentum=0.7, kfac_lr=2e-3, grad_clip=300.0,
                max_steps=1000, warmup_steps=200, seed=42, factor_update_freq=20)

# Damping ladder — span enough to draw a curve.  Skip values we already have.
DAMPINGS = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# Cells: (label, wgso)
CELLS = [("vered", False), ("vered_wgso", True)]


def out_path(label, d):
    return OUT / f"per_step_bf16_damp_{label}_d{d:.0e}_s1000.json"


def run_one(label, wgso, damping, tlf, vl, vocab, pad, device, hw):
    p = out_path(label, damping)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    enable_bf16(wgso=wgso)
    try:
        torch.manual_seed(CFG_BASE["seed"])
        model = SmallGPT(vocab_size=vocab).to(device)
        kfac, emb, _ = make_optimizers(
            "VeredKFAC", model, CFG_BASE["kfac_lr"], damping, CFG_BASE["momentum"],
            grad_clip=CFG_BASE["grad_clip"], gamma=CFG_BASE["gamma"],
            factor_update_freq=CFG_BASE["factor_update_freq"],
        )
        w = CFG_BASE["warmup_steps"]
        sched = torch.optim.lr_scheduler.SequentialLR(kfac, schedulers=[
            torch.optim.lr_scheduler.LinearLR(kfac, start_factor=0.1, end_factor=1.0, total_iters=w),
            torch.optim.lr_scheduler.ConstantLR(kfac, factor=1.0, total_iters=CFG_BASE["max_steps"]),
        ], milestones=[w])
        esched = torch.optim.lr_scheduler.ConstantLR(emb, factor=1.0, total_iters=CFG_BASE["max_steps"])

        it = iter(tlf())
        recs, prev = [], None
        print(f"\n=== bf16 damping sweep  {label}  damp={damping:.0e} ===")
        t0 = time.perf_counter()
        for step in range(1, CFG_BASE["max_steps"] + 1):
            try: x, y = next(it)
            except StopIteration: it = iter(tlf()); x, y = next(it)
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1), ignore_index=pad)
            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step} NaN, aborting"); break
            model.zero_grad(); loss.backward()
            kfac.step(); emb.step(); sched.step(); esched.step()
            dl = (lv - prev) if prev is not None else 0.0; prev = lv
            recs.append({"step": step, "loss": lv, "delta_loss": dl,
                         "refreshed": (step-1) % CFG_BASE["factor_update_freq"] == 0,
                         "steps_since_refresh": (step-1) % CFG_BASE["factor_update_freq"]})

        fp = evaluate_ppl(model, vl, device, pad)
        wall = time.perf_counter() - t0
        out = {"label": label, "variant": "VeredKFAC", "wgso": wgso,
               "damping": damping, "precision": "bf16_kfac_only",
               "config": CFG_BASE, "hw": hw, "wall_s": wall,
               "final_ppl": fp, "per_step": recs}
        p.write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> ppl={fp:.0f}  wall={wall/60:.1f}m")
    finally:
        try:
            kfac.cleanup()
        except Exception:
            pass
        del kfac, emb, model
        disable_bf16()
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def main():
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device); pad = vocab - 1

    results = []
    total = len(CELLS) * len(DAMPINGS)
    done = 0
    for label, wgso in CELLS:
        for d in DAMPINGS:
            done += 1
            print(f"\n[{done}/{total}]")
            results.append(run_one(label, wgso, d, tlf, vl, vocab, pad, device, hw))

    # ---- Summary table -----------------------------------------------------
    print(f"\n=== bf16 damping-robustness summary ===")
    print(f"  {'damping':>10}  {'vered ppl':>10}  {'wgso ppl':>10}")
    by_cell = {}
    for r in results:
        by_cell.setdefault(r["label"], {})[r["damping"]] = r.get("final_ppl")
    for d in DAMPINGS:
        v = by_cell.get("vered", {}).get(d)
        w = by_cell.get("vered_wgso", {}).get(d)
        v_s = f"{v:.0f}" if v is not None and math.isfinite(v) else "DIV"
        w_s = f"{w:.0f}" if w is not None and math.isfinite(w) else "DIV"
        print(f"  {d:>10.0e}  {v_s:>10}  {w_s:>10}")


if __name__ == "__main__":
    main()
