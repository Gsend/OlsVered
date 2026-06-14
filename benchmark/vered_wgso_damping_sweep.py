"""
benchmark/vered_wgso_damping_sweep.py

WGSO + diminishing damping sweep on VeredKFAC at the matched-screen champion.

Per-sample weighting in the streaming TSQR:
    w_i = 1 / (|x_i|^2 + eps * median(|x|^2))      # smoothed inverse-magnitude
Rows are multiplied by sqrt(w_i) before TSQR accumulation, so
    R_w^T R_w = sum_i w_i x_i x_i^T
This is the weighted Gram (inverse-variance-optimal under |x|^4 noise).

Hypothesis: with the noise floor lowered by WGSO, smaller damping
becomes viable and may expose Vered's kappa^1 numerical advantage as
training benefit.

Cell:     (mom=0.7, lr=2e-3, gamma=0.9, clip=300, freq=20, schedule=constant_warmup)
Dampings: 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7   (diminishing from the optimum)
Compare to: existing Vered (unweighted) at same dampings already in
    benchmark/results/vered_damp_m0.70_lr2e-03_d*_s1000_const.json

Wall: ~90 minutes (6 cells x ~15 min).
Output: benchmark/results/vered_wgso_m0.70_lr2e-03_d{damp}_s1000_const.json
"""
from __future__ import annotations
import json, sys, time, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from benchmark.stability_benchmark import build_data, run_probe, OUT
from benchmark.gpu_benchmark import get_device, get_hardware_info
import optimizer.raw_activation_hooks as rah


# ----- Monkey-patch the hooks to apply WGSO weighting before TSQR ----------

WGSO_EPS_FRAC = 1.0   # smoothing relative to median(|x|^2); 1.0 = "median floor"


def _wgso_weight_rows(M: torch.Tensor) -> torch.Tensor:
    """Multiply each row by sqrt(w_i) where w_i = 1/(|x_i|^2 + eps*median(|x|^2))."""
    sq_norms = (M * M).sum(dim=1)             # (n,)
    med = sq_norms.median()
    w = 1.0 / (sq_norms + WGSO_EPS_FRAC * med + 1e-30)
    sqrt_w = w.sqrt().unsqueeze(1)            # (n, 1)
    return M * sqrt_w


_orig_fwd = rah.RawActivationHooks._forward_hook
_orig_bwd = rah.RawActivationHooks._backward_hook


def _wgso_forward_hook(self, module, inputs, output):
    # Wrap streaming_tsqr_update to weight rows.
    orig_update = rah.streaming_tsqr_update
    def weighted_update(running, x):
        return orig_update(running, _wgso_weight_rows(x))
    rah.streaming_tsqr_update = weighted_update
    try:
        _orig_fwd(self, module, inputs, output)
    finally:
        rah.streaming_tsqr_update = orig_update


def _wgso_backward_hook(self, module, grad_input, grad_output):
    orig_update = rah.streaming_tsqr_update
    def weighted_update(running, x):
        return orig_update(running, _wgso_weight_rows(x))
    rah.streaming_tsqr_update = weighted_update
    try:
        _orig_bwd(self, module, grad_input, grad_output)
    finally:
        rah.streaming_tsqr_update = orig_update


def enable_wgso():
    rah.RawActivationHooks._forward_hook = _wgso_forward_hook
    rah.RawActivationHooks._backward_hook = _wgso_backward_hook


def disable_wgso():
    rah.RawActivationHooks._forward_hook = _orig_fwd
    rah.RawActivationHooks._backward_hook = _orig_bwd


# ----- Sweep ---------------------------------------------------------------

FIXED = {
    "variant":     "VeredKFAC",
    "gamma":       0.9,
    "momentum":    0.7,
    "kfac_lr":     2e-3,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "lr_schedule": "constant_warmup",
}

DAMPINGS = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7]


def out_path(damp: float) -> Path:
    tag = f"m{FIXED['momentum']:.2f}_lr{FIXED['kfac_lr']:.0e}_d{damp:.0e}_s1000_const"
    return OUT / f"vered_wgso_{tag}.json"


def run_one(damp, tlf, vl, vocab, pad, device, hw):
    p = out_path(damp)
    if p.exists():
        try:
            d = json.loads(p.read_text())
            print(f"  [skip] {p.name}  final_ppl={d['result'].get('final_ppl')}")
            return d
        except Exception:
            pass

    print(f"\n== WGSO Vered  damp={damp:.0e} ==")
    cfg = dict(FIXED, damping=damp, wgso=True, wgso_eps_frac=WGSO_EPS_FRAC)
    t0 = time.perf_counter()
    try:
        res = run_probe(
            variant=cfg["variant"], kfac_lr=cfg["kfac_lr"], damping=cfg["damping"],
            momentum=cfg["momentum"], max_steps=cfg["max_steps"],
            vocab_size=vocab, train_loader_factory=tlf, val_loader=vl, pad_id=pad,
            device=device, seed=42, record_natgrad=True, record_condition=True,
            condition_log_every=200, print_progress=True,
            grad_clip=cfg["grad_clip"], gamma=cfg["gamma"],
            lr_schedule=cfg["lr_schedule"],
        )
    except Exception as e:
        res = {"status": "diverged_crash", "error": f"{type(e).__name__}: {e}",
               "final_ppl": None, "val_ppls": [], "val_times": []}
    wall = time.perf_counter() - t0

    vps = res.get("val_ppls") or []
    slope = (vps[-3] - vps[-1]) / 2.0 if len(vps) >= 3 and all(
        v and v < 5000 for v in vps[-3:]) else None
    saved = {"config": cfg, "hw": hw, "wall_s": wall, "result": res,
             "slope_per_100": slope}
    p.write_text(json.dumps(saved, indent=2, default=str))
    print(f"  -> final_ppl={res.get('final_ppl')}  slope/100={slope}")
    return saved


def main():
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    print(f"WGSO enabled: w_i = 1/(|x_i|^2 + {WGSO_EPS_FRAC}*median(|x|^2))")
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1

    enable_wgso()
    try:
        runs = [run_one(d, tlf, vl, vocab, pad, device, hw) for d in DAMPINGS]
    finally:
        disable_wgso()

    print("\n=== summary (WGSO Vered vs unweighted baseline) ===")
    print(f"  {'damping':>10}  {'WGSO_ppl':>10}  {'unweighted_ppl':>15}  {'delta':>+7}")
    for d in DAMPINGS:
        r = next((rr for rr in runs if rr and abs(rr['config']['damping']-d)<1e-30), None)
        wgso_ppl = r["result"].get("final_ppl") if r else None
        # Look up unweighted baseline if it exists.
        base_path = OUT / f"vered_damp_m0.70_lr2e-03_d{d:.0e}_s1000_const.json"
        base_ppl = None
        if base_path.exists():
            try:
                base_ppl = json.loads(base_path.read_text())["result"].get("final_ppl")
            except Exception:
                pass
        w = f"{wgso_ppl:.0f}" if wgso_ppl else "DIV/--"
        b = f"{base_ppl:.0f}" if base_ppl else "--"
        delta = (wgso_ppl - base_ppl) if (wgso_ppl and base_ppl) else None
        ds = f"{delta:+.0f}" if delta is not None else "--"
        print(f"  {d:>10.0e}  {w:>10}  {b:>15}  {ds:>7}")


if __name__ == "__main__":
    main()
