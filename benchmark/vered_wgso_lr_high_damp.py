"""
benchmark/vered_wgso_lr_high_damp.py

WGSO Vered sweep at higher (damping, lr) pairs at the matched-screen
champion cell (mom=0.7, gamma=0.9).  Hypothesis: higher damping shrinks
the natgrad magnitude, so higher LR is needed to take comparable steps;
the cleaner WGSO curvature may make those larger steps land productively
where they otherwise overshoot.

Grid:
    damping=1e-3 x lr in {4e-3, 8e-3, 2e-2}
    damping=2e-3 x lr in {8e-3, 2e-2, 4e-2}
6 cells x ~15 min = ~90 min.

WGSO weight: w_i = 1 / (|x_i|^2 + median(|x|^2)).

Output: benchmark/results/vered_wgso_d{damp}_lr{lr}_s1000_const.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from benchmark.stability_benchmark import build_data, run_probe, OUT
from benchmark.gpu_benchmark import get_device, get_hardware_info
import optimizer.raw_activation_hooks as rah


# ----- WGSO monkey-patch ---------------------------------------------------

WGSO_EPS_FRAC = 1.0

def _wgso_weight_rows(M):
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


FIXED = {
    "variant":     "VeredKFAC",
    "gamma":       0.9,
    "momentum":    0.7,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "lr_schedule": "constant_warmup",
}

# (damping, lr) pairs to sweep
CELLS = [
    (1e-3, 4e-3),
    (1e-3, 8e-3),
    (1e-3, 2e-2),
    (2e-3, 8e-3),
    (2e-3, 2e-2),
    (2e-3, 4e-2),
]


def out_path(damp, lr):
    tag = f"d{damp:.0e}_lr{lr:.0e}_s{FIXED['max_steps']}_const"
    return OUT / f"vered_wgso_{tag}.json"


def run_one(damp, lr, tlf, vl, vocab, pad, device, hw):
    p = out_path(damp, lr)
    if p.exists():
        try:
            d = json.loads(p.read_text())
            print(f"  [skip] {p.name}  final_ppl={d['result'].get('final_ppl')}")
            return d
        except Exception:
            pass

    print(f"\n== WGSO Vered  damp={damp:.0e}  lr={lr:.0e} ==")
    cfg = dict(FIXED, damping=damp, kfac_lr=lr, wgso=True, wgso_eps_frac=WGSO_EPS_FRAC)
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
    print(f"WGSO eps_frac={WGSO_EPS_FRAC}")
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1

    enable_wgso()
    try:
        runs = [run_one(d, lr, tlf, vl, vocab, pad, device, hw) for d, lr in CELLS]
    finally:
        disable_wgso()

    print("\n=== summary ===")
    print(f"  {'damping':>10}  {'lr':>10}  {'final_ppl':>10}  {'slope/100':>10}")
    for d, lr in CELLS:
        r = next((rr for rr in runs if rr
                  and abs(rr['config']['damping']-d)<1e-30
                  and abs(rr['config']['kfac_lr']-lr)<1e-12), None)
        fin = r["result"].get("final_ppl") if r else None
        sl = r.get("slope_per_100") if r else None
        fin_str = f"{fin:.0f}" if fin else "DIV/--"
        sl_str = f"{sl:.1f}" if sl is not None else "--"
        print(f"  {d:>10.0e}  {lr:>10.0e}  {fin_str:>10}  {sl_str:>10}")


if __name__ == "__main__":
    main()
