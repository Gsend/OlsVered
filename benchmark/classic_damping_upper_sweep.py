"""
benchmark/classic_damping_upper_sweep.py

Classic K-FAC damping sweep at gamma=0.9 matched-screen operating point,
upper-range only (1e-4 .. 2e-3).  The Vered sweep showed monotonic
preference for larger damping at gamma=0.9; this asks whether Classic
shares that preference at the same cells.

Cells:
    A_champion       (mom=0.7, lr=2e-3)  matched-screen 922 @ damp=1e-4
    B_high_eff_lr    (mom=0.9, lr=8e-3)  matched-screen 1667 @ damp=1e-4

Wall: ~2 hours (2 cells x 4 dampings x ~15 min).
Resume-friendly per cell.
"""
from __future__ import annotations
import json, sys, time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data, run_probe, OUT, time_to_ppl
from benchmark.gpu_benchmark import get_device, get_hardware_info


FIXED = {
    "variant":     "ClassicKFAC",
    "gamma":       0.9,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "lr_schedule": "constant_warmup",
}

CELLS = [
    ("A_champion",    0.7, 2e-3),
    ("B_high_eff_lr", 0.9, 8e-3),
]

DAMPINGS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]


def out_path(mom: float, lr: float, damp: float) -> Path:
    tag = f"m{mom:.2f}_lr{lr:.0e}_d{damp:.0e}_s{FIXED['max_steps']}_const"
    return OUT / f"classic_damp_{tag}.json"


def run_one(label, mom, lr, damp, tlf, vl, vocab, pad, device, hw):
    p = out_path(mom, lr, damp)
    if p.exists():
        try:
            d = json.loads(p.read_text())
            ppl = d["result"].get("final_ppl")
            print(f"  [skip] {p.name}  final_ppl={ppl}")
            return d
        except Exception:
            pass

    print(f"\n== {label}: mom={mom} lr={lr:.0e} damp={damp:.0e} ==")
    cfg = dict(FIXED, momentum=mom, kfac_lr=lr, damping=damp, cell_label=label)
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
    fin = res.get("final_ppl")
    print(f"  -> final_ppl={fin}  slope/100={slope}")
    return saved


def main():
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1

    runs = []
    for (label, mom, lr), damp in product(CELLS, DAMPINGS):
        runs.append(run_one(label, mom, lr, damp, tlf, vl, vocab, pad, device, hw))

    print("\n=== summary ===")
    for label, mom, lr in CELLS:
        print(f"\n  {label}: mom={mom} lr={lr:.0e}")
        print(f"    {'damping':>10}  {'final_ppl':>10}")
        for damp in DAMPINGS:
            r = next((rr for rr in runs if rr and abs(rr['config']['momentum']-mom)<1e-9
                      and abs(rr['config']['kfac_lr']-lr)<1e-12
                      and abs(rr['config']['damping']-damp)<1e-30), None)
            if r is None: continue
            fin = r["result"].get("final_ppl")
            print(f"    {damp:>10.0e}  {fin}")


if __name__ == "__main__":
    main()
