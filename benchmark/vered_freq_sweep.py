"""
benchmark/vered_freq_sweep.py

Vered (no WGSO) factor_update_freq sweep at the champion cell
(lr=2e-3, damping=1e-4, mom=0.7, gamma=0.9). Tests whether less frequent
K-FAC refresh changes the result. The matched-screen baseline at freq=20
is 921 ppl; this sweep checks freq in {50, 100, 200, 500}.

freq=20 omitted -- matched-screen data already covers it.

4 cells x ~15 min = ~60 min.

Output: benchmark/results/vered_freq{freq}_d1e-04_lr2e-03_s1000_const.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data, run_probe, OUT
from benchmark.gpu_benchmark import get_device, get_hardware_info


FIXED = {
    "variant":     "VeredKFAC",
    "gamma":       0.9,
    "momentum":    0.7,
    "damping":     1e-4,
    "kfac_lr":     2e-3,
    "grad_clip":   300.0,
    "max_steps":   1000,
    "lr_schedule": "constant_warmup",
}

FREQS = [50, 100, 200, 500]


def out_path(freq):
    return OUT / f"vered_freq{freq}_d1e-04_lr2e-03_s1000_const.json"


def run_one(freq, tlf, vl, vocab, pad, device, hw):
    p = out_path(freq)
    if p.exists():
        try:
            d = json.loads(p.read_text())
            print(f"  [skip] {p.name}  final_ppl={d['result'].get('final_ppl')}")
            return d
        except Exception:
            pass

    print(f"\n== Vered (no WGSO)  freq={freq} ==")
    cfg = dict(FIXED, factor_update_freq=freq)
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
            factor_update_freq=freq,
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
    print(f"Vered (no WGSO) freq sweep at lr={FIXED['kfac_lr']}, damping={FIXED['damping']}")
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1

    runs = [run_one(f, tlf, vl, vocab, pad, device, hw) for f in FREQS]

    print("\n=== summary (Vered no WGSO, lr=2e-3, damping=1e-4) ===")
    print(f"  {'freq':>6}  {'final_ppl':>10}  {'slope/100':>10}")
    print(f"  {'20':>6}  {'921':>10}  {'(matched-screen baseline)':>25}")
    for f in FREQS:
        r = next((rr for rr in runs if rr
                  and rr['config'].get('factor_update_freq') == f), None)
        fin = r["result"].get("final_ppl") if r else None
        sl = r.get("slope_per_100") if r else None
        fin_str = f"{fin:.0f}" if fin else "DIV/--"
        sl_str = f"{sl:.1f}" if sl is not None else "--"
        print(f"  {f:>6}  {fin_str:>10}  {sl_str:>10}")


if __name__ == "__main__":
    main()
