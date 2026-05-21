"""
benchmark/vered_diagnostic_single.py

Runs a single (momentum, lr) cell with the exact same machinery as
vered_2d_mom_lr_screen.py — same fixed config, same run_probe call,
same output JSON path.  Useful when you want to measure one specific
cell quickly without waiting for the full screen iteration order.

Default cell: mom=0.0, lr=8e-3 — the post-fix reproduction of the
historical Vered "winner" config (vered_clip_g0.90_c300.json, which
was nominally mom=0.3 but effectively mom=0.0 due to the pre-fix
momentum bug).  Diagnostic question: does this reproduce the
historical ~1000 ppl @ step 1000?

Usage:
    python benchmark/vered_diagnostic_single.py
    python benchmark/vered_diagnostic_single.py --mom 0.7 --lr 5e-4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data
from benchmark.gpu_benchmark import get_device, get_hardware_info
from benchmark.vered_2d_mom_lr_screen import run_one, out_path, FIXED


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mom", type=float, default=0.0,
                    help="momentum (default 0.0)")
    ap.add_argument("--lr", type=float, default=8e-3,
                    help="kfac_lr (default 8e-3)")
    args = ap.parse_args()

    mom, lr = args.mom, args.lr

    print("=" * 72)
    print(f"  Vered single-cell diagnostic: mom={mom:g}  lr={lr:.0e}")
    print(f"  Fixed:  gamma={FIXED['gamma']}, lambda={FIXED['damping']:.0e}, "
          f"clip={FIXED['grad_clip']:g}, steps={FIXED['max_steps']}")
    print(f"  Output JSON: {out_path(mom, lr).name}")
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1

    saved = run_one(mom, lr, train_loader_factory, val_loader,
                    vocab_size, pad_id, device, hw)

    if saved is None:
        print("\n  [!] run_one returned None")
        return

    res = saved.get("result", {})
    val_ppls = res.get("val_ppls") or []
    final_ppl = res.get("final_ppl")
    slope = saved.get("slope_per_100")

    print()
    print("=" * 72)
    print("  Diagnostic summary")
    print("=" * 72)
    print(f"  final_ppl @ step {FIXED['max_steps']}: "
          f"{final_ppl:.1f}" if final_ppl is not None else "  final_ppl: DIVERGED")
    print(f"  slope/100 (last ~200 steps): {slope}")
    if val_ppls:
        last5 = val_ppls[-5:]
        last5_str = ", ".join(f"{v:.0f}" for v in last5 if v is not None)
        print(f"  last 5 val_ppls: {last5_str}")

    # Compare against the historical reference, if applicable.
    if abs(mom - 0.0) < 1e-9 and abs(lr - 8e-3) / 8e-3 < 1e-3:
        print()
        print("  Historical reference (vered_clip_g0.90_c300.json, pre-fix):")
        print("    ppl ~1150 @ step  938")
        print("    ppl ~1014 @ step 1094")
        if final_ppl is not None:
            if final_ppl < 1300:
                verdict = "REPRODUCES historical baseline"
            elif final_ppl < 1700:
                verdict = "PARTIAL reproduction (1300-1700 band)"
            else:
                verdict = "REGRESSION vs. historical baseline -- bisect needed"
            print(f"    Verdict: {verdict}")

    print("=" * 72)


if __name__ == "__main__":
    main()
