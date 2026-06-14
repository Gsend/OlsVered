"""
benchmark/vered_freshness_sweep.py

Tests whether Vered's kappa^1 stability advantage manifests when the
K-FAC factors are kept fresh.  Three configurations sweep the
(batch, factor_update_freq, max_steps) joint so that:

    * Total token budget is approximately fixed at ~8.2M tokens, the same
      as the matched-screen 1000-step Vered runs.
    * Number of factor refreshes is fixed at 50, so the EMA's effective
      memory in "refresh count" is the same.
    * What varies is the factor-age profile each step sees:
        baseline    -> factor age cycles 0..19 over the freq=20 window
        medium      -> factor age cycles 0..4
        large_batch -> factor age is always 0 (every step IS a refresh)

The hypothesis under test (from the conversation about A_ema noise and
inherited K-FAC frequency defaults): if the cached-factor staleness is
the binding constraint at the matched-screen champion cell, going to
the large_batch row should drop final_ppl materially below the 921 ppl
baseline.  If it doesn't, staleness isn't binding at this operating
point and the (freq, batch) defaults inherited from Classic K-FAC
conventions are fine for Vered too.

Cell (fixed across all rows):
    variant=VeredKFAC, gamma=0.9, mom=0.7, lr=2e-3, damping=1e-6,
    grad_clip=300, lr_schedule=constant_warmup, seed=42.

Configurations:
    | label       | batch | freq | steps | tokens | warmup |
    |-------------|-------|------|-------|--------|--------|
    | baseline    |   64  |  20  | 1000  |  8.2M  |  200   |
    | medium      |  256  |   5  |  250  |  8.2M  |   50   |
    | large_batch | 1280  |   1  |   50  |  8.2M  |   10   |

Notes:
    * `batch=1280` is at the upper edge of what fits on a 17GB 3080;
      reduce to 1024 if OOM (and adjust steps to keep token budget).
    * Warmup is scaled as ~20% of total steps so the schedule shape is
      comparable across rows.
    * eval_every_samples=10_000 is samples-based not step-based, so it
      naturally gives the same count of evals (~6) across rows.

Wall time estimate: ~75 minutes total (baseline ~15 min, medium ~25 min,
large_batch ~35 min; large-batch steps are slower because each does a
full factor refresh and processes 20x the data).

Outputs:
    benchmark/results/vered_freshness_{label}_s{steps}_const.json

Usage:
    python benchmark/vered_freshness_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.stability_benchmark import build_data, run_probe, OUT, time_to_ppl
from benchmark.gpu_benchmark import get_device, get_hardware_info


# ---- Configuration --------------------------------------------------------

# Fixed cell across all rows (matched-screen champion).
CELL = {
    "variant":     "VeredKFAC",
    "gamma":       0.9,
    "momentum":    0.7,
    "kfac_lr":     2e-3,
    "damping":     1e-6,
    "grad_clip":   300.0,
    "lr_schedule": "constant_warmup",
    "seed":        42,
}

# Per-row freshness config: (label, batch, freq, max_steps, warmup_steps)
ROWS = [
    ("baseline",    64,   20, 1000, 200),
    ("medium",     256,    5,  250,  50),
    ("large_batch", 1280,   1,   50,  10),
]


def out_path(label: str, steps: int) -> Path:
    return OUT / f"vered_freshness_{label}_s{steps}_const.json"


def run_one(label: str, batch: int, freq: int, max_steps: int,
            warmup: int, device, hw: Dict) -> Optional[Dict]:
    p = out_path(label, max_steps)

    if p.exists():
        try:
            data = json.loads(p.read_text())
            ppl = data["result"].get("final_ppl")
            ppl_str = f"{ppl:.0f}" if ppl is not None else "DIVERGED"
            print(f"  [skip] {p.name} exists.  final_ppl={ppl_str}")
            return data
        except Exception as e:
            print(f"  [warn] failed to load {p.name}: {e}; re-running")

    # Fresh data loader at the requested batch size for this row.
    train_loader_factory, val_loader, vocab_size = build_data(device, batch_size=batch)
    pad_id = vocab_size - 1

    cfg = dict(CELL, batch=batch, factor_update_freq=freq,
               max_steps=max_steps, warmup_steps=warmup,
               freshness_row=label)
    total_tokens = batch * 128 * max_steps   # seq_len=128
    print()
    print("=" * 72)
    print(f"  Vered freshness sweep [{label}]:")
    print(f"    batch={batch}  freq={freq}  max_steps={max_steps}  "
          f"warmup={warmup}")
    print(f"    total_tokens ~ {total_tokens/1e6:.1f}M  "
          f"refreshes ~ {max_steps // freq}")
    print(f"    cell: mom={CELL['momentum']}, lr={CELL['kfac_lr']:.0e}, "
          f"damping={CELL['damping']:.0e}")
    print("=" * 72)

    t0 = time.perf_counter()
    try:
        res = run_probe(
            variant=CELL["variant"],
            kfac_lr=CELL["kfac_lr"],
            damping=CELL["damping"],
            momentum=CELL["momentum"],
            max_steps=max_steps,
            vocab_size=vocab_size,
            train_loader_factory=train_loader_factory,
            val_loader=val_loader,
            pad_id=pad_id,
            device=device,
            seed=CELL["seed"],
            record_natgrad=True,
            record_condition=True,
            condition_log_every=max(1, max_steps // 5),
            print_progress=True,
            grad_clip=CELL["grad_clip"],
            gamma=CELL["gamma"],
            lr_schedule=CELL["lr_schedule"],
            factor_update_freq=freq,
            warmup_steps=warmup,
        )
    except Exception as e:
        crash_msg = f"{type(e).__name__}: {e}"
        print(f"\n  [!] CRASH during run_probe: {crash_msg}")
        res = {
            "status":         "diverged_crash",
            "error":          crash_msg,
            "final_ppl":      None,
            "val_ppls":       [],
            "val_times":      [],
            "median_step_ms": None,
        }
    wall = time.perf_counter() - t0

    val_ppls = res.get("val_ppls") or []
    slope: Optional[float] = None
    if len(val_ppls) >= 3:
        recent = val_ppls[-3:]
        if all(v is not None and v < 5000 for v in recent):
            slope = (recent[0] - recent[-1]) / 2.0

    try:
        ttt = {
            f"ppl<={int(t)}": time_to_ppl(res.get("val_ppls", []),
                                          res.get("val_times", []), t)
            for t in [3000.0, 2000.0, 1500.0, 1200.0, 1000.0]
        }
    except Exception:
        ttt = {}

    saved = {
        "config":         cfg,
        "hw":             hw,
        "wall_s":         wall,
        "result":         res,
        "slope_per_100":  slope,
        "time_to_target": ttt,
    }
    p.write_text(json.dumps(saved, indent=2, default=str))
    fin = res.get("final_ppl")
    fin_str = f"{fin:.0f}" if fin is not None else "DIV"
    sl_str = f"{slope:.1f}" if slope is not None else "n/a"
    print(f"  Saved -> {p.name}   final_ppl={fin_str}   slope/100={sl_str}")
    return saved


# ---- Main -----------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Vered freshness sweep — kappa^1 advantage under fresh curvature")
    print("=" * 72)
    print(f"  Fixed cell: variant={CELL['variant']}, gamma={CELL['gamma']}, "
          f"mom={CELL['momentum']}, lr={CELL['kfac_lr']:.0e}, "
          f"damping={CELL['damping']:.0e}")
    print(f"  Rows ({len(ROWS)}): " +
          ", ".join(f"{label}(b={b},f={f},s={s})" for label, b, f, s, _ in ROWS))
    print("=" * 72)

    device = get_device()
    hw = get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name')}  CUDA {hw.get('cuda_version')}  "
          f"torch {hw.get('torch_version')}")

    runs: List[Dict] = []
    for label, batch, freq, steps, warmup in ROWS:
        saved = run_one(label, batch, freq, steps, warmup, device, hw)
        if saved is not None:
            runs.append(saved)

    # ----- Summary --------------------------------------------------------
    print()
    print("=" * 72)
    print("  Freshness sweep summary")
    print("=" * 72)
    print(f"  {'row':>12}  {'batch':>6}  {'freq':>5}  {'steps':>6}  "
          f"{'final_ppl':>10}  {'slope/100':>10}  {'wall':>7}")
    print("  " + "-" * 70)
    finals: Dict[str, Optional[float]] = {}
    for r in runs:
        c = r["config"]
        res = r["result"]
        label = c["freshness_row"]
        fin = res.get("final_ppl")
        sl = r.get("slope_per_100")
        fin_str = f"{fin:.0f}" if fin is not None else "DIV"
        sl_str = f"{sl:.1f}" if sl is not None else "n/a"
        wall_s = r.get("wall_s", 0)
        print(f"  {label:>12}  {c['batch']:>6}  {c['factor_update_freq']:>5}  "
              f"{c['max_steps']:>6}  {fin_str:>10}  {sl_str:>10}  "
              f"{wall_s/60:>5.1f}m")
        finals[label] = fin

    # ----- Verdict --------------------------------------------------------
    print()
    print("=" * 72)
    print("  Verdict (matched-screen baseline at this cell = 921 ppl)")
    print("=" * 72)
    baseline_ppl = finals.get("baseline")
    large_ppl    = finals.get("large_batch")
    medium_ppl   = finals.get("medium")
    if baseline_ppl is None or large_ppl is None:
        print("  Incomplete data; verdict deferred.")
        return

    print(f"    baseline    (freq=20):  {baseline_ppl:.0f} ppl")
    if medium_ppl is not None:
        print(f"    medium      (freq=5):   {medium_ppl:.0f} ppl  "
              f"(delta vs baseline: {medium_ppl - baseline_ppl:+.0f})")
    print(f"    large_batch (freq=1):   {large_ppl:.0f} ppl  "
          f"(delta vs baseline: {large_ppl - baseline_ppl:+.0f})")
    print()

    delta_lb = large_ppl - baseline_ppl
    if delta_lb < -50:
        print("  -> FRESHNESS HELPS: large_batch beats baseline by >50 ppl.")
        print("     kappa^1 advantage materializes under fresh curvature.")
        print("     Inherited (freq=20) default is suboptimal for Vered.")
    elif delta_lb > 50:
        print("  -> FRESHNESS HURTS: large_batch loses to baseline by >50 ppl.")
        print("     Likely: large-batch needs different LR, or step-count is")
        print("     too low for the warmup+constant schedule to reach LR.")
    else:
        print("  -> NEUTRAL: large_batch within 50 ppl of baseline.")
        print("     Cached-factor staleness is NOT the binding constraint at")
        print("     this operating point.  Vered's kappa^1 advantage does not")
        print("     manifest as better training here.")
    print("=" * 72)


if __name__ == "__main__":
    main()
