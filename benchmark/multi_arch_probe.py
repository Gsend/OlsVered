"""
benchmark/multi_arch_probe.py

Tier 1 generalization probe: run each K-FAC variant on multiple model
architectures and aggregate results into a unified summary.

Architectures covered:
    - MLP on MNIST       (via benchmark/training_benchmark.py)
    - ConvNet on CIFAR-10 (via benchmark/training_benchmark.py)
    - SmallGPT on WT-2   (covered by classic_full_grid.py and
                          vered_full_grid.py; this script reads their
                          existing JSON results rather than re-running)

How it works:
    For MLP and CIFAR, subprocess-calls benchmark/training_benchmark.py with
    --optimizers limited to the three K-FAC variants (plus Adam baseline).
    For SmallGPT, reads the existing winner JSONs from benchmark/results/.

IMPORTANT METHODOLOGICAL CAVEAT:
    benchmark/training_benchmark.py uses HARDCODED hyperparameters per
    variant (see its top-of-file CONFIGS).  Those hardcoded values predate
    the Vered full-grid result and predate Phase 2 (Classic full grid).
    For a properly fair comparison, the hardcoded configs in
    training_benchmark.py should be updated to match each variant's tuned
    optimum from the SmallGPT grids:
        ClassicKFAC: <fill in after running classic_full_grid.py>
        OlsSMKFAC:   <fill in after running olssm_full_grid.py if/when run>
        VeredKFAC:   gamma=0.3, momentum=0.3, damping=1e-4, grad_clip=60.0,
                     lr=8e-3 (from vered_full_grid.py winner)

    Until those updates happen, this probe is a "Tier 0" baseline:
    it tests generalization at literature-default configs, not at the
    tuned configs from the SmallGPT grids.  Results from this probe should
    be interpreted as "do K-FAC variants generalize across architectures at
    their default settings" -- a weaker but still useful claim.

Usage:
    python benchmark/multi_arch_probe.py [--archs mnist cifar smallgpt]
                                          [--optimizers ClassicKFAC OlsSMKFAC VeredKFAC]
                                          [--steps N]

Wall time:
    MNIST MLP (per variant):    ~10-15 min
    CIFAR ConvNet (per variant): ~30-60 min
    SmallGPT (already done):    0 (reads existing files)
    Total for 3 variants on MNIST + CIFAR: ~3-4 hours.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"

DEFAULT_ARCHS     = ["mnist", "cifar", "smallgpt"]
DEFAULT_OPT_NAMES = ["ClassicKFAC", "OlsSMKFAC", "VeredKFAC"]


# ---- Per-architecture runners ---------------------------------------------

def run_training_benchmark(dataset: str, optimizers: List[str],
                           steps: Optional[int] = None) -> Dict:
    """Run benchmark/training_benchmark.py for MLP/MNIST or CIFAR.

    Returns the parsed contents of training_results.json.
    """
    cmd = [sys.executable, str(ROOT / "benchmark" / "training_benchmark.py"),
           "--dataset", dataset,
           "--optimizers"] + optimizers
    if steps is not None:
        cmd += ["--steps", str(steps)]

    print(f"\n  Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    wall = time.perf_counter() - t0
    print(f"  Subprocess exited with code {proc.returncode}; wall = {wall/60:.1f} min")

    out_json = RESULTS_DIR / "training_results.json"
    if not out_json.exists():
        print(f"  [warn] expected {out_json} not produced")
        return {}
    try:
        return json.loads(out_json.read_text())
    except Exception as e:
        print(f"  [warn] failed to parse {out_json}: {e}")
        return {}


def read_smallgpt_results(optimizers: List[str]) -> Dict:
    """Read existing SmallGPT/WT-2 results from the grid JSONs.

    For VeredKFAC: reads vered_grid_g0.30_m0.30_l1e-04.json (winner from
                   vered_full_grid.py).
    For ClassicKFAC: reads the lowest-ppl classic_grid_*.json file (winner
                     from classic_full_grid.py, if run).
    For OlsSMKFAC: not yet covered by any grid; reads any olssm_grid_* if
                   present, else returns None.
    """
    results = {}

    # VeredKFAC winner
    if "VeredKFAC" in optimizers:
        p = RESULTS_DIR / "vered_grid_g0.30_m0.30_l1e-04.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                results["VeredKFAC"] = {
                    "final_metric": data["result"].get("final_ppl"),
                    "metric_name":  "val_ppl",
                    "config":       data["config"],
                    "wall_min":     data["wall_s"] / 60,
                    "source":       p.name,
                }
            except Exception as e:
                print(f"  [warn] failed to read {p}: {e}")

    # ClassicKFAC: pick the best classic_grid_*.json file
    if "ClassicKFAC" in optimizers:
        candidates = list(RESULTS_DIR.glob("classic_grid_*.json"))
        best = None
        best_ppl = float("inf")
        for c in candidates:
            try:
                data = json.loads(c.read_text())
                ppl = data["result"].get("final_ppl")
                if ppl is not None and ppl < best_ppl and ppl < 5000:
                    best_ppl = ppl
                    best = (c, data)
            except Exception:
                continue
        if best is not None:
            c, data = best
            results["ClassicKFAC"] = {
                "final_metric": data["result"].get("final_ppl"),
                "metric_name":  "val_ppl",
                "config":       data["config"],
                "wall_min":     data["wall_s"] / 60,
                "source":       c.name,
            }

    # OlsSMKFAC: optional
    if "OlsSMKFAC" in optimizers:
        candidates = list(RESULTS_DIR.glob("olssm_grid_*.json"))
        best = None
        best_ppl = float("inf")
        for c in candidates:
            try:
                data = json.loads(c.read_text())
                ppl = data["result"].get("final_ppl")
                if ppl is not None and ppl < best_ppl and ppl < 5000:
                    best_ppl = ppl
                    best = (c, data)
            except Exception:
                continue
        if best is not None:
            c, data = best
            results["OlsSMKFAC"] = {
                "final_metric": data["result"].get("final_ppl"),
                "metric_name":  "val_ppl",
                "config":       data["config"],
                "wall_min":     data["wall_s"] / 60,
                "source":       c.name,
            }

    return results


# ---- Aggregation ----------------------------------------------------------

def aggregate(arch_to_raw: Dict[str, Dict], optimizers: List[str]) -> Dict:
    """Normalize each architecture's result format into a unified table:
        { (arch, variant) -> {final_metric, metric_name, ...} }
    """
    table = {}

    # MNIST and CIFAR -- training_benchmark.py output format
    for arch in ("mnist", "cifar"):
        raw = arch_to_raw.get(arch, {})
        if not raw:
            continue
        for variant in optimizers:
            entry = raw.get(variant) or raw.get(variant.lower()) or {}
            if not entry:
                # The variant may be keyed differently in training_results.json
                # (e.g., "VeredKFAC" vs "vered_kfac"); try fuzzy match.
                for k, v in raw.items():
                    if variant.lower().replace("kfac", "") in k.lower().replace("_", ""):
                        entry = v
                        break
            if entry:
                final_acc = entry.get("final_val_acc")
                table[(arch, variant)] = {
                    "final_metric": final_acc,
                    "metric_name":  "val_acc",
                    "wall_min":     entry.get("time_to_target", 0) / 60 if entry.get("time_to_target") else None,
                    "steps":        entry.get("steps_to_target") or len(entry.get("step", [])),
                }

    # SmallGPT -- read from grid result files
    smallgpt_raw = arch_to_raw.get("smallgpt", {})
    for variant, entry in smallgpt_raw.items():
        if variant in optimizers:
            table[("smallgpt", variant)] = {
                "final_metric": entry["final_metric"],
                "metric_name":  entry["metric_name"],
                "wall_min":     entry["wall_min"],
                "source":       entry.get("source"),
            }

    return table


def print_summary(table: Dict, archs: List[str], optimizers: List[str]):
    print()
    print("=" * 72)
    print("  Multi-architecture summary")
    print("=" * 72)
    # Header
    header = f"  {'Variant':<14}" + "".join(
        f"  {a:<22}" for a in archs)
    print(header)
    print("  " + "-" * (14 + 24 * len(archs)))
    for variant in optimizers:
        row = f"  {variant:<14}"
        for a in archs:
            entry = table.get((a, variant))
            if entry is None:
                row += f"  {'(missing)':<22}"
            else:
                m = entry["final_metric"]
                n = entry["metric_name"]
                if m is None:
                    cell = "DIVERGED"
                elif n == "val_acc":
                    cell = f"{n}={m:.4f}"
                else:
                    cell = f"{n}={m:.1f}"
                row += f"  {cell:<22}"
        print(row)

    print()
    print("  Notes:")
    print("    - val_acc: higher is better (MNIST, CIFAR)")
    print("    - val_ppl: lower is better (SmallGPT)")
    print("    - 'missing' = no result for that (arch, variant) cell")
    print()
    print("  IMPORTANT: MNIST/CIFAR results use training_benchmark.py's")
    print("  hardcoded hyperparameters, which were set BEFORE the SmallGPT")
    print("  grid search.  For a fully fair Tier 1 comparison, update those")
    print("  hardcoded values to match each variant's grid winner, then re-run")
    print("  this probe.  See top-of-file docstring for details.")


# ---- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-architecture generalization probe for K-FAC variants")
    ap.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS,
                    choices=DEFAULT_ARCHS,
                    help="architectures to probe (default: all three)")
    ap.add_argument("--optimizers", nargs="+", default=DEFAULT_OPT_NAMES,
                    help="K-FAC variants to compare")
    ap.add_argument("--steps", type=int, default=None,
                    help="override --steps for training_benchmark.py")
    args = ap.parse_args()

    print("=" * 72)
    print("  Multi-architecture K-FAC variant probe")
    print(f"  Architectures: {args.archs}")
    print(f"  Optimizers:    {args.optimizers}")
    print("=" * 72)

    arch_to_raw: Dict[str, Dict] = {}

    if "mnist" in args.archs:
        print("\n## MNIST (4-layer MLP)")
        arch_to_raw["mnist"] = run_training_benchmark(
            "mnist", args.optimizers, args.steps)

    if "cifar" in args.archs:
        print("\n## CIFAR-10 (3-conv + 2-fc)")
        arch_to_raw["cifar"] = run_training_benchmark(
            "cifar10", args.optimizers, args.steps)

    if "smallgpt" in args.archs:
        print("\n## SmallGPT/WikiText-2 (reading existing grid results)")
        arch_to_raw["smallgpt"] = read_smallgpt_results(args.optimizers)
        for variant, entry in arch_to_raw["smallgpt"].items():
            print(f"  {variant}: ppl={entry['final_metric']:.1f} "
                  f"(from {entry.get('source')})")
        missing = [v for v in args.optimizers
                   if v not in arch_to_raw["smallgpt"]]
        if missing:
            print(f"  [warn] missing SmallGPT results for: {missing}")
            print(f"         run vered_full_grid.py / classic_full_grid.py first")

    table = aggregate(arch_to_raw, args.optimizers)
    print_summary(table, args.archs, args.optimizers)

    # Save unified summary
    out_path = RESULTS_DIR / "multi_arch_probe_summary.json"
    save_obj = {
        "archs":      args.archs,
        "optimizers": args.optimizers,
        "table":      {f"{a}|{v}": e for (a, v), e in table.items()},
    }
    out_path.write_text(json.dumps(save_obj, indent=2, default=str))
    print(f"\n  Saved unified summary to: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
