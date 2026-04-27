"""
Run all benchmarks: MNIST MLP, Transformer LM, and BERT fine-tuning.

Runs in order:
  1. MNIST MLP      — training_benchmark.py --dataset mnist
  2. Transformer LM — gpu_benchmark.py --task transformer
  3. BERT SST-2     — gpu_benchmark.py --task bert

Usage:
    python -m benchmark.run_all
    python -m benchmark.run_all --optimizers OlsSMKFAC VeredKFAC Adam
    python -m benchmark.run_all --skip-mnist
    python -m benchmark.run_all --skip-bert
    python -m benchmark.run_all --steps 500
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(cmd: list, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{label}] finished in {elapsed/60:.1f} min — {status}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks: MNIST, Transformer, and BERT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--optimizers", nargs="+",
        default=["OlsSMKFAC", "VeredKFAC", "Adam"],
        metavar="NAME",
        help="Optimizers to include (default: OlsSMKFAC VeredKFAC Adam).",
    )
    parser.add_argument(
        "--skip-mnist", action="store_true",
        help="Skip the MNIST MLP benchmark.",
    )
    parser.add_argument(
        "--skip-transformer", action="store_true",
        help="Skip the transformer LM benchmark.",
    )
    parser.add_argument(
        "--skip-bert", action="store_true",
        help="Skip the BERT fine-tuning benchmark.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override max steps for MNIST (training_benchmark tasks).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = parser.parse_args()

    py = sys.executable
    opt_args = ["--optimizers"] + args.optimizers
    log_args = ["--log-level", args.log_level]
    step_args = ["--steps", str(args.steps)] if args.steps else []

    # gpu_benchmark uses --skip flags to exclude optimizers
    all_opts = {"Adam", "OlsSMKFAC", "ClassicKFAC", "VeredKFAC"}
    requested = set(args.optimizers)
    skip_opts = [o.lower() for o in all_opts if o not in requested]
    gpu_skip_args = [arg for o in skip_opts for arg in ("--skip", o)]

    failures = []
    t_total = time.perf_counter()

    if not args.skip_mnist:
        rc = run(
            [py, "-m", "benchmark.training_benchmark",
             "--dataset", "mnist", *opt_args, *log_args, *step_args],
            "MNIST MLP",
        )
        if rc != 0:
            failures.append("MNIST MLP")

    if not args.skip_transformer:
        rc = run(
            [py, "-m", "benchmark.gpu_benchmark",
             "--task", "transformer", *gpu_skip_args],
            "Transformer LM",
        )
        if rc != 0:
            failures.append("Transformer LM")

    if not args.skip_bert:
        rc = run(
            [py, "-m", "benchmark.gpu_benchmark",
             "--task", "bert", *gpu_skip_args],
            "BERT SST-2",
        )
        if rc != 0:
            failures.append("BERT SST-2")

    total_min = (time.perf_counter() - t_total) / 60
    print(f"\n{'='*60}")
    if failures:
        print(f"  DONE in {total_min:.1f} min — FAILURES: {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"  ALL DONE in {total_min:.1f} min — all benchmarks passed.")


if __name__ == "__main__":
    main()
