"""
Run all benchmarks except BERT.

Runs in order:
  1. MNIST MLP      — training_benchmark.py --dataset mnist
  2. CIFAR-10       — training_benchmark.py --dataset cifar10
  3. Transformer LM — gpu_benchmark.py --task transformer

Usage:
    python -m benchmark.run_all_except_bert
    python -m benchmark.run_all_except_bert --optimizers OlsSMKFAC VeredKFAC Adam
    python -m benchmark.run_all_except_bert --skip-mnist
    python -m benchmark.run_all_except_bert --steps 500
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
        description="Run all benchmarks except BERT.",
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
        "--skip-cifar", action="store_true",
        help="Skip the CIFAR-10 benchmark.",
    )
    parser.add_argument(
        "--skip-transformer", action="store_true",
        help="Skip the transformer LM benchmark.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override max steps for training_benchmark tasks.",
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

    if not args.skip_cifar:
        rc = run(
            [py, "-m", "benchmark.training_benchmark",
             "--dataset", "cifar10", *opt_args, *log_args, *step_args],
            "CIFAR-10 ConvNet",
        )
        if rc != 0:
            failures.append("CIFAR-10")

    if not args.skip_transformer:
        # gpu_benchmark uses its own optimizer selection; pass skip flags for
        # any optimizers not in the requested set.
        all_opts = {"Adam", "OlsSMKFAC", "ClassicKFAC", "VeredKFAC"}
        requested = set(args.optimizers)
        skip_opts = [o.lower() for o in all_opts if o not in requested]
        skip_args = []
        for o in skip_opts:
            skip_args += ["--skip", o]
        rc = run(
            [py, "-m", "benchmark.gpu_benchmark",
             "--task", "transformer", *skip_args],
            "Transformer LM",
        )
        if rc != 0:
            failures.append("Transformer LM")

    total_min = (time.perf_counter() - t_total) / 60
    print(f"\n{'='*60}")
    if failures:
        print(f"  DONE in {total_min:.1f} min — FAILURES: {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"  ALL DONE in {total_min:.1f} min — all benchmarks passed.")
