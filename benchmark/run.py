#!/usr/bin/env python3
"""
Main benchmark script — compares Adam, ClassicKFAC, and OlsSMKFAC.

Usage:
    python benchmark/run.py --optimizer adam --steps 200 --model transformer
    python benchmark/run.py --optimizer olssm_kfac --steps 200 --model transformer
    python benchmark/run.py --optimizer classic_kfac --steps 200 --model transformer
    python benchmark/run.py --all --steps 200   # run all three and compare

Results are saved to benchmark/results/<optimizer>_<model>_<seed>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.models import SmallTransformer, SimpleMLP
from benchmark.data import get_synthetic_lm_loader, SyntheticClassificationData
from benchmark.metrics import BenchmarkResult, Timer
from optimizer.backend import get_backend_name

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model(model_name: str, device: torch.device) -> nn.Module:
    """Create model by name."""
    if model_name == "transformer":
        model = SmallTransformer(
            vocab_size=1000, d_model=256, n_heads=4,
            n_layers=4, d_ff=512, max_seq_len=128,
        )
    elif model_name == "mlp":
        model = SimpleMLP(input_dim=784, hidden_dim=256, output_dim=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

def create_optimizer(opt_name: str, model: nn.Module, lr: float,
                     damping: float, factor_freq: int, momentum: float):
    """Create optimizer by name."""
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "olssm_kfac":
        from optimizer.olssm_kfac import OlsSMKFAC
        return OlsSMKFAC(
            model, lr=lr, damping=damping,
            factor_update_freq=factor_freq,
            inv_update_freq=factor_freq,
            momentum=momentum,
        )
    elif opt_name == "classic_kfac":
        from optimizer.classic_kfac import ClassicKFAC
        return ClassicKFAC(
            model, lr=lr, damping=damping,
            factor_update_freq=factor_freq,
            inv_update_freq=factor_freq,
            momentum=momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def train_transformer(
    model: nn.Module,
    optimizer,
    opt_name: str,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    log_interval: int = 10,
) -> BenchmarkResult:
    """Train a transformer LM and collect benchmark metrics."""
    loader = get_synthetic_lm_loader(
        num_samples=max(steps * batch_size, 5000),
        seq_len=128, vocab_size=1000,
        batch_size=batch_size, seed=seed,
    )
    data_iter = iter(loader)

    backend = get_backend_name() if "olssm" in opt_name else (
        "torch.linalg.inv" if "classic" in opt_name else "adam (first-order)"
    )

    result = BenchmarkResult(
        model_name="SmallTransformer",
        optimizer_name=opt_name,
        backend=backend,
        total_steps=steps,
        total_time_s=0,
        hyperparams={
            "lr": optimizer.defaults.get("lr", optimizer.param_groups[0]["lr"]),
            "damping": getattr(optimizer, "damping", None),
            "factor_update_freq": getattr(optimizer, "factor_update_freq", None),
            "batch_size": batch_size,
        },
        model_params=model.count_parameters(),
        model_linear_layers=model.count_linear_layers(),
        device=str(device),
        seed=seed,
    )

    model.train()
    t_start = time.perf_counter()

    for step in range(1, steps + 1):
        # Get next batch (cycle if exhausted)
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)

        with Timer() as step_timer:
            optimizer.zero_grad()
            logits = model(inputs)  # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        wall_time = time.perf_counter() - t_start
        result.add_step(
            step=step,
            loss=loss.item(),
            wall_time=wall_time,
            step_time_ms=step_timer.elapsed_ms,
            lr=optimizer.param_groups[0]["lr"],
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        )

        if step % log_interval == 0 or step == 1:
            print(f"  [{opt_name}] step {step:4d}/{steps}  "
                  f"loss={loss.item():.4f}  "
                  f"step_ms={step_timer.elapsed_ms:.1f}  "
                  f"wall={wall_time:.1f}s")

    result.total_time_s = time.perf_counter() - t_start

    # Collect optimizer-specific timing
    if hasattr(optimizer, "get_timing_stats"):
        result.optimizer_timing = optimizer.get_timing_stats()
    if hasattr(optimizer, "cleanup"):
        optimizer.cleanup()

    return result

def run_benchmark(args):
    """Run a single benchmark configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n{'='*60}")
    print(f"Benchmark: {args.optimizer} on {args.model}")
    print(f"Device: {device}  |  Backend: {get_backend_name()}")
    print(f"Steps: {args.steps}  |  Seed: {args.seed}")
    print(f"{'='*60}\n")

    set_seed(args.seed)
    model = create_model(args.model, device)
    print(f"Model: {model.count_parameters():,} params, "
          f"{model.count_linear_layers()} Linear layers\n")

    optimizer = create_optimizer(
        args.optimizer, model,
        lr=args.lr, damping=args.damping,
        factor_freq=args.factor_freq,
        momentum=args.momentum,
    )

    if args.model == "transformer":
        result = train_transformer(
            model, optimizer, args.optimizer,
            steps=args.steps, batch_size=args.batch_size,
            device=device, seed=args.seed,
            log_interval=args.log_interval,
        )
    else:
        raise ValueError(f"Training loop for {args.model} not implemented")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"{args.optimizer}_{args.model}_seed{args.seed}.json"
    result.save(str(result_path))

    # Print summary
    summary = result.summary()
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.optimizer}")
    print(f"{'='*60}")
    print(f"  Total time:    {summary['total_time_s']:.2f}s")
    print(f"  Final loss:    {summary['final_loss']}")
    print(f"  Min loss:      {summary['min_loss']}")
    print(f"  Mean step:     {summary['mean_step_ms']:.1f}ms")
    print(f"  P50 step:      {summary['p50_step_ms']:.1f}ms")
    print(f"  P99 step:      {summary['p99_step_ms']:.1f}ms")
    if summary["optimizer_timing"]:
        print(f"  Optimizer breakdown:")
        for key, stats in summary["optimizer_timing"].items():
            print(f"    {key}: mean={stats['mean_ms']:.2f}ms, "
                  f"total={stats['total_s']:.2f}s")
    print(f"  Saved: {result_path}\n")

    return result

def run_all(args):
    """Run all three optimizers and print comparison."""
    results = {}
    for opt in ["adam", "classic_kfac", "olssm_kfac"]:
        args.optimizer = opt
        # Use appropriate LR defaults
        if opt == "adam":
            args.lr = args.adam_lr
        else:
            args.lr = args.kfac_lr
        set_seed(args.seed)
        results[opt] = run_benchmark(args)

    # Comparison table
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Optimizer':<20} {'Time(s)':>8} {'Final Loss':>11} {'Min Loss':>9} "
          f"{'Mean Step':>10} {'P99 Step':>9}")
    print(f"{'-'*70}")
    for opt, res in results.items():
        s = res.summary()
        print(f"{opt:<20} {s['total_time_s']:>8.1f} {s['final_loss']:>11.4f} "
              f"{s['min_loss']:>9.4f} {s['mean_step_ms']:>8.1f}ms "
              f"{s['p99_step_ms']:>7.1f}ms")
    print()

def main():
    parser = argparse.ArgumentParser(description="olssm K-FAC Benchmark")

    parser.add_argument("--optimizer", type=str, default="olssm_kfac",
                        choices=["adam", "classic_kfac", "olssm_kfac"],
                        help="Optimizer to benchmark")
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "mlp"],
                        help="Model architecture")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if GPU available")

    # Optimizer hyperparams
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam-lr", type=float, default=3e-4,
                        help="LR for Adam (used in --all mode)")
    parser.add_argument("--kfac-lr", type=float, default=1e-3,
                        help="LR for K-FAC variants (used in --all mode)")
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--factor-freq", type=int, default=10,
                        help="Factor/inverse update frequency (K-FAC only)")
    parser.add_argument("--momentum", type=float, default=0.9)

    # Output
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--all", action="store_true",
                        help="Run all three optimizers and compare")

    args = parser.parse_args()

    if args.all:
        run_all(args)
    else:
        run_benchmark(args)

if __name__ == "__main__":
    main()
