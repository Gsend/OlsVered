#!/usr/bin/env python3
"""
Core benchmark: olssm LU inversion vs classical inversion for K-FAC Gram matrices.

This benchmark tests the EXACT bottleneck that matters — Gram matrix inversion
as it occurs in K-FAC / Shampoo optimizers. No GPU or PyTorch required.

For each matrix size (simulating real transformer layer dimensions):
  1. Generate realistic Gram matrices (XᵀX from random activations)
  2. Time classical inversion (numpy.linalg.inv — equivalent to torch.linalg.inv)
  3. Time olssm-style LU inversion (scipy LU factor + solve)
  4. Compare numerical accuracy (condition number, inverse residual)
  5. Test near-singular matrices (the stability advantage scenario)

Usage:
    python benchmark/bench_inversion.py
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import linalg as sp_linalg

# ---------------------------------------------------------------------------
# Inversion methods (these mirror what happens in K-FAC)
# ---------------------------------------------------------------------------

def classical_inverse(gram: np.ndarray, damping: float) -> np.ndarray:
    """Classical explicit inverse: (A + λI)⁻¹ via numpy (= torch.linalg.inv)."""
    damped = gram + damping * np.eye(gram.shape[0])
    return np.linalg.inv(damped)

def lu_inverse(gram: np.ndarray, damping: float) -> np.ndarray:
    """olssm-style LU inverse: solve (A + λI) X = I via LU factorisation."""
    damped = gram + damping * np.eye(gram.shape[0])
    lu, piv = sp_linalg.lu_factor(damped)
    return sp_linalg.lu_solve((lu, piv), np.eye(gram.shape[0]))

def lu_solve_direct(gram: np.ndarray, rhs: np.ndarray, damping: float) -> np.ndarray:
    """olssm-style direct solve: (A + λI) X = rhs, no explicit inverse."""
    damped = gram + damping * np.eye(gram.shape[0])
    lu, piv = sp_linalg.lu_factor(damped)
    return sp_linalg.lu_solve((lu, piv), rhs)

# ---------------------------------------------------------------------------
# Matrix generators (simulate real K-FAC Gram matrices)
# ---------------------------------------------------------------------------

def make_gram_from_activations(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Build A = (1/n) XᵀX from random activations — realistic Gram matrix."""
    X = rng.standard_normal((n_samples, dim))
    return (X.T @ X) / n_samples

def make_near_singular_gram(dim: int, rank_deficiency: int, rng: np.random.Generator) -> np.ndarray:
    """Build a near-singular Gram matrix (simulates collapsed attention heads)."""
    effective_rank = dim - rank_deficiency
    # Low-rank factor
    F = rng.standard_normal((dim, effective_rank))
    gram = F @ F.T / effective_rank
    # Add tiny noise to make it technically full-rank but ill-conditioned
    gram += 1e-10 * np.eye(dim)
    return gram

# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

@dataclass
class InversionResult:
    method: str
    dim: int
    damping: float
    time_ms: float
    residual_norm: float      # ||A @ A⁻¹ - I||_F
    max_residual: float       # max |A @ A⁻¹ - I|
    condition_number: float
    is_near_singular: bool = False

def benchmark_single(gram: np.ndarray, damping: float, n_warmup: int = 3,
                     n_trials: int = 20, is_near_singular: bool = False
                     ) -> List[InversionResult]:
    """Benchmark all methods on a single Gram matrix."""
    dim = gram.shape[0]
    damped = gram + damping * np.eye(dim)
    cond = np.linalg.cond(damped)
    identity = np.eye(dim)

    # Also benchmark the direct solve path (no explicit inverse)
    rhs = np.random.default_rng(0).standard_normal((dim, dim))

    results = []

    for method_name, method_fn in [
        ("classical_inv", lambda: classical_inverse(gram, damping)),
        ("lu_inverse", lambda: lu_inverse(gram, damping)),
        ("lu_solve_direct", lambda: lu_solve_direct(gram, rhs, damping)),
    ]:
        # Warmup
        for _ in range(n_warmup):
            _ = method_fn()

        # Timed trials
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            result = method_fn()
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000

        # Accuracy check (skip for direct solve — different semantics)
        if method_name != "lu_solve_direct":
            product = damped @ result
            residual = product - identity
            res_norm = np.linalg.norm(residual, 'fro')
            max_res = np.abs(residual).max()
        else:
            check = damped @ result
            residual = check - rhs
            res_norm = np.linalg.norm(residual, 'fro')
            max_res = np.abs(residual).max()

        results.append(InversionResult(
            method=method_name,
            dim=dim,
            damping=damping,
            time_ms=round(median_ms, 4),
            residual_norm=float(res_norm),
            max_residual=float(max_res),
            condition_number=float(cond),
            is_near_singular=is_near_singular,
        ))

    return results

# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------

def run_full_benchmark():
    """Run the complete benchmark across all configurations."""
    rng = np.random.default_rng(42)

    # Dimensions matching real transformer layers:
    #   256 = small model hidden dim
    #   512 = FFN inner dim (small)
    #   768 = GPT-2 small / BERT base
    #  1024 = GPT-2 medium
    #  2048 = larger models
    #  4096 = GPT-2 large / modern LLMs
    dims = [64, 128, 256, 512, 768, 1024]

    # Damping values to test
    dampings = [1e-2, 1e-4]

    all_results = []

    # --- Well-conditioned Gram matrices ---
    print("=" * 70)
    print("BENCHMARK: Gram Matrix Inversion — Well-Conditioned")
    print("=" * 70)
    print(f"\n{'Dim':>6} {'Damping':>8} {'Method':<18} {'Time(ms)':>10} "
          f"{'Residual':>12} {'Max Res':>12} {'Cond#':>12}")
    print("-" * 82)

    for dim in dims:
        gram = make_gram_from_activations(max(dim * 4, 256), dim, rng)
        for damping in dampings:
            results = benchmark_single(gram, damping)
            for r in results:
                all_results.append(r)
                print(f"{r.dim:>6} {r.damping:>8.0e} {r.method:<18} "
                      f"{r.time_ms:>10.3f} {r.residual_norm:>12.2e} "
                      f"{r.max_residual:>12.2e} {r.condition_number:>12.1f}")
        print()

    # --- Near-singular Gram matrices (the stability test) ---
    print("\n" + "=" * 70)
    print("BENCHMARK: Near-Singular Gram Matrices (stability advantage test)")
    print("=" * 70)
    print(f"\n{'Dim':>6} {'Rank Def':>9} {'Damping':>8} {'Method':<18} "
          f"{'Time(ms)':>10} {'Residual':>12} {'Max Res':>12} {'Cond#':>14}")
    print("-" * 100)

    for dim in [128, 256, 512]:
        for rank_def in [dim // 2]:
            gram = make_near_singular_gram(dim, rank_def, rng)
            for damping in [1e-2, 1e-4, 1e-6]:
                results = benchmark_single(gram, damping, is_near_singular=True)
                for r in results:
                    all_results.append(r)
                    print(f"{r.dim:>6} {rank_def:>9} {r.damping:>8.0e} "
                          f"{r.method:<18} {r.time_ms:>10.3f} "
                          f"{r.residual_norm:>12.2e} {r.max_residual:>12.2e} "
                          f"{r.condition_number:>14.1e}")
            print()

    # --- K-FAC simulation: time a full preconditioner update ---
    print("\n" + "=" * 70)
    print("BENCHMARK: Simulated K-FAC Preconditioner Update (full layer set)")
    print("=" * 70)

    # Simulate a small transformer: 4 layers × 6 Linear each (Q,K,V,O,FFN1,FFN2)
    layer_dims = [
        (256, 256),   # Q,K,V projections: d_model → d_model
        (256, 256),   # O projection
        (256, 512),   # FFN1: d_model → d_ff
        (512, 256),   # FFN2: d_ff → d_model
    ] * 4  # 4 transformer blocks = 16 layers

    n_samples = 1024
    damping = 1e-2

    print(f"\nSimulating {len(layer_dims)} Linear layers, "
          f"{n_samples} samples, damping={damping}")
    print(f"Layer dims: {set(layer_dims)}\n")

    for method_name, inv_fn in [
        ("classical_inv", classical_inverse),
        ("lu_inverse", lu_inverse),
    ]:
        # Build all Gram matrices first
        grams_A = []
        grams_G = []
        for d_in, d_out in layer_dims:
            grams_A.append(make_gram_from_activations(n_samples, d_in, rng))
            grams_G.append(make_gram_from_activations(n_samples, d_out, rng))

        # Time the full preconditioner update
        times = []
        for trial in range(10):
            t0 = time.perf_counter()
            for A, G in zip(grams_A, grams_G):
                A_inv = inv_fn(A, damping)
                G_inv = inv_fn(G, damping)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        mean_ms = np.mean(times) * 1000
        print(f"  {method_name:<18}: median={median_ms:.1f}ms, "
              f"mean={mean_ms:.1f}ms  ({len(layer_dims)} layers × 2 matrices)")

    # --- Save results ---
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "inversion_benchmark.json"

    output = {
        "metadata": {
            "description": "Gram matrix inversion benchmark: classical vs LU-based (olssm)",
            "n_warmup": 3,
            "n_trials": 20,
            "numpy_version": np.__version__,
        },
        "results": [asdict(r) for r in all_results],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # --- Summary comparison ---
    print("\n" + "=" * 70)
    print("SUMMARY: Speed Ratio (classical / lu_inverse)")
    print("=" * 70)

    for dim in dims:
        for damping in [1e-2]:
            classical = [r for r in all_results
                         if r.dim == dim and r.damping == damping
                         and r.method == "classical_inv"
                         and not r.is_near_singular]
            lu = [r for r in all_results
                  if r.dim == dim and r.damping == damping
                  and r.method == "lu_inverse"
                  and not r.is_near_singular]
            if classical and lu:
                ratio = classical[0].time_ms / lu[0].time_ms
                faster = "LU faster" if ratio > 1 else "Classical faster"
                print(f"  dim={dim:>5}, damping={damping:.0e}: "
                      f"ratio={ratio:.2f}x  ({faster})")

    # Stability comparison for near-singular
    print("\n" + "=" * 70)
    print("SUMMARY: Numerical Stability (near-singular, damping=1e-4)")
    print("=" * 70)

    for dim in [128, 256, 512, 1024]:
        classical = [r for r in all_results
                     if r.dim == dim and r.damping == 1e-4
                     and r.method == "classical_inv"
                     and r.is_near_singular]
        lu = [r for r in all_results
              if r.dim == dim and r.damping == 1e-4
              and r.method == "lu_inverse"
              and r.is_near_singular]
        if classical and lu:
            print(f"  dim={dim:>5}: classical residual={classical[0].max_residual:.2e}, "
                  f"LU residual={lu[0].max_residual:.2e}")

if __name__ == "__main__":
    run_full_benchmark()
