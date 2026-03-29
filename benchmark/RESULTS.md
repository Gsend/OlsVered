# olsvered K-FAC Benchmark: Honest Results

## What We Tested

We benchmarked three approaches to Gram matrix inversion — the core bottleneck in K-FAC and Shampoo second-order optimizers:

1. **Classical inversion** (`numpy.linalg.inv`) — equivalent to `torch.linalg.inv`
2. **LU-based inversion** (`scipy.linalg.lu_factor` + `lu_solve`) — the olsvered approach
3. **LU direct solve** (no explicit inverse formed) — the ideal olsvered path

Tested on matrix dimensions matching real transformer layers (64–1024), across well-conditioned and near-singular Gram matrices, with damping values from 1e-2 to 1e-6.

---

## Key Finding: Speed Is a Wash in Python/NumPy

On CPU with numpy/scipy (which both call the same underlying LAPACK routines):

| Dim | Classical (ms) | LU Inverse (ms) | Ratio |
|-----|---------------|-----------------|-------|
| 64  | 0.08          | 0.10            | 0.8×  |
| 128 | 0.51          | 0.55            | 0.9×  |
| 256 | 3.84          | 1.95            | 2.0×  |
| 512 | 12.3          | 15.7            | 0.8×  |
| 768 | 36.7          | 37.6            | 1.0×  |
| 1024| 74.8          | 74.4            | 1.0×  |

At medium sizes (~256), LU shows a 2× advantage. At large sizes, both converge to similar performance because numpy's `inv` internally uses LU factorization anyway. **The raw speed difference is not the value proposition at the Python/LAPACK level.**

---

## Key Finding: Numerical Stability Is Equivalent With Damping

For near-singular matrices with moderate damping (λ ≥ 1e-4):

| Dim | Rank Deficiency | Classical Max Residual | LU Max Residual |
|-----|----------------|----------------------|-----------------|
| 128 | 50%            | 6.78e-12             | 5.50e-12        |
| 256 | 50%            | 6.22e-12             | 6.22e-12        |
| 512 | 50%            | 1.00e-11             | 1.00e-11        |

Both methods produce identical or nearly identical accuracy. This is expected: with sufficient damping, neither method encounters true numerical difficulty.

---

## Where olsvered's Value Actually Lies

The benchmark reveals that the advantage is NOT in raw Python/numpy speed — both call LAPACK under the hood. The real value of olsvered's Rust implementation is:

### 1. Lower-level control
nalgebra's LU with partial pivoting gives explicit access to pivot information, condition estimates, and factorization internals that numpy's `inv` hides. This enables adaptive damping strategies.

### 2. Compiled Rust vs Python overhead
The numpy↔Python conversion overhead becomes significant in K-FAC's hot path (called thousands of times per training run across all layers). A pure Rust path eliminates this entirely.

### 3. The direct-solve path
When K-FAC's update frequency equals 1 (every step), the `lu_solve_direct` path avoids forming the explicit inverse at all — computing `G⁻¹ · ∇L · A⁻¹` as two sequential LU solves. This is architecturally different from the classical approach and becomes advantageous when the inverse is not cached.

### 4. Enabling lower damping in practice
The stability advantage manifests not in residual norms but in **allowing practitioners to use lower λ values** without catastrophic failure. This is a training quality improvement, not a benchmark number — and requires end-to-end training experiments to measure.

---

## Simulated K-FAC Full Update (16-layer transformer)

| Method | Median Time | Mean Time |
|--------|------------|-----------|
| Classical | 202.6 ms | 202.6 ms |
| LU Inverse | 214.2 ms | 211.8 ms |

For a complete preconditioner update across 16 Linear layers (32 Gram matrices total), both methods are within 6% of each other. The preconditioner update represents ~200ms overhead per K-FAC update step.

---

## Honest Assessment

This benchmark shows that at the numpy/scipy level, olsvered's LU approach does not provide a dramatic speed advantage over classical inversion. Both methods use the same LAPACK primitives.

**The case for olsvered in K-FAC rests on:**

1. A compiled Rust backend that eliminates Python overhead in the hot path
2. Architectural flexibility (direct solve vs cached inverse)
3. Enabling lower damping → better curvature estimates → fewer training steps
4. A clean, well-tested foundation for building second-order optimizer tooling

**To prove the full value proposition, the next step must be an end-to-end training benchmark** on a real model, comparing Adam vs K-FAC with olsvered backend, measuring total wall-clock time to a target loss — not just inversion speed.

---

## Reproducing

```bash
python benchmark/bench_inversion.py
```

Results saved to `benchmark/results/inversion_benchmark.json`.

## Environment

- CPU-only (no GPU)
- Python 3.10, NumPy 2.2.6, SciPy 1.15.3
- All timings are median of 20 trials after 3 warmup runs
