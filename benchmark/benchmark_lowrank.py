"""
Isolated micro-benchmark: Classic K-FAC apply vs Low-rank eigen apply.

Measures pure apply cost (no backward pass, no Python overhead) across
representative layer shapes and rank values.

Run from the project root:
    python benchmark/benchmark_lowrank.py

No GPU required. Uses single-threaded numpy matmuls for reproducibility.
"""
import time
import numpy as np

try:
    import olssm
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print("[WARNING] Rust backend not installed — using numpy fallback.")

DAMPING = 1e-2
WARMUP  = 50
REPS    = 300

rng = np.random.default_rng(42)

def make_gram(n: int) -> np.ndarray:
    """Random SPD matrix of size n×n."""
    A = rng.standard_normal((n, n)).astype(np.float32)
    return A @ A.T / n + np.eye(n, dtype=np.float32) * DAMPING

def time_fn(fn, warmup: int = WARMUP, reps: int = REPS) -> float:
    """Return mean call time in milliseconds."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000

LAYERS = [
    ("fc1  784→512", 512, 784),
    ("fc2  512→256", 256, 512),
    ("fc3  256→128", 128, 256),
    ("fc4  128→64",   64, 128),
]

K_VALUES = [16, 32, 64, 128]

print()
print("K-FAC apply cost: Classic (cached dense inverse) vs Low-rank eigen")
print("=" * 80)
print(f"  Layer shape = (d_out, d_in), damping={DAMPING}, reps={REPS}")
print()

hdr = f"{'Layer':<20}  {'Classic':>9}"
for k in K_VALUES:
    hdr += f"  {'k='+str(k):>9}"
print(hdr)
print("-" * len(hdr))

for name, d_out, d_in in LAYERS:
    A_np = make_gram(d_in)
    G_np = make_gram(d_out)
    grad = rng.standard_normal((d_out, d_in)).astype(np.float32)

    # Classic: cached n×n inverses, 2 matmuls
    if _HAS_RUST:
        A_inv = olssm.lu_damped_inverse_f32(A_np, DAMPING)
        G_inv = olssm.lu_damped_inverse_f32(G_np, DAMPING)
    else:
        A_inv = np.linalg.inv(A_np + DAMPING * np.eye(d_in, dtype=np.float32))
        G_inv = np.linalg.inv(G_np + DAMPING * np.eye(d_out, dtype=np.float32))

    classic_ms = time_fn(lambda G=G_inv, A=A_inv: G @ grad @ A)
    row = f"{name:<20}  {classic_ms:>7.3f}ms"

    for k in K_VALUES:
        k_eff = min(k, min(d_out, d_in))  # clamp to matrix size
        if _HAS_RUST:
            Qg, ig = olssm.eigh_topk_f32(G_np, k_eff, DAMPING)
            Qa, ia = olssm.eigh_topk_f32(A_np, k_eff, DAMPING)
        else:
            lam_g, Qg_full = np.linalg.eigh(G_np.astype(np.float64))
            lam_a, Qa_full = np.linalg.eigh(A_np.astype(np.float64))
            Qg = Qg_full[:, -k_eff:].astype(np.float32)
            Qa = Qa_full[:, -k_eff:].astype(np.float32)
            ig = (1.0 / np.maximum(lam_g[-k_eff:] + DAMPING, 1e-8)).astype(np.float32)
            ia = (1.0 / np.maximum(lam_a[-k_eff:] + DAMPING, 1e-8)).astype(np.float32)

        def apply_lr(Qg=Qg, Qa=Qa, ig=ig, ia=ia):
            tmp = Qg.T @ grad @ Qa
            tmp *= np.outer(ig, ia)
            return Qg @ tmp @ Qa.T

        lr_ms = time_fn(apply_lr)
        speedup = classic_ms / lr_ms
        row += f"  {lr_ms:>6.3f}ms({speedup:.1f}x)"

    print(row)

print()
print("Speedup = Classic ms / Low-rank ms  (higher is better for low-rank)")
print("Note: low-rank approximation trades accuracy for speed.")
print("      k=32–64 typically retains >95% of curvature information.")
print()
