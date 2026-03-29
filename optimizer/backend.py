"""
Backend abstraction for Gram matrix operations.

Tries to import the compiled olsvered Rust module first.
Falls back to numpy-based implementations that are API-compatible.
This allows development and benchmarking without a Rust compiler,
while seamlessly using the Rust backend when available.
"""

import numpy as np
from scipy import linalg as sp_linalg

# ---------------------------------------------------------------------------
# Try importing the compiled Rust module
# ---------------------------------------------------------------------------
try:
    import olsvered as _rust_backend

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# ---------------------------------------------------------------------------
# Numpy fallback implementations
# ---------------------------------------------------------------------------

def _np_lu_solve_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve gram @ X = rhs via LU factorisation (scipy)."""
    lu, piv = sp_linalg.lu_factor(gram)
    if rhs.ndim == 1:
        return sp_linalg.lu_solve((lu, piv), rhs)
    # Handle matrix RHS column-by-column for stability
    result = np.empty_like(rhs)
    for j in range(rhs.shape[1]):
        result[:, j] = sp_linalg.lu_solve((lu, piv), rhs[:, j])
    return result


def _np_lu_inverse_gram(gram: np.ndarray) -> np.ndarray:
    """Compute gram⁻¹ via LU factorisation (scipy)."""
    lu, piv = sp_linalg.lu_factor(gram)
    identity = np.eye(gram.shape[0])
    return sp_linalg.lu_solve((lu, piv), identity)


# ---------------------------------------------------------------------------
# Public API — dispatches to Rust or numpy
# ---------------------------------------------------------------------------

def lu_solve_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve gram @ X = rhs via LU factorisation.

    Uses the compiled olsvered Rust backend if available,
    otherwise falls back to scipy.linalg.lu_factor/lu_solve.

    Parameters
    ----------
    gram : np.ndarray, shape (p, p)
        Symmetric positive-(semi)definite Gram matrix.
    rhs : np.ndarray, shape (p, k) or (p,)
        Right-hand side matrix or vector.

    Returns
    -------
    np.ndarray, shape (p, k) or (p,)
        Solution X such that gram @ X ≈ rhs.
    """
    if _HAS_RUST and rhs.ndim == 2:
        return _rust_backend.lu_solve_gram(
            np.ascontiguousarray(gram, dtype=np.float64),
            np.ascontiguousarray(rhs, dtype=np.float64),
        )
    if _HAS_RUST and rhs.ndim == 1:
        return _rust_backend.lu_solve_gram_vec(
            np.ascontiguousarray(gram, dtype=np.float64),
            np.ascontiguousarray(rhs, dtype=np.float64),
        )
    return _np_lu_solve_gram(gram.astype(np.float64), rhs.astype(np.float64))


def lu_inverse_gram(gram: np.ndarray) -> np.ndarray:
    """Compute gram⁻¹ via LU factorisation.

    Uses the compiled olsvered Rust backend if available,
    otherwise falls back to scipy.linalg.lu_factor/lu_solve.

    Parameters
    ----------
    gram : np.ndarray, shape (p, p)
        Square matrix.

    Returns
    -------
    np.ndarray, shape (p, p)
        Inverse matrix gram⁻¹.
    """
    if _HAS_RUST:
        return _rust_backend.lu_inverse_gram(
            np.ascontiguousarray(gram, dtype=np.float64),
        )
    return _np_lu_inverse_gram(gram.astype(np.float64))


def lu_damped_inverse_f32(gram: np.ndarray, damping: float) -> np.ndarray:
    """Compute ``(gram + damping·I)⁻¹`` on f32 data — fast path for K-FAC.

    Uses the compiled olsvered Rust backend if available (zero dtype cast,
    damping applied inside Rust, only 2 copies vs 6+).
    Falls back to scipy on float64 when Rust is unavailable.

    Parameters
    ----------
    gram : np.ndarray, shape (n, n), float32 or float64
        Symmetric positive-(semi)definite Gram matrix.
    damping : float
        Scalar λ added to the diagonal before inversion.

    Returns
    -------
    np.ndarray, shape (n, n), same dtype as input (float32 if Rust path taken)
    """
    if _HAS_RUST:
        g32 = np.ascontiguousarray(gram, dtype=np.float32)
        return _rust_backend.lu_damped_inverse_f32(g32, float(damping))
    # numpy/scipy fallback — use float64 for stability
    g64 = gram.astype(np.float64)
    n = g64.shape[0]
    g64 += damping * np.eye(n)
    return _np_lu_inverse_gram(g64).astype(gram.dtype)


def get_backend_name() -> str:
    """Return the name of the active backend."""
    return "olsvered (Rust)" if _HAS_RUST else "numpy/scipy (fallback)"
