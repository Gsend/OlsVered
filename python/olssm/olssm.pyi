"""
olssm — Closed-form OLS and K-FAC Gram matrix operations (Rust backend).

All arrays must be ``numpy.float64`` (dtype=np.float64 / np.double) unless
a function name ends in ``_f32``, in which case ``numpy.float32`` is expected.
Row-major (C-contiguous) layout is assumed; call ``.ascontiguousarray()``
on Fortran-order inputs before passing them to any function.
"""

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Float64Array1D = NDArray[np.float64]
Float64Array2D = NDArray[np.float64]


def modified_cholesky(
    x: Float64Array2D,  # shape (n, p)
    y: Float64Array1D,  # shape (n,)
) -> Float64Array2D:
    """
    LU-based Gram matrix decomposition with row normalisation.

    Augments [X | y], computes the Gram matrix G = [X|y]ᵀ[X|y],
    LU-decomposes G, and returns the row-normalised upper triangular
    factor C (diagonal entries = 1).

    Args:
        x: Design matrix of shape ``(n, p)``, dtype float64.
        y: Response vector of shape ``(n,)``, dtype float64.

    Returns:
        C matrix of shape ``(p+1, p+1)``, upper triangular, diagonal = 1.
        Pass to :func:`back_substitute` to recover OLS coefficients.

    Raises:
        ValueError: on dimension mismatch or (near-)singular Gram matrix.
    """
    ...


def back_substitute(
    c: Float64Array2D,  # shape (p+1, p+1)
) -> Float64Array1D:
    """
    Back-substitute C matrix → OLS beta coefficients.

    Args:
        c: ``(p+1, p+1)`` upper triangular matrix from :func:`modified_cholesky`.

    Returns:
        beta: OLS coefficient vector of shape ``(p,)``.

    Raises:
        ValueError: on invalid input.
    """
    ...


def solve_ols(
    x: Float64Array2D,  # shape (n, p)
    y: Float64Array1D,  # shape (n,)
) -> Float64Array1D:
    """
    Full OLS solver — equivalent to :func:`modified_cholesky` + :func:`back_substitute`.

    Args:
        x: Design matrix of shape ``(n, p)``, dtype float64.
        y: Response vector of shape ``(n,)``, dtype float64.

    Returns:
        beta: OLS coefficient vector of shape ``(p,)``.

    Raises:
        ValueError: on dimension mismatch or singular matrix.
    """
    ...


def simplified_gram_schmidt(
    x: Float64Array2D,  # shape (n, p)
) -> Float64Array2D:
    """
    Non-normalised Gram-Schmidt orthogonalisation (SGSO).

    For each column j:
        q_j = x_j  -  Σ_{i<j} (x_j · q_i) / (q_i · q_i)  *  q_i

    No square-root operations are performed.

    Args:
        x: Input matrix of shape ``(n, p)``, dtype float64.

    Returns:
        Q: Matrix of shape ``(n, p)`` with mutually orthogonal (un-normalised) columns.

    Raises:
        ValueError: on invalid input.
    """
    ...


def weighted_generalized_inverse(
    x: Float64Array2D,  # shape (n, p)
    w: Float64Array2D,  # shape (n, n)
) -> Float64Array2D:
    """
    Weighted generalised inverse ``(XᵀWX)⁻¹ Xᵀ W``.

    Computes the weighted least-squares coefficient matrix without explicitly
    inverting XᵀWX (uses LU solve internally).

    Args:
        x: Design matrix of shape ``(n, p)``, dtype float64.
        w: Positive-definite weight matrix of shape ``(n, n)``, dtype float64.

    Returns:
        G: Weighted generalised inverse of shape ``(p, n)``.
           Weighted OLS coefficients: ``beta = G @ y``.

    Raises:
        ValueError: on dimension mismatch or singular matrix.
    """
    ...


# ---------------------------------------------------------------------------
# K-FAC Gram matrix operations (float32, for optimizer use)
# ---------------------------------------------------------------------------

def lu_solve_gram(
    gram: Float64Array2D,   # shape (p, p)
    rhs: Float64Array2D,    # shape (p, k)
) -> Float64Array2D:
    """Solve ``gram @ X = rhs`` via LU factorisation. Returns X of shape (p, k)."""
    ...


def lu_solve_gram_vec(
    gram: Float64Array2D,   # shape (p, p)
    rhs: Float64Array1D,    # shape (p,)
) -> Float64Array1D:
    """Solve ``gram @ x = rhs`` via LU factorisation (vector RHS). Returns x of shape (p,)."""
    ...


def lu_inverse_gram(
    gram: Float64Array2D,   # shape (p, p)
) -> Float64Array2D:
    """Compute ``gram⁻¹`` via LU factorisation. Returns (p, p) inverse."""
    ...


def lu_damped_inverse_f32(
    gram: Float32Array,     # shape (n, n)
    damping: float,
) -> Float32Array:
    """Compute ``(gram + damping·I)⁻¹`` on float32 data. Returns (n, n) float32."""
    ...


def eigh_f32(
    gram: Float32Array,     # shape (n, n)
    damping: float,
) -> tuple[Float32Array, Float32Array]:
    """
    Symmetric eigendecomposition of a Gram matrix.

    Returns:
        q:          eigenvector matrix of shape (n, n), float32
        inv_lambda: damped inverse eigenvalues ``1/(λᵢ + δ)``, shape (n,), float32
    """
    ...


def eigh_topk_f32(
    gram: Float32Array,     # shape (n, n)
    k: int,
    damping: float,
) -> tuple[Float32Array, Float32Array]:
    """
    Top-k symmetric eigendecomposition.

    Returns:
        q_k:          top-k eigenvectors of shape (n, k), float32
        inv_lambda_k: damped inverse eigenvalues of shape (k,), float32
    """
    ...


def apply_kfac_eigen_f32(
    q_g: Float32Array,        # (d_out, d_out)
    inv_lam_g: Float32Array,  # (d_out,)
    grad: Float32Array,       # (d_out, d_in)
    q_a: Float32Array,        # (d_in, d_in)
    inv_lam_a: Float32Array,  # (d_in,)
) -> Float32Array:
    """
    Apply K-FAC eigen-basis preconditioner in a single fused call.

    Computes: ΔW = Q_G · diag(d_G) · Q_Gᵀ · grad · Q_A · diag(d_A) · Q_Aᵀ

    Returns:
        Preconditioned gradient of shape (d_out, d_in), float32.
    """
    ...


def apply_kfac_lowrank_f32(
    q_g_k: Float32Array,        # (d_out, k_g)
    inv_lam_g_k: Float32Array,  # (k_g,)
    grad: Float32Array,         # (d_out, d_in)
    q_a_k: Float32Array,        # (d_in, k_a)
    inv_lam_a_k: Float32Array,  # (k_a,)
) -> Float32Array:
    """
    Apply low-rank K-FAC preconditioner.

    Uses rank-k approximations of A and G.

    Returns:
        Approximate preconditioned gradient of shape (d_out, d_in), float32.
    """
    ...


def randomized_eigh_f32(
    gram: Float32Array,     # shape (n, n)
    k: int,
    n_iter: int,
    damping: float,
) -> tuple[Float32Array, Float32Array]:
    """
    Randomized symmetric EVD — approximate top-k eigenvectors in O(k·n²).

    Args:
        gram:   (n, n) symmetric positive-semidefinite matrix, float32.
        k:      Number of top eigenvectors to approximate.
        n_iter: Power-iteration passes (0 = pure random projection; 1–2 recommended).
        damping: Scalar δ — returns ``1 / max(λᵢ + δ, 1e-8)``.

    Returns:
        q_k:          approximate top-k eigenvectors, shape (n, k), float32
        inv_lambda_k: damped inverse eigenvalues, shape (k,), float32
    """
    ...
