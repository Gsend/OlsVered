"""
olsvered — Closed-form OLS without inversion or normalisation.

Reference: "Solving The Ordinary Least Squares in Closed Form, Without
Inversion or Normalization" — Vered Senderovich Madar & Sandra Batista.
https://arxiv.org/abs/2301.01854

All arrays must be ``numpy.float64`` (dtype=np.float64 / np.double).
Row-major (C-contiguous) layout is assumed; call ``.ascontiguousarray()``
on Fortran-order inputs before passing them to any function.
"""

import numpy as np
from numpy.typing import NDArray

Float64Array1D = NDArray[np.float64]
Float64Array2D = NDArray[np.float64]


def modified_cholesky(
    x: Float64Array2D,  # shape (n, p)
    y: Float64Array1D,  # shape (n,)
) -> Float64Array2D:
    """
    Algorithm 1: LU-based Gram matrix decomposition with row normalisation.

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
    Algorithm 2: Non-normalised Gram-Schmidt orthogonalisation (SGSO).

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
    Algorithm 3: Weighted generalised inverse ``(XᵀWX)⁻¹ Xᵀ W``.

    Computes the weighted least-squares coefficient matrix without explicitly
    inverting XᵀWX (uses LU solve internally).

    Args:
        x: Design matrix of shape ``(n, p)``, dtype float64.
        w: Positive-definite weight matrix of shape ``(n, n)``, dtype float64.
           For GWAS/genomics use cases this is typically a kinship matrix.

    Returns:
        G: Weighted generalised inverse of shape ``(p, n)``.
           Weighted OLS coefficients: ``beta = G @ y``.

    Raises:
        ValueError: on dimension mismatch or singular matrix.
    """
    ...
