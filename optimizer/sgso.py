"""
QR decomposition utilities for Vered K-FAC.

Three levels of implementation:

1. sgso() — Sequential Gram-Schmidt Orthogonalization (pure Python/NumPy).
   Reference implementation for debugging.  O(p · n²) FLOPs.  Never use
   in production — its sequential column loop kills GPU utilisation.

2. tsqr() — Tile-based TSQR (PyTorch).
   Tall-Skinny QR via a tile + tree-reduction using torch.linalg.qr.
   GPU-friendly: leaf QRs are batched across tiles; merge rounds are
   O(log(p/tile)) sequential steps of small (2n × n) QRs.
   Preferred single-GPU path when M does not fit in one qr call.

3. streaming_tsqr_update() — Online R-accumulator.
   Folds one new chunk into an existing partial R.  Used inside hooks
   so raw activations are discarded immediately — memory stays O(n²).

apply_vered() — Back-substitution preconditioner apply.
   Given R_X (n_in × n_in) and R_G (n_out × n_out) from QR of X and δ,
   computes the natural gradient without ever forming A⁻¹ or G⁻¹:

       nat_grad = (R_Gᵀ R_G)⁻¹ · grad_W · (R_Xᵀ R_X)⁻¹

   via four triangular solves (two per Kronecker factor).

Numerical note
--------------
Storing R from QR(X) rather than forming A = XᵀX avoids squaring the
condition number:  cond(R) = cond(X)  vs  cond(A) = cond(X)².
Classic K-FAC error scales as κ(X)⁴ ε; Vered scales as κ(X)¹ ε.

Damping (ridge augmentation)
-----------------------------
Classic K-FAC adds λI to the Gram matrix before inversion: (XᵀX + λI)⁻¹.
Vered achieves the same effect by augmenting X with √λ · I_n rows before QR:
    X_aug = [X ; √λ · I_n]   →   R_aug such that R_augᵀ R_aug = XᵀX + λI
This keeps the R factor well-conditioned without breaking the triangular solve.
Augmentation is applied in get_factors() / finalize_R(), not inside the hooks.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _TS:
    """Lazy tensor summary — zero cost unless DEBUG logging is active.

    Python evaluates all arguments to logger.debug() before the logging level
    is checked, so passing _tensor_summary(t) directly would compute norm/min/max
    even at INFO level.  This wrapper stores only a reference; __str__ (which
    does the actual tensor ops) is only called when the LogRecord is formatted,
    i.e. only when the logger level is DEBUG.
    """
    __slots__ = ("_t", "_name")

    def __init__(self, t: torch.Tensor, name: str = ""):
        self._t = t
        self._name = name

    def __str__(self) -> str:
        t, name = self._t, self._name
        prefix = f"{name}=" if name else ""
        if t.numel() == 0:
            return f"{prefix}empty"
        return (
            f"{prefix}shape={tuple(t.shape)} "
            f"dtype={t.dtype} "
            f"device={t.device} "
            f"norm={t.norm().item():.4g} "
            f"min={t.min().item():.4g} "
            f"max={t.max().item():.4g}"
        )


# ---------------------------------------------------------------------------
# 1. SGSO — CPU reference only
# ---------------------------------------------------------------------------

def sgso(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential Gram-Schmidt Orthogonalization on the columns of M.

    Reference implementation — used in tests to cross-check TSQR.
    Do NOT call on GPU or large matrices; it is inherently sequential.

    Parameters
    ----------
    M : (p, n) float tensor, p >= n

    Returns
    -------
    Q : (p, n)  orthonormal columns
    R : (n, n)  upper-triangular factor  (diagonal guaranteed positive)

    Raises
    ------
    ValueError if p < n (rank-deficient case — Vered requirement violated).
    AssertionError if any column of M is numerically zero (degenerate input).
    """
    p, n = M.shape
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("sgso() called: %s", _TS(M, "M"))
    if p < n:
        raise ValueError(
            f"sgso requires p >= n (got p={p}, n={n}).  "
            "Vered K-FAC needs batch_size × seq_len >= layer_input_dim."
        )
    device, dtype = M.device, M.dtype
    Q = torch.zeros(p, n, device=device, dtype=dtype)
    R = torch.zeros(n, n, device=device, dtype=dtype)

    for j in range(n):
        v = M[:, j].clone()                    # column j, shape (p,)
        for k in range(j):                     # deflate against previous basis
            R[k, j] = torch.dot(Q[:, k], v)
            v = v - R[k, j] * Q[:, k]
        norm_v = v.norm()
        if norm_v < 1e-12:
            warnings.warn(
                f"sgso: column {j} is near-zero after deflation "
                f"(norm={norm_v:.2e}).  Matrix may be rank-deficient.",
                RuntimeWarning,
                stacklevel=2,
            )
            norm_v = torch.tensor(1.0, device=device, dtype=dtype)
        R[j, j] = norm_v                       # positive diagonal by construction
        Q[:, j] = v / norm_v
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("sgso() col %d/%d: R[%d,%d]=%.4g", j, n - 1, j, j, norm_v.item())

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("sgso() done: %s  diag_min=%.4g diag_max=%.4g",
                     _TS(R, "R"), R.diag().min().item(), R.diag().max().item())
    return Q, R


# ---------------------------------------------------------------------------
# 2. TSQR — GPU-friendly tile + tree reduction
# ---------------------------------------------------------------------------

def tsqr(M: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
    """Tall-Skinny QR via tile batching + tree reduction.

    Returns only the R factor — Q is never materialised.

    Algorithm
    ---------
    1. Split M into tiles of shape (tile_size, n).
    2. Compute all leaf QRs in a single batched call (parallel on GPU).
    3. Merge pairs of R matrices (each 2n × n) up the tree until one R remains.

    Cost: O(tile_size · n² · num_tiles)  for leaves  (parallel)
          + O(n³ · log(num_tiles))        for merges   (sequential rounds)

    For typical K-FAC dimensions (n ≤ 5120, p ≤ 16384, tile_size=512):
    - num_tiles ≤ 32, log rounds ≤ 5 — merge overhead is negligible.

    Parameters
    ----------
    M         : (p, n) float tensor, p >= n
    tile_size : rows per leaf tile (default 512)

    Returns
    -------
    R : (n, n) upper-triangular, diagonal positive.
    """
    p, n = M.shape
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("tsqr() called: %s  tile_size=%d  n_tiles=%d",
                     _TS(M, "M"), tile_size, -(-p // tile_size))
    if p < n:
        raise ValueError(
            f"tsqr requires p >= n (got p={p}, n={n}).  "
            "Vered K-FAC needs batch_size × seq_len >= layer_input_dim."
        )

    # ---- Leaf QRs (batched, parallel) ----
    tiles = M.split(tile_size, dim=0)           # list of (tile_size, n) tensors

    if len(tiles) == 1:
        # Single tile — direct QR, no tree needed
        _, R = torch.linalg.qr(tiles[0], mode="reduced")
        return _positive_diagonal_R(R)

    # Pad last tile if shorter than tile_size so we can stack uniformly
    last = tiles[-1]
    if last.shape[0] < tile_size:
        pad = torch.zeros(tile_size - last.shape[0], n, device=M.device, dtype=M.dtype)
        last_padded = torch.cat([last, pad], dim=0)
        tiles = tiles[:-1] + (last_padded,)

    tile_stack = torch.stack(list(tiles), dim=0)      # (num_tiles, tile_size, n)
    _, R_batch = torch.linalg.qr(tile_stack, mode="reduced")  # (num_tiles, n, n)
    Rs: list[torch.Tensor] = list(R_batch.unbind(0))  # num_tiles × (n, n)

    # ---- Tree merge ----
    merge_round = 0
    while len(Rs) > 1:
        merged: list[torch.Tensor] = []
        for i in range(0, len(Rs), 2):
            if i + 1 < len(Rs):
                pair = torch.cat([Rs[i], Rs[i + 1]], dim=0)   # (2n, n)
                _, R_merged = torch.linalg.qr(pair, mode="reduced")
                merged.append(R_merged)
            else:
                merged.append(Rs[i])   # odd one out — carry forward
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("tsqr() merge round %d: %d → %d R-factors",
                         merge_round, len(Rs), len(merged))
        Rs = merged
        merge_round += 1

    R_final = _positive_diagonal_R(Rs[0])
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("tsqr() done: %s  diag_min=%.4g diag_max=%.4g",
                     _TS(R_final, "R"),
                     R_final.diag().min().item(), R_final.diag().max().item())
    return R_final


def _positive_diagonal_R(R: torch.Tensor) -> torch.Tensor:
    """Flip columns/rows so that R has a positive diagonal.

    QR is not unique — any sign flip on a column of Q (and corresponding
    row of R) gives an equally valid factorisation.  Normalising to positive
    diagonal makes TSQR and SGSO directly comparable in tests.
    """
    signs = R.diag().sign()
    signs[signs == 0] = 1          # avoid multiplying by 0
    return R * signs.unsqueeze(1)  # broadcast: each row of R scaled by sign


# ---------------------------------------------------------------------------
# 3. Streaming TSQR — for use inside hooks
# ---------------------------------------------------------------------------

def streaming_tsqr_update(
    running_R: Optional[torch.Tensor],
    new_chunk: torch.Tensor,
) -> torch.Tensor:
    """Fold one new chunk of raw activations into the running R factor.

    This is the incremental building block used in hooks.  Each call:
      1. Computes the leaf QR of new_chunk → R_new  (n × n).
      2. Stacks [running_R ; R_new] → (2n × n) and computes a merge QR.
      3. Returns the merged R_merged  (n × n).

    Memory: keeps only two (n × n) matrices — O(n²) constant regardless
    of how many chunks are folded in.

    Parameters
    ----------
    running_R : (n, n) upper-triangular from previous calls, or None on first call.
    new_chunk : (p_i, n) raw activation / gradient rows for this mini-batch.

    Returns
    -------
    R_updated : (n, n) upper-triangular, diagonal positive.

    Notes
    -----
    If new_chunk has fewer rows than n (possible for very small batches), the
    QR will still run but R will have zeros on some diagonal entries — the
    caller should check this and fall back to Classic K-FAC for that layer.
    """
    p_i, n = new_chunk.shape
    is_first = running_R is None
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("streaming_tsqr_update() called: %s  running_R=%s",
                     _TS(new_chunk, "chunk"),
                     "None (first chunk)" if is_first
                     else f"shape={tuple(running_R.shape)}")

    # Leaf QR of the new chunk
    _, R_new = torch.linalg.qr(new_chunk, mode="reduced")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("streaming_tsqr_update() leaf QR: R_new.shape=%s  "
                     "rank=%d/%d (p_i=%d)",
                     tuple(R_new.shape), R_new.shape[0], n, p_i)

    if is_first:
        result = _positive_diagonal_R(R_new)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("streaming_tsqr_update() done (first chunk): %s",
                         _TS(result, "R"))
        return result

    # Merge step
    pair = torch.cat([running_R, R_new], dim=0)               # (2n, n)
    _, R_merged = torch.linalg.qr(pair, mode="reduced")       # (n, n)
    result = _positive_diagonal_R(R_merged)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("streaming_tsqr_update() done (merged): %s  "
                     "diag_min=%.4g diag_max=%.4g",
                     _TS(result, "R"),
                     result.diag().min().item(), result.diag().max().item())
    return result


def finalize_R(
    running_R: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Apply Tikhonov damping via ridge augmentation and return the final R.

    Appends √λ · I_n to the running R and runs one more QR merge:
        [running_R ; √λ · I_n]  →  R_damped

    The resulting R satisfies R_dampedᵀ R_damped = XᵀX + λI (approximately,
    up to the approximation quality of the streaming TSQR).

    Parameters
    ----------
    running_R : (n, n) upper-triangular accumulated R so far.
    damping   : λ — Tikhonov damping scalar (must be > 0).

    Returns
    -------
    R_damped : (n, n) upper-triangular, diagonal positive.
    """
    n = running_R.shape[0]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("finalize_R() called: %s  damping=%.4g",
                     _TS(running_R, "running_R"), damping)
    damp_rows = (damping ** 0.5) * torch.eye(
        n, device=running_R.device, dtype=running_R.dtype
    )
    pair = torch.cat([running_R, damp_rows], dim=0)           # (2n, n)
    _, R_damped = torch.linalg.qr(pair, mode="reduced")       # (n, n)
    result = _positive_diagonal_R(R_damped)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("finalize_R() done: %s  diag_min=%.4g diag_max=%.4g",
                     _TS(result, "R_damped"),
                     result.diag().min().item(), result.diag().max().item())
    return result


# ---------------------------------------------------------------------------
# 4. apply_vered — back-substitution preconditioner
# ---------------------------------------------------------------------------

def apply_vered(
    grad_W: torch.Tensor,
    R_X: torch.Tensor,
    R_G: torch.Tensor,
) -> torch.Tensor:
    """Apply the Vered K-FAC preconditioner to grad_W via triangular solves.

    Computes the natural gradient without forming explicit inverses:

        nat_grad = (R_Gᵀ R_G)⁻¹ · grad_W · (R_Xᵀ R_X)⁻¹
                 = G⁻¹ · grad_W · A⁻¹

    where G = R_Gᵀ R_G ≈ δᵀδ and A = R_Xᵀ R_X ≈ XᵀX (with damping folded in).

    Implementation uses four triangular solves — two for the left factor (G)
    and two for the right factor (A):

        Left (G⁻¹ · grad_W):
            T1 = solve(R_Gᵀ,  grad_W, lower=True)    # R_Gᵀ T1 = grad_W
            T2 = solve(R_G,   T1,     upper=True)     # R_G T2 = T1

        Right (T2 · A⁻¹):
            T3ᵀ = solve(R_Xᵀ, T2ᵀ,   lower=True)    # R_Xᵀ T3ᵀ = T2ᵀ
            T4ᵀ = solve(R_X,  T3ᵀ,   upper=True)     # R_X T4ᵀ = T3ᵀ
            nat_grad = T4ᵀᵀ

    Parameters
    ----------
    grad_W : (n_out, n_in) — weight gradient from autograd.
    R_X    : (n_in,  n_in) upper-triangular — from QR of input activations X.
    R_G    : (n_out, n_out) upper-triangular — from QR of gradient signals δ.

    Returns
    -------
    nat_grad : (n_out, n_in)

    Notes
    -----
    Numerical cost: O(n_out² · n_in  +  n_in² · n_out)  ≈ O(n²d).
    This matches Classic K-FAC's apply cost, but avoids the O(n³) explicit
    inversion and the κ(X)⁴ condition number amplification.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("apply_vered() called: %s  %s  %s",
                     _TS(grad_W, "grad_W"), _TS(R_X, "R_X"), _TS(R_G, "R_G"))

    # ---- Left: multiply by G⁻¹ = (R_Gᵀ R_G)⁻¹ ----
    # Step 1: solve R_Gᵀ T1 = grad_W   (lower triangular)
    T1 = torch.linalg.solve_triangular(R_G.T, grad_W, upper=False)   # (n_out, n_in)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("apply_vered() T1 = R_G⁻ᵀ · grad_W: %s", _TS(T1, "T1"))
    # Step 2: solve R_G T2 = T1         (upper triangular)
    T2 = torch.linalg.solve_triangular(R_G, T1, upper=True)          # (n_out, n_in)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("apply_vered() T2 = G⁻¹ · grad_W: %s", _TS(T2, "T2"))

    # ---- Right: multiply by A⁻¹ = (R_Xᵀ R_X)⁻¹ ----
    # T2 · A⁻¹ = T2 · R_X⁻¹ · R_Xᵀ⁻¹
    # Transposing: (A⁻¹ᵀ · T2ᵀ)ᵀ  — work column-wise via transposed problems
    #
    # Step 3: solve R_Xᵀ T3ᵀ = T2ᵀ   (lower triangular)
    T3_T = torch.linalg.solve_triangular(R_X.T, T2.T, upper=False)   # (n_in, n_out)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("apply_vered() T3ᵀ = R_X⁻ᵀ · T2ᵀ: %s", _TS(T3_T, "T3_T"))
    # Step 4: solve R_X T4ᵀ = T3ᵀ     (upper triangular)
    T4_T = torch.linalg.solve_triangular(R_X, T3_T, upper=True)      # (n_in, n_out)

    nat_grad = T4_T.T                                                  # (n_out, n_in)
    if logger.isEnabledFor(logging.DEBUG):
        scale = (nat_grad.norm() / (grad_W.norm() + 1e-12)).item()
        logger.debug("apply_vered() done: %s  scale_vs_grad=%.4g×",
                     _TS(nat_grad, "nat_grad"), scale)
    return nat_grad


def apply_vered_bias(
    grad_b: torch.Tensor,
    R_G: torch.Tensor,
) -> torch.Tensor:
    """Apply G⁻¹ to a bias gradient vector.

    For bias terms the gradient is just a vector (n_out,), so only the
    G factor is applied (A = 1 for a scalar input):

        nat_grad_b = G⁻¹ · grad_b = (R_Gᵀ R_G)⁻¹ · grad_b

    Parameters
    ----------
    grad_b : (n_out,) bias gradient.
    R_G    : (n_out, n_out) upper-triangular.

    Returns
    -------
    nat_grad_b : (n_out,)
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("apply_vered_bias() called: %s  %s",
                     _TS(grad_b, "grad_b"), _TS(R_G, "R_G"))
    gb = grad_b.unsqueeze(1)                                           # (n_out, 1)
    T1 = torch.linalg.solve_triangular(R_G.T, gb, upper=False)        # (n_out, 1)
    T2 = torch.linalg.solve_triangular(R_G, T1, upper=True)           # (n_out, 1)
    result = T2.squeeze(1)                                             # (n_out,)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("apply_vered_bias() done: %s", _TS(result, "nat_grad_b"))
    return result
