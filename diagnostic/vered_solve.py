"""
diagnostic/vered_solve.py
=========================

Inversion-free solver for SPD linear systems Gx = r.

This module implements the same "factor-then-solve" pattern used in
OlsSMKFAC._decompose_lu (optimizer/olssm_kfac.py:326-394): never form G^{-1}
explicitly, only Cholesky-factor and back-substitute. Progressive damping
retry with eigh PSD-projection fallback handles ill-conditioned inputs
from accumulated streaming Gram matrices.

Design rationale
----------------
The Vered paper's Algorithm 1 is a modified-Cholesky / LDL^T variant that
avoids square roots on the diagonal. The substantive property — *no explicit
matrix inversion* — is shared with standard Cholesky-solve. We use
torch.linalg.cholesky here for two reasons:

1. Consistency with OlsSMKFAC's existing solver path (the project's
   battle-tested code, tuned on Blackwell GPU for streaming Gram matrices).
2. PyTorch's cholesky + cholesky_solve are heavily optimized and dispatch
   to cuSOLVER on CUDA, beating any hand-rolled LDL^T in practice.

The LDL^T-without-sqrt variant from Vered's paper can be plugged in later
as an alternative backend; the interface here (`vered_solve`) does not
care which internal factorization is used.

Public API
----------
vered_decompose(G, damping=...) -> CholFactor
vered_solve(G, r, damping=...) -> x
vered_solve_batched(G, R) -> X    (factor once, solve many)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass for the factored representation
# ---------------------------------------------------------------------------

@dataclass
class CholFactor:
    """A Cholesky factor L of (G + damping*I), with metadata.

    Attributes
    ----------
    L : (d, d) lower-triangular torch.Tensor
        The factor satisfying (G + damping_used * I) = L L^T.
    damping_used : float
        Effective damping applied. Equal to base damping × multiplier from
        the progressive retry loop (1, 10, 100, 1000) or the eigh-fallback
        floor.
    used_eigh_fallback : bool
        True if Cholesky failed at all damping levels and we fell back to
        eigh-based PSD projection.
    """

    L: torch.Tensor
    damping_used: float
    used_eigh_fallback: bool = False


# ---------------------------------------------------------------------------
# Decomposition: Cholesky with progressive damping + eigh fallback
# ---------------------------------------------------------------------------

def vered_decompose(
    G: torch.Tensor,
    damping: float = 1e-6,
    damping_multipliers: Tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
    eigh_fallback: bool = True,
) -> CholFactor:
    """Factor an SPD matrix G into (G + λI) = L L^T without forming the inverse.

    Mirrors OlsSMKFAC._decompose_lu: tries Cholesky at increasing damping
    multipliers; falls back to eigh-based PSD projection if all damping
    levels fail.

    Parameters
    ----------
    G : (d, d) torch.Tensor
        Symmetric matrix. Will be symmetrized internally as (G + G^T)/2 for
        exact symmetry. Should be PSD; near-PSD with small negative eigenvalues
        is handled by damping/fallback.
    damping : float
        Base damping λ added to the diagonal. Final damping used is
        damping * multiplier_that_succeeded.
    damping_multipliers : tuple of float
        Progressive retry schedule. Each multiplier is tried in order.
    eigh_fallback : bool
        If True, fall back to eigh PSD projection if all damping levels fail.
        If False, raise the original LinAlgError.

    Returns
    -------
    CholFactor with L, damping_used, used_eigh_fallback.

    Raises
    ------
    torch.linalg.LinAlgError
        If all damping levels fail and eigh_fallback is False, or if eigh
        itself fails (extremely rare).
    """
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square 2D matrix, got shape {tuple(G.shape)}")

    G = (G + G.transpose(-1, -2)) * 0.5  # enforce exact symmetry
    n = G.shape[0]
    eye_n = torch.eye(n, device=G.device, dtype=G.dtype)

    # Sanity check for NaN/Inf inputs
    if not torch.isfinite(G).all():
        logger.warning(
            "vered_decompose: G has non-finite values; returning identity Cholesky "
            "(downstream solves will reduce to scaled identity)"
        )
        L = torch.linalg.cholesky(eye_n * (1.0 + damping))
        return CholFactor(L=L, damping_used=1.0 + damping, used_eigh_fallback=True)

    # Progressive damping retry
    last_err: Optional[Exception] = None
    for mult in damping_multipliers:
        damp_eff = damping * mult
        try:
            G_damped = G + damp_eff * eye_n
            L = torch.linalg.cholesky(G_damped)
            return CholFactor(L=L, damping_used=damp_eff, used_eigh_fallback=False)
        except torch.linalg.LinAlgError as e:
            last_err = e
            continue

    # Fallback: eigh-based PSD projection
    if not eigh_fallback:
        raise last_err  # type: ignore[misc]

    try:
        logger.warning(
            "vered_decompose: Cholesky failed at all damping multipliers "
            f"{damping_multipliers}; using eigh PSD-projection fallback"
        )
        eigvals, eigvecs = torch.linalg.eigh(G)
        floor = max(damping, 1e-8)
        eigvals_clamped = eigvals.clamp(min=floor)
        G_psd = (eigvecs * eigvals_clamped) @ eigvecs.transpose(-1, -2)
        G_psd = (G_psd + G_psd.transpose(-1, -2)) * 0.5
        L = torch.linalg.cholesky(G_psd)
        return CholFactor(L=L, damping_used=floor, used_eigh_fallback=True)
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        logger.warning(
            f"vered_decompose: eigh fallback failed ({type(e).__name__}: {e}); "
            "returning identity Cholesky as last resort"
        )
        L = torch.linalg.cholesky(eye_n * (1.0 + damping))
        return CholFactor(L=L, damping_used=1.0 + damping, used_eigh_fallback=True)


# ---------------------------------------------------------------------------
# Solve: factor + cholesky_solve (no explicit inverse)
# ---------------------------------------------------------------------------

def vered_solve(
    G: torch.Tensor,
    r: torch.Tensor,
    damping: float = 1e-6,
    damping_multipliers: Tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
) -> torch.Tensor:
    """Solve (G + λI) x = r without forming G^{-1}.

    One-shot factor + solve. For repeated solves with the same G but
    different RHS, prefer `vered_decompose` once + `vered_apply` many times.

    Parameters
    ----------
    G : (d, d) symmetric tensor.
    r : (d,) or (d, k) RHS.
    damping, damping_multipliers : see vered_decompose.

    Returns
    -------
    x : same shape as r — solution to (G + λI) x = r where λ is the
        effective damping that succeeded.
    """
    factor = vered_decompose(G, damping=damping, damping_multipliers=damping_multipliers)
    return vered_apply(factor, r)


def vered_apply(
    factor: CholFactor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Apply a pre-computed Cholesky factor to a RHS via cholesky_solve.

    Parameters
    ----------
    factor : CholFactor from vered_decompose.
    r : (d,) or (d, k) RHS.

    Returns
    -------
    x : solution to (L L^T) x = r, same shape as r.
    """
    L = factor.L
    if r.ndim == 1:
        r2 = r.unsqueeze(-1)
        x = torch.cholesky_solve(r2, L)
        return x.squeeze(-1)
    return torch.cholesky_solve(r, L)


def vered_solve_batched(
    G: torch.Tensor,
    R: torch.Tensor,
    damping: float = 1e-6,
    damping_multipliers: Tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
) -> torch.Tensor:
    """Factor G once, solve for multiple RHS columns at once.

    Equivalent to `vered_solve(G, R)` when R is 2D, but the function name
    documents the "factor once, apply to a batch" intent.

    Parameters
    ----------
    G : (d, d) symmetric tensor.
    R : (d, k) batch of RHS as columns.

    Returns
    -------
    X : (d, k) solutions.
    """
    if R.ndim != 2:
        raise ValueError(f"R must be 2D (d, k); got shape {tuple(R.shape)}")
    factor = vered_decompose(G, damping=damping, damping_multipliers=damping_multipliers)
    return vered_apply(factor, R)
