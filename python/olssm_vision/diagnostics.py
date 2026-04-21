"""Degeneracy, gauge, and leverage diagnostics for BA / SLAM systems.

All entry points accept either a Rust-backed factorisation result or a plain
numpy information matrix. The Rust extension is used when available; a
numpy/scipy fallback keeps the module usable during development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from olssm_vision._rust import (  # type: ignore
        whitened_cholesky as _rust_wc,
        pivot_profile as _rust_pp,
        detect_near_singular as _rust_nsing,
    )
    _HAVE_RUST = True
except ImportError:
    _HAVE_RUST = False


@dataclass
class DegeneracyReport:
    """Parameter indices whose conditional variance is near-zero, i.e. those
    directions are unobservable from the current measurement set."""

    weak_columns: List[Tuple[int, float]]
    """Sorted weakest-first list of (parameter_index, pivot_magnitude)."""

    pivot_profile: np.ndarray
    """Diagonal of the whitened Cholesky factor — one pivot per parameter."""

    threshold: float

    def is_degenerate(self) -> bool:
        return len(self.weak_columns) > 0

    def summarise(self) -> str:
        if not self.weak_columns:
            return f"No parameters below threshold {self.threshold:.2e}."
        lines = [
            f"{len(self.weak_columns)} near-singular directions (threshold {self.threshold:.2e}):"
        ]
        for idx, piv in self.weak_columns[:10]:
            lines.append(f"  param[{idx}] pivot={piv:.3e}")
        if len(self.weak_columns) > 10:
            lines.append(f"  ... +{len(self.weak_columns) - 10} more")
        return "\n".join(lines)


def analyse_information(
    information: np.ndarray,
    threshold: float = 1e-3,
) -> DegeneracyReport:
    """Compute whitened-Cholesky pivots of the information matrix and flag
    columns below `threshold`.

    Parameters
    ----------
    information : (n, n) symmetric positive-definite array
        The BA / SLAM information matrix, e.g. the reduced camera Hessian `S`
        produced by Schur elimination (plus any LM damping).
    threshold : float
        Pivot magnitude below which a parameter direction is flagged as
        near-unobservable. Default `1e-3` is a reasonable starting point for
        problems pre-scaled by their diagonal.

    Returns
    -------
    DegeneracyReport
    """
    information = np.ascontiguousarray(information, dtype=np.float64)
    if _HAVE_RUST:
        lw, _d = _rust_wc(information)
        pp = _rust_pp(lw)
        weak = _rust_nsing(lw, threshold)
    else:
        lw, _d = _np_whitened_cholesky(information)
        pp = np.diag(lw).copy()
        weak = sorted(
            [(i, float(pp[i])) for i in range(len(pp)) if pp[i] < threshold],
            key=lambda t: t[1],
        )
    return DegeneracyReport(
        weak_columns=[(int(i), float(p)) for i, p in weak],
        pivot_profile=np.asarray(pp, dtype=np.float64),
        threshold=threshold,
    )


def _np_whitened_cholesky(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-numpy fallback for `whitened_cholesky`."""
    d = np.sqrt(np.diag(a))
    r = a / np.outer(d, d)
    lw = np.linalg.cholesky(r)
    return lw, d


def leverage(jacobian: np.ndarray) -> np.ndarray:
    """Per-observation leverage ``h_ii`` of the hat matrix ``J (JᵀJ)⁻¹ Jᵀ``.

    Large ``h_ii`` identifies observations that disproportionately influence
    the BA solution. Useful as a principled robust-weighting signal.
    """
    # QR-based implementation (stable and fast enough for diagnostic use).
    q, _ = np.linalg.qr(jacobian)
    return np.einsum("ij,ij->i", q, q)
