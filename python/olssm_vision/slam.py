"""Online covariance-on-demand for SLAM / AR.

Given the converged or near-converged information matrix `A` of a BA / SLAM
system, query marginal variances, cross-covariances, and 3×3 landmark
covariance blocks at a fraction of the cost of forming `A⁻¹` in full.

Typical usage pattern at the BA / SLAM outer loop:

    from olssm_vision.slam import CovarianceOracle

    oracle = CovarianceOracle.from_information(S)       # once per LM iteration
    sigma_cam0 = oracle.parameter_variance(0)           # scalar
    sigma_pt   = oracle.point_covariance(point_dof=[21, 22, 23])  # 3×3
    cross      = oracle.cross_covariance(i=0, j=21)     # scalar

The oracle caches the whitened Cholesky factor so repeated queries in the
same iteration are cheap (one forward+back substitution each).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    from olssm_vision._rust import (  # type: ignore
        whitened_cholesky as _rust_wc,
        covariance_column as _rust_col,
        covariance_diagonal as _rust_diag,
        covariance_block as _rust_block,
    )
    _HAVE_RUST = True
except ImportError:
    _HAVE_RUST = False


@dataclass
class CovarianceOracle:
    """Holds the cached Cholesky factor for on-demand covariance queries."""

    l_full: np.ndarray
    """Lower-triangular `L` such that `L Lᵀ = A` (not whitened)."""

    def _column(self, j: int) -> np.ndarray:
        if _HAVE_RUST:
            return _rust_col(self.l_full, j)
        e = np.zeros(self.l_full.shape[0])
        e[j] = 1.0
        y = np.linalg.solve(self.l_full, e)
        x = np.linalg.solve(self.l_full.T, y)
        return x

    @classmethod
    def from_information(cls, information: np.ndarray) -> "CovarianceOracle":
        information = np.ascontiguousarray(information, dtype=np.float64)
        # We use the full (not whitened) Cholesky for the solver path so that
        # queries are directly in the parameter scale.
        l_full = np.linalg.cholesky(information)
        return cls(l_full=l_full)

    def parameter_variance(self, index: int) -> float:
        return float(self._column(index)[index])

    def cross_covariance(self, i: int, j: int) -> float:
        return float(self._column(j)[i])

    def all_marginal_variances(self) -> np.ndarray:
        if _HAVE_RUST:
            return _rust_diag(self.l_full)
        n = self.l_full.shape[0]
        return np.array([self.parameter_variance(i) for i in range(n)])

    def point_covariance(self, point_dof: Sequence[int]) -> np.ndarray:
        """Return the `k × k` covariance block for a set of parameter indices.

        For a 3-D landmark this is the 3×3 ellipsoid used by AR anchor
        stability, NeRF / 3DGS uncertainty export, and Mahalanobis gating.
        """
        indices = list(point_dof)
        if _HAVE_RUST:
            return _rust_block(self.l_full, indices)
        k = len(indices)
        out = np.zeros((k, k))
        for a, ja in enumerate(indices):
            col = self._column(ja)
            for b, ib in enumerate(indices):
                out[b, a] = col[ib]
        return out
