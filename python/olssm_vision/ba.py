"""Minimal bundle-adjust outer loop.

The heavy lifting (Schur complement, damped solve, selected-inverse covariance)
happens in the Rust extension. This module provides a thin LM wrapper that:

  * accepts pre-linearised `U`, `V`, `B`, `g_c`, `g_p` blocks per iteration,
  * calls the Rust `build_schur` + `covariance_*` routines,
  * returns the parameter updates plus an optional covariance report.

Full-featured BA (Jacobian construction from (pose, point) blocks, robust
norms, trust-region control) is a Phase 2 concern. This module exists so the
Rust inner solver can be driven from a Python test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from olssm_vision._rust import (  # type: ignore
        build_schur as _rust_build_schur,
        backsubstitute_points as _rust_back,
    )
    _HAVE_RUST = True
except ImportError:
    _HAVE_RUST = False


@dataclass
class BAStep:
    delta_c: np.ndarray
    """Camera parameter update, length `6 * num_cameras`."""

    delta_p: np.ndarray
    """Point parameter update, length `3 * num_points`."""

    s: np.ndarray
    """Reduced camera Hessian — available for downstream covariance queries."""


def solve_ba_step(
    u: np.ndarray,
    v: np.ndarray,
    b: np.ndarray,
    g_c: np.ndarray,
    g_p: np.ndarray,
    damping: float = 0.0,
) -> BAStep:
    """One LM inner step on a pre-linearised BA problem.

    Parameters
    ----------
    u : (6Nc, 6Nc) float64
    v : (3Np, 3Np) float64  — block-diagonal per point
    b : (6Nc, 3Np) float64
    g_c : (6Nc,) float64
    g_p : (3Np,) float64
    damping : float — Levenberg-Marquardt damping added to the diagonal.

    Returns
    -------
    BAStep
    """
    u = np.ascontiguousarray(u, dtype=np.float64)
    v = np.ascontiguousarray(v, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    g_c = np.ascontiguousarray(g_c, dtype=np.float64)
    g_p = np.ascontiguousarray(g_p, dtype=np.float64)

    if _HAVE_RUST:
        sys_ = _rust_build_schur(u, v, b, g_c, g_p, damping)
        s = sys_["s"]
        g_tilde = sys_["g_tilde"]
        delta_c = np.linalg.solve(s, g_tilde)
        delta_p = _rust_back(delta_c, sys_["v_inv_block_diag"], sys_["b"], sys_["g_p"])
        return BAStep(delta_c=delta_c, delta_p=delta_p, s=s)

    # Fallback: pure numpy path, same math.
    n_cam = u.shape[0]
    n_pt = v.shape[0]
    assert n_pt % 3 == 0, "V must be (3Np) x (3Np)"
    n_points = n_pt // 3

    u_d = u + damping * np.eye(n_cam)
    v_d = v + damping * np.eye(n_pt)

    v_inv = np.zeros_like(v_d)
    for i in range(n_points):
        base = 3 * i
        v_inv[base:base + 3, base:base + 3] = np.linalg.inv(
            v_d[base:base + 3, base:base + 3]
        )

    b_vinv = b @ v_inv
    s = u_d - b_vinv @ b.T
    g_tilde = g_c - b_vinv @ g_p
    delta_c = np.linalg.solve(s, g_tilde)
    delta_p = v_inv @ (g_p - b.T @ delta_c)
    return BAStep(delta_c=delta_c, delta_p=delta_p, s=s)
