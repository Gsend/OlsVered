"""
diagnostic/inversion.py
=======================

Two methods for inverting a single neural-network layer in the back-target
direction: recovering an estimate of the input activation from a target
output activation.

Method N (Naive)
    Minimum-norm Moore-Penrose pseudo-inverse, computed via vered_solve
    (no explicit matrix inversion).

        a_hat = arg min ||a||^2  subject to  W a = c

    where c = t_pre - b. The minimum-norm solution is a_hat = W^+ c.

Method K (K-FAC-A regularized)
    Conditional Gaussian posterior mean under a Gaussian prior
    N(mu_a, Sigma_a) on the input activation, with the constraint W a = c
    imposed exactly (modulo damping).

        a_hat = mu_a + Sigma_a W^T (W Sigma_a W^T + sigma^2 I)^-1 (c - W mu_a)

    The prior covariance Sigma_a is the empirical activation covariance from
    the forward pass — the same matrix used as the K-FAC A factor. The
    formula recovers the constrained linear combinations exactly and
    imputes the unconstrained directions using the prior.

Both methods use diagnostic.vered_solve.vered_solve internally; no
explicit matrix inversion is performed.

Activation-function inversion
    invert_activation maps a post-activation target back through the
    inverse of an element-wise activation function (ReLU / Tanh /
    Sigmoid / GELU / SiLU / LeakyReLU / ELU). Mirrors the per-activation
    rules in OlsSMLayerRetrainer._inverse_activation but as a pure
    function (no optimizer state).

Top-level driver
    invert_layer ties activation-inverse + linear-inverse together
    for a complete one-layer back-target step.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from diagnostic.layers import LNReLU
from diagnostic.vered_solve import vered_solve


# ---------------------------------------------------------------------------
# Linear inverse — Method N (naive pseudo-inverse, via vered_solve)
# ---------------------------------------------------------------------------

def naive_pinv_inverse(
    W: torch.Tensor,                       # (d_out, d_in)
    b: Optional[torch.Tensor],             # (d_out,) or None
    t_pre: torch.Tensor,                   # (N, d_out)
    eps: float = 1e-4,
) -> torch.Tensor:                          # (N, d_in)
    """Method N — minimum-norm pseudo-inverse, computed without explicit inversion.

    For each row j of t_pre, recovers a_hat^(j) such that
    W a_hat^(j) + b ~= t_pre^(j) using the damped Moore-Penrose pseudo-inverse:

        a_hat^(j) = W^+ (t_pre^(j) - b)

    The aspect-ratio of W determines the formulation:
      - d_in >= d_out (typical, going backward to a wider layer):
            a_hat = W^T (W W^T + lambda I)^-1 c              minimum-norm
      - d_in < d_out (going backward to a narrower layer):
            a_hat = (W^T W + lambda I)^-1 W^T c              least-squares

    In both cases the inverse is computed via Cholesky-factor-then-solve
    using vered_solve (no explicit inversion).

    Parameters
    ----------
    W : (d_out, d_in) layer weight matrix.
    b : (d_out,) bias, or None.
    t_pre : (N, d_out) pre-activation target.
    eps : Damping. Scaled by mean diagonal of the formed Gram matrix
        internally to be problem-size-aware.

    Returns
    -------
    a_hat : (N, d_in) recovered input activations.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2D (d_out, d_in); got shape {tuple(W.shape)}")
    if t_pre.ndim != 2:
        raise ValueError(f"t_pre must be 2D (N, d_out); got shape {tuple(t_pre.shape)}")
    d_out, d_in = W.shape
    if t_pre.shape[1] != d_out:
        raise ValueError(
            f"t_pre.shape[1] = {t_pre.shape[1]} must equal d_out = {d_out}"
        )

    if b is not None:
        c = t_pre - b.unsqueeze(0)
    else:
        c = t_pre

    if d_in >= d_out:
        # Minimum-norm: a_hat = W^T (W W^T + lambda I)^-1 c
        WWt = W @ W.transpose(0, 1)
        damping = eps * (WWt.diagonal().mean().item() + 1e-30)
        X = vered_solve(WWt, c.transpose(0, 1), damping=damping)
        a_hat = W.transpose(0, 1) @ X
        return a_hat.transpose(0, 1)
    else:
        # Least-squares: a_hat = (W^T W + lambda I)^-1 W^T c
        WtW = W.transpose(0, 1) @ W
        damping = eps * (WtW.diagonal().mean().item() + 1e-30)
        Wt_c = W.transpose(0, 1) @ c.transpose(0, 1)
        a_hat_T = vered_solve(WtW, Wt_c, damping=damping)
        return a_hat_T.transpose(0, 1)


# ---------------------------------------------------------------------------
# Linear inverse — Method K (K-FAC-A regularized, via vered_solve)
# ---------------------------------------------------------------------------

def kfac_a_inverse(
    W: torch.Tensor,                       # (d_out, d_in)
    b: Optional[torch.Tensor],             # (d_out,) or None
    t_pre: torch.Tensor,                   # (N, d_out)
    mu_a: torch.Tensor,                    # (d_in,) prior mean
    Sigma_a: torch.Tensor,                 # (d_in, d_in) prior covariance (K-FAC A factor)
    sigma2: Optional[float] = None,
) -> torch.Tensor:                          # (N, d_in)
    """Method K — K-FAC-A regularized inverse via Gaussian-posterior mean.

    Treats a_hat as having a Gaussian prior N(mu_a, Sigma_a) and imposes the
    constraint W a_hat = c = t_pre - b. Returns the posterior mean (which is
    also the maximum-likelihood point in the constraint-satisfying affine
    subspace under the Gaussian prior):

        a_hat^(j) = mu_a + Sigma_a W^T (W Sigma_a W^T + sigma^2 I)^-1
                                            (c^(j) - W mu_a)

    For each sample j, a_hat^(j) satisfies the constraint W a_hat^(j) ~= c^(j)
    exactly (modulo sigma^2 damping), AND lies as close as possible to mu_a
    in the Mahalanobis metric Sigma_a^-1. The prior shapes the answer only
    in directions where the constraint is silent (null space of W).

    Parameters
    ----------
    W : (d_out, d_in) layer weight matrix.
    b : (d_out,) bias, or None.
    t_pre : (N, d_out) pre-activation target.
    mu_a : (d_in,) prior mean — empirical mean of forward-pass input activations.
    Sigma_a : (d_in, d_in) prior covariance — empirical covariance of
        forward-pass input activations. Same matrix as the K-FAC A factor
        (up to bias-column convention).
    sigma2 : Observation-noise variance for the Kalman update. If None, a
        heuristic value 1e-4 * trace(W Sigma_a W^T) / d_out is used. Set to 0
        (or very small) for hard-constraint behaviour; larger values relax
        the constraint into a soft penalty.

    Returns
    -------
    a_hat : (N, d_in) recovered input activations (per-sample conditional means).
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2D (d_out, d_in); got shape {tuple(W.shape)}")
    if t_pre.ndim != 2:
        raise ValueError(f"t_pre must be 2D (N, d_out); got shape {tuple(t_pre.shape)}")
    d_out, d_in = W.shape
    if t_pre.shape[1] != d_out:
        raise ValueError(
            f"t_pre.shape[1] = {t_pre.shape[1]} must equal d_out = {d_out}"
        )
    if mu_a.ndim != 1 or mu_a.shape[0] != d_in:
        raise ValueError(
            f"mu_a must be 1D with length d_in = {d_in}; got shape {tuple(mu_a.shape)}"
        )
    if Sigma_a.shape != (d_in, d_in):
        raise ValueError(
            f"Sigma_a must be ({d_in}, {d_in}); got shape {tuple(Sigma_a.shape)}"
        )

    # Sigma_a W^T : (d_in, d_out)
    SWt = Sigma_a @ W.transpose(0, 1)

    # G = W Sigma_a W^T : (d_out, d_out)
    G = W @ SWt

    if sigma2 is None:
        sigma2_eff = 1e-4 * (G.diagonal().abs().mean().item() + 1e-30)
    else:
        sigma2_eff = float(sigma2)

    if b is not None:
        c = t_pre - b.unsqueeze(0)
    else:
        c = t_pre
    W_mu = W @ mu_a
    r = c - W_mu.unsqueeze(0)

    # Solve (G + sigma^2 I) X = r^T  =>  X : (d_out, N)
    X = vered_solve(G, r.transpose(0, 1), damping=sigma2_eff)

    # a_hat^(j) = mu_a + Sigma_a W^T X[:, j]  =>  shape (d_in, N) -> transpose
    a_hat = mu_a.unsqueeze(1) + SWt @ X
    return a_hat.transpose(0, 1)


# ---------------------------------------------------------------------------
# Activation inverse — element-wise inverse mapping
# ---------------------------------------------------------------------------

def invert_activation(
    a_post: torch.Tensor,                  # (N, d_out) post-activation target
    a_pre_forward: torch.Tensor,           # (N, d_out) forward-pass pre-activation
    activation: Optional[nn.Module],
) -> torch.Tensor:                          # (N, d_out) pre-activation target
    """Map a post-activation target back through the inverse of an activation.

    Handles common activation functions:

    - None: pass-through (linear / identity case).
    - Tanh: exact inverse arctanh, clipped to (-0.9999, 0.9999) for stability.
    - Sigmoid: exact inverse logit, clipped to (1e-4, 1-1e-4).
    - ReLU / ReLU6: mask-based. Live units (a_post > 0) pass through;
      dead units (a_post == 0) use the forward-pass pre-activation
      (Option A from the design doc — mask preservation).
    - GELU / SiLU: mask-based with threshold-aware live/dead split.
    - LeakyReLU: scales negative side by 1/negative_slope.
    - ELU: log(1 + a/alpha) on the negative side, identity on the positive side.

    Parameters
    ----------
    a_post : (N, d_out) post-activation target.
    a_pre_forward : (N, d_out) forward-pass pre-activation values, used as
        fallback for dead units in mask-based inverses.
    activation : nn.Module instance (ReLU, Tanh, ...) or None.

    Returns
    -------
    t_pre : (N, d_out) pre-activation target.
    """
    if activation is None:
        return a_post.clone()

    if isinstance(activation, nn.Tanh):
        return torch.arctanh(a_post.clamp(-0.9999, 0.9999))

    if isinstance(activation, nn.Sigmoid):
        a_c = a_post.clamp(1e-4, 1 - 1e-4)
        return torch.log(a_c / (1.0 - a_c))

    if isinstance(activation, (nn.ReLU, nn.ReLU6)):
        mask = (a_post > 0).to(a_post.dtype)
        return a_post * mask + a_pre_forward.detach() * (1.0 - mask)

    if isinstance(activation, nn.GELU):
        mask = (a_post > 0).to(a_post.dtype)
        return a_post * mask + a_pre_forward.detach() * (1.0 - mask)

    if isinstance(activation, nn.SiLU):
        mask = (a_post > -0.278).to(a_post.dtype)
        return a_post * mask + a_pre_forward.detach() * (1.0 - mask)

    if isinstance(activation, nn.LeakyReLU):
        slope = float(activation.negative_slope)
        if slope <= 0:
            mask = (a_post > 0).to(a_post.dtype)
            return a_post * mask + a_pre_forward.detach() * (1.0 - mask)
        mask_pos = (a_post >= 0).to(a_post.dtype)
        return a_post * mask_pos + (a_post / slope) * (1.0 - mask_pos)

    if isinstance(activation, nn.ELU):
        alpha = float(activation.alpha)
        mask_pos = (a_post >= 0).to(a_post.dtype)
        neg_part = torch.log(torch.clamp(1.0 + a_post / max(alpha, 1e-12), min=1e-8))
        return a_post * mask_pos + neg_part * (1.0 - mask_pos)

    if isinstance(activation, LNReLU):
        # Step 1: ReLU inverse with mask (live units pass through; dead units
        # fall back to the forward-pass LN output, preserving pre-activation).
        mask_live = (a_post > 0).to(a_post.dtype)
        with torch.no_grad():
            # LayerNorm has parameters — move a_pre_forward to its device.
            ln_device = activation.norm.weight.device
            a_ln_fwd = activation.norm(
                a_pre_forward.float().to(ln_device)
            ).cpu().to(a_post.dtype)
        a_ln_target = a_post * mask_live + a_ln_fwd * (1.0 - mask_live)

        # Step 2: LN inverse — recover x from y = (x - mu)/std * gamma + beta
        # using per-sample statistics from the forward pre-activation.
        mu = a_pre_forward.mean(dim=-1, keepdim=True)
        var = ((a_pre_forward - mu) ** 2).mean(dim=-1, keepdim=True)
        std = (var + activation.norm.eps).sqrt()
        # Move gamma/beta to the same device+dtype as a_post for the inverse.
        gamma = activation.norm.weight.to(device=a_post.device, dtype=a_post.dtype)
        beta = activation.norm.bias.to(device=a_post.device, dtype=a_post.dtype)
        # Guard near-zero gamma (gamma starts at 1; rarely near-zero after training)
        safe_gamma = torch.where(gamma.abs() < 1e-6, torch.ones_like(gamma), gamma)
        t_pre = (a_ln_target - beta) / safe_gamma * std + mu
        return t_pre.to(a_post.dtype)

    # Unknown activation — pass through
    return a_post.clone()


# ---------------------------------------------------------------------------
# Top-level driver: complete one-layer back-target step
# ---------------------------------------------------------------------------

def invert_layer(
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    a_out_post: torch.Tensor,              # (N, d_out) post-activation target
    a_pre_forward: torch.Tensor,           # (N, d_out) forward-pass pre-activation
    activation: Optional[nn.Module],
    method: Literal["naive", "kfac_a"],
    mu_a: Optional[torch.Tensor] = None,
    Sigma_a: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
    sigma2: Optional[float] = None,
) -> torch.Tensor:                          # (N, d_in)
    """Complete one-layer back-target step: activation-inverse then linear-inverse.

    Parameters
    ----------
    W, b : Layer weight and bias.
    a_out_post : (N, d_out) post-activation target.
    a_pre_forward : (N, d_out) forward-pass pre-activation (for mask preservation).
    activation : The activation module (nn.ReLU, nn.Tanh, ...) or None.
    method : 'naive' for Method N, 'kfac_a' for Method K.
    mu_a, Sigma_a : Required when method='kfac_a'. Prior mean and covariance
        of the input activation distribution.
    eps : Damping for Method N.
    sigma2 : Observation noise for Method K. Defaults to a heuristic.

    Returns
    -------
    a_hat : (N, d_in) recovered input activations.
    """
    t_pre = invert_activation(a_out_post, a_pre_forward, activation)

    if method == "naive":
        return naive_pinv_inverse(W, b, t_pre, eps=eps)
    elif method == "kfac_a":
        if mu_a is None or Sigma_a is None:
            raise ValueError("Method 'kfac_a' requires mu_a and Sigma_a")
        return kfac_a_inverse(W, b, t_pre, mu_a, Sigma_a, sigma2=sigma2)
    else:
        raise ValueError(
            f"Unknown method '{method}'; expected 'naive' or 'kfac_a'"
        )
