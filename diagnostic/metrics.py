"""
diagnostic/metrics.py
=====================

Drift metrics for comparing back-propagated activations (â) to ground-truth
forward activations (a*).

Three tiers of comparison:

Tier 1 — Aggregate distortion (one number per layer × method)
    relative_l2_error           — per-sample reconstruction quality
    mean_drift                  — ||mu_hat - mu_star|| / ||mu_star||
    covariance_frobenius        — ||Sigma_hat - Sigma_star||_F / ||Sigma_star||_F
    gaussian_kl_symmetric       — symmetric KL of Gaussian fits (with ridge)
    wasserstein2_gaussian       — closed-form W2 between Gaussian fits

Tier 2 — Spectral diagnosis (decompose where the distortion lives)
    eigenvalue_spectrum_match   — sorted-spectrum correlation + KL on the
                                  eigenvalue distribution
    principal_subspace_angles   — angles between top-k eigenvector subspaces

Tier 3 — Functional fidelity (the load-bearing metric)
    downstream_loss             — feed â through the remaining network and
                                  measure prediction error vs labels and
                                  vs the prediction we'd get from a*

Sanity-test helpers
    constraint_residual         — ||W â + b - t_pre|| / ||t_pre||
    predicted_projection_covariance — analytic prediction of cov(â) under
                                  Method K's conditional-Gaussian formula
                                  (rank d_out projection of Sigma_a)

All functions are pure (no torch state); operate on (N, d) batched activation
tensors and (d, d) covariance matrices.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Tier 1 — per-sample fidelity
# ---------------------------------------------------------------------------

def relative_l2_error(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    eps: float = 1e-30,
) -> torch.Tensor:                          # (N,)
    """Per-sample relative L2 error: ||â^(j) - a*^(j)||_2 / ||a*^(j)||_2.

    Returns a 1D tensor of shape (N,). Take .mean() or .quantile(0.95) to
    summarize.
    """
    _validate_2d_pair(a_hat, a_star)
    num = (a_hat - a_star).norm(dim=-1)
    den = a_star.norm(dim=-1).clamp(min=eps)
    return num / den


def cosine_similarity_per_sample(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    eps: float = 1e-30,
) -> torch.Tensor:                          # (N,)
    """Per-sample cosine similarity in [-1, 1]."""
    _validate_2d_pair(a_hat, a_star)
    return torch.nn.functional.cosine_similarity(a_hat, a_star, dim=-1, eps=eps)


# ---------------------------------------------------------------------------
# Tier 1 — distributional fidelity
# ---------------------------------------------------------------------------

def mean_drift(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    eps: float = 1e-30,
) -> float:
    """Relative drift of empirical mean: ||mu_hat - mu_star|| / ||mu_star||."""
    _validate_2d_pair(a_hat, a_star)
    mu_hat = a_hat.mean(dim=0)
    mu_star = a_star.mean(dim=0)
    return ((mu_hat - mu_star).norm() / mu_star.norm().clamp(min=eps)).item()


def covariance_frobenius(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    eps: float = 1e-30,
) -> float:
    """Relative Frobenius distance between empirical covariances:
    ||Sigma_hat - Sigma_star||_F / ||Sigma_star||_F.
    """
    _validate_2d_pair(a_hat, a_star)
    S_hat = _empirical_cov(a_hat)
    S_star = _empirical_cov(a_star)
    num = (S_hat - S_star).norm(p="fro")
    den = S_star.norm(p="fro").clamp(min=eps)
    return (num / den).item()


def gaussian_kl_symmetric(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    ridge: float = 1e-6,
) -> float:
    """Symmetric KL divergence between Gaussian fits to the two samples:

        0.5 * (KL(N_hat || N_star) + KL(N_star || N_hat))

    A ridge term is added to both covariances before inversion to handle
    rank-deficiency when N <= d. The ridge is scaled by trace(Sigma)/d.

    Closed form for one direction:
        KL(N_1 || N_2) = 0.5 * [tr(Σ₂⁻¹ Σ₁) + (μ₂ - μ₁)ᵀ Σ₂⁻¹ (μ₂ - μ₁)
                                - d + log(det Σ₂ / det Σ₁)]
    """
    _validate_2d_pair(a_hat, a_star)
    d = a_hat.shape[1]
    mu_hat = a_hat.mean(dim=0)
    mu_star = a_star.mean(dim=0)
    S_hat = _empirical_cov(a_hat)
    S_star = _empirical_cov(a_star)

    # Ridge scaled by trace/d for shape invariance
    eye = torch.eye(d, dtype=a_hat.dtype, device=a_hat.device)
    r_hat = ridge * (S_hat.diagonal().abs().mean().item() + 1e-30)
    r_star = ridge * (S_star.diagonal().abs().mean().item() + 1e-30)
    S_hat_r = S_hat + r_hat * eye
    S_star_r = S_star + r_star * eye

    kl_1to2 = _gaussian_kl_one_way(mu_hat, S_hat_r, mu_star, S_star_r)
    kl_2to1 = _gaussian_kl_one_way(mu_star, S_star_r, mu_hat, S_hat_r)
    return 0.5 * (kl_1to2 + kl_2to1)


def wasserstein2_gaussian(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    ridge: float = 1e-6,
) -> float:
    """Squared 2-Wasserstein distance between Gaussian fits:

        W₂²(N_1, N_2) = ||μ_1 - μ_2||² + tr(Σ_1 + Σ_2 - 2 (Σ_1^½ Σ_2 Σ_1^½)^½)

    Uses eigendecomposition for matrix square roots (stable on SPD matrices).
    """
    _validate_2d_pair(a_hat, a_star)
    d = a_hat.shape[1]
    mu_hat = a_hat.mean(dim=0)
    mu_star = a_star.mean(dim=0)
    S_hat = _empirical_cov(a_hat)
    S_star = _empirical_cov(a_star)

    eye = torch.eye(d, dtype=a_hat.dtype, device=a_hat.device)
    r_hat = ridge * (S_hat.diagonal().abs().mean().item() + 1e-30)
    r_star = ridge * (S_star.diagonal().abs().mean().item() + 1e-30)
    S_hat_r = S_hat + r_hat * eye
    S_star_r = S_star + r_star * eye

    # Mean term
    mean_term = ((mu_hat - mu_star) ** 2).sum().item()

    # Trace term: tr(S_1 + S_2 - 2 (S_1^½ S_2 S_1^½)^½)
    S_hat_sqrt = _matrix_sqrt_psd(S_hat_r)
    inner = S_hat_sqrt @ S_star_r @ S_hat_sqrt
    inner_sqrt = _matrix_sqrt_psd(inner)
    trace_term = (
        S_hat_r.diagonal().sum() + S_star_r.diagonal().sum()
        - 2 * inner_sqrt.diagonal().sum()
    ).item()
    return max(mean_term + trace_term, 0.0)


# ---------------------------------------------------------------------------
# Tier 2 — spectral diagnosis
# ---------------------------------------------------------------------------

def eigenvalue_spectrum_match(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compare eigenvalue spectra of empirical covariances.

    Returns dict with:
      'pearson'         — Pearson correlation between sorted eigenvalues
      'kl_on_spectrum'  — KL divergence between sorted spectra treated as
                          discrete probability distributions (after normalization)
      'effective_rank_hat'  — exp(entropy of normalized eigenvalues of S_hat)
      'effective_rank_star' — exp(entropy of normalized eigenvalues of S_star)
    """
    _validate_2d_pair(a_hat, a_star)
    S_hat = _empirical_cov(a_hat)
    S_star = _empirical_cov(a_star)

    eig_hat = torch.linalg.eigvalsh(S_hat).clamp(min=0.0)
    eig_star = torch.linalg.eigvalsh(S_star).clamp(min=0.0)

    # Sort descending for comparison
    eig_hat_sorted, _ = torch.sort(eig_hat, descending=True)
    eig_star_sorted, _ = torch.sort(eig_star, descending=True)

    # Pearson correlation
    pearson = _pearson_correlation(eig_hat_sorted, eig_star_sorted)

    # KL between sorted-spectrum distributions
    p = (eig_hat_sorted + eps) / (eig_hat_sorted.sum() + eps * eig_hat_sorted.numel())
    q = (eig_star_sorted + eps) / (eig_star_sorted.sum() + eps * eig_star_sorted.numel())
    kl_on_spectrum = (p * (torch.log(p) - torch.log(q))).sum().item()

    return {
        "pearson": pearson,
        "kl_on_spectrum": kl_on_spectrum,
        "effective_rank_hat": _effective_rank(eig_hat_sorted),
        "effective_rank_star": _effective_rank(eig_star_sorted),
    }


def principal_subspace_angles(
    a_hat: torch.Tensor,                   # (N, d)
    a_star: torch.Tensor,                  # (N, d)
    k: int,
) -> torch.Tensor:                          # (k,)
    """Principal angles (in radians) between the top-k eigenvector
    subspaces of cov(â) and cov(a*).

    Small angles → subspaces agree → variation directions preserved.
    """
    _validate_2d_pair(a_hat, a_star)
    d = a_hat.shape[1]
    if k < 1 or k > d:
        raise ValueError(f"k must be in [1, {d}]; got {k}")
    S_hat = _empirical_cov(a_hat)
    S_star = _empirical_cov(a_star)

    # eigh returns eigenvalues in ascending order; take top-k (last k columns)
    _, V_hat = torch.linalg.eigh(S_hat)
    _, V_star = torch.linalg.eigh(S_star)
    V_hat_top = V_hat[:, -k:]
    V_star_top = V_star[:, -k:]

    # Singular values of V_star^T V_hat are cos(theta_i)
    sv = torch.linalg.svdvals(V_star_top.T @ V_hat_top)
    sv_clamped = sv.clamp(-1.0, 1.0)
    return torch.arccos(sv_clamped)


# ---------------------------------------------------------------------------
# Tier 3 — functional fidelity
# ---------------------------------------------------------------------------

def downstream_loss(
    a_hat: torch.Tensor,                   # (N, d_in)
    a_star: torch.Tensor,                  # (N, d_in)
    forward_remainder: Callable[[torch.Tensor], torch.Tensor],
    labels: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """Run the rest of the network from layer-(L-1)'s output forward.

    Computes three functional-fidelity metrics:

      'logit_mse'         — relative L2 between forward_remainder(â) and
                            forward_remainder(a*).
      'logit_mse_abs'     — absolute MSE.
      'prediction_agreement' — fraction of samples where argmax matches.
      'ce_drift'          — (CE(â, labels) - CE(a*, labels)) if labels given.
                            Otherwise None.

    Parameters
    ----------
    a_hat, a_star : Activation tensors at the same layer (one from inversion,
        one from forward pass).
    forward_remainder : Function that takes (N, d) activations at a given layer
        and returns the model's output (logits or final predictions) by running
        the remainder of the network.
    labels : Optional ground-truth labels for the CE drift computation.
    loss_fn : Optional loss function for the CE drift computation. Defaults to
        torch.nn.functional.cross_entropy.
    """
    _validate_2d_pair(a_hat, a_star)

    with torch.no_grad():
        y_hat = forward_remainder(a_hat)
        y_star = forward_remainder(a_star)

    diff = (y_hat - y_star)
    logit_mse_abs = (diff ** 2).mean().item()
    denom = (y_star ** 2).mean().item() + 1e-30
    logit_mse = logit_mse_abs / denom

    # Prediction agreement (classification only)
    if y_hat.ndim >= 2 and y_hat.shape[-1] > 1:
        pred_hat = y_hat.argmax(dim=-1)
        pred_star = y_star.argmax(dim=-1)
        agreement = (pred_hat == pred_star).float().mean().item()
    else:
        agreement = float("nan")

    result: Dict[str, float] = {
        "logit_mse": logit_mse,
        "logit_mse_abs": logit_mse_abs,
        "prediction_agreement": agreement,
    }

    if labels is not None:
        if loss_fn is None:
            loss_fn = torch.nn.functional.cross_entropy
        ce_hat = loss_fn(y_hat, labels).item()
        ce_star = loss_fn(y_star, labels).item()
        result["ce_drift"] = ce_hat - ce_star
        result["ce_hat"] = ce_hat
        result["ce_star"] = ce_star

    return result


# ---------------------------------------------------------------------------
# Sanity-test helpers
# ---------------------------------------------------------------------------

def constraint_residual(
    W: torch.Tensor,                       # (d_out, d_in)
    b: Optional[torch.Tensor],             # (d_out,) or None
    a_hat: torch.Tensor,                   # (N, d_in)
    t_pre: torch.Tensor,                   # (N, d_out)
    eps: float = 1e-30,
) -> float:
    """Relative Frobenius norm of (W â + b - t_pre).

    For the conditional Gaussian inversion with no damping, should be at
    machine precision (~1e-12 for fp64). With damping σ², residual scales
    with σ² / λ_min(W Σ_a W^T).
    """
    if W.ndim != 2:
        raise ValueError("W must be 2D")
    pred = a_hat @ W.transpose(0, 1)
    if b is not None:
        pred = pred + b.unsqueeze(0)
    return ((pred - t_pre).norm() / t_pre.norm().clamp(min=eps)).item()


def predicted_projection_covariance(
    Sigma_a: torch.Tensor,                 # (d_in, d_in) prior covariance
    W: torch.Tensor,                       # (d_out, d_in)
    ridge: float = 1e-12,
) -> torch.Tensor:                          # (d_in, d_in)
    """Closed-form covariance of Method K's point estimates when the targets
    are generated by passing forward activations through the layer:

        Cov(â)_predicted = Σ_a W^T (W Σ_a W^T)^{-1} W Σ_a

    This is the rank-d_out oblique projection of Σ_a onto the row space of W
    in the Σ_a-metric. It is what `cov(â)` from Method K should empirically
    match when sigma² → 0.
    """
    if W.ndim != 2:
        raise ValueError("W must be 2D")
    d_out = W.shape[0]
    SWt = Sigma_a @ W.transpose(0, 1)
    G = W @ SWt
    eye = torch.eye(d_out, dtype=W.dtype, device=W.device)
    G_inv = torch.linalg.inv(G + ridge * eye)
    return SWt @ G_inv @ SWt.transpose(0, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_2d_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Both inputs must be 2D (N, d); got shapes {tuple(a.shape)}, {tuple(b.shape)}"
        )
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )


def _empirical_cov(x: torch.Tensor) -> torch.Tensor:
    """Empirical covariance with N-1 normalization. Returns (d, d)."""
    n = x.shape[0]
    if n < 2:
        return torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1) @ centered / (n - 1)


def _gaussian_kl_one_way(
    mu1: torch.Tensor, S1: torch.Tensor,
    mu2: torch.Tensor, S2: torch.Tensor,
) -> float:
    """KL(N(mu1, S1) || N(mu2, S2)) for SPD S1, S2 (must already be ridged)."""
    d = mu1.shape[0]
    # Use Cholesky of S2 for trace and log-det terms
    L2 = torch.linalg.cholesky(S2)
    # tr(S2^-1 S1): solve S2 X = S1 (via cholesky_solve), then trace
    X = torch.cholesky_solve(S1, L2)
    trace_term = X.diagonal().sum()
    # (mu2 - mu1)^T S2^-1 (mu2 - mu1)
    diff = (mu2 - mu1).unsqueeze(-1)
    quad = (diff.T @ torch.cholesky_solve(diff, L2)).squeeze()
    # log(det S2 / det S1) via log-det of Cholesky factors
    L1 = torch.linalg.cholesky(S1)
    logdet_S1 = 2.0 * torch.log(L1.diagonal()).sum()
    logdet_S2 = 2.0 * torch.log(L2.diagonal()).sum()
    return 0.5 * (trace_term + quad - d + logdet_S2 - logdet_S1).item()


def _matrix_sqrt_psd(S: torch.Tensor) -> torch.Tensor:
    """Principal square root of an SPD matrix via eigendecomposition.
    Negative eigenvalues from numerical error are clamped to zero.
    """
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals_sqrt = eigvals.clamp(min=0.0).sqrt()
    return (eigvecs * eigvals_sqrt) @ eigvecs.transpose(-1, -2)


def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1D tensors."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Pearson inputs must be 1D")
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    den = (x_c.norm() * y_c.norm()).clamp(min=1e-30)
    return (num / den).item()


def _effective_rank(eigvals: torch.Tensor) -> float:
    """exp(entropy of normalized eigenvalues). Bounded above by len(eigvals).
    A measure of how many 'effective' dimensions the covariance occupies.
    """
    pos = eigvals.clamp(min=0.0)
    s = pos.sum()
    if s.item() <= 0:
        return 0.0
    p = pos / s
    p_clipped = p.clamp(min=1e-30)
    entropy = -(p_clipped * torch.log(p_clipped)).sum()
    return torch.exp(entropy).item()
