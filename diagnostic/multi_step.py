"""
diagnostic/multi_step.py
========================

Multi-step back-target chain - the real H3 test.

Phase 1's single-layer run_drift_experiment cannot distinguish Method N
from Method K on functional grounds because both methods enforce the
per-layer constraint by construction. The real test is back-propagating
through k > 1 layers without ground-truth reset: each step's recovered
a_hat becomes the next step's a_post target.

Optional fixes for the chain's prior-target mismatch:

  correct_target_mean
      Before feeding step k's recovered a_hat as step k+1's target,
      shift its empirical mean to match the forward-pass mean at that
      level. Mean correction only.

  correct_target_cov  (requires correct_target_mean=True)
      Apply full moment matching: whitening by the recovered output's
      empirical covariance, then recoloring by the forward-pass
      covariance. Matches both mean AND second-moment structure of
      the target distribution.

  n_passes > 1
      Run the chain n_passes times. Pass 1 uses the forward-pass prior.
      Pass k>1 uses pass (k-1)'s empirical output statistics as the prior
      at each step (except step 0).

  Hybrid
      Any combination of the above.

Method N never uses a prior; n_passes short-circuits to 1 for N.
Target corrections are still applied to N's chained targets so the
comparison stays fair across methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from diagnostic.capture import collect_activations
from diagnostic.inversion import invert_layer
from diagnostic.metrics import (
    cosine_similarity_per_sample,
    covariance_frobenius,
    gaussian_kl_symmetric,
    mean_drift,
    relative_l2_error,
    wasserstein2_gaussian,
)


@dataclass
class ChainStepReport:
    chain_idx: int
    method: str
    layer_idx: int
    layer_label: str
    d_in: int
    d_out: int
    n_samples: int
    sample_rel_err_mean: float
    sample_rel_err_p95: float
    cos_sim_mean: float
    mean_drift: float
    cov_frob: float
    gauss_kl_sym: float
    w2_gauss: float
    cov_frob_growth: Optional[float] = None
    mean_drift_growth: Optional[float] = None
    dead_unit_fraction: float = 0.0
    activation_name: str = "None"
    n_passes: int = 1
    correct_target_mean: bool = False
    correct_target_cov: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _empirical_mean_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0)
    centered = x - mu.unsqueeze(0)
    n = max(x.shape[0] - 1, 1)
    Sigma = centered.T @ centered / n
    return mu, Sigma


def _shift_mean_to(x: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
    """Re-center x so its empirical mean equals target_mean."""
    cur_mean = x.mean(dim=0)
    return x + (target_mean - cur_mean).unsqueeze(0)


def _moment_match(
    x: torch.Tensor,
    target_mean: torch.Tensor,
    target_Sigma: torch.Tensor,
    rank_eps: float = 1e-8,
) -> torch.Tensor:
    """Transform x to have empirical mean = target_mean and covariance
    structure matching target_Sigma (rescaled within x's effective rank).

    Whitening + recoloring:
        1. Center x by its own mean.
        2. Whiten by source covariance (pseudo-inverse of source Sigma^(1/2)).
        3. Recolor by target Sigma^(1/2).
        4. Add target_mean.

    For rank-deficient source covariance (which K's recovered outputs always
    are), the whitening uses a pseudo-inverse — directions with
    near-zero source variance get zero contribution. The result has rank
    bounded by min(rank(source_Sigma), rank(target_Sigma)). This means the
    transformation reshapes existing variance but does NOT inflate rank.

    Parameters
    ----------
    x : (n, d) input batch
    target_mean : (d,) desired empirical mean
    target_Sigma : (d, d) desired empirical covariance
    rank_eps : Eigenvalue threshold for source-covariance pseudo-inverse.
        Eigenvalues at or below rank_eps * max_eigenvalue are treated as zero.
    """
    n, d = x.shape
    source_mu, source_Sigma = _empirical_mean_cov(x)
    centered = x - source_mu.unsqueeze(0)

    # Source whitening matrix via eigendecomposition
    eig_s, vec_s = torch.linalg.eigh(
        (source_Sigma + source_Sigma.T) * 0.5
    )
    max_eig_s = eig_s.abs().max().item()
    eig_threshold = rank_eps * max(max_eig_s, 1e-30)
    # Inverse sqrt for non-null directions, zero for null directions
    inv_sqrt = torch.where(
        eig_s > eig_threshold,
        1.0 / torch.sqrt(eig_s.clamp(min=eig_threshold)),
        torch.zeros_like(eig_s),
    )
    W_whiten = vec_s @ torch.diag(inv_sqrt) @ vec_s.T  # (d, d)

    # Target recoloring matrix
    eig_t, vec_t = torch.linalg.eigh(
        (target_Sigma + target_Sigma.T) * 0.5
    )
    sqrt_eig_t = torch.sqrt(eig_t.clamp(min=0.0))
    W_color = vec_t @ torch.diag(sqrt_eig_t) @ vec_t.T  # (d, d)

    # Compose: x_new = (centered @ W_whiten.T) @ W_color.T + target_mean
    transformed = centered @ W_whiten.T @ W_color.T
    return transformed + target_mean.unsqueeze(0)


def run_multi_step_chain(
    model: nn.Module,
    layers_descending: List[nn.Linear],
    dataloader: Iterable,
    *,
    methods: Tuple[str, ...] = ("naive", "kfac_a"),
    eps: float = 1e-4,
    sigma2: Optional[float] = None,
    max_samples: int = 4096,
    seq_subsample: int = 4096,
    device: Optional[torch.device] = None,
    forward_fn: Optional[Callable] = None,
    layer_labels: Optional[List[str]] = None,
    activation_overrides: Optional[Dict[nn.Linear, Optional[nn.Module]]] = None,
    correct_target_mean: bool = False,
    correct_target_cov: bool = False,
    n_passes: int = 1,
) -> List[ChainStepReport]:
    if not layers_descending:
        raise ValueError("layers_descending must be non-empty")
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    if correct_target_cov and not correct_target_mean:
        raise ValueError(
            "correct_target_cov=True requires correct_target_mean=True"
        )
    if layer_labels is None:
        layer_labels = [str(l) for l in layers_descending]
    if len(layer_labels) != len(layers_descending):
        raise ValueError(
            f"layer_labels has {len(layer_labels)} entries; "
            f"expected {len(layers_descending)}"
        )

    captured = collect_activations(
        model, dataloader, layers_descending,
        max_samples=max_samples, seq_subsample=seq_subsample,
        device=device, forward_fn=forward_fn,
        activation_overrides=activation_overrides,
    )

    # Forward-pass priors per step
    forward_priors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for step_idx, layer in enumerate(layers_descending):
        a_in = captured[layer]["a_in"].double()
        forward_priors[step_idx] = _empirical_mean_cov(a_in)

    reports: List[ChainStepReport] = []

    for method in methods:
        effective_passes = 1 if method == "naive" else n_passes
        prior_overrides: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for pass_idx in range(effective_passes):
            is_final_pass = (pass_idx == effective_passes - 1)
            prev_cov_frob: Optional[float] = None
            prev_mean_drift: Optional[float] = None
            cur_target_a_post: Optional[torch.Tensor] = None
            this_pass_recoveries: Dict[int, torch.Tensor] = {}

            for step_idx, layer in enumerate(layers_descending):
                cap = captured[layer]
                a_in_truth = cap["a_in"].double()
                a_pre_truth = cap["a_pre"].double()
                a_post_truth = cap["a_post"].double()
                activation = cap["activation"]

                # Choose prior
                if step_idx == 0:
                    mu_a, Sigma_a = forward_priors[step_idx]
                elif step_idx in prior_overrides:
                    mu_a, Sigma_a = prior_overrides[step_idx]
                else:
                    mu_a, Sigma_a = forward_priors[step_idx]

                target_device = a_in_truth.device
                target_dtype = a_in_truth.dtype
                W = layer.weight.detach().to(
                    device=target_device, dtype=target_dtype)
                b = (
                    layer.bias.detach().to(
                        device=target_device, dtype=target_dtype)
                    if layer.bias is not None else None
                )

                if step_idx == 0:
                    cur_target_a_post = a_post_truth.clone()

                if method == "naive":
                    a_hat = invert_layer(
                        W, b, cur_target_a_post,
                        a_pre_forward=a_pre_truth,
                        activation=activation,
                        method="naive", eps=eps,
                    )
                else:
                    a_hat = invert_layer(
                        W, b, cur_target_a_post,
                        a_pre_forward=a_pre_truth,
                        activation=activation,
                        method="kfac_a",
                        mu_a=mu_a, Sigma_a=Sigma_a, sigma2=sigma2,
                    )

                if is_final_pass:
                    rel_l2 = relative_l2_error(a_hat, a_in_truth)
                    cos = cosine_similarity_per_sample(a_hat, a_in_truth)
                    md_val = mean_drift(a_hat, a_in_truth)
                    cf_val = covariance_frobenius(a_hat, a_in_truth)
                    try:
                        kl_val = gaussian_kl_symmetric(a_hat, a_in_truth)
                    except Exception:
                        kl_val = float("nan")
                    try:
                        w2_val = wasserstein2_gaussian(a_hat, a_in_truth)
                    except Exception:
                        w2_val = float("nan")

                    cov_growth = None if prev_cov_frob is None else (
                        cf_val / prev_cov_frob if prev_cov_frob > 0
                        else float("inf"))
                    mean_growth = None if prev_mean_drift is None else (
                        md_val / prev_mean_drift if prev_mean_drift > 0
                        else float("inf"))

                    if isinstance(activation, (nn.ReLU, nn.ReLU6)):
                        dead_frac = (a_post_truth == 0).to(
                            torch.float32).mean().item()
                    else:
                        dead_frac = 0.0

                    reports.append(ChainStepReport(
                        chain_idx=step_idx,
                        method=method,
                        layer_idx=step_idx,
                        layer_label=layer_labels[step_idx],
                        d_in=int(a_in_truth.shape[1]),
                        d_out=int(a_pre_truth.shape[1]),
                        n_samples=int(a_in_truth.shape[0]),
                        sample_rel_err_mean=float(rel_l2.mean().item()),
                        sample_rel_err_p95=float(rel_l2.quantile(0.95).item()),
                        cos_sim_mean=float(cos.mean().item()),
                        mean_drift=md_val,
                        cov_frob=cf_val,
                        gauss_kl_sym=kl_val,
                        w2_gauss=w2_val,
                        cov_frob_growth=cov_growth,
                        mean_drift_growth=mean_growth,
                        dead_unit_fraction=dead_frac,
                        activation_name=(
                            type(activation).__name__
                            if activation is not None else "None"),
                        n_passes=effective_passes,
                        correct_target_mean=correct_target_mean,
                        correct_target_cov=correct_target_cov,
                    ))
                    prev_cov_frob = cf_val
                    prev_mean_drift = md_val

                this_pass_recoveries[step_idx] = a_hat.detach().clone()

                # Prepare target for next step
                next_target = a_hat
                if correct_target_mean and step_idx + 1 < len(layers_descending):
                    target_mean_for_correction = a_in_truth.mean(dim=0)
                    if correct_target_cov:
                        # Full moment matching: mean + covariance
                        _, target_Sigma_for_correction = _empirical_mean_cov(a_in_truth)
                        next_target = _moment_match(
                            a_hat,
                            target_mean_for_correction,
                            target_Sigma_for_correction,
                        )
                    else:
                        # Mean correction only
                        next_target = _shift_mean_to(a_hat, target_mean_for_correction)
                cur_target_a_post = next_target

            # End of pass: build prior_overrides for the next pass
            for step_idx_, a_hat_ in this_pass_recoveries.items():
                if step_idx_ == 0:
                    continue
                mu_o, Sigma_o = _empirical_mean_cov(a_hat_)
                prior_overrides[step_idx_] = (mu_o, Sigma_o)

    return reports
