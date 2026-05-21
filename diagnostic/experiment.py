"""
diagnostic/experiment.py
========================

Top-level driver for the inversion-drift diagnostic.

Ties together:
  - diagnostic.capture     : forward-pass activation capture
  - diagnostic.inversion   : Method N and Method K back-target inversion
  - diagnostic.metrics     : drift metrics (per-sample, distributional, functional)
  - diagnostic.vered_solve : inner SPD solver (used by inversion methods)

Public API
----------
LayerDriftReport
    Dataclass holding all measurements for one (layer, method) pair.
run_drift_experiment(model, layers, dataloader, ...)
    The main driver: captures activations, runs both inversion methods per
    layer, computes drift metrics, returns a list of reports.
save_reports(reports, path)
    Persist results as JSON in the same convention as benchmark/results/*.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diagnostic.capture import collect_activations
from diagnostic.inversion import invert_layer
from diagnostic.metrics import (
    constraint_residual,
    cosine_similarity_per_sample,
    covariance_frobenius,
    downstream_loss,
    eigenvalue_spectrum_match,
    gaussian_kl_symmetric,
    mean_drift,
    predicted_projection_covariance,
    principal_subspace_angles,
    relative_l2_error,
    wasserstein2_gaussian,
)


# ---------------------------------------------------------------------------
# Per-layer drift report
# ---------------------------------------------------------------------------

@dataclass
class LayerDriftReport:
    """All measurements for one (layer, method) pair on one batch.

    Numeric fields are kept as plain Python floats for JSON-friendliness.
    """

    # Identification
    layer_idx: int
    layer_label: str
    d_in: int
    d_out: int
    method: str                    # "naive" | "kfac_a"

    # Hyperparameters
    eps: float
    sigma2: Optional[float]
    n_samples: int                 # rows used in this analysis

    # Per-sample fidelity
    sample_rel_err_mean: float
    sample_rel_err_p95: float
    cos_sim_mean: float
    cos_sim_p5: float

    # Distributional fidelity (Tier 1)
    mean_drift: float
    cov_frob: float
    gauss_kl_sym: float
    w2_gauss: float

    # Spectral diagnosis (Tier 2)
    eig_pearson: float
    eig_kl_on_spectrum: float
    effective_rank_hat: float
    effective_rank_star: float
    subspace_angle_top1: float
    subspace_angle_top5: float

    # Functional fidelity (Tier 3) — populated when remaining_forward_fn given
    logit_mse: Optional[float] = None
    logit_mse_abs: Optional[float] = None
    prediction_agreement: Optional[float] = None
    ce_drift: Optional[float] = None

    # Sanity tests
    constraint_residual: float = 0.0
    cov_predicted_match: Optional[float] = None  # ||emp_cov - predicted_cov|| / ||predicted_cov||

    # Per-layer context
    dead_unit_fraction: float = 0.0   # for ReLU layers
    activation_name: str = "None"

    # Free-form notes / debug info
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-(layer, method) analysis
# ---------------------------------------------------------------------------

def _analyze_one(
    *,
    layer: nn.Linear,
    layer_idx: int,
    layer_label: str,
    a_in: torch.Tensor,
    a_pre: torch.Tensor,
    a_post: torch.Tensor,
    activation: Optional[nn.Module],
    method: str,
    eps: float,
    sigma2: Optional[float],
    use_predicted_cov_check: bool,
    remaining_forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    labels: Optional[torch.Tensor],
) -> LayerDriftReport:
    """Run one inversion method on one layer and collect all drift metrics."""
    # Move W and b to the same device + dtype as the (CPU-stored) activations
    target_device = a_in.device
    target_dtype = a_in.dtype
    W = layer.weight.detach().to(device=target_device, dtype=target_dtype)
    b = (
        layer.bias.detach().to(device=target_device, dtype=target_dtype)
        if layer.bias is not None else None
    )

    # Prior estimates from forward-pass activations
    mu_a = a_in.mean(dim=0)
    # Centered empirical covariance with n-1 normalization
    centered = a_in - mu_a.unsqueeze(0)
    Sigma_a = centered.T @ centered / max(a_in.shape[0] - 1, 1)

    # Run inversion: target is the forward-pass post-activation
    if method == "naive":
        a_hat = invert_layer(
            W, b, a_post, a_pre_forward=a_pre, activation=activation,
            method="naive", eps=eps,
        )
    elif method == "kfac_a":
        a_hat = invert_layer(
            W, b, a_post, a_pre_forward=a_pre, activation=activation,
            method="kfac_a", mu_a=mu_a, Sigma_a=Sigma_a, sigma2=sigma2,
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Per-sample fidelity
    rel_l2 = relative_l2_error(a_hat, a_in)
    cos = cosine_similarity_per_sample(a_hat, a_in)

    # Distributional metrics
    md = mean_drift(a_hat, a_in)
    cf = covariance_frobenius(a_hat, a_in)
    kl = gaussian_kl_symmetric(a_hat, a_in)
    w2 = wasserstein2_gaussian(a_hat, a_in)

    # Spectral diagnosis
    spec = eigenvalue_spectrum_match(a_hat, a_in)
    k1_angles = principal_subspace_angles(a_hat, a_in, k=1)
    k5_top = min(5, a_in.shape[1])
    k5_angles = principal_subspace_angles(a_hat, a_in, k=k5_top)

    # Sanity: constraint residual on the pre-activation
    # Reconstruct t_pre from inverse-activation and compare
    from diagnostic.inversion import invert_activation
    t_pre = invert_activation(a_post, a_pre, activation)
    constraint_res = constraint_residual(W, b, a_hat, t_pre)

    # Sanity: predicted projection covariance match (for Method K only and
    # only if requested — it's an O(d^3) inversion-based check)
    cov_predicted_match: Optional[float] = None
    if use_predicted_cov_check and method == "kfac_a":
        S_pred = predicted_projection_covariance(Sigma_a, W)
        centered_hat = a_hat - a_hat.mean(dim=0, keepdim=True)
        S_emp = centered_hat.T @ centered_hat / max(a_hat.shape[0] - 1, 1)
        denom = S_pred.norm(p="fro").clamp(min=1e-30)
        cov_predicted_match = ((S_emp - S_pred).norm(p="fro") / denom).item()

    # Functional fidelity
    logit_mse = logit_mse_abs = agreement = ce_drift = None
    if remaining_forward_fn is not None:
        try:
            fr = downstream_loss(a_hat, a_in, remaining_forward_fn, labels=labels)
            logit_mse = fr["logit_mse"]
            logit_mse_abs = fr["logit_mse_abs"]
            agreement = fr["prediction_agreement"]
            ce_drift = fr.get("ce_drift")
        except Exception as exc:
            # Don't kill the experiment over a downstream failure; record extras
            logit_mse = logit_mse_abs = agreement = float("nan")
            extras_err = str(exc)
        else:
            extras_err = None
    else:
        extras_err = None

    # Dead-unit fraction (for ReLU-like activations)
    if isinstance(activation, (nn.ReLU, nn.ReLU6)):
        dead_frac = (a_post == 0).to(torch.float32).mean().item()
    else:
        dead_frac = 0.0

    extras: Dict[str, Any] = {}
    if extras_err:
        extras["downstream_error"] = extras_err

    return LayerDriftReport(
        layer_idx=layer_idx,
        layer_label=layer_label,
        d_in=int(a_in.shape[1]),
        d_out=int(a_pre.shape[1]),
        method=method,
        eps=float(eps),
        sigma2=None if sigma2 is None else float(sigma2),
        n_samples=int(a_in.shape[0]),
        sample_rel_err_mean=float(rel_l2.mean().item()),
        sample_rel_err_p95=float(rel_l2.quantile(0.95).item()),
        cos_sim_mean=float(cos.mean().item()),
        cos_sim_p5=float(cos.quantile(0.05).item()),
        mean_drift=md,
        cov_frob=cf,
        gauss_kl_sym=kl,
        w2_gauss=w2,
        eig_pearson=spec["pearson"],
        eig_kl_on_spectrum=spec["kl_on_spectrum"],
        effective_rank_hat=spec["effective_rank_hat"],
        effective_rank_star=spec["effective_rank_star"],
        subspace_angle_top1=float(k1_angles[0].item()),
        subspace_angle_top5=float(k5_angles.mean().item()),
        logit_mse=logit_mse,
        logit_mse_abs=logit_mse_abs,
        prediction_agreement=agreement,
        ce_drift=ce_drift,
        constraint_residual=float(constraint_res),
        cov_predicted_match=cov_predicted_match,
        dead_unit_fraction=float(dead_frac),
        activation_name=type(activation).__name__ if activation is not None else "None",
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_drift_experiment(
    model: nn.Module,
    layers: List[nn.Linear],
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
    remaining_forward_fns: Optional[Dict[nn.Linear, Callable]] = None,
    labels_per_layer: Optional[Dict[nn.Linear, torch.Tensor]] = None,
    use_predicted_cov_check: bool = True,
) -> List[LayerDriftReport]:
    """Run the inversion-drift experiment for each (layer, method) pair.

    Steps:
    1. Collect raw activations for all requested layers via forward hooks.
    2. For each layer × method:
        a. Compute the prior (mu_a, Sigma_a) from captured activations.
        b. Invert layer (a_post -> a_hat).
        c. Compute all drift metrics comparing a_hat to ground-truth a_in.
        d. Compute sanity-check metrics (constraint residual, predicted-cov match).
    3. Optionally compute functional metrics if remaining_forward_fns provided.
    4. Return a list of LayerDriftReport — one per (layer, method) pair.

    Parameters
    ----------
    model : The network to analyze.
    layers : nn.Linear modules to invert.
    dataloader : Iterable of batches for activation capture.
    methods : Which inversion methods to run. Default both.
    eps, sigma2 : Damping params for Method N and Method K respectively.
    max_samples, seq_subsample : Caps for activation capture.
    device : Device for forward passes (defaults to model's device).
    forward_fn : Optional custom forward call (for BERT-style batches).
    layer_labels : Optional list of human-readable labels, one per layer.
        Defaults to ``str(layer)``.
    activation_overrides : Force a particular activation for a layer
        (overrides auto-detection).
    remaining_forward_fns : Per-layer function that runs the rest of the
        network from that layer's output forward to model output. Used for
        functional-fidelity metrics. If absent, those metrics are None.
    labels_per_layer : Per-layer ground-truth labels for CE drift computation.
    use_predicted_cov_check : If True, run the predicted-projection-cov
        sanity test for Method K (O(d^3) per layer, adds a few seconds at
        BERT scale).

    Returns
    -------
    List[LayerDriftReport] with len(layers) * len(methods) entries, sorted
    first by layer order then by method.
    """
    if layer_labels is None:
        layer_labels = [str(l) for l in layers]
    if len(layer_labels) != len(layers):
        raise ValueError(
            f"layer_labels has {len(layer_labels)} entries; expected {len(layers)}"
        )

    # 1. Capture activations once
    captured = collect_activations(
        model, dataloader, layers,
        max_samples=max_samples, seq_subsample=seq_subsample,
        device=device, forward_fn=forward_fn,
        activation_overrides=activation_overrides,
    )

    # 2 & 3. Per-layer × per-method analysis
    reports: List[LayerDriftReport] = []
    for layer_idx, layer in enumerate(layers):
        captured_layer = captured[layer]
        a_in = captured_layer["a_in"]
        a_pre = captured_layer["a_pre"]
        a_post = captured_layer["a_post"]
        activation = captured_layer["activation"]
        label = layer_labels[layer_idx]
        rem_fn = (remaining_forward_fns or {}).get(layer)
        lbl = (labels_per_layer or {}).get(layer)
        for method in methods:
            report = _analyze_one(
                layer=layer,
                layer_idx=layer_idx,
                layer_label=label,
                a_in=a_in.double(),
                a_pre=a_pre.double(),
                a_post=a_post.double(),
                activation=activation,
                method=method,
                eps=eps,
                sigma2=sigma2,
                use_predicted_cov_check=use_predicted_cov_check,
                remaining_forward_fn=rem_fn,
                labels=lbl,
            )
            reports.append(report)
    return reports



# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_reports(
    reports: List[LayerDriftReport],
    path: Union[str, Path],
    *,
    model_name: str = "unknown",
    task: str = "unknown",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a list of LayerDriftReport to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import datetime
    payload: Dict[str, Any] = {
        "model": model_name,
        "task": task,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "n_reports": len(reports),
        "reports": [r.to_dict() for r in reports],
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for non-standard types."""
    import math
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_reports(path: Union[str, Path]) -> List[LayerDriftReport]:
    """Load a previously-saved JSON file back into LayerDriftReport objects."""
    path = Path(path)
    with open(path) as fh:
        payload = json.load(fh)
    return [LayerDriftReport(**r) for r in payload["reports"]]
