"""
diagnostic/plotting.py
======================

Matplotlib helpers for visualizing inversion-drift experiment results.

All plotters take a list of LayerDriftReport (from diagnostic.experiment) or
raw torch tensors and write PNG files. Uses the Agg backend so it works in
headless contexts (CI, remote benchmarks).

Plot catalog
------------
plot_drift_vs_depth        — one panel per drift metric, two curves (N vs K)
                              indexed by layer depth.
plot_eigenvalue_spectrum   — log-scale eigenvalue overlay (â vs a*) for one layer.
plot_per_sample_error_histogram — distribution of per-sample relative L2
                              error, Method N vs Method K overlaid.
plot_drift_vs_dim_ratio    — drift metric as a function of d_in/d_out ratio.
                              Tests hypothesis H4: more lossy layers → more
                              drift.

All plotters return the matplotlib Figure for further customization, and
also save to disk if out_path is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # Headless-safe; must come before pyplot import.
import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnostic.experiment import LayerDriftReport


# ---------------------------------------------------------------------------
# Visual conventions
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "naive": "#d62728",   # red
    "kfac_a": "#1f77b4",  # blue
}
METHOD_LABELS = {
    "naive": "Method N (naive pseudo-inverse)",
    "kfac_a": "Method K (K-FAC-A regularized)",
}
METHOD_MARKERS = {
    "naive": "o",
    "kfac_a": "s",
}


# ---------------------------------------------------------------------------
# Plot 1 — drift vs layer depth (one metric per call)
# ---------------------------------------------------------------------------

# Metrics with their human-readable axis labels and log-y advice
_METRIC_META = {
    "sample_rel_err_mean":  ("per-sample rel L2 (mean)", True),
    "sample_rel_err_p95":   ("per-sample rel L2 (p95)",  True),
    "cos_sim_mean":         ("cosine similarity (mean)", False),
    "mean_drift":           ("mean drift (relative)",    True),
    "cov_frob":             ("covariance Frobenius (rel)", True),
    "gauss_kl_sym":         ("Gaussian symmetric KL",    True),
    "w2_gauss":             ("2-Wasserstein² (Gaussian fit)", True),
    "eig_pearson":          ("eigenvalue spectrum Pearson r", False),
    "eig_kl_on_spectrum":   ("KL on sorted spectrum",    True),
    "effective_rank_hat":   ("effective rank (â)",       False),
    "effective_rank_star":  ("effective rank (a*)",      False),
    "subspace_angle_top1":  ("principal subspace angle (top 1)", False),
    "subspace_angle_top5":  ("principal subspace angle (top 5 mean)", False),
    "constraint_residual":  ("constraint residual ||Wâ+b−t_pre||",   True),
    "cov_predicted_match":  ("Σ̂ vs predicted Σ projection (rel)",     True),
    "logit_mse":            ("downstream logit MSE (rel)",            True),
    "logit_mse_abs":        ("downstream logit MSE (abs)",            True),
    "prediction_agreement": ("prediction agreement", False),
    "ce_drift":             ("CE drift (â vs a*)",   False),
    "dead_unit_fraction":   ("dead unit fraction",   False),
}


def plot_drift_vs_depth(
    reports: List[LayerDriftReport],
    metric: str,
    out_path: Optional[Union[str, Path]] = None,
    *,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> "plt.Figure":
    """Plot a drift metric as a function of layer index, with one curve per method.

    Parameters
    ----------
    reports : List of LayerDriftReport. Should contain at least one entry per
        (layer, method) pair.
    metric : Field name of LayerDriftReport to plot (e.g., 'cov_frob').
    out_path : If given, save the figure to this path.
    title : Optional plot title; defaults to "<metric> vs layer depth".
    figsize : matplotlib figure size in inches.
    """
    if metric not in _METRIC_META:
        # Allow unknown metrics — fall back to default formatting
        ylabel = metric
        use_log_y = False
    else:
        ylabel, use_log_y = _METRIC_META[metric]

    # Group reports by method, sorted by layer_idx
    by_method: dict = {}
    for r in reports:
        by_method.setdefault(r.method, []).append(r)
    for method, rs in by_method.items():
        rs.sort(key=lambda r: r.layer_idx)

    fig, ax = plt.subplots(figsize=figsize)
    for method, rs in by_method.items():
        xs = [r.layer_idx for r in rs]
        ys = [getattr(r, metric) for r in rs]
        # Filter out None values (e.g., functional metrics when not computed)
        xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not xy:
            continue
        xs2, ys2 = zip(*xy)
        ax.plot(
            xs2, ys2,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            marker=METHOD_MARKERS.get(method, "x"),
            linewidth=2, markersize=8,
        )

    ax.set_xlabel("layer index (in retraining order)")
    ax.set_ylabel(ylabel)
    if use_log_y:
        # Only set log if all values are positive
        all_y = [getattr(r, metric) for rs in by_method.values() for r in rs
                 if getattr(r, metric) is not None]
        if all_y and all(y > 0 for y in all_y):
            ax.set_yscale("log")
    ax.set_title(title or f"{metric} vs layer depth")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — eigenvalue spectrum overlay
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    a_hat: torch.Tensor,
    a_star: torch.Tensor,
    out_path: Optional[Union[str, Path]] = None,
    *,
    title: Optional[str] = None,
    method_name: str = "method",
    figsize: tuple = (8, 5),
) -> "plt.Figure":
    """Log-scale overlay of empirical covariance eigenvalues.

    Visualizes whether the inversion method preserves or collapses the
    eigenvalue tail relative to the ground-truth activations.
    """
    def _cov_eigvals(x: torch.Tensor) -> np.ndarray:
        n = x.shape[0]
        if n < 2:
            return np.zeros(x.shape[1])
        c = x - x.mean(dim=0, keepdim=True)
        S = c.T @ c / (n - 1)
        return torch.linalg.eigvalsh(S).clamp(min=0).cpu().numpy()

    eig_hat = np.sort(_cov_eigvals(a_hat))[::-1]
    eig_star = np.sort(_cov_eigvals(a_star))[::-1]
    ranks = np.arange(1, len(eig_star) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    # Tiny floor so log-scale doesn't choke on zero eigenvalues
    floor = max(eig_star.max() * 1e-12, 1e-30) if eig_star.max() > 0 else 1e-30
    ax.semilogy(ranks, np.maximum(eig_star, floor),
                label="a* (ground truth)", color="black", linewidth=2)
    ax.semilogy(ranks, np.maximum(eig_hat, floor),
                label=f"â ({method_name})",
                color=METHOD_COLORS.get(method_name, "tab:orange"),
                linewidth=2, linestyle="--")
    ax.set_xlabel("rank (eigenvalue index, sorted descending)")
    ax.set_ylabel("eigenvalue magnitude")
    ax.set_title(title or "Eigenvalue spectrum overlay")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — per-sample error histogram
# ---------------------------------------------------------------------------

def plot_per_sample_error_histogram(
    errs_naive: torch.Tensor,
    errs_kfac: torch.Tensor,
    out_path: Optional[Union[str, Path]] = None,
    *,
    title: Optional[str] = None,
    bins: int = 50,
    figsize: tuple = (8, 5),
) -> "plt.Figure":
    """Histogram of per-sample relative L2 errors, both methods overlaid.

    Useful for spotting heavy-tail differences: even if mean error is similar,
    K may have lower p95 / max error if the prior helps with worst cases.
    """
    errs_naive_np = errs_naive.cpu().numpy()
    errs_kfac_np = errs_kfac.cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    # Shared bin edges
    all_errs = np.concatenate([errs_naive_np, errs_kfac_np])
    max_e = float(np.percentile(all_errs, 99.5))  # clip extreme tail for readability
    edges = np.linspace(0, max_e, bins + 1)
    ax.hist(errs_naive_np, bins=edges, alpha=0.5,
            color=METHOD_COLORS["naive"], label=METHOD_LABELS["naive"])
    ax.hist(errs_kfac_np, bins=edges, alpha=0.5,
            color=METHOD_COLORS["kfac_a"], label=METHOD_LABELS["kfac_a"])
    ax.set_xlabel("per-sample relative L2 error")
    ax.set_ylabel("count")
    ax.set_title(title or "Per-sample error distribution")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — drift vs dimension ratio (H4 test)
# ---------------------------------------------------------------------------

def plot_drift_vs_dim_ratio(
    reports: List[LayerDriftReport],
    metric: str = "cov_frob",
    out_path: Optional[Union[str, Path]] = None,
    *,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> "plt.Figure":
    """Scatter of a drift metric vs d_in/d_out ratio of each layer.

    Tests hypothesis H4: drift should grow with the aspect ratio
    (more under-determined → more reliance on the prior in the free directions).
    """
    ylabel, use_log_y = _METRIC_META.get(metric, (metric, False))

    by_method: dict = {}
    for r in reports:
        by_method.setdefault(r.method, []).append(r)

    fig, ax = plt.subplots(figsize=figsize)
    for method, rs in by_method.items():
        xs = [r.d_in / max(r.d_out, 1) for r in rs]
        ys = [getattr(r, metric) for r in rs]
        xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not xy:
            continue
        xs2, ys2 = zip(*xy)
        ax.scatter(
            xs2, ys2,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            marker=METHOD_MARKERS.get(method, "x"),
            s=80, alpha=0.8, edgecolors="black",
        )

    ax.set_xlabel("d_in / d_out (aspect ratio)")
    ax.set_ylabel(ylabel)
    if use_log_y:
        all_y = [getattr(r, metric) for rs in by_method.values() for r in rs
                 if getattr(r, metric) is not None]
        if all_y and all(y > 0 for y in all_y):
            ax.set_yscale("log")
    ax.set_title(title or f"{metric} vs aspect ratio (H4 test)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate the full default plot suite
# ---------------------------------------------------------------------------

def generate_default_plots(
    reports: List[LayerDriftReport],
    out_dir: Union[str, Path],
    *,
    raw_per_layer: Optional[dict] = None,
) -> List[Path]:
    """Generate the standard set of plots for an experiment report and write
    them to `out_dir`.

    Produces:
      - drift_vs_depth__<metric>.png for each of the core drift metrics
      - drift_vs_dim_ratio.png
      - (optionally) eigenvalue_spectrum_layer<i>.png and per_sample_error_histogram_layer<i>.png
        if raw_per_layer is supplied. raw_per_layer should be a dict
        layer_idx -> {'a_in': tensor, 'a_hat_naive': tensor, 'a_hat_kfac': tensor,
                      'errs_naive': tensor, 'errs_kfac': tensor}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    core_metrics = [
        "cov_frob", "mean_drift", "gauss_kl_sym", "w2_gauss",
        "sample_rel_err_mean", "eig_pearson", "effective_rank_hat",
        "logit_mse", "prediction_agreement",
    ]
    for m in core_metrics:
        # Skip if no report has that metric populated
        vals = [getattr(r, m) for r in reports if getattr(r, m, None) is not None]
        if not vals:
            continue
        p = out_dir / f"drift_vs_depth__{m}.png"
        fig = plot_drift_vs_depth(reports, m, out_path=p)
        plt.close(fig)
        produced.append(p)

    p = out_dir / "drift_vs_dim_ratio.png"
    fig = plot_drift_vs_dim_ratio(reports, metric="cov_frob", out_path=p)
    plt.close(fig)
    produced.append(p)

    if raw_per_layer is not None:
        for layer_idx, data in raw_per_layer.items():
            if "a_hat_kfac" in data and "a_in" in data:
                p = out_dir / f"eigenvalue_spectrum_layer{layer_idx}.png"
                fig = plot_eigenvalue_spectrum(
                    data["a_hat_kfac"], data["a_in"],
                    out_path=p, method_name="kfac_a",
                    title=f"Eigenvalue spectrum — layer {layer_idx}",
                )
                plt.close(fig)
                produced.append(p)
            if "errs_naive" in data and "errs_kfac" in data:
                p = out_dir / f"per_sample_error_histogram_layer{layer_idx}.png"
                fig = plot_per_sample_error_histogram(
                    data["errs_naive"], data["errs_kfac"],
                    out_path=p,
                    title=f"Per-sample error distribution — layer {layer_idx}",
                )
                plt.close(fig)
                produced.append(p)
    return produced
