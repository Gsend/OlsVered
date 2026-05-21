"""
diagnostic/target_prop_retrainer.py
====================================

Target-propagation + per-layer OLS retraining.

Single-pass version:
  1. Capture forward activations on a batch of data.
  2. Back-propagate a target through the layers being retrained, using
     either Method K (K-FAC-A) or Method N (naive pseudo-inverse) for the
     inversion, with optional moment-matching correction applied at each
     chained step.
  3. For each layer being retrained, solve OLS to find new weights:
        W_new = argmin || X @ W^T - T_pre ||^2 + lambda ||W||^2
  4. Apply the new weights to the model.

Multi-pass version (n_iterations > 1):
  After the single-pass update, re-capture activations from the UPDATED
  model and repeat the back-prop + OLS-fit step. The chain's starting
  target (deepest layer's a_post) stays fixed at the ORIGINAL model's
  forward output — this drives the model back toward the original
  behavior, fixing the chain-noise distortion introduced in the first pass.

Public API
----------
RetrainerResult
    Holds the retrained model, per-layer info, and metadata.
solve_ols_layer(X, T_pre, ols_lambda)
    Pure OLS solve for one layer's weights.
retrain_via_target_prop(model, layers_to_retrain, dataloader, ...)
    End-to-end driver with optional n_iterations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from diagnostic.capture import collect_activations
from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.multi_step import (
    _empirical_mean_cov,
    _moment_match,
    _shift_mean_to,
)
from diagnostic.vered_solve import vered_solve


@dataclass
class RetrainerResult:
    """Output of retrain_via_target_prop."""
    model: nn.Module
    per_layer_info: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    method: str = "kfac_a"
    correct_target_mean: bool = False
    correct_target_cov: bool = False
    ols_lambda: float = 1e-4
    n_iterations: int = 1
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)


def solve_ols_layer(
    X: torch.Tensor,
    T_pre: torch.Tensor,
    *,
    with_bias: bool = True,
    ols_lambda: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Solve per-layer OLS: argmin ||X @ W^T + b - T_pre||^2 + lambda ||W||^2."""
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d_in); got shape {tuple(X.shape)}")
    if T_pre.ndim != 2:
        raise ValueError(f"T_pre must be (n, d_out); got shape {tuple(T_pre.shape)}")
    if X.shape[0] != T_pre.shape[0]:
        raise ValueError(
            f"X and T_pre must have same n; got {X.shape[0]} vs {T_pre.shape[0]}"
        )

    n, d_in = X.shape

    if with_bias:
        ones = torch.ones(n, 1, dtype=X.dtype, device=X.device)
        X_aug = torch.cat([X, ones], dim=1)
    else:
        X_aug = X

    XtX = X_aug.T @ X_aug
    XtT = X_aug.T @ T_pre
    damping = ols_lambda * (XtX.diagonal().abs().mean().item() + 1e-30)
    Z = vered_solve(XtX, XtT, damping=damping)

    if with_bias:
        W = Z[:d_in, :].T
        b = Z[d_in, :]
    else:
        W = Z.T
        b = None
    return W, b


def _one_iteration(
    *,
    capture_model: nn.Module,
    work_layers: List[nn.Linear],
    chain_layers: List[nn.Linear],
    dataloader: Iterable,
    method: str,
    correct_target_mean: bool,
    correct_target_cov: bool,
    eps: float,
    sigma2: Optional[float],
    ols_lambda: float,
    max_samples: int,
    seq_subsample: int,
    device: Optional[torch.device],
    forward_fn: Optional[Callable],
    activation_overrides: Optional[Dict[nn.Linear, Optional[nn.Module]]],
    original_deepest_a_post: Optional[torch.Tensor] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[nn.Linear, torch.Tensor]]:
    """Run one iteration: capture, back-prop, OLS-fit, update weights in work_layers.

    Returns
    -------
    per_layer_info : Dict with diagnostics for each layer
    new_targets    : Dict mapping each chain_layer to its a_post target this
                     iteration (so caller can stash the original deepest a_post)
    """
    # 1. Capture from the (current) capture_model
    captured = collect_activations(
        capture_model, dataloader, chain_layers,
        max_samples=max_samples, seq_subsample=seq_subsample,
        device=device, forward_fn=forward_fn,
        activation_overrides=activation_overrides,
    )

    # 2. Forward priors per chain layer
    forward_priors: Dict[nn.Linear, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in chain_layers:
        a_in = captured[layer]["a_in"].double()
        forward_priors[layer] = _empirical_mean_cov(a_in)

    # 3. Back-prop chain
    targets_per_layer: Dict[nn.Linear, torch.Tensor] = {}
    cur_target_a_post: Optional[torch.Tensor] = None

    for step_idx, layer in enumerate(chain_layers):
        cap = captured[layer]
        a_in_truth = cap["a_in"].double()
        a_pre_truth = cap["a_pre"].double()
        a_post_truth = cap["a_post"].double()
        activation = cap["activation"]
        mu_a, Sigma_a = forward_priors[layer]

        if step_idx == 0:
            if original_deepest_a_post is not None:
                # Multi-iter: use the ORIGINAL model's deepest a_post.
                # Crop/pad to match current sample count if needed.
                n_orig = original_deepest_a_post.shape[0]
                n_now = a_post_truth.shape[0]
                if n_orig == n_now:
                    cur_target_a_post = original_deepest_a_post.clone()
                else:
                    # Mismatched batch — fall back to current-model a_post
                    # (rare; only if dataloader yields different counts)
                    cur_target_a_post = a_post_truth.clone()
            else:
                cur_target_a_post = a_post_truth.clone()

        targets_per_layer[layer] = cur_target_a_post.clone()

        target_device = a_in_truth.device
        target_dtype = a_in_truth.dtype
        W_old = layer.weight.detach().to(
            device=target_device, dtype=target_dtype)
        b_old = (layer.bias.detach().to(
            device=target_device, dtype=target_dtype)
            if layer.bias is not None else None)

        if method == "naive":
            a_hat = invert_layer(
                W_old, b_old, cur_target_a_post,
                a_pre_forward=a_pre_truth,
                activation=activation, method="naive", eps=eps,
            )
        else:
            a_hat = invert_layer(
                W_old, b_old, cur_target_a_post,
                a_pre_forward=a_pre_truth,
                activation=activation, method="kfac_a",
                mu_a=mu_a, Sigma_a=Sigma_a, sigma2=sigma2,
            )

        next_target = a_hat
        if correct_target_mean and step_idx + 1 < len(chain_layers):
            tgt_mean = a_in_truth.mean(dim=0)
            if correct_target_cov:
                _, tgt_Sigma = _empirical_mean_cov(a_in_truth)
                next_target = _moment_match(a_hat, tgt_mean, tgt_Sigma)
            else:
                next_target = _shift_mean_to(a_hat, tgt_mean)
        cur_target_a_post = next_target

    # 4. Per-layer OLS solve — FORWARD SWEEP (shallowest -> deepest).
    # The entry layer of the retrained chain (shallowest among retrained) uses
    # its captured a_in (output of frozen upstream or raw model input). Every
    # subsequent retrained layer uses the post-activation output of the
    # just-rebuilt previous retrained layer, NOT the captured a_in. This
    # assumes contiguous retraining chains.
    #
    # chain_layers is deepest-first ([fc4, fc3, fc2, fc1]); we iterate
    # reversed (shallowest-first: [fc1, fc2, fc3, fc4]).
    per_layer_info: Dict[int, Dict[str, Any]] = {}
    n_chain = len(chain_layers)
    prev_a_post_new: Optional[torch.Tensor] = None

    for sweep_pos in range(n_chain):
        # sweep_pos = 0 means shallowest (last entry of chain_layers);
        # chain_idx (used for per_layer_info key) preserves the original
        # deepest-first convention used by callers.
        chain_idx = n_chain - 1 - sweep_pos
        chain_layer = chain_layers[chain_idx]
        work_layer = work_layers[chain_idx]

        cap = captured[chain_layer]
        a_pre_forward = cap["a_pre"].double()
        activation_after = cap["activation"]
        target_a_post = targets_per_layer[chain_layer]

        if sweep_pos == 0:
            X = cap["a_in"].double()
        else:
            X = prev_a_post_new

        t_pre = invert_activation(target_a_post, a_pre_forward, activation_after)
        has_bias = work_layer.bias is not None
        W_new, b_new = solve_ols_layer(
            X, t_pre, with_bias=has_bias, ols_lambda=ols_lambda)

        target_device = work_layer.weight.device
        target_dtype = work_layer.weight.dtype
        with torch.no_grad():
            old_W = work_layer.weight.detach().clone()
            new_W = W_new.to(device=target_device, dtype=target_dtype)
            work_layer.weight.copy_(new_W)
            w_change = (new_W - old_W).norm().item()
            b_change = 0.0
            if has_bias and b_new is not None:
                old_b = work_layer.bias.detach().clone()
                new_b = b_new.to(device=target_device, dtype=target_dtype)
                work_layer.bias.copy_(new_b)
                b_change = (new_b - old_b).norm().item()

        with torch.no_grad():
            pred = X @ W_new.T
            if has_bias and b_new is not None:
                pred = pred + b_new.unsqueeze(0)
            residual = (pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30)

            # Compute this layer's new a_post (= next retrained layer's X).
            a_pre_new = pred  # X @ W_new.T (+ b_new) was just computed
            if activation_after is None or isinstance(activation_after, nn.Identity):
                prev_a_post_new = a_pre_new
            else:
                try:
                    act_param = next(activation_after.parameters())
                    prev_a_post_new = activation_after(
                        a_pre_new.to(device=act_param.device, dtype=act_param.dtype)
                    ).to(device=a_pre_new.device, dtype=a_pre_new.dtype)
                except StopIteration:
                    prev_a_post_new = activation_after(a_pre_new)

        per_layer_info[chain_idx] = {
            "d_in": int(X.shape[1]),
            "d_out": int(t_pre.shape[1]),
            "weight_change_frob": float(w_change),
            "bias_change_frob": float(b_change),
            "target_residual": float(residual.item()),
            "n_samples": int(X.shape[0]),
        }

    return per_layer_info, targets_per_layer


def retrain_via_target_prop(
    model: nn.Module,
    layers_to_retrain_deepest_first: List[nn.Linear],
    dataloader: Iterable,
    *,
    method: str = "kfac_a",
    correct_target_mean: bool = False,
    correct_target_cov: bool = False,
    eps: float = 1e-4,
    sigma2: Optional[float] = None,
    ols_lambda: float = 1e-4,
    max_samples: int = 4096,
    seq_subsample: int = 4096,
    device: Optional[torch.device] = None,
    forward_fn: Optional[Callable] = None,
    activation_overrides: Optional[Dict[nn.Linear, Optional[nn.Module]]] = None,
    return_copy: bool = True,
    n_iterations: int = 1,
    eval_fn: Optional[Callable[[nn.Module], float]] = None,
) -> RetrainerResult:
    """Retrain layers via target propagation + per-layer OLS, with optional iteration.

    Parameters
    ----------
    n_iterations : Number of full retraining passes. Each pass after the first
        re-captures activations from the updated model and re-fits weights.
        The chain's deepest-target stays fixed at the original model's
        forward output, driving convergence back toward original behavior.
    eval_fn : If provided, called after each iteration with the (current)
        retrained model; the returned float is logged in iteration_history.
        Useful for accuracy tracking per iteration.
    Other params : Same as before.
    """
    if not layers_to_retrain_deepest_first:
        raise ValueError("layers_to_retrain_deepest_first must be non-empty")
    if method not in ("naive", "kfac_a"):
        raise ValueError(f"method must be 'naive' or 'kfac_a', got {method!r}")
    if correct_target_cov and not correct_target_mean:
        raise ValueError(
            "correct_target_cov=True requires correct_target_mean=True"
        )
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1; got {n_iterations}")

    chain_layers = layers_to_retrain_deepest_first
    work_model = copy.deepcopy(model) if return_copy else model

    # Build mapping from chain (original-model) layers to work_model layers
    if return_copy:
        orig_names = {id(m): n for n, m in model.named_modules()
                      if isinstance(m, nn.Linear)}
        work_by_name = {n: m for n, m in work_model.named_modules()
                        if isinstance(m, nn.Linear)}
        work_layers = [work_by_name[orig_names[id(l)]] for l in chain_layers]
    else:
        work_layers = list(chain_layers)

    # We need the dataloader N times (once per iteration). If it's a generator,
    # materialize it to a list so we can iterate.
    if hasattr(dataloader, "__iter__") and not isinstance(dataloader, list):
        try:
            dataloader_iter = list(dataloader)
        except TypeError:
            # Already a fixed-size container
            dataloader_iter = dataloader
    else:
        dataloader_iter = list(dataloader)

    # Capture the ORIGINAL deepest layer's a_post (this is the unchanging
    # reference target across all iterations).
    deepest_layer = chain_layers[0]
    initial_capture = collect_activations(
        model, dataloader_iter, [deepest_layer],
        max_samples=max_samples, seq_subsample=seq_subsample,
        device=device, forward_fn=forward_fn,
        activation_overrides=activation_overrides,
    )
    original_deepest_a_post = initial_capture[deepest_layer]["a_post"].double()

    iteration_history: List[Dict[str, Any]] = []
    final_per_layer_info: Dict[int, Dict[str, Any]] = {}

    for iteration in range(n_iterations):
        # Iter 0 captures from original model, iter k>0 captures from updated work_model
        capture_model = model if iteration == 0 else work_model

        per_layer_info, _ = _one_iteration(
            capture_model=capture_model,
            work_layers=work_layers,
            chain_layers=chain_layers if iteration == 0 else work_layers,
            dataloader=dataloader_iter,
            method=method,
            correct_target_mean=correct_target_mean,
            correct_target_cov=correct_target_cov,
            eps=eps,
            sigma2=sigma2,
            ols_lambda=ols_lambda,
            max_samples=max_samples,
            seq_subsample=seq_subsample,
            device=device,
            forward_fn=forward_fn,
            activation_overrides=activation_overrides,
            original_deepest_a_post=original_deepest_a_post,
        )
        final_per_layer_info = per_layer_info

        iter_record = {"iteration": iteration, "per_layer": per_layer_info}
        if eval_fn is not None:
            try:
                iter_record["accuracy"] = float(eval_fn(work_model))
            except Exception as exc:
                iter_record["eval_error"] = str(exc)
        iteration_history.append(iter_record)

    return RetrainerResult(
        model=work_model,
        per_layer_info=final_per_layer_info,
        method=method,
        correct_target_mean=correct_target_mean,
        correct_target_cov=correct_target_cov,
        ols_lambda=ols_lambda,
        n_iterations=n_iterations,
        iteration_history=iteration_history,
    )
