"""
OlsSMLayerRetrainer — Block Coordinate Descent retrainer for the last N layers.

Algorithm overview
------------------
Instead of gradient descent, each layer is retrained with an *exact* OLS solve:

    W* = (XᵀX + λI)⁻¹ XᵀY

where X is the layer's input activation matrix (accumulated streaming over the
full dataset) and Y is a target matrix derived from:
  - Ground-truth labels for the *last* retrained layer.
  - Backward-projected targets for earlier layers via the damped pseudo-inverse:
        Y_i = Y_{i+1} @ (Wᵀ (WW ᵀ + λI)⁻¹)ᵀ

BCD convergence
---------------
The algorithm alternates: solve layer 1, then 2, ..., then N, then repeat.
This is Block Coordinate Descent with exact sub-solvers (Jacobi variant).
For near-trained models, 2–5 sweeps typically match hundreds of GD steps.

OLS + LoRA mode  (lora_rank > 0)
---------------------------------
After the BCD phase, an optional second stage fits a rank-r residual adapter:

    W_final = W_ols + B @ A        (A: r×d_in,  B: d_out×r)

using Alternating Least Squares on ΔY = Y_true - W_ols @ X.
Because OLS already removed the bulk of the adaptation gap, the residual is
small and genuinely low-rank — so a tiny r (1–8) usually suffices.

Reuse from existing codebase
-----------------------------
- Layer identification pattern from optimizer/hooks.py
- lu_solve_gram() from optimizer/backend.py (Rust LU or scipy fallback)
- Numpy ↔ torch bridging consistent with the rest of the optimizer package.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from optimizer.backend import lu_solve_gram

# ---------------------------------------------------------------------------
# Forward-pass helpers — handle plain tensors and dict inputs (transformers)
# ---------------------------------------------------------------------------

def _to_device(batch_x, device: torch.device):
    """Move batch_x to device, preserving dtype (token IDs must stay Long)."""
    if isinstance(batch_x, dict):
        return {k: v.to(device) for k, v in batch_x.items()}
    return batch_x.to(device)


def _model_forward(model: torch.nn.Module, batch_x):
    """Call model(batch_x) or model(**batch_x) for dict inputs (e.g. BERT)."""
    if isinstance(batch_x, dict):
        return model(**batch_x)
    return model(batch_x)


# ---------------------------------------------------------------------------
# Internal state containers
# ---------------------------------------------------------------------------

@dataclass
class GramState:
    """Streaming accumulator for a single layer's OLS matrices.

    Accumulates batches incrementally — memory is O(d_in² + d_in·d_out)
    regardless of dataset size (no raw activation storage).
    """
    XtX: torch.Tensor         # (d_in_aug, d_in_aug)  — augmented if bias
    XtY: torch.Tensor         # (d_in_aug, d_out)
    n_samples: int = 0

    def accumulate(self, x_aug: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch.  x_aug: (B, d_in_aug),  y: (B, d_out)."""
        self.XtX.add_(x_aug.T @ x_aug)
        self.XtY.add_(x_aug.T @ y)
        self.n_samples += x_aug.shape[0]

    def reset(self) -> None:
        self.XtX.zero_()
        self.XtY.zero_()
        self.n_samples = 0

@dataclass
class LoraAdapter:
    """Rank-r residual adapter:  output += B @ A @ input."""
    A: torch.Tensor   # (r, d_in)
    B: torch.Tensor   # (d_out, r)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in)  →  (B, d_out)."""
        return x @ self.A.T @ self.B.T

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OlsSMLayerRetrainer:
    """Retrain the last N nn.Linear layers of a model using exact OLS.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model.  Only nn.Linear layers are retrained; all other
        layer types (Conv2d, LayerNorm, Embedding, …) are left frozen.
    n_layers : int
        Number of linear layers counted from the end to retrain.
        N=1 is optimal (single-pass, no BCD iteration needed).
        N=2–4 is the recommended range for fine-tuning.
    lambda_reg : float
        Tikhonov regularisation λ added to the Gram diagonal:
            W* = (XᵀX + λI)⁻¹ XᵀY
        Larger → more stable but less aggressive correction.  Default: 1e-4.
    max_sweeps : int
        Maximum BCD iterations.  N=1 always runs exactly one sweep.
        Default: 5.
    tol : float
        Early-stop threshold on max weight change per sweep:
            max(|W_new − W_old|) < tol  →  stop.
        Set to 0.0 to always run max_sweeps.  Default: 1e-4.
    lora_rank : int
        If > 0, fit a rank-r residual adapter after the BCD phase.
        0 (default) disables the LoRA stage.
    lora_lambda : float
        Regularisation for the ALS sub-solves inside the LoRA stage.
        Default: same as lambda_reg.
    lora_sweeps : int
        ALS iterations for the LoRA adapter.  2–3 is usually enough.
        Default: 3.
    max_gram_dim : int
        Skip retraining any layer whose output dimension exceeds this value.
        Mirrors the same parameter in KFACHooks to exclude large-vocab heads.
        0 (default) = no limit.
    device : torch.device, optional
        Device for Gram accumulation.  Inferred from model if not given.
    dtype : torch.dtype
        Numeric dtype for accumulators.  Default: torch.float32.
    verbose : bool
        Print sweep-level progress.  Default: True.

    Usage
    -----
    ::

        retrainer = OlsSMLayerRetrainer(model, n_layers=3, lambda_reg=1e-3)
        history = retrainer.retrain(train_loader, target_fn=lambda y: F.one_hot(y, 10).float())
        retrainer.remove_hooks()   # clean up when done
    """

    def __init__(
        self,
        model: nn.Module,
        n_layers: int = 1,
        lambda_reg: float = 1e-4,
        max_sweeps: int = 5,
        tol: float = 1e-4,
        lora_rank: int = 0,
        lora_lambda: Optional[float] = None,
        lora_sweeps: int = 3,
        max_gram_dim: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ):
        self.model = model
        self.lambda_reg = lambda_reg
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.lora_rank = lora_rank
        self.lora_lambda = lora_lambda if lora_lambda is not None else lambda_reg
        self.lora_sweeps = lora_sweeps
        self.verbose = verbose
        self.device = device or next(model.parameters()).device
        self.dtype = dtype

        # ── Identify retrained layers ────────────────────────────────────────
        all_linear = [
            m for m in model.modules()
            if isinstance(m, nn.Linear)
            and (max_gram_dim == 0 or m.out_features <= max_gram_dim)
        ]
        if not all_linear:
            raise ValueError("No eligible nn.Linear layers found in model.")
        if n_layers > len(all_linear):
            warnings.warn(
                f"n_layers={n_layers} > available linear layers ({len(all_linear)}). "
                f"Retraining all {len(all_linear)} layers.",
                UserWarning, stacklevel=2,
            )
            n_layers = len(all_linear)

        self.n_layers = n_layers
        self._retrained_layers: List[nn.Linear] = all_linear[-n_layers:]

        if self.verbose:
            print(f"[OlsSMLayerRetrainer] Retraining {n_layers} layer(s):")
            for i, layer in enumerate(self._retrained_layers):
                print(f"  [{i}] Linear({layer.in_features} → {layer.out_features}"
                      f"{', bias' if layer.bias is not None else ''})")

        # ── Gram accumulators (one per retrained layer) ──────────────────────
        self._grams: List[GramState] = []
        for layer in self._retrained_layers:
            d_in_aug = layer.in_features + (1 if layer.bias is not None else 0)
            d_out = layer.out_features
            self._grams.append(GramState(
                XtX=torch.zeros(d_in_aug, d_in_aug, device=self.device, dtype=self.dtype),
                XtY=torch.zeros(d_in_aug, d_out,    device=self.device, dtype=self.dtype),
            ))

        # ── Forward hooks to capture input activations ───────────────────────
        # {layer_index: input_activation_tensor}
        self._act_cache: Dict[int, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._register_activation_hooks()

        # ── LoRA adapters (initialised lazily in _fit_lora) ──────────────────
        self._lora: List[Optional[LoraAdapter]] = [None] * n_layers

    # -----------------------------------------------------------------------
    # Hook management
    # -----------------------------------------------------------------------

    def _register_activation_hooks(self) -> None:
        """Register forward pre-hooks to capture layer input activations."""
        for idx, layer in enumerate(self._retrained_layers):
            def _hook(module, input, output, _idx=idx):  # noqa: E306
                # input is a tuple; element 0 is the activation tensor
                self._act_cache[_idx] = input[0].detach().to(
                    device=self.device, dtype=self.dtype
                )
            h = layer.register_forward_hook(_hook)
            self._hook_handles.append(h)

    def remove_hooks(self) -> None:
        """Remove all forward hooks.  Call when retraining is complete."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def retrain(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        target_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Run the full BCD retraining loop, then optionally fit LoRA residual.

        Parameters
        ----------
        dataloader : iterable of (batch_x, batch_y)
            Training data.  batch_y is passed to target_fn to produce targets
            for the *last* retrained layer.
        target_fn : callable(batch_y) → Tensor, optional
            Transforms ground-truth labels into a target tensor with shape
            (B, last_layer.out_features).  Defaults to ``lambda y: y.float()``.
            Examples:
              - Classification (int labels): ``lambda y: F.one_hot(y, C).float()``
              - Regression: ``lambda y: y.float()``
              - Language model: ``lambda y: y.float()`` on logit targets

        Returns
        -------
        dict with keys:
            ``converged``   bool — whether tol was reached before max_sweeps
            ``n_sweeps``    int  — BCD sweeps executed
            ``deltas``      list of lists — max |ΔW| per layer per sweep
            ``lora_fitted`` bool — whether a LoRA adapter was fitted
        """
        if target_fn is None:
            target_fn = lambda y: y.float()  # noqa: E731

        history: Dict[str, Any] = {
            "converged": False,
            "n_sweeps": 0,
            "deltas": [],
            "lora_fitted": False,
        }

        # Special path: N=1 is provably optimal in one pass
        if self.n_layers == 1:
            return self._single_layer_retrain(dataloader, target_fn)

        # ── BCD loop ─────────────────────────────────────────────────────────
        for sweep in range(self.max_sweeps):
            deltas = self._bcd_sweep(dataloader, target_fn)
            history["deltas"].append(deltas)
            history["n_sweeps"] += 1

            max_delta = max(deltas)
            if self.verbose:
                delta_str = "  ".join(f"L{i}:{d:.2e}" for i, d in enumerate(deltas))
                print(f"[BCD sweep {sweep + 1}/{self.max_sweeps}]  "
                      f"max|ΔW| = {max_delta:.2e}  ({delta_str})")

            if max_delta < self.tol:
                history["converged"] = True
                if self.verbose:
                    print(f"  → Converged (tol={self.tol:.1e})")
                break

        # ── OLS + LoRA residual stage ─────────────────────────────────────────
        if self.lora_rank > 0:
            if self.verbose:
                print(f"\n[OLS+LoRA] Fitting rank-{self.lora_rank} residual adapters …")
            self._fit_lora_residual(dataloader, target_fn)
            history["lora_fitted"] = True

        return history

    # -----------------------------------------------------------------------
    # BCD internals
    # -----------------------------------------------------------------------

    def _single_layer_retrain(
        self,
        dataloader: Iterable,
        target_fn: Callable,
    ) -> Dict[str, Any]:
        """Optimal single-layer path: one pass, one solve, no iteration."""
        self._reset_grams()
        layer = self._retrained_layers[0]
        gram = self._grams[0]

        for batch_x, batch_y in dataloader:
            # Move to device but preserve original dtype — token IDs must stay Long,
            # float inputs stay float.  Hooks cast captured activations to self.dtype.
            batch_x = _to_device(batch_x, self.device)
            batch_y = batch_y.to(self.device)  # keep original dtype for target_fn

            with torch.no_grad():
                self._act_cache.clear()
                _model_forward(self.model, batch_x)   # populates self._act_cache[0]

            x_in = self._act_cache.get(0)
            if x_in is None:
                raise RuntimeError(
                    "Activation hook did not fire — ensure the retrained layer "
                    "is reachable during a forward pass on your input data."
                )
            y = target_fn(batch_y).to(self.device, dtype=self.dtype)
            if y.shape[-1] != layer.out_features:
                raise ValueError(
                    f"target_fn output has shape {y.shape}, but last retrained layer "
                    f"has out_features={layer.out_features}."
                )

            x_aug = self._augment(x_in, layer)
            gram.accumulate(x_aug, y)

        old_W = layer.weight.data.clone()
        self._solve_and_update(0)
        delta = (layer.weight.data - old_W).abs().max().item()

        if self.verbose:
            print(f"[OlsSMLayerRetrainer] Single-layer solve done. "
                  f"max|ΔW| = {delta:.2e}  (n={gram.n_samples})")

        history: Dict[str, Any] = {
            "converged": True,
            "n_sweeps": 1,
            "deltas": [[delta]],
            "lora_fitted": False,
        }
        if self.lora_rank > 0:
            if self.verbose:
                print(f"[OLS+LoRA] Fitting rank-{self.lora_rank} residual adapter …")
            self._fit_lora_residual(dataloader, target_fn)
            history["lora_fitted"] = True

        return history

    def _bcd_sweep(
        self,
        dataloader: Iterable,
        target_fn: Callable,
    ) -> List[float]:
        """One BCD sweep: full dataset pass → solve all retrained layers.

        Returns list of max|ΔW| per layer.
        """
        self._reset_grams()

        # Snapshot current weights for target propagation (Jacobi-style BCD)
        weight_snapshot = [
            layer.weight.data.clone() for layer in self._retrained_layers
        ]

        # ── Dataset pass: accumulate Gram matrices ────────────────────────────
        for batch_x, batch_y in dataloader:
            batch_x = _to_device(batch_x, self.device)
            batch_y = batch_y.to(self.device)  # keep original dtype for target_fn

            # Full forward pass → activation hooks fire → fill _act_cache
            self._act_cache.clear()
            with torch.no_grad():
                _model_forward(self.model, batch_x)

            # Compute targets for each retrained layer via backward propagation
            targets_list = self._backward_propagate_targets(
                target_fn(batch_y).to(self.device, dtype=self.dtype),
                weight_snapshot,
            )

            # Accumulate Gram for each retrained layer
            for i, layer in enumerate(self._retrained_layers):
                x_in = self._act_cache.get(i)
                if x_in is None:
                    continue
                x_aug = self._augment(x_in, layer)
                targets_list[i] = targets_list[i].to(self.device, dtype=self.dtype)
                self._grams[i].accumulate(x_aug, targets_list[i])

        # ── Solve and update each layer ───────────────────────────────────────
        deltas = []
        for i, layer in enumerate(self._retrained_layers):
            if self._grams[i].n_samples == 0:
                warnings.warn(
                    f"No samples accumulated for retrained layer {i} — "
                    f"activation hook may not have fired.",
                    UserWarning, stacklevel=3,
                )
                deltas.append(0.0)
                continue
            old_W = layer.weight.data.clone()
            self._solve_and_update(i)
            delta = (layer.weight.data - old_W).abs().max().item()
            deltas.append(delta)

        return deltas

    # -----------------------------------------------------------------------
    # OLS + LoRA residual stage
    # -----------------------------------------------------------------------

    def _fit_lora_residual(
        self,
        dataloader: Iterable,
        target_fn: Callable,
    ) -> None:
        """Fit rank-r LoRA adapters to the OLS residual via ALS.

        For each retrained layer i, after OLS solved W_ols:
            ΔY = Y_true - W_ols @ X
        Then alternately solve A and B:
            B* = (ΔYᵀZ)(ZᵀZ + λI)⁻¹     where Z = XAᵀ  (B × r)
            A* = (XᵀX + λI)⁻¹ Xᵀ(ΔY Bᵀ)
        """
        r = self.lora_rank

        # Snapshot current weights (already updated by OLS)
        weight_snapshot = [layer.weight.data.clone() for layer in self._retrained_layers]

        # Initialise LoRA factors with small random values.
        # Note: gradient-descent LoRA uses B=0 init so that BA=0 at step 0.
        # ALS requires both A and B to be nonzero: if B=0, the A sub-solve
        # receives a zero RHS and sets A=0, breaking all subsequent updates.
        for i, layer in enumerate(self._retrained_layers):
            d_in, d_out = layer.in_features, layer.out_features
            self._lora[i] = LoraAdapter(
                A=torch.randn(r, d_in, device=self.device, dtype=self.dtype) * 0.02,
                B=torch.randn(d_out, r, device=self.device, dtype=self.dtype) * 0.02,
            )

        for als_iter in range(self.lora_sweeps):
            # Accumulators for ALS sub-solves (one set per layer):
            #   ZtZ[i]:  (r, r)       where Z = X @ A^T
            #   ZtDY[i]: (r, d_out)   cross-term for B solve
            #   XtX[i]:  (d_in, d_in) reused from Gram (could be cached)
            #   XtE[i]:  (d_in, r)    cross-term for A solve  (E = ΔY @ B)
            ZtZ  = [torch.zeros(r, r,                  device=self.device, dtype=self.dtype)
                    for _ in self._retrained_layers]
            ZtDY = [torch.zeros(r, layer.out_features, device=self.device, dtype=self.dtype)
                    for layer in self._retrained_layers]
            XtX_als = [torch.zeros(layer.in_features, layer.in_features,
                                   device=self.device, dtype=self.dtype)
                       for layer in self._retrained_layers]
            XtE  = [torch.zeros(layer.in_features, r, device=self.device, dtype=self.dtype)
                    for layer in self._retrained_layers]

            # ── Accumulate residual statistics ────────────────────────────────
            for batch_x, batch_y in dataloader:
                batch_x = _to_device(batch_x, self.device)
                batch_y = batch_y.to(self.device)  # keep original dtype for target_fn

                self._act_cache.clear()
                with torch.no_grad():
                    _model_forward(self.model, batch_x)

                targets_list = self._backward_propagate_targets(
                    target_fn(batch_y).to(self.device, dtype=self.dtype),
                    weight_snapshot,
                )

                for i, layer in enumerate(self._retrained_layers):
                    x_in = self._act_cache.get(i)
                    if x_in is None:
                        continue
                    x = x_in.reshape(x_in.shape[0], -1)        # (B, d_in)
                    y = targets_list[i].to(self.device, dtype=self.dtype)

                    # Residual: ΔY = Y - W_ols @ X
                    delta_y = y - x @ layer.weight.data.T       # (B, d_out)
                    if layer.bias is not None:
                        delta_y = delta_y - layer.bias.data.unsqueeze(0)

                    ada = self._lora[i]
                    Z = x @ ada.A.T                             # (B, r)

                    # For B solve
                    ZtZ[i].add_(Z.T @ Z)
                    ZtDY[i].add_(Z.T @ delta_y)

                    # For A solve
                    XtX_als[i].add_(x.T @ x)
                    E = delta_y @ ada.B                         # (B, r)  = ΔY @ B
                    XtE[i].add_(x.T @ E)

            # ── Update B (fix A, solve B) ─────────────────────────────────────
            for i, layer in enumerate(self._retrained_layers):
                ada = self._lora[i]
                ZtZ_reg = ZtZ[i].cpu().numpy().astype(np.float64)
                ZtZ_reg += self.lora_lambda * np.eye(r)
                ZtDY_np = ZtDY[i].cpu().numpy().astype(np.float64)
                # Solve (ZᵀZ + λI) Bᵀ = ZᵀΔY  →  Bᵀ: (r, d_out)
                B_T_new = lu_solve_gram(ZtZ_reg, ZtDY_np)
                ada.B = torch.from_numpy(B_T_new.T.astype(np.float32)).to(self.device, dtype=self.dtype)

            # ── Update A (fix B, solve A) ─────────────────────────────────────
            for i, layer in enumerate(self._retrained_layers):
                ada = self._lora[i]
                XtX_reg = XtX_als[i].cpu().numpy().astype(np.float64)
                XtX_reg += self.lora_lambda * np.eye(layer.in_features)
                XtE_np = XtE[i].cpu().numpy().astype(np.float64)
                # Solve (XᵀX + λI) Aᵀ = XᵀE  →  Aᵀ: (d_in, r)
                A_T_new = lu_solve_gram(XtX_reg, XtE_np)
                ada.A = torch.from_numpy(A_T_new.T.astype(np.float32)).to(self.device, dtype=self.dtype)

            if self.verbose:
                print(f"  [LoRA ALS iter {als_iter + 1}/{self.lora_sweeps}] done.")

        # ── Merge LoRA into weights (W_final = W_ols + B @ A) ────────────────
        for i, layer in enumerate(self._retrained_layers):
            ada = self._lora[i]
            if ada is not None:
                delta_W = ada.B @ ada.A   # (d_out, d_in)
                # Safety: clip adapter if it dwarfs the original weight
                # (indicates numerical blow-up from noisy targets)
                w_norm = layer.weight.data.norm().item()
                d_norm = delta_W.norm().item()
                if w_norm > 0 and d_norm > 10.0 * w_norm:
                    scale = (10.0 * w_norm) / d_norm
                    delta_W = delta_W * scale
                    if self.verbose:
                        print(f"  [LoRA] Layer {i}: adapter clipped "
                              f"(‖BA‖={d_norm:.2e} > 10×‖W‖={w_norm:.2e}), "
                              f"scale={scale:.3f}")
                layer.weight.data.add_(delta_W)
                if self.verbose:
                    print(f"  [LoRA] Layer {i}: merged adapter  "
                          f"‖BA‖ = {delta_W.norm().item():.3e}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _augment(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        """Flatten to 2D and optionally append a bias column of ones.

        Handles sequence tensors (B, seq, d) by folding seq into batch dim.
        """
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        if layer.bias is not None:
            ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return torch.cat([x, ones], dim=1)   # (B, d_in+1)
        return x

    def _backward_propagate_targets(
        self,
        last_targets: torch.Tensor,
        weight_snapshot: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Compute per-layer targets via backward pseudo-inverse projection.

        Starting from targets for the *last* retrained layer, project backward
        through each layer's pseudo-inverse:

            Y_i = Y_{i+1} @ (Wᵀ (WWᵀ + λI)⁻¹)ᵀ
                = Y_{i+1} @ (WWᵀ + λI)⁻¹ W

        Parameters
        ----------
        last_targets : (B, d_out_last)  — targets for the final retrained layer
        weight_snapshot : list of weight tensors from the START of this sweep

        Returns
        -------
        List of target tensors, one per retrained layer (index 0 = first/earliest).
        """
        N = len(self._retrained_layers)
        targets_list: List[Optional[torch.Tensor]] = [None] * N
        targets_list[N - 1] = last_targets

        for i in range(N - 2, -1, -1):
            W = weight_snapshot[i + 1]   # (d_out, d_in) of the NEXT layer
            targets_list[i] = self._pseudo_inverse_project(W, targets_list[i + 1])

        return targets_list  # type: ignore[return-value]

    def _pseudo_inverse_project(
        self,
        W: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Project targets backward through a layer via its damped pseudo-inverse.

        Computes:  Y_in = Y_out @ (WWᵀ + λI)⁻¹ W

        Shapes:
            W       : (d_out, d_in)
            targets : (B, d_out)
            return  : (B, d_in)

        Uses torch.linalg.solve (stays on GPU) rather than the Rust backend
        since this is a tall-and-thin solve (d_out × d_out, usually small).
        """
        d_out = W.shape[0]
        W_f = W.to(dtype=torch.float32)
        t_f = targets.to(dtype=torch.float32)

        # WWᵀ + λI : (d_out, d_out)
        WWt = W_f @ W_f.T
        WWt.diagonal().add_(self.lambda_reg)

        # Solve (WWᵀ + λI) X = targetsᵀ  → X : (d_out, B)
        # then Y_in = Wᵀ @ X             → (d_in, B)  → transposed
        X = torch.linalg.solve(WWt, t_f.T)   # (d_out, B)
        Y_in = W_f.T @ X                      # (d_in, B)
        return Y_in.T.to(dtype=self.dtype)    # (B, d_in)

    def _solve_and_update(self, layer_idx: int) -> None:
        """Solve OLS for layer `layer_idx` and update its weights in-place.

        Solves:  (XᵀX + λI) Wᵀ = XᵀY

        If the layer has a bias, the augmented solution has shape
        (d_in+1, d_out); the last row is the new bias vector.
        """
        layer = self._retrained_layers[layer_idx]
        gram = self._grams[layer_idx]

        d_in_aug = gram.XtX.shape[0]
        d_in = layer.in_features

        # Convert to float64 numpy for the Rust/scipy solver
        XtX_np = gram.XtX.cpu().numpy().astype(np.float64)
        XtY_np = gram.XtY.cpu().numpy().astype(np.float64)

        # Apply Tikhonov damping
        XtX_np += self.lambda_reg * np.eye(d_in_aug)

        # Solve: (XᵀX + λI) @ W_aug^T = XᵀY  →  W_aug^T : (d_in_aug, d_out)
        W_aug_T = lu_solve_gram(XtX_np, XtY_np)   # (d_in_aug, d_out)
        W_aug = W_aug_T.T                           # (d_out, d_in_aug)
        W_aug_t = torch.from_numpy(W_aug.astype(np.float32)).to(self.device, dtype=self.dtype)

        # Extract weight and optional bias
        W_weight = W_aug_t[:, :d_in]              # (d_out, d_in)
        layer.weight.data.copy_(W_weight)

        if layer.bias is not None and d_in_aug == d_in + 1:
            W_bias = W_aug_t[:, d_in]             # (d_out,)
            layer.bias.data.copy_(W_bias)

    def _reset_grams(self) -> None:
        """Zero all Gram accumulators."""
        for g in self._grams:
            g.reset()

    # -----------------------------------------------------------------------
    # Introspection helpers
    # -----------------------------------------------------------------------

    def get_retrained_layers(self) -> List[nn.Linear]:
        """Return the list of layers being retrained."""
        return list(self._retrained_layers)

    def get_all_linear_layers(self) -> List[nn.Linear]:
        """Return all nn.Linear layers found in the model (in traversal order)."""
        return [m for m in self.model.modules() if isinstance(m, nn.Linear)]

    def summary(self) -> str:
        """Short text summary of the retrainer configuration."""
        lines = [
            f"OlsSMLayerRetrainer",
            f"  n_layers    : {self.n_layers}",
            f"  lambda_reg  : {self.lambda_reg}",
            f"  max_sweeps  : {self.max_sweeps}  (tol={self.tol})",
            f"  lora_rank   : {self.lora_rank}"
            + (f"  (lora_sweeps={self.lora_sweeps})" if self.lora_rank > 0 else "  (disabled)"),
            f"  device/dtype: {self.device} / {self.dtype}",
            f"  layers:",
        ]
        for i, layer in enumerate(self._retrained_layers):
            lines.append(
                f"    [{i}] Linear({layer.in_features} → {layer.out_features}"
                + (", bias)" if layer.bias is not None else ")")
            )
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Checkpoint support
    # -----------------------------------------------------------------------

    def get_gram_state(self) -> dict:
        """Serialise accumulated Gram matrices for checkpointing.

        Captures the current state of all XtX / XtY accumulators so a
        partially-accumulated sweep can be resumed without replaying the data.

        Returns
        -------
        dict with integer keys 0 … n_layers-1, each mapping to::

            {
                "XtX":      np.ndarray (d_in_aug, d_in_aug) float32,
                "XtY":      np.ndarray (d_in_aug, d_out)    float32,
                "n_samples": int,
            }

        Usage::

            state = retrainer.get_gram_state()
            torch.save(state, "gram_checkpoint.pt")
            # later …
            retrainer.load_gram_state(torch.load("gram_checkpoint.pt"))
        """
        return {
            i: {
                "XtX":      g.XtX.cpu().numpy(),
                "XtY":      g.XtY.cpu().numpy(),
                "n_samples": g.n_samples,
            }
            for i, g in enumerate(self._grams)
        }

    def load_gram_state(self, state: dict) -> None:
        """Restore Gram accumulators from a checkpoint.

        Parameters
        ----------
        state : dict
            Output of a previous :meth:`get_gram_state` call.

        Raises
        ------
        CheckpointError
            If the state dict has a different number of layers or
            incompatible matrix shapes.
        """
        import numpy as np
        from optimizer.errors import CheckpointError

        if len(state) != self.n_layers:
            raise CheckpointError(
                f"Checkpoint has {len(state)} layer(s), "
                f"but retrainer has n_layers={self.n_layers}."
            )
        for i, data in state.items():
            i = int(i)
            if i >= len(self._grams):
                raise CheckpointError(f"Layer index {i} out of range.")
            g = self._grams[i]
            XtX = torch.from_numpy(np.array(data["XtX"])).to(self.device, dtype=self.dtype)
            XtY = torch.from_numpy(np.array(data["XtY"])).to(self.device, dtype=self.dtype)
            if XtX.shape != g.XtX.shape:
                raise CheckpointError(
                    f"Layer {i}: checkpoint XtX shape {XtX.shape} != "
                    f"expected {g.XtX.shape}."
                )
            if XtY.shape != g.XtY.shape:
                raise CheckpointError(
                    f"Layer {i}: checkpoint XtY shape {XtY.shape} != "
                    f"expected {g.XtY.shape}."
                )
            g.XtX.copy_(XtX)
            g.XtY.copy_(XtY)
            g.n_samples = int(data["n_samples"])
