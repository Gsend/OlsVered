"""
Raw-activation hooks for Vered K-FAC.

Key difference from KFACHooks
------------------------------
KFACHooks accumulates  A_sum += xᵀx  in each forward hook — it discards the
raw activation x immediately after forming the outer product.  This is
memory-efficient (O(n²) per layer) but squares the condition number.

RawActivationHooks instead runs a streaming TSQR update in each hook:

    running_R_X ← streaming_tsqr_update(running_R_X, x_chunk)

Only the O(n²) upper-triangular factor is retained; the raw x rows are
discarded as soon as each chunk is processed.  The final R satisfies:

    Rᵀ R  ≈  Σ xᵢᵀ xᵢ  =  Xᵀ X

with condition number κ(R) = κ(X)  (vs κ(X²) for the Gram approach).

At get_factors() time, Tikhonov damping is applied via ridge augmentation:
    R_damped = tsqr([R_undamped ; √λ · I_n])
so that R_dampedᵀ R_damped = XᵀX + λI without ever touching X directly.

p >= n requirement
------------------
Vered K-FAC requires the total accumulated batch rows p >= n_in (or n_out).
For layers where this is not satisfied, get_factors() raises VeredRankError
and the optimizer falls back to Classic K-FAC for that layer.

Conv2d support
--------------
Conv2d layers use the same im2col unfolding as KFACHooks, giving an
effective (B·L, C_in·kH·kW) activation matrix.  Typically B·L >> C_in·kH·kW
so the p >= n constraint is easy to satisfy.

Bias handling
-------------
When module.bias is not None, the input activation x is augmented with a
column of ones — [x, 1] — so that n_in effectively becomes n_in + 1 and the
bias Kronecker factor is handled implicitly.  Disabled by augment_bias=False.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _module_tag(module: nn.Module) -> str:
    """Short human-readable identifier for a module, used in log messages."""
    if isinstance(module, nn.Linear):
        return f"Linear({module.in_features}→{module.out_features})"
    if isinstance(module, nn.Conv2d):
        return (f"Conv2d({module.in_channels}→{module.out_channels}, "
                f"k={module.kernel_size})")
    return type(module).__name__
import torch.nn.functional as F

from optimizer.gram_estimator import GramMatrixEstimator
from optimizer.sgso import streaming_tsqr_update, finalize_R

# Suppress the benign PyTorch warning about backward hooks on layers
# whose inputs don't require grad (e.g. the first layer).
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing",
    category=UserWarning,
)


class VeredRankError(RuntimeError):
    """Raised when accumulated p < n for a layer, making QR rank-deficient."""


class RawActivationHooks(GramMatrixEstimator):
    """Forward/backward hooks that maintain streaming TSQR R-factors.

    Implements GramMatrixEstimator so VeredKFAC can share the same
    hook lifecycle interface as OlsSMKFAC / ClassicKFAC.

    get_factors() returns (R_X, R_G) — upper-triangular factors — rather than
    the Gram matrices (A, G) returned by KFACHooks.  VeredKFAC calls
    apply_vered() which consumes these R factors directly via triangular solves.

    Parameters
    ----------
    model : nn.Module
    damping : float
        Tikhonov damping λ applied at finalization (ridge augmentation).
    max_out_dim : int
        Skip layers whose output dimension exceeds this value (e.g. LM heads).
        0 = disabled (all layers tracked).
    augment_bias : bool
        If True, append a column of ones to input activations for layers with
        bias, so the bias Kronecker factor is handled implicitly.
        Default: False — matches ClassicKFAC's bias handling and avoids the
        centred-covariance contamination the augmentation introduces.
    """

    def __init__(
        self,
        model: nn.Module,
        damping: float = 1e-2,
        max_out_dim: int = 0,
        augment_bias: bool = False,
        max_conv_rows: int = 512,
    ):
        self.model = model
        self.damping = damping
        self.augment_bias = augment_bias
        self.max_conv_rows = max_conv_rows  # cap Conv2d patch rows per batch (0 = no cap)

        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._enabled = False

        # Running upper-triangular R factors, updated each hook call.
        # None means no data accumulated yet for this layer.
        self._R_X: Dict[nn.Module, Optional[torch.Tensor]] = {}
        self._R_G: Dict[nn.Module, Optional[torch.Tensor]] = {}

        # Total accumulated rows (for p >= n validation).
        self._n_rows_X: Dict[nn.Module, int] = {}
        self._n_rows_G: Dict[nn.Module, int] = {}

        self._linear_layers: List[nn.Module] = []

        for module in model.modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            if max_out_dim > 0:
                out_dim = (
                    module.out_features
                    if isinstance(module, nn.Linear)
                    else module.out_channels
                )
                if out_dim > max_out_dim:
                    continue
            self._linear_layers.append(module)
            self._R_X[module] = None
            self._R_G[module] = None
            self._n_rows_X[module] = 0
            self._n_rows_G[module] = 0

    # ------------------------------------------------------------------
    # GramMatrixEstimator interface
    # ------------------------------------------------------------------

    @property
    def linear_layers(self) -> List[nn.Module]:
        return self._linear_layers

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        """Register forward and backward hooks on all tracked layers."""
        if self._enabled:
            return
        for module in self._linear_layers:
            h_fwd = module.register_forward_hook(self._forward_hook)
            h_bwd = module.register_full_backward_hook(self._backward_hook)
            self._handles.extend([h_fwd, h_bwd])
        self._enabled = True

    def get_factors(self) -> Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        """Return damped R factors (R_X, R_G) for each layer with data.

        Applies ridge augmentation for damping before returning.  Raises
        VeredRankError if any layer has p < n (caller should fall back to
        Classic K-FAC for that layer).

        Returns
        -------
        dict : module → (R_X, R_G)
            R_X : (n_in,  n_in)  upper-triangular  (input factor)
            R_G : (n_out, n_out) upper-triangular  (gradient factor)
        """
        logger.debug("get_factors() called: %d tracked layers", len(self._linear_layers))
        factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        for module in self._linear_layers:
            R_X_raw = self._R_X.get(module)
            R_G_raw = self._R_G.get(module)
            if R_X_raw is None or R_G_raw is None:
                logger.debug("get_factors() [%s]: skipped — no data yet",
                             _module_tag(module))
                continue   # no data yet

            # When p < n, torch.linalg.qr(mode='reduced') returns a non-square
            # R of shape (p, n) instead of (n, n).  Detect this by checking
            # whether the stored R is square — if not, p < n was violated.
            rows_X = self._n_rows_X[module]
            rows_G = self._n_rows_G[module]

            if R_X_raw.shape[0] != R_X_raw.shape[1]:
                n_in_actual = R_X_raw.shape[1]
                raise VeredRankError(
                    f"Layer {module}: accumulated {rows_X} rows for X, "
                    f"but need >= n_in={n_in_actual}.  "
                    "Increase batch_size × seq_len or set factor_update_freq "
                    "higher to accumulate more rows before factorisation."
                )
            if R_G_raw.shape[0] != R_G_raw.shape[1]:
                n_out_actual = R_G_raw.shape[1]
                raise VeredRankError(
                    f"Layer {module}: accumulated {rows_G} rows for δ, "
                    f"but need >= n_out={n_out_actual}."
                )

            n_in  = R_X_raw.shape[0]
            n_out = R_G_raw.shape[0]

            logger.debug(
                "get_factors() [%s]: rows_X=%d rows_G=%d  "
                "R_X_raw=%s  R_G_raw=%s  damping=%.4g",
                _module_tag(module), rows_X, rows_G,
                tuple(R_X_raw.shape), tuple(R_G_raw.shape), self.damping,
            )

            # BUG FIX: normalize by sqrt(n_rows) so that RᵀR ≈ XᵀX/n (per-sample
            # mean), matching ClassicKFAC which divides A_sum by n_rows in
            # get_factors().  Without this, RᵀR = XᵀX (sum), making the
            # preconditioner n_rows times too small and the effective lr ~1280×
            # too small for typical KFAC_FREQ=20, batch=64 settings.
            R_X_scaled = R_X_raw / math.sqrt(rows_X)
            R_G_scaled = R_G_raw / math.sqrt(rows_G)

            R_X = finalize_R(R_X_scaled, self.damping)
            R_G = finalize_R(R_G_scaled, self.damping)

            logger.debug(
                "get_factors() [%s]: done → R_X=%s  R_G=%s",
                _module_tag(module), tuple(R_X.shape), tuple(R_G.shape),
            )
            factors[module] = (R_X, R_G)

        logger.debug("get_factors() returning %d factor pairs", len(factors))
        return factors

    def clear(self):
        """Reset all R-factor accumulators."""
        for module in self._linear_layers:
            self._R_X[module] = None
            self._R_G[module] = None
            self._n_rows_X[module] = 0
            self._n_rows_G[module] = 0

    def remove(self):
        """Detach all hooks and free state."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._enabled = False
        self.clear()

    def n_samples_accumulated(self, module: Optional[nn.Module] = None) -> int:
        target = module or (self._linear_layers[0] if self._linear_layers else None)
        if target is None:
            return 0
        return self._n_rows_X.get(target, 0)

    # ------------------------------------------------------------------
    # Conv2d helpers (same as KFACHooks — im2col unfolding)
    # ------------------------------------------------------------------

    @staticmethod
    def _unfold_conv_input(x: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
        x_unf = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )   # (B, C_in·kH·kW, L)
        B, C_kk, L = x_unf.shape
        return x_unf.permute(0, 2, 1).reshape(B * L, C_kk)

    @staticmethod
    def _reshape_conv_grad(delta: torch.Tensor) -> torch.Tensor:
        B, C_out, H_out, W_out = delta.shape
        return delta.permute(0, 2, 3, 1).reshape(B * H_out * W_out, C_out)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """Streaming TSQR update for input activations X."""
        if not self._enabled:
            return
        x = input[0].detach()
        raw_shape = tuple(x.shape)

        if isinstance(module, nn.Conv2d):
            x = self._unfold_conv_input(x, module)   # (B·L, C_in·kH·kW)
            # Spatial patches are highly correlated — subsample to cap the leaf
            # QR size.  Without this, a CIFAR-10 conv layer produces 8192 rows
            # per batch (128 images × 64 spatial locations), making streaming
            # TSQR the per-step bottleneck even on GPU.
            if self.max_conv_rows > 0 and x.shape[0] > self.max_conv_rows:
                idx = torch.randperm(x.shape[0], device=x.device)[:self.max_conv_rows]
                x = x[idx]
        elif x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])            # (B·T, d_in)

        # Bias augmentation: append column of ones so bias is handled implicitly
        if self.augment_bias and isinstance(module, nn.Linear) and module.bias is not None:
            ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ones], dim=1)           # (p, n_in + 1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_forward_hook [%s]: raw_shape=%s → chunk=%s  "
                "total_rows_X=%d  running_R_X=%s",
                _module_tag(module), raw_shape, tuple(x.shape),
                self._n_rows_X[module] + x.shape[0],
                "None" if self._R_X[module] is None
                else str(tuple(self._R_X[module].shape)),
            )

        # Streaming TSQR update
        self._R_X[module] = streaming_tsqr_update(self._R_X[module], x)
        self._n_rows_X[module] += x.shape[0]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_forward_hook [%s]: R_X updated → shape=%s  "
                "diag_min=%.4g diag_max=%.4g",
                _module_tag(module),
                tuple(self._R_X[module].shape),
                self._R_X[module].diag().min().item(),
                self._R_X[module].diag().max().item(),
            )

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ):
        """Streaming TSQR update for gradient signals δ."""
        if not self._enabled:
            return
        delta = grad_output[0].detach()
        raw_shape = tuple(delta.shape)

        if isinstance(module, nn.Conv2d):
            delta = self._reshape_conv_grad(delta)    # (B·L, C_out)
        elif delta.ndim > 2:
            delta = delta.reshape(-1, delta.shape[-1])  # (B·T, d_out)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_backward_hook [%s]: raw_shape=%s → chunk=%s  "
                "total_rows_G=%d  running_R_G=%s",
                _module_tag(module), raw_shape, tuple(delta.shape),
                self._n_rows_G[module] + delta.shape[0],
                "None" if self._R_G[module] is None
                else str(tuple(self._R_G[module].shape)),
            )

        self._R_G[module] = streaming_tsqr_update(self._R_G[module], delta)
        self._n_rows_G[module] += delta.shape[0]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_backward_hook [%s]: R_G updated → shape=%s  "
                "diag_min=%.4g diag_max=%.4g",
                _module_tag(module),
                tuple(self._R_G[module].shape),
                self._R_G[module].diag().min().item(),
                self._R_G[module].diag().max().item(),
            )
