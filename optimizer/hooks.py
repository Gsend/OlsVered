"""
Forward and backward hooks for capturing activation and gradient statistics.

Shared infrastructure used by both OlsveredKFAC and ClassicKFAC.
Registers hooks on nn.Linear layers to capture:
  - Input activations x (forward hook)  → accumulates A_sum = XᵀX
  - Output gradients δ (backward hook)  → accumulates G_sum = δᵀδ

Design: accumulate directly into Gram sums each step
---------------------------------------------------------
The previous implementation stored the raw activation/gradient tensors and
only computed Gram matrices when get_factors() was called.  This meant:

  - Only the *last* step's batch was used (overwrite bug)
  - Raw tensors from all accumulated steps were kept in memory

The correct K-FAC approach accumulates running outer-product sums in the
hooks themselves:

    A_sum += xᵀx   (every forward pass)
    G_sum += δᵀδ   (every backward pass)

get_factors() then just divides by the sample count and returns.
Memory cost is O(d_in² + d_out²) per layer — two fixed matrices, not growing
with the number of accumulation steps.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class KFACHooks:
    """Manages forward/backward hooks on Linear layers for K-FAC factor capture.

    Gram matrices are accumulated incrementally every step.  Call get_factors()
    to read the current averages, then clear() to reset for the next window.

    Usage::

        hooks = KFACHooks(model)
        hooks.enable()
        for step in range(update_freq):
            loss = model(x)
            loss.backward()          # triggers hooks
        factors = hooks.get_factors()
        hooks.clear()
        # ... when training is done:
        hooks.remove()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        # Running Gram sums, accumulated across steps.
        # Keyed by module; values are (d_in × d_in) and (d_out × d_out) tensors.
        self._A_sum: Dict[nn.Module, torch.Tensor] = {}
        self._G_sum: Dict[nn.Module, torch.Tensor] = {}

        # Total number of *samples* (not steps) accumulated since last clear().
        self._n_A: Dict[nn.Module, int] = {}
        self._n_G: Dict[nn.Module, int] = {}

        self._enabled = False
        self._linear_layers: List[nn.Linear] = []

        for module in model.modules():
            if isinstance(module, nn.Linear):
                self._linear_layers.append(module)

    @property
    def linear_layers(self) -> List[nn.Linear]:
        return self._linear_layers

    def enable(self):
        """Register hooks on all Linear layers."""
        if self._enabled:
            return
        for module in self._linear_layers:
            h_fwd = module.register_forward_hook(self._forward_hook)
            self._handles.append(h_fwd)
            h_bwd = module.register_full_backward_hook(self._backward_hook)
            self._handles.append(h_bwd)
        self._enabled = True

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """Accumulate A_sum += xᵀx from this step's input activations."""
        if not self._enabled:
            return
        x = input[0].detach()
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])     # (N, d_in)
        # x.T @ x is (d_in, d_in); accumulate in-place when possible
        gram_a = x.t().mm(x)                    # (d_in, d_in)
        if module in self._A_sum:
            self._A_sum[module].add_(gram_a)
            self._n_A[module] += x.shape[0]
        else:
            self._A_sum[module] = gram_a
            self._n_A[module] = x.shape[0]

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ):
        """Accumulate G_sum += δᵀδ from this step's output gradients."""
        if not self._enabled:
            return
        delta = grad_output[0].detach()
        if delta.ndim > 2:
            delta = delta.reshape(-1, delta.shape[-1])  # (N, d_out)
        gram_g = delta.t().mm(delta)                     # (d_out, d_out)
        if module in self._G_sum:
            self._G_sum[module].add_(gram_g)
            self._n_G[module] += delta.shape[0]
        else:
            self._G_sum[module] = gram_g
            self._n_G[module] = delta.shape[0]

    def get_factors(self) -> Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        """Return averaged Gram matrices (A, G) for each Linear layer.

        A = A_sum / n_A     shape (d_in,  d_in)
        G = G_sum / n_G     shape (d_out, d_out)

        Call clear() afterwards to reset accumulators for the next window.
        Returns only layers that have received at least one forward+backward pass.
        """
        factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        for module in self._linear_layers:
            if module not in self._A_sum or module not in self._G_sum:
                continue
            A = self._A_sum[module] / self._n_A[module]
            G = self._G_sum[module] / self._n_G[module]
            factors[module] = (A, G)
        return factors

    def n_samples_accumulated(self, module: Optional[nn.Module] = None) -> int:
        """Return how many samples have been accumulated since last clear().

        If module is None, returns the count for the first layer (as a proxy
        for the whole network — all layers see the same batch).
        """
        target = module or (self._linear_layers[0] if self._linear_layers else None)
        if target is None:
            return 0
        return self._n_A.get(target, 0)

    def clear(self):
        """Reset all Gram accumulators.  Call after get_factors() to start a
        fresh accumulation window."""
        self._A_sum.clear()
        self._G_sum.clear()
        self._n_A.clear()
        self._n_G.clear()

    def remove(self):
        """Remove all hooks from the model and free cached state."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._enabled = False
        self.clear()
