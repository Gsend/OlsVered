"""
Forward and backward hooks for capturing activation and gradient statistics.

Shared infrastructure used by both OlsveredKFAC and ClassicKFAC.
Registers hooks on nn.Linear layers to capture:
  - Input activations x (forward hook)  → used to build A = (1/n) XᵀX
  - Output gradients δ (backward hook)  → used to build G = (1/n) δᵀδ
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class KFACHooks:
    """Manages forward/backward hooks on Linear layers for K-FAC factor capture.

    Usage:
        hooks = KFACHooks(model)
        hooks.enable()
        # ... forward + backward pass ...
        factors = hooks.get_factors()
        hooks.clear()
        # ... when done:
        hooks.remove()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._activations: Dict[nn.Module, torch.Tensor] = {}
        self._gradients: Dict[nn.Module, torch.Tensor] = {}
        self._enabled = False
        self._linear_layers: List[nn.Linear] = []

        # Discover all Linear layers
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
            # Forward hook: capture input activation
            h_fwd = module.register_forward_hook(self._forward_hook)
            self._handles.append(h_fwd)

            # Backward hook: capture gradient w.r.t. output
            h_bwd = module.register_full_backward_hook(self._backward_hook)
            self._handles.append(h_bwd)

        self._enabled = True

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...],
                      output: torch.Tensor):
        """Capture input activations for Gram matrix A = XᵀX."""
        if not self._enabled:
            return
        # input is a tuple; first element is the activation tensor
        x = input[0].detach()
        # Flatten batch and sequence dims: (batch, ..., d_in) → (N, d_in)
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        self._activations[module] = x

    def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...],
                       grad_output: Tuple[torch.Tensor, ...]):
        """Capture output gradients for Gram matrix G = δᵀδ."""
        if not self._enabled:
            return
        delta = grad_output[0].detach()
        # Flatten batch and sequence dims: (batch, ..., d_out) → (N, d_out)
        if delta.ndim > 2:
            delta = delta.reshape(-1, delta.shape[-1])
        self._gradients[module] = delta

    def get_factors(self) -> Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute and return Gram matrix factors (A, G) for each Linear layer.

        Returns:
            Dict mapping each nn.Linear module to (A, G) where:
                A = (1/n) XᵀX, shape (d_in, d_in)
                G = (1/n) δᵀδ, shape (d_out, d_out)
        """
        factors = {}
        for module in self._linear_layers:
            if module not in self._activations or module not in self._gradients:
                continue
            x = self._activations[module]   # (N, d_in)
            delta = self._gradients[module]  # (N, d_out)
            n = x.shape[0]

            A = (x.t() @ x) / n   # (d_in, d_in)
            G = (delta.t() @ delta) / n  # (d_out, d_out)

            factors[module] = (A, G)
        return factors

    def clear(self):
        """Clear cached activations and gradients to free memory."""
        self._activations.clear()
        self._gradients.clear()

    def remove(self):
        """Remove all hooks from the model."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._enabled = False
        self.clear()
