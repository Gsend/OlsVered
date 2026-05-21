"""
diagnostic/capture.py
=====================

Raw per-sample activation capture for the inversion-drift diagnostic.

Unlike OlsSMKFAC's KFACHooks which stores per-layer Gram sums (XᵀX),
this module stores the *raw activation tensors themselves* (concatenated
across batches up to a cap), because the diagnostic needs to:

1. Compute mu_a, Sigma_a as Method K's prior — derivable from XᵀX, but the
   raw activations are also needed for…
2. Construct the ground-truth a_in to compare back-propagated inverse against.
3. Compute a_pre (= W a_in + b) per sample as the "forward mask" used by
   activation inversion for ReLU dead units.

For sequence-like inputs of shape (B, T, d) (transformers), the hook
auto-flattens to (B*T, d) and subsamples down to a per-call cap to keep
memory bounded.

Public API
----------
RawActivationHooks(model, layers, max_samples, seq_subsample)
    Hook manager. Call enable(), run forward passes, then get(layer)
    to retrieve (a_in, a_pre).

collect_activations(model, dataloader, layers, ...)
    Convenience one-shot helper. Runs the model on the dataloader with
    hooks enabled, returns per-layer dict of (a_in, a_pre, a_post).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnostic.layers import LNReLU


# ---------------------------------------------------------------------------
# Activation-module detection (mirrors OlsSMLayerRetrainer._detect_following_activations)
# ---------------------------------------------------------------------------

_KNOWN_ACTIVATIONS = (
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU,
    nn.Tanh, nn.Sigmoid,
    nn.GELU, nn.SiLU,
    nn.PReLU, nn.SELU,
    LNReLU,   # composite LayerNorm+ReLU for LNMLP (E2)
)


def detect_following_activation(
    model: nn.Module,
    layer: Union[nn.Linear, nn.Conv2d],
) -> Optional[nn.Module]:
    """Find the activation module that immediately follows `layer` in the
    model's topological forward order.

    Heuristic: walks the model's named modules in declaration order; finds
    the position of `layer`; returns the next module that is an instance of
    a known activation function.

    Returns None if no activation follows (e.g., layer is a final classifier
    head with logits as outputs).
    """
    saw_layer = False
    modules = list(model.named_modules())
    for _, mod in modules:
        if mod is layer:
            saw_layer = True
            continue
        if saw_layer:
            if isinstance(mod, _KNOWN_ACTIVATIONS):
                return mod
            # Skip Sequential / module containers and look inside
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                # Hit the next parametric layer without finding an activation
                return None
    return None


# ---------------------------------------------------------------------------
# Raw activation capture
# ---------------------------------------------------------------------------

class RawActivationHooks:
    """Forward-hook manager that captures raw (N, d) activation tensors.

    For each tracked nn.Linear `L`:
      - a_in[L]  : (n_collected, d_in)  — input to L (post-activation of L-1
                                          or raw model input)
      - a_pre[L] : (n_collected, d_out) — output of L (pre-activation, i.e.
                                          before the following activation
                                          function is applied)

    For 3D inputs (B, T, d_in) (typical for transformers), tensors are
    auto-flattened to (B*T, d) before storage.

    Per-batch subsampling caps the rows per call at `seq_subsample`; the
    overall buffer is capped at `max_samples` per layer total (LIFO eviction
    if exceeded, though typical use stays under the cap).

    Usage
    -----
        hooks = RawActivationHooks(model, layers=[lin1, lin2], max_samples=4096)
        hooks.enable()
        for batch in dataloader:
            with torch.no_grad():
                model(batch)
        a_in, a_pre = hooks.get(lin1)
        hooks.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: List[nn.Linear],
        max_samples: int = 4096,
        seq_subsample: int = 4096,
        rng_seed: Optional[int] = 0,
    ):
        if not isinstance(layers, (list, tuple)):
            raise ValueError("layers must be a list of nn.Linear modules")
        for layer in layers:
            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                raise ValueError(
                    f"Only nn.Linear and nn.Conv2d are supported; got {type(layer).__name__}"
                )
        self.model = model
        self.layers = list(layers)
        self.max_samples = int(max_samples)
        self.seq_subsample = int(seq_subsample)
        # Per-layer accumulators
        self._a_in: Dict[nn.Linear, List[torch.Tensor]] = {l: [] for l in self.layers}
        self._a_pre: Dict[nn.Linear, List[torch.Tensor]] = {l: [] for l in self.layers}
        self._counts: Dict[nn.Linear, int] = {l: 0 for l in self.layers}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._enabled = False
        self._rng = torch.Generator()
        if rng_seed is not None:
            self._rng.manual_seed(int(rng_seed))

    # ----------------------------------------------------------------------

    def enable(self) -> None:
        """Register forward hooks on all tracked layers."""
        if self._enabled:
            return
        for layer in self.layers:
            handle = layer.register_forward_hook(self._make_hook(layer))
            self._handles.append(handle)
        self._enabled = True

    def remove(self) -> None:
        """Unregister all hooks. Captured tensors remain available via get()."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def clear(self) -> None:
        """Reset all per-layer buffers (does not change hook state)."""
        for layer in self.layers:
            self._a_in[layer] = []
            self._a_pre[layer] = []
            self._counts[layer] = 0

    def get(self, layer: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated (a_in, a_pre) for the given layer.

        Returns 2D tensors of shape (n_collected, d_in) and (n_collected, d_out).
        """
        if layer not in self._a_in:
            raise KeyError(f"layer not tracked by these hooks: {layer}")
        a_in_list = self._a_in[layer]
        a_pre_list = self._a_pre[layer]
        if not a_in_list:
            if isinstance(layer, nn.Conv2d):
                kH, kW = (layer.kernel_size if isinstance(layer.kernel_size, tuple)
                          else (layer.kernel_size, layer.kernel_size))
                d_in = layer.in_channels * kH * kW
                d_out = layer.out_channels
            else:
                d_in = layer.in_features
                d_out = layer.out_features
            return torch.empty(0, d_in), torch.empty(0, d_out)
        a_in = torch.cat(a_in_list, dim=0)
        a_pre = torch.cat(a_pre_list, dim=0)
        return a_in, a_pre

    def count(self, layer: nn.Linear) -> int:
        """How many rows are currently stored for this layer."""
        return self._counts.get(layer, 0)

    # ----------------------------------------------------------------------

    def _make_hook(self, layer: Union[nn.Linear, nn.Conv2d]) -> Callable:
        """Build a closure that captures a_in (layer input) and a_pre (layer output).

        For Conv2d layers, applies im2col (F.unfold) so both tensors are stored
        as 2D matrices: (N*H_out*W_out, C_in*kH*kW) and (N*H_out*W_out, C_out).
        """

        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if not self._enabled:
                return
            x_in = inputs[0].detach()
            x_pre = output.detach()

            if isinstance(module, nn.Conv2d):
                # im2col: (B, C_in, H, W) → (B, C_in*kH*kW, H_out*W_out)
                unf = F.unfold(
                    x_in,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                # (B, H_out*W_out, C_in*kH*kW) → (B*H_out*W_out, C_in*kH*kW)
                x_in = unf.permute(0, 2, 1).reshape(-1, unf.shape[1])
                # (B, C_out, H_out, W_out) → (B*H_out*W_out, C_out)
                x_pre = x_pre.permute(0, 2, 3, 1).reshape(-1, x_pre.shape[1])
            elif x_in.ndim > 2:
                x_in = x_in.reshape(-1, x_in.shape[-1])
                x_pre = x_pre.reshape(-1, x_pre.shape[-1])
            elif x_in.ndim < 2:
                raise ValueError(
                    f"layer input has ndim={x_in.ndim}; expected >= 2"
                )

            n = x_in.shape[0]
            # Per-call subsample
            if n > self.seq_subsample:
                idx = torch.randperm(n, generator=self._rng)[: self.seq_subsample]
                x_in = x_in[idx]
                x_pre = x_pre[idx]
                n = self.seq_subsample

            # Cap total buffer at max_samples
            cur = self._counts[layer]
            if cur >= self.max_samples:
                return
            avail = self.max_samples - cur
            if n > avail:
                x_in = x_in[:avail]
                x_pre = x_pre[:avail]
                n = avail

            self._a_in[layer].append(x_in.float().cpu())
            self._a_pre[layer].append(x_pre.float().cpu())
            self._counts[layer] = cur + n

        return hook


# ---------------------------------------------------------------------------
# One-shot convenience helper
# ---------------------------------------------------------------------------

def collect_activations(
    model: nn.Module,
    dataloader: Iterable,
    layers: List[Union[nn.Linear, nn.Conv2d]],
    *,
    max_samples: int = 4096,
    seq_subsample: int = 4096,
    device: Optional[torch.device] = None,
    forward_fn: Optional[Callable] = None,
    activation_overrides: Optional[Dict[nn.Linear, Optional[nn.Module]]] = None,
) -> Dict[nn.Linear, Dict[str, object]]:
    """Run the model on the dataloader and return per-layer raw activations.

    For each layer L in `layers`, returns a dict with keys:
      'a_in'  : (n, d_in) input tensor
      'a_pre' : (n, d_out) pre-activation (= W a_in + b)
      'a_post': (n, d_out) post-activation (= activation(a_pre))
      'activation': nn.Module or None — the activation that follows L
                    (auto-detected, or supplied via activation_overrides[L])

    Parameters
    ----------
    model : The model to capture from.
    dataloader : Iterable of batches. Each batch is passed to forward_fn(model, batch)
        if supplied, else to model(batch).
    layers : List of nn.Linear modules to track.
    max_samples : Cap on total rows per layer.
    seq_subsample : Cap on rows captured per forward call.
    device : Device for forward pass. If None, uses model's device.
    forward_fn : Optional custom forward call. Signature: forward_fn(model, batch).
        Defaults to `model(batch)`.
    activation_overrides : Optional mapping from layer to its following
        activation module. If a layer is in the mapping, its value overrides
        the auto-detected activation.
    """
    if device is None:
        device = next(model.parameters()).device
    if forward_fn is None:
        def forward_fn(m, batch):
            if isinstance(batch, (list, tuple)):
                return m(batch[0])
            if isinstance(batch, dict):
                return m(**batch)
            return m(batch)

    # Detect following activation for each layer
    activations: Dict[nn.Linear, Optional[nn.Module]] = {}
    for layer in layers:
        if activation_overrides and layer in activation_overrides:
            activations[layer] = activation_overrides[layer]
        else:
            activations[layer] = detect_following_activation(model, layer)

    hooks = RawActivationHooks(
        model, layers, max_samples=max_samples, seq_subsample=seq_subsample,
    )
    hooks.enable()
    try:
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Move tensors in batch to device when possible
                batch = _move_batch_to_device(batch, device)
                forward_fn(model, batch)
                # Stop early if all layers are saturated
                if all(hooks.count(l) >= max_samples for l in layers):
                    break
    finally:
        hooks.remove()

    # Assemble result
    result: Dict[nn.Linear, Dict[str, object]] = {}
    for layer in layers:
        a_in, a_pre = hooks.get(layer)
        act = activations[layer]
        if act is None:
            a_post = a_pre.clone()
        else:
            with torch.no_grad():
                # For activations with parameters (e.g. LNReLU), move a_pre
                # to the activation's device before applying it.
                try:
                    act_device = next(act.parameters()).device
                    a_pre_act = a_pre.to(act_device)
                except StopIteration:
                    a_pre_act = a_pre  # no parameters; any device works
                a_post = act(a_pre_act).cpu()
        result[layer] = {
            "a_in": a_in,
            "a_pre": a_pre,
            "a_post": a_post,
            "activation": act,
        }
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _move_batch_to_device(batch, device):
    """Move tensors in a batch (tensor / list / tuple / dict) to a device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [(v.to(device) if isinstance(v, torch.Tensor) else v) for v in batch]
        return type(batch)(moved)
    return batch
