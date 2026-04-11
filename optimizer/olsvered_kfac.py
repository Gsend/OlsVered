"""
K-FAC optimizer using olsvered LU backend for Gram matrix operations.

Natural gradient update:  ΔW = G⁻¹ · ∇L · A⁻¹
where A = E[xxᵀ] (input Gram) and G = E[δδᵀ] (gradient Gram).

Instead of torch.linalg.inv, uses olsvered's LU-based solver which:
  - Avoids forming explicit inverse when possible
  - Provides better numerical stability near singularity
  - Enables lower damping → more faithful curvature estimate
"""

import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from optimizer.backend import eigh_f32, apply_kfac_eigen_f32, eigh_topk_f32, randomized_eigh_f32
from optimizer.hooks import KFACHooks


class OlsveredKFAC(torch.optim.Optimizer):
    """K-FAC optimizer with olsvered LU backend.

    Parameters
    ----------
    model : nn.Module
        The model whose Linear layers will be preconditioned.
    lr : float
        Learning rate. Default: 1e-3.
    damping : float
        Tikhonov damping λ added to Gram matrices before inversion.
        Lower values → more faithful curvature (olsvered enables this).
        Default: 1e-2.
    factor_update_freq : int
        How often (in steps) to recompute Gram matrices A, G.
        Default: 10.
    inv_update_freq : int
        How often (in steps) to recompute cached inverses A⁻¹, G⁻¹.
        Must be ≥ factor_update_freq. Default: 10.
    weight_decay : float
        L2 regularisation coefficient. Default: 0.
    momentum : float
        SGD-style momentum on the preconditioned gradient. Default: 0.9.
    rank : int or None
        If set, use a rank-k approximation of A and G instead of the full
        eigen basis.  Reduces apply cost from O(4·n·d_out·d_in) to
        O((rank_g + rank_a)·d_out·d_in) — genuinely faster for rank << n.
        None (default) uses the full eigen basis.  Ignored when adaptive=True.
    randomized : bool
        When True (default) and rank is set (or adaptive=True), use randomized
        EVD to find the top-k eigenvectors in O(k·n²) instead of O(n³).
        Has no effect when neither rank nor adaptive is set.
    n_power_iter : int
        Number of power-iteration passes for the randomized EVD (default 1).
        More passes = more accurate but more expensive.  1 is sufficient for
        K-FAC Gram matrices whose eigenvalues decay rapidly.
    adaptive : bool
        When True, automatically choose the rank for each Gram matrix based on
        its size rather than applying a single global rank.  Rules applied
        per matrix:
          - n < adaptive_min_n  →  full EVD (no low-rank approximation)
          - n >= adaptive_min_n →  k = min(adaptive_rank_budget, n)
        This avoids the overhead of low-rank EVD on small layers (where it
        can actually be slower) while still speeding up large layers.
        Default: False.  Overrides the ``rank`` parameter when True.
    adaptive_min_n : int
        Minimum Gram matrix dimension to apply low-rank approximation.
        Matrices smaller than this use full EVD regardless of rank budget.
        Default: 256 (empirically safe — layers with n < 256 are too small
        to benefit from truncation at typical batch sizes).
    adaptive_rank_budget : int
        Maximum rank k to use when adaptive=True and n >= adaptive_min_n.
        Should match roughly the batch size used for training (the effective
        rank of a Gram matrix from a batch of B samples is at most B).
        Default: 64.
    gamma : float
        EMA decay for Kronecker factors: A ← γ·A_old + (1−γ)·A_batch.
        0.0 (default) disables EMA — factors are replaced each update.
        Higher γ smooths out per-batch noise at the cost of slower adaptation.
        OlsveredKFAC can safely use γ=0.95 because its EVD handles the
        near-singular matrices that aggressive smoothing can produce; ClassicKFAC
        should use a lower γ (≤0.9) since direct inversion is less stable.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        damping: float = 1e-2,
        factor_update_freq: int = 10,
        inv_update_freq: int = 10,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        rank: Optional[int] = None,
        randomized: bool = True,
        n_power_iter: int = 1,
        adaptive: bool = False,
        adaptive_min_n: int = 256,
        adaptive_rank_budget: int = 64,
        grad_clip: Optional[float] = None,
        gamma: float = 0.0,
    ):
        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay,
                        momentum=momentum)
        # Collect Linear and Conv2d layer parameters
        params = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                params.append({"params": module.parameters()})
        super().__init__(params, defaults)

        self.model = model
        self.damping = damping
        self.factor_update_freq = factor_update_freq
        self.inv_update_freq = inv_update_freq
        self.rank = rank
        self.randomized = randomized      # use randomized EVD when rank is set
        self.n_power_iter = n_power_iter  # power-iteration passes for randomized EVD
        self.adaptive = adaptive
        self.adaptive_min_n = adaptive_min_n
        self.adaptive_rank_budget = adaptive_rank_budget
        self.grad_clip = grad_clip  # max L2 norm per natural-gradient matrix (None=off)
        self.gamma = gamma          # EMA decay for Gram matrices (0 = disabled)

        # Per-layer rank choices recorded during _update_inverses for inspection.
        # Keys are module objects; values are (k_a, k_g) — None means full EVD.
        self.layer_ranks_: Dict[nn.Module, Tuple[Optional[int], Optional[int]]] = {}

        # Hook infrastructure
        self.hooks = KFACHooks(model)
        self.hooks.enable()

        # Cached factors and eigen decompositions.
        # _inverses stores (Q_A, inv_λ_A, Q_G, inv_λ_G) as torch f32 tensors.
        # The apply step uses these directly — no extra copies or dtype casts.
        self._factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._inverses: Dict[nn.Module, Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._momentum_buffers: Dict[nn.Module, torch.Tensor] = {}

        # Step counter
        self._step_count = 0

        # Timing instrumentation — bounded deques prevent unbounded memory growth
        _maxlen = 1000
        self.timing = {
            "factor_compute": deque(maxlen=_maxlen),
            "inversion":      deque(maxlen=_maxlen),
            "precondition":   deque(maxlen=_maxlen),
            "total_step":     deque(maxlen=_maxlen),
        }

    def _update_factors(self):
        """Recompute Gram matrix factors A, G, optionally EMA-smoothed.

        When gamma > 0, each new batch estimate is blended with the running
        average:  A ← γ·A_old + (1−γ)·A_batch.  This smooths per-batch noise
        at the cost of slower adaptation — critical on harder tasks (CIFAR-10,
        BERT) where single-batch Gram matrices are too noisy to precondition well.
        """
        t0 = time.perf_counter()
        new_factors = self.hooks.get_factors()
        self.hooks.clear()
        if self.gamma > 0.0 and self._factors:
            for module, (A_new, G_new) in new_factors.items():
                if module in self._factors:
                    A_old, G_old = self._factors[module]
                    new_factors[module] = (
                        self.gamma * A_old + (1.0 - self.gamma) * A_new,
                        self.gamma * G_old + (1.0 - self.gamma) * G_new,
                    )
        self._factors = new_factors
        self.timing["factor_compute"].append(time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Rank-selection helpers
    # ------------------------------------------------------------------

    def _effective_rank(self, n: int) -> Optional[int]:
        """Return the rank k to use for an n×n Gram matrix, or None for full EVD.

        Priority:
          1. adaptive=True   → per-matrix rule based on n
          2. rank is not None → fixed global rank (clamped to n)
          3. fallback        → None (full EVD)
        """
        if self.adaptive:
            if n < self.adaptive_min_n:
                return None  # small matrix: full EVD avoids overhead
            return min(self.adaptive_rank_budget, n)
        if self.rank is not None:
            return min(self.rank, n)
        return None  # full EVD

    def _decompose_torch(
        self, mat: torch.Tensor, k: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """EVD entirely on the tensor's device (GPU or CPU).

        Replaces the old numpy/Rust path so that on CUDA the decomposition
        runs via cuSOLVER (torch.linalg.eigh) with zero host–device transfers.

        Parameters
        ----------
        mat : (n, n) symmetric float32 tensor on target device
        k   : None → full EVD (n×n Q); int → low-rank (n×k Q)

        Returns
        -------
        Q       : (n, n) or (n, k) float32 — eigenvectors as columns
        inv_lam : (n,)  or (k,)  float32 — 1 / (λ + damping)
        """
        # Ensure symmetry (numerical drift can break eigh)
        mat = (mat + mat.T) * 0.5

        if k is None or k >= mat.shape[0]:
            # Full EVD — uses cuSOLVER on CUDA, LAPACK on CPU
            eigenvalues, Q = torch.linalg.eigh(mat)
            inv_lam = 1.0 / (eigenvalues + self.damping).clamp(min=1e-8)
            return Q, inv_lam

        if self.randomized:
            # Halko-Martinsson-Tropp randomized EVD — stays on GPU
            n = mat.shape[0]
            Omega = torch.randn(n, k, device=mat.device, dtype=mat.dtype)
            Y = mat @ Omega
            for _ in range(self.n_power_iter):
                Y = mat @ (mat @ Y)
            Q_basis, _ = torch.linalg.qr(Y)          # (n, k) orthonormal
            B = Q_basis.T @ mat @ Q_basis              # (k, k) small sketch
            B = (B + B.T) * 0.5
            eigenvalues, V = torch.linalg.eigh(B)
            Q = Q_basis @ V                            # (n, k) back to full space
            inv_lam = 1.0 / (eigenvalues + self.damping).clamp(min=1e-8)
            return Q, inv_lam

        # Deterministic top-k: full eigh then slice (cheaper than scipy for GPU)
        eigenvalues, Q_full = torch.linalg.eigh(mat)
        eigenvalues = eigenvalues[-k:]
        Q = Q_full[:, -k:]
        inv_lam = 1.0 / (eigenvalues + self.damping).clamp(min=1e-8)
        return Q, inv_lam

    # ------------------------------------------------------------------

    def _update_inverses(self):
        """Recompute cached eigen decompositions A = QΛQᵀ, G = QΛQᵀ.

        Runs entirely on the model's device (GPU via cuSOLVER when available).
        No CPU/numpy roundtrip — tensors stay on-device throughout.

        Damping is applied in eigenvalue space: caches 1/(λᵢ + δ) instead of
        materialising the full inverse matrix.
        """
        t0 = time.perf_counter()
        for module, (A, G) in self._factors.items():
            device = module.weight.device
            dtype  = module.weight.dtype

            # Cast to float32 for numerical stability; keep on device
            A_f = A.to(dtype=torch.float32)
            G_f = G.to(dtype=torch.float32)

            # Guard: skip corrupt Gram matrices (NaN/inf from diverged training)
            if not (torch.isfinite(A_f).all() and torch.isfinite(G_f).all()):
                continue

            k_a = self._effective_rank(A_f.shape[0])
            k_g = self._effective_rank(G_f.shape[0])

            Q_A, inv_lam_A = self._decompose_torch(A_f, k_a)
            Q_G, inv_lam_G = self._decompose_torch(G_f, k_g)

            # Record for external inspection (e.g. print_layer_ranks())
            self.layer_ranks_[module] = (k_a, k_g)

            # Cast back to model dtype and store — apply step uses these directly
            self._inverses[module] = (
                Q_A.to(dtype=dtype),
                inv_lam_A.to(dtype=dtype),
                Q_G.to(dtype=dtype),
                inv_lam_G.to(dtype=dtype),
            )

        self.timing["inversion"].append(time.perf_counter() - t0)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single K-FAC optimisation step.

        The natural gradient update for each Linear layer is:
            ΔW = G⁻¹ · ∇L_W · A⁻¹
            Δb = G⁻¹ · ∇L_b   (bias uses only gradient preconditioning)
        """
        t_total = time.perf_counter()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Update factors periodically
        if self._step_count % self.factor_update_freq == 1 or self.factor_update_freq == 1:
            self._update_factors()

        # Update inverses periodically
        if self._step_count % self.inv_update_freq == 1 or self.inv_update_freq == 1:
            if self._factors:
                self._update_inverses()

        # Apply preconditioned update to each Linear layer
        t_precond = time.perf_counter()
        for module in self.hooks.linear_layers:
            if module not in self._inverses:
                # No preconditioning available yet — fall back to SGD
                for p in module.parameters():
                    if p.grad is None:
                        continue
                    grad = p.grad
                    # Weight decay
                    for group in self.param_groups:
                        if any(p is pp for pp in group["params"]):
                            wd = group["weight_decay"]
                            lr = group["lr"]
                            mom = group["momentum"]
                            break
                    if wd > 0:
                        grad = p.grad.add(p.data, alpha=wd)
                    p.data.add_(grad, alpha=-lr)
                continue

            # Eigen factors cached by _update_inverses
            Q_A, inv_lam_A, Q_G, inv_lam_G = self._inverses[module]

            # Get hyperparams for this layer
            for group in self.param_groups:
                if any(p is module.weight for p in group["params"]):
                    lr = group["lr"]
                    wd = group["weight_decay"]
                    mom = group["momentum"]
                    break

            # --- Weight update: ΔW = Q_G d_G Q_Gᵀ ∇W Q_A d_A Q_Aᵀ ---
            if module.weight.grad is not None:
                # Conv2d weight is (C_out, C_in, kH, kW) — flatten to 2D for nat-grad,
                # then reshape back.  Linear weight is already (d_out, d_in).
                is_conv = isinstance(module, nn.Conv2d)
                raw_grad = module.weight.grad
                if is_conv:
                    raw_grad = raw_grad.view(module.weight.shape[0], -1)
                if wd > 0:
                    grad_w = raw_grad.add(module.weight.data.view_as(raw_grad), alpha=wd)
                else:
                    grad_w = raw_grad

                # Apply in eigen basis — 4 matmuls + element-wise scale
                tmp = Q_G.T @ grad_w @ Q_A                               # rotate in
                tmp = tmp * (inv_lam_G.unsqueeze(1) * inv_lam_A.unsqueeze(0))  # scale
                nat_grad = Q_G @ tmp @ Q_A.T                             # rotate out

                # Optional gradient clipping — prevents divergence on early steps
                # when Gram matrices are rank-deficient (few samples seen so far)
                if self.grad_clip is not None:
                    grad_norm = nat_grad.norm()
                    if grad_norm > self.grad_clip:
                        nat_grad = nat_grad * (self.grad_clip / grad_norm)

                # Momentum
                if mom > 0:
                    if module not in self._momentum_buffers:
                        self._momentum_buffers[module] = torch.zeros_like(nat_grad)
                    buf = self._momentum_buffers[module]
                    buf.mul_(mom).add_(nat_grad)
                    nat_grad = buf

                # Reshape back to original weight shape for Conv2d
                if is_conv:
                    nat_grad = nat_grad.view_as(module.weight)
                module.weight.data.add_(nat_grad, alpha=-lr)

            # --- Bias update: Δb = Q_G d_G Q_Gᵀ ∇b ---
            if module.bias is not None and module.bias.grad is not None:
                if wd > 0:
                    grad_b = module.bias.grad.add(module.bias.data, alpha=wd)
                else:
                    grad_b = module.bias.grad  # (d_out,)
                tmp_b = Q_G.T @ grad_b
                tmp_b = tmp_b * inv_lam_G
                nat_grad_b = Q_G @ tmp_b
                module.bias.data.add_(nat_grad_b, alpha=-lr)

        self.timing["precondition"].append(time.perf_counter() - t_precond)
        self.timing["total_step"].append(time.perf_counter() - t_total)

        return loss

    def print_layer_ranks(self):
        """Print a summary of the rank chosen for each layer's Gram matrices.

        Useful for verifying adaptive mode selections and estimating the
        actual compute savings vs full-rank K-FAC.

        Example output (adaptive=True, adaptive_min_n=256, adaptive_rank_budget=64)::

            Layer ranks after _update_inverses:
              Linear(784→512)  A(784×784): k=64  G(512×512): k=64
              Linear(512→256)  A(512×512): k=64  G(256×256): k=64
              Linear(256→128)  A(256×256): k=64  G(128×128): full EVD
        """
        if not self.layer_ranks_:
            print("No layer ranks recorded yet — call step() at least once.")
            return
        print("Layer ranks (last _update_inverses call):")
        for module, (k_a, k_g) in self.layer_ranks_.items():
            if isinstance(module, nn.Conv2d):
                kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) \
                          else (module.kernel_size, module.kernel_size)
                n_a = module.in_channels * kH * kW   # A dim: C_in·kH·kW
                n_g = module.out_channels             # G dim: C_out
                label = f"Conv2d({module.in_channels}→{n_g}, k={kH}×{kW})"
            else:
                n_a = module.weight.shape[1]
                n_g = module.weight.shape[0]
                label = f"Linear({n_a}→{n_g})"
            ka_str = f"k={k_a}" if k_a is not None else "full"
            kg_str = f"k={k_g}" if k_g is not None else "full"
            print(f"  {label}"
                  f"  A({n_a}×{n_a}): {ka_str}"
                  f"  G({n_g}×{n_g}): {kg_str}")

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Return timing statistics for profiling."""
        stats = {}
        for key, times in self.timing.items():
            if times:
                arr = np.array(times)
                stats[key] = {
                    "mean_ms": float(arr.mean() * 1000),
                    "p50_ms": float(np.percentile(arr, 50) * 1000),
                    "p99_ms": float(np.percentile(arr, 99) * 1000),
                    "total_s": float(arr.sum()),
                    "count": len(times),
                }
        return stats

    def kfac_state_dict(self) -> dict:
        """Serialize K-FAC curvature state for warm-start checkpointing.

        Saves the Gram matrices (_factors) and their EVDs (_inverses) keyed by
        layer index rather than module object, so the dict is pickle-safe.

        Usage::

            torch.save(opt.kfac_state_dict(), "kfac_state.pt")

        Returns
        -------
        dict with keys:
            "step_count" : int
            "factors"    : {layer_idx: (A_cpu, G_cpu)}
            "inverses"   : {layer_idx: (Q_A_cpu, il_A_cpu, Q_G_cpu, il_G_cpu)}
        """
        mod_to_idx = {mod: i for i, mod in enumerate(self.hooks.linear_layers)}
        state: dict = {"step_count": self._step_count, "factors": {}, "inverses": {}}
        for mod, (A, G) in self._factors.items():
            if mod in mod_to_idx:
                state["factors"][mod_to_idx[mod]] = (A.cpu(), G.cpu())
        for mod, (Q_A, il_A, Q_G, il_G) in self._inverses.items():
            if mod in mod_to_idx:
                state["inverses"][mod_to_idx[mod]] = (
                    Q_A.cpu(), il_A.cpu(), Q_G.cpu(), il_G.cpu())
        return state

    def load_kfac_state_dict(self, state: dict, device=None):
        """Restore K-FAC curvature state from a checkpoint.

        Call this after constructing the optimizer but before the first step.
        The model must have the same architecture as when the state was saved.

        Parameters
        ----------
        state  : dict returned by kfac_state_dict()
        device : torch.device or None — defaults to the model's current device
        """
        if device is None:
            device = next(self.model.parameters()).device
        idx_to_mod = {i: mod for i, mod in enumerate(self.hooks.linear_layers)}

        self._step_count = int(state.get("step_count", 0))

        self._factors = {}
        for i, (A, G) in state.get("factors", {}).items():
            mod = idx_to_mod.get(int(i))
            if mod is not None:
                self._factors[mod] = (A.to(device), G.to(device))

        self._inverses = {}
        for i, (Q_A, il_A, Q_G, il_G) in state.get("inverses", {}).items():
            mod = idx_to_mod.get(int(i))
            if mod is not None:
                self._inverses[mod] = (
                    Q_A.to(device), il_A.to(device),
                    Q_G.to(device), il_G.to(device),
                )

    def cleanup(self):
        """Remove hooks and free cached state."""
        self.hooks.remove()
        self._factors.clear()
        self._inverses.clear()
        self._momentum_buffers.clear()
        self.layer_ranks_.clear()
