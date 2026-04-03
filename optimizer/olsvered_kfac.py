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
    ):
        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay,
                        momentum=momentum)
        # Collect only Linear layer parameters
        params = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
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

        # Timing instrumentation
        self.timing = {
            "factor_compute": [],
            "inversion": [],
            "precondition": [],
            "total_step": [],
        }

    def _update_factors(self):
        """Recompute Gram matrix factors A, G from cached activations/gradients."""
        t0 = time.perf_counter()
        self._factors = self.hooks.get_factors()
        self.hooks.clear()
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

    def _decompose(self, mat_np: np.ndarray, k: Optional[int]):
        """Run EVD on a Gram matrix and return (Q, inv_lam) as numpy arrays.

        Parameters
        ----------
        mat_np : (n, n) float32 contiguous array
        k      : None → full EVD returning Q (n×n); int → low-rank returning Q (n×k)

        Returns
        -------
        Q        : (n, n) or (n, k) float32
        inv_lam  : (n,) or (k,) float32 — 1 / (λ + damping)
        """
        if k is None:
            return eigh_f32(mat_np, self.damping)
        if self.randomized:
            return randomized_eigh_f32(mat_np, k, self.n_power_iter, self.damping)
        return eigh_topk_f32(mat_np, k, self.damping)

    # ------------------------------------------------------------------

    def _update_inverses(self):
        """Recompute cached eigen decompositions A = QΛQᵀ, G = QΛQᵀ.

        Uses faer's SIMD-accelerated self-adjoint EVD (f32 throughout).
        Damping is applied in eigenvalue space: caches 1/(λᵢ + δ) instead of
        materialising the full inverse matrix.  This lets us:

          1. Avoid forming the dense n×n inverse (saves O(n³) flops vs LU inverse).
          2. Clamp near-zero eigenvalues for better stability near singularity.
          3. Change the damping coefficient without re-factorising Q.

        When adaptive=True, each Gram matrix independently gets a rank chosen
        by _effective_rank(): small matrices use full EVD, large matrices use
        low-rank EVD with at most adaptive_rank_budget eigenvectors.

        The cached (Q_A, inv_λ_A, Q_G, inv_λ_G) tuples are stored as f32 torch
        tensors so the hot apply path never touches numpy after this point.
        """
        t0 = time.perf_counter()
        for module, (A, G) in self._factors.items():
            device = module.weight.device
            dtype  = module.weight.dtype

            A_np = np.ascontiguousarray(A.cpu().numpy(), dtype=np.float32)
            G_np = np.ascontiguousarray(G.cpu().numpy(), dtype=np.float32)

            k_a = self._effective_rank(A_np.shape[0])
            k_g = self._effective_rank(G_np.shape[0])

            Q_A, inv_lam_A = self._decompose(A_np, k_a)
            Q_G, inv_lam_G = self._decompose(G_np, k_g)

            # Record choices for external inspection (e.g. print_layer_ranks())
            self.layer_ranks_[module] = (k_a, k_g)

            # Cache as torch tensors; move to target device/dtype once
            self._inverses[module] = (
                torch.from_numpy(Q_A).to(device=device, dtype=dtype),
                torch.from_numpy(inv_lam_A).to(device=device, dtype=dtype),
                torch.from_numpy(Q_G).to(device=device, dtype=dtype),
                torch.from_numpy(inv_lam_G).to(device=device, dtype=dtype),
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
                        grad = grad + wd * p.data
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
                grad_w = module.weight.grad  # (d_out, d_in)
                if wd > 0:
                    grad_w = grad_w + wd * module.weight.data

                # Apply in eigen basis — 4 matmuls + element-wise scale
                tmp = Q_G.T @ grad_w @ Q_A                               # rotate in
                tmp = tmp * (inv_lam_G.unsqueeze(1) * inv_lam_A.unsqueeze(0))  # scale
                nat_grad = Q_G @ tmp @ Q_A.T                             # rotate out

                # Momentum
                if mom > 0:
                    if module not in self._momentum_buffers:
                        self._momentum_buffers[module] = torch.zeros_like(nat_grad)
                    buf = self._momentum_buffers[module]
                    buf.mul_(mom).add_(nat_grad)
                    nat_grad = buf

                module.weight.data.add_(nat_grad, alpha=-lr)

            # --- Bias update: Δb = Q_G d_G Q_Gᵀ ∇b ---
            if module.bias is not None and module.bias.grad is not None:
                grad_b = module.bias.grad  # (d_out,)
                if wd > 0:
                    grad_b = grad_b + wd * module.bias.data
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
            n_a = module.weight.shape[1]  # d_in  → A is n_a × n_a
            n_g = module.weight.shape[0]  # d_out → G is n_g × n_g
            ka_str = f"k={k_a}" if k_a is not None else "full"
            kg_str = f"k={k_g}" if k_g is not None else "full"
            print(f"  Linear({n_a}→{n_g})"
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

    def cleanup(self):
        """Remove hooks and free cached state."""
        self.hooks.remove()
        self._factors.clear()
        self._inverses.clear()
        self._momentum_buffers.clear()
        self.layer_ranks_.clear()
