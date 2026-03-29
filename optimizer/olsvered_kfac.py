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

from optimizer.backend import eigh_f32, apply_kfac_eigen_f32
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

    def _update_inverses(self):
        """Recompute cached eigen decompositions A = QΛQᵀ, G = QΛQᵀ.

        Uses faer's SIMD-accelerated self-adjoint EVD (f32 throughout).
        Damping is applied in eigenvalue space: caches 1/(λᵢ + δ) instead of
        materialising the full inverse matrix.  This lets us:

          1. Avoid forming the dense n×n inverse (saves O(n³) flops vs LU inverse).
          2. Clamp near-zero eigenvalues for better stability near singularity.
          3. Change the damping coefficient without re-factorising Q.

        The cached (Q_A, inv_λ_A, Q_G, inv_λ_G) tuples are stored as f32 torch
        tensors so the hot apply path never touches numpy after this point.
        """
        t0 = time.perf_counter()
        for module, (A, G) in self._factors.items():
            device = module.weight.device
            dtype  = module.weight.dtype

            A_np = np.ascontiguousarray(A.cpu().numpy(), dtype=np.float32)
            G_np = np.ascontiguousarray(G.cpu().numpy(), dtype=np.float32)

            # faer symmetric EVD — returns (Q, inv_λ) both f32
            Q_A, inv_lam_A = eigh_f32(A_np, self.damping)
            Q_G, inv_lam_G = eigh_f32(G_np, self.damping)

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
