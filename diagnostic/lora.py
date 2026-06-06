"""
diagnostic/lora.py
==================

Closed-form LoRA fitting via Reduced-Rank Regression (RRR).

Public API
----------
solve_lora_layer(X, T, W0, b0, r, alpha, lam, metric)
    Given input X (n, d_in) and pre-activation target T (n, d_out), the frozen
    base weight W0 (d_out, d_in) and bias b0 (d_out,) or None, and desired
    LoRA rank r, return (A, B, residual) where:
        W_eff = W0 + s*B@A,  s = alpha/r
    and residual = ||X @ (s*B@A).T - T'|| / ||T'|| (T' = T - X@W0.T - b0).

    Two metric variants (for rank truncation):
      "output"   : eigenvectors of Yhat.T @ Yhat  (output-space Gram)
      "whitened" : eigenvectors of M @ inv(X.T@X + lam*I) @ M.T

LoRALinear(nn.Module)
    Wrapper: forward = x @ (W0 + s*B@A).T + bias0.
    Call set_lora(A, B) to install fitted adapters.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from diagnostic.target_prop_retrainer import solve_ols_layer


def solve_lora_layer(
    X: torch.Tensor,
    T: torch.Tensor,
    W0: torch.Tensor,
    b0: Optional[torch.Tensor],
    r: int,
    alpha: float,
    lam: float = 1e-4,
    metric: str = "output",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Closed-form LoRA adapter fitting via Reduced-Rank Regression.

    Parameters
    ----------
    X   : (n, d_in) input activations (CPU, float64)
    T   : (n, d_out) pre-activation targets (CPU, float64)
    W0  : (d_out, d_in) frozen base weight (CPU, float64)
    b0  : (d_out,) frozen base bias, or None
    r   : LoRA rank
    alpha : LoRA scaling (s = alpha/r)
    lam : ridge regularisation for the OLS solve
    metric : "output" or "whitened" — which Gram matrix to eigen-decompose

    Returns
    -------
    A   : (r, d_in) LoRA A matrix
    B   : (d_out, r) LoRA B matrix
    residual : float — ||X(sBA)^T - T'||_F / ||T'||_F
    """
    X = X.double().cpu()
    T = T.double().cpu()
    W0 = W0.double().cpu()

    # residual target: T' = T - X@W0^T - b0
    T_prime = T - X @ W0.T
    if b0 is not None:
        T_prime = T_prime - b0.double().cpu().unsqueeze(0)

    n, d_in = X.shape
    d_out = T_prime.shape[1]
    r = min(r, d_in, d_out)

    # Step 1: unconstrained OLS on T' (bias=False — bias absorbed into W0 b0)
    M, _ = solve_ols_layer(X, T_prime, with_bias=False, ols_lambda=lam)
    # M : (d_out, d_in)

    s = alpha / r   # LoRA scaling factor

    if metric == "whitened":
        # Whitened metric: eigen-decompose M @ (X.T X + lam*tr(XTX)/d_in * I)^{-1} @ M.T
        XtX = X.T @ X                                   # (d_in, d_in)
        damping = lam * (XtX.diagonal().abs().mean().item() + 1e-30)
        reg = XtX + damping * torch.eye(d_in, dtype=X.dtype)
        # Use cholesky solve for stability
        try:
            L = torch.linalg.cholesky(reg)
            MLinv = torch.linalg.solve_triangular(L, M.T, upper=False)  # (d_in, d_out)
            G = MLinv.T @ MLinv                          # (d_out, d_out)
        except Exception:
            G = M @ torch.linalg.pinv(reg) @ M.T
    else:
        # Output metric: eigen-decompose Yhat.T @ Yhat (equivalent to M @ X.T @ X @ M.T)
        Yhat = X @ M.T                                   # (n, d_out)
        G = Yhat.T @ Yhat                               # (d_out, d_out)

    # Step 2: top-r eigenvectors of G (eigh returns ascending order)
    vals, vecs = torch.linalg.eigh(G)                   # vecs: (d_out, d_out)
    Vr = vecs[:, -r:]                                   # (d_out, r)  top-r

    # Step 3: rank-r weight and LoRA factors
    Wr = Vr @ (Vr.T @ M)                               # (d_out, d_in)
    B = Vr / s                                          # (d_out, r)
    A = Vr.T @ M                                        # (r, d_in)   = A so sBA = Wr

    # Residual of the rank-r fit
    Yhat_r = X @ Wr.T                                   # (n, d_out)
    res = float((Yhat_r - T_prime).norm() / T_prime.norm().clamp(min=1e-30))

    return A.float(), B.float(), res


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight and a trainable low-rank adapter.

    forward: x -> x @ (W0 + s*B@A).T + bias0
    """

    def __init__(self, in_features: int, out_features: int, r: int, alpha: float,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.s = alpha / r

        self.W0 = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.bias0 = (nn.Parameter(torch.zeros(out_features), requires_grad=False)
                      if bias else None)
        # Standard LoRA init: A ~ N(0, 0.02), B = 0  →  W_eff = W0 at init, gradients non-zero
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.normal_(self.A, std=0.02)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, alpha: float) -> "LoRALinear":
        """Wrap an existing nn.Linear, copying its weights as the frozen base."""
        out_f, in_f = linear.weight.shape
        lora = cls(in_f, out_f, r, alpha, bias=(linear.bias is not None))
        with torch.no_grad():
            lora.W0.copy_(linear.weight)
            if linear.bias is not None and lora.bias0 is not None:
                lora.bias0.copy_(linear.bias)
        return lora

    def set_lora(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """Install closed-form fitted LoRA weights."""
        with torch.no_grad():
            self.A.copy_(A.to(self.A.device, self.A.dtype))
            self.B.copy_(B.to(self.B.device, self.B.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.W0 + self.s * (self.B @ self.A)
        out = x @ W_eff.T
        if self.bias0 is not None:
            out = out + self.bias0
        return out

    def effective_weight(self) -> torch.Tensor:
        return self.W0 + self.s * (self.B @ self.A)
