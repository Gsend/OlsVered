"""
optimizer/bf16_linalg.py

Hand-rolled bf16 linear algebra primitives — built on bf16 tensor-core matmul
since cuSOLVER does not implement geqrf, triangular_solve, cholesky, or inv
for the bfloat16 dtype (verified by tests/test_bf16_linalg_support.py).

Scope:
  - householder_qr_bf16        : reduced QR via Householder reflections
  - solve_triangular_bf16      : forward / back substitution
  - cholesky_bf16              : down-looking Cholesky factorization
  - inv_bf16                   : inverse via Cholesky (for SPD only)

Precision policy:
  - Storage is bf16.  Matmuls run on bf16 tensor cores (~2× fp32 throughput,
    fp32-accumulating internally — that's how NVIDIA tensor cores work).
  - Norms and reciprocals are computed in fp32 then cast to bf16 to avoid
    overflow / underflow.  This is standard mixed-precision practice
    (Higham 2002 §3.4 — "scaled accumulation for sums of squares").

For the κ-sweep (matrices up to ~64×64) Python loops over n are fine.
A blocked variant for larger matrices (transformer / CNN K-FAC factors
up to ~1024×1024) is left for a future iteration; the operation budget at
those sizes wants Triton or CUDA kernels.

References:
  - Householder (1958), Golub (1965) for the QR algorithm
  - Higham (2002) §19 for Householder backward error analysis
  - Higham (2002) §10 for Cholesky precision considerations
"""
from __future__ import annotations

import torch


# ---- Norm-safe primitives --------------------------------------------------

def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    """Frobenius norm computed in fp32 to avoid bf16 overflow / underflow,
    then cast back to bf16.  For vectors and matrices alike."""
    return x.float().norm().to(x.dtype)


def _safe_div(num: torch.Tensor, denom: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """Element-wise division with fp32 intermediate.  Returns same dtype as num.

    Magnitude-clamp (not value-clamp): if |denom| < eps, replace with
    sign(denom) * eps to avoid div-by-zero without corrupting sign.  This
    is critical for triangular-solve correctness — the triangular diagonal
    can carry either sign.
    """
    denom_fp32 = denom.float()
    abs_d = denom_fp32.abs()
    # sign(0) is 0; map to +1 for that edge case so sign * eps stays positive
    sgn = torch.sign(denom_fp32)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    safe = torch.where(abs_d < eps, sgn * eps, denom_fp32)
    return (num.float() / safe).to(num.dtype)


# ---- Householder QR --------------------------------------------------------

def householder_qr_bf16(A: torch.Tensor) -> torch.Tensor:
    """Reduced Householder QR.  Input A: (m, n) bf16 with m ≥ n.
    Returns R: (n, n) bf16 upper-triangular, such that A = Q R for some
    implicit orthogonal Q (Q is never materialized — K-FAC doesn't need it).

    Numerical regime: matmuls in bf16 (tensor-core path), norms in fp32.
    """
    assert A.dtype == torch.bfloat16, f"expected bf16 input, got {A.dtype}"
    m, n = A.shape
    assert m >= n, f"need m >= n, got ({m}, {n})"
    A = A.clone()
    for i in range(n):
        x = A[i:, i:i+1]                        # (m-i, 1) bf16
        alpha = -torch.sign(x[0, 0]) * _safe_norm(x)
        v = x.clone()
        v[0, 0] = v[0, 0] - alpha
        v_norm = _safe_norm(v)
        if v_norm.item() == 0.0:
            continue
        v = _safe_div(v, v_norm)                # (m-i, 1) bf16 unit Householder vector
        # Reflection: A[i:, i:] = A[i:, i:] - 2 v (vᵀ A[i:, i:])
        # The inner matmul (v.T @ block) runs on bf16 tensor cores.
        block = A[i:, i:]                       # (m-i, n-i) bf16
        coeffs = v.t() @ block                  # (1, n-i)  bf16 tensor-core matmul
        A[i:, i:] = block - 2.0 * v @ coeffs    # bf16 update
    return A[:n, :n].triu()


# ---- Triangular solve ------------------------------------------------------

def solve_triangular_bf16(R: torch.Tensor, B: torch.Tensor,
                           upper: bool = True) -> torch.Tensor:
    """Solve R · X = B for X by forward (upper=False) or back (upper=True)
    substitution.  R is (n, n) triangular bf16; B is (n, k) bf16.
    Returns X: (n, k) bf16.

    Implementation: column-by-column substitution.  Each step is a bf16
    matmul against the already-solved rows + a bf16 diagonal divide.
    The diagonal divide uses fp32 intermediate (see _safe_div)."""
    assert R.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    n = R.shape[0]
    assert R.shape[1] == n, f"R must be square, got {R.shape}"
    assert B.shape[0] == n, f"B rows must match R, got {B.shape}"
    X = torch.empty_like(B)
    if upper:
        # Back substitution: solve from i = n-1 down to 0
        for i in range(n - 1, -1, -1):
            # tail contribution: R[i, i+1:] @ X[i+1:, :]
            if i + 1 < n:
                tail = R[i:i+1, i+1:] @ X[i+1:, :]   # (1, k) bf16 matmul
                rhs = B[i:i+1, :] - tail
            else:
                rhs = B[i:i+1, :]
            X[i:i+1, :] = _safe_div(rhs, R[i, i])
    else:
        # Forward substitution: solve from i = 0 up to n-1
        for i in range(n):
            if i > 0:
                head = R[i:i+1, :i] @ X[:i, :]
                rhs = B[i:i+1, :] - head
            else:
                rhs = B[i:i+1, :]
            X[i:i+1, :] = _safe_div(rhs, R[i, i])
    return X


# ---- Cholesky (for Classic K-FAC control) ----------------------------------

def cholesky_bf16(A: torch.Tensor) -> torch.Tensor:
    """Down-looking Cholesky factorization.  A: (n, n) SPD bf16.
    Returns L: (n, n) lower-triangular bf16 such that L Lᵀ = A.

    Used only for the Classic K-FAC control in the κ-sweep; production
    Classic K-FAC inverts via torch.linalg.inv at fp32."""
    assert A.dtype == torch.bfloat16
    n = A.shape[0]
    L = torch.zeros_like(A)
    for j in range(n):
        # Diagonal element: L[j, j] = sqrt(A[j, j] - sum(L[j, :j]^2))
        sum_sq_fp32 = (L[j, :j].float() ** 2).sum()
        diag_fp32 = (A[j, j].float() - sum_sq_fp32).clamp_min(1e-7).sqrt()
        L[j, j] = diag_fp32.to(torch.bfloat16)
        # Off-diagonal column: L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
        if j + 1 < n:
            # vectorized matmul over remaining rows — bf16 tensor cores
            below = A[j+1:, j:j+1]                                 # (n-j-1, 1) bf16
            head  = L[j+1:, :j] @ L[j:j+1, :j].t()                 # (n-j-1, 1) bf16
            L[j+1:, j:j+1] = _safe_div(below - head, L[j, j])
    return L


# ---- Inverse via Cholesky (SPD only) ---------------------------------------

def inv_spd_bf16(A: torch.Tensor) -> torch.Tensor:
    """Compute A⁻¹ for SPD A by solving A X = I via Cholesky.
    Returns X: (n, n) bf16."""
    L = cholesky_bf16(A)
    n = A.shape[0]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    Y = solve_triangular_bf16(L,   I, upper=False)
    X = solve_triangular_bf16(L.t().contiguous(), Y, upper=True)
    return X


# ---- Self-test (run as script) ---------------------------------------------

def _self_test():
    import math
    print("Sanity checks against fp32 references...\n")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # 1. Householder QR
    print("[1] Householder QR")
    for n in (8, 32, 64):
        A_fp32 = torch.randn(2*n, n, device=device, dtype=torch.float32)
        A_bf16 = A_fp32.to(torch.bfloat16)
        R_ours = householder_qr_bf16(A_bf16).float()
        _, R_torch = torch.linalg.qr(A_fp32, mode="reduced")
        # Normalize sign so we can compare R's directly
        sgn_ours  = torch.sign(torch.diagonal(R_ours))
        sgn_torch = torch.sign(torch.diagonal(R_torch))
        R_ours_n  = R_ours  * sgn_ours.unsqueeze(1)
        R_torch_n = R_torch * sgn_torch.unsqueeze(1)
        err = ((R_ours_n - R_torch_n).norm() / R_torch_n.norm()).item()
        print(f"  n={n:>3d}  relative |R_bf16 - R_fp32|/|R_fp32| = {err:.3e}")

    # 2. Triangular solve
    print("\n[2] Triangular solve")
    for n in (8, 32, 64):
        R_fp32 = torch.triu(torch.randn(n, n, device=device, dtype=torch.float32))
        R_fp32 = R_fp32 + torch.sign(torch.diagonal(R_fp32)).diag() * 1.0
        B_fp32 = torch.randn(n, 4, device=device, dtype=torch.float32)
        X_ours  = solve_triangular_bf16(R_fp32.to(torch.bfloat16),
                                          B_fp32.to(torch.bfloat16), upper=True).float()
        X_torch = torch.linalg.solve_triangular(R_fp32, B_fp32, upper=True)
        err = ((X_ours - X_torch).norm() / X_torch.norm()).item()
        print(f"  n={n:>3d}  relative |X_bf16 - X_fp32|/|X_fp32| = {err:.3e}")

    # 3. Cholesky
    print("\n[3] Cholesky (SPD)")
    for n in (8, 32, 64):
        Z = torch.randn(2*n, n, device=device, dtype=torch.float32)
        A_fp32 = Z.t() @ Z + torch.eye(n, device=device)
        L_ours  = cholesky_bf16(A_fp32.to(torch.bfloat16)).float()
        L_torch = torch.linalg.cholesky(A_fp32)
        err = ((L_ours - L_torch).norm() / L_torch.norm()).item()
        print(f"  n={n:>3d}  relative |L_bf16 - L_fp32|/|L_fp32| = {err:.3e}")

    # 4. SPD inverse via bf16 Cholesky
    print("\n[4] SPD inverse via Cholesky")
    for n in (8, 32, 64):
        Z = torch.randn(2*n, n, device=device, dtype=torch.float32)
        A_fp32 = Z.t() @ Z + torch.eye(n, device=device)
        Ainv_ours  = inv_spd_bf16(A_fp32.to(torch.bfloat16)).float()
        Ainv_torch = torch.linalg.inv(A_fp32)
        err = ((Ainv_ours - Ainv_torch).norm() / Ainv_torch.norm()).item()
        print(f"  n={n:>3d}  relative |Ainv_bf16 - Ainv_fp32|/|Ainv_fp32| = {err:.3e}")


if __name__ == "__main__":
    _self_test()
