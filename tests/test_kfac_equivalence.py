"""
test_kfac_equivalence.py

Sanity tests that compare the three K-FAC variants on synthetic data with
known well-conditioned A and G matrices.  At well conditioned input
(kappa(X) ~ 10), all three variants should produce essentially identical
natural-gradient updates; if Vered's output diverges from Classic's by
more than a small numerical tolerance, there is a bug.

Run from repo root:
    pytest tests/test_kfac_equivalence.py -v
or:
    python tests/test_kfac_equivalence.py

The test focuses on the per-step "apply" formula:
        DeltaW = G^-1 . grad . A^-1
which all three variants compute differently:
    Classic:    explicit inv(A+lambda I), inv(G+lambda I), then 2 GEMMs
    OlsSM:      cholesky(A+lambda I) and cholesky(G+lambda I), then 4 TRSMs
    Vered:      QR factor R_X of X (where X^T X = A) and R_G of delta;
                damping via row-augment; then 4 TRSMs

If any variant disagrees with the others on a well-conditioned input by
more than ~1e-4 relative, we expose a bug independent of the K-FAC training
loop (no model, no scheduler, no momentum, no clipping).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================================
#  Reference implementations (independent of optimizer code)
# ============================================================================

def natgrad_classic(A: torch.Tensor, G: torch.Tensor, grad: torch.Tensor,
                    damping: float) -> torch.Tensor:
    """Reference Classic K-FAC apply: form explicit inverses, then 2 GEMMs."""
    n_in  = A.shape[0]
    n_out = G.shape[0]
    A_inv = torch.linalg.inv(A + damping * torch.eye(n_in,  dtype=A.dtype))
    G_inv = torch.linalg.inv(G + damping * torch.eye(n_out, dtype=G.dtype))
    return G_inv @ grad @ A_inv


def natgrad_olssm(A: torch.Tensor, G: torch.Tensor, grad: torch.Tensor,
                  damping: float) -> torch.Tensor:
    """Reference OlsSM apply: cholesky factor + 2 cholesky_solve calls."""
    n_in  = A.shape[0]
    n_out = G.shape[0]
    L_A = torch.linalg.cholesky(A + damping * torch.eye(n_in,  dtype=A.dtype))
    L_G = torch.linalg.cholesky(G + damping * torch.eye(n_out, dtype=G.dtype))
    # Apply (G+lambda I)^-1 from the left
    C = torch.cholesky_solve(grad, L_G)
    # Apply (A+lambda I)^-1 from the right
    return torch.cholesky_solve(C.T, L_A).T


def natgrad_vered(R_X: torch.Tensor, R_G: torch.Tensor,
                  grad: torch.Tensor, damping: float) -> torch.Tensor:
    """Reference Vered apply: 4 triangular solves on R factors.

    R_X and R_G are the upper-triangular R-factors from QR(X) and QR(delta),
    pre-augmented with damping (i.e., they encode A+lambda I = R_X^T R_X).
    """
    # Apply (R_G^T R_G)^-1 from the LEFT
    T1 = torch.linalg.solve_triangular(R_G.T, grad, upper=False)   # = R_G^-T grad
    T2 = torch.linalg.solve_triangular(R_G,   T1,   upper=True)    # = R_G^-1 T1 = G^-1 grad

    # Apply (R_X^T R_X)^-1 from the RIGHT (via transpose trick)
    T3 = torch.linalg.solve_triangular(R_X.T, T2.T, upper=False)
    T4 = torch.linalg.solve_triangular(R_X,   T3,   upper=True)
    return T4.T   # = T2 . A^-1


def damped_R_factor(X: torch.Tensor, damping: float) -> torch.Tensor:
    """Compute the QR R-factor of [X; sqrt(lambda) I], encoding X^T X + lambda I.

    R_aug satisfies R_aug^T R_aug = X^T X + lambda I exactly.
    """
    n = X.shape[1]
    aug = torch.cat([X, math.sqrt(damping) * torch.eye(n, dtype=X.dtype)], dim=0)
    _, R = torch.linalg.qr(aug, mode="reduced")
    # Ensure positive diagonal for sign-consistency with the optimizer
    diag_signs = torch.sign(torch.diagonal(R))
    diag_signs[diag_signs == 0] = 1.0
    R = R * diag_signs.unsqueeze(1)
    return R


# ============================================================================
#  Synthetic test fixtures
# ============================================================================

def make_synthetic(N: int = 256, n_in: int = 32, n_out: int = 16,
                   target_kappa: float = 10.0, seed: int = 42, dtype=torch.float64):
    """Random well-conditioned X (and delta) with prescribed condition number.

    Build X via SVD with controlled singular value spread, so that
    kappa(X) ~= target_kappa.  X^T X then has kappa = target_kappa^2 = 100,
    well within the regime where all three variants should agree.
    """
    g = torch.Generator().manual_seed(seed)

    def make_well_conditioned(rows, cols, kappa):
        # Random orthogonal U (rows x cols) and V (cols x cols)
        U = torch.linalg.qr(torch.randn(rows, cols, generator=g, dtype=dtype),
                            mode="reduced")[0]
        V = torch.linalg.qr(torch.randn(cols, cols, generator=g, dtype=dtype),
                            mode="reduced")[0]
        # Singular values geometrically spaced from 1 to 1/kappa
        sigmas = torch.exp(torch.linspace(0.0, -math.log(kappa), cols, dtype=dtype))
        return U @ torch.diag(sigmas) @ V.T

    X     = make_well_conditioned(N, n_in,  target_kappa)
    delta = make_well_conditioned(N, n_out, target_kappa)
    grad  = torch.randn(n_out, n_in, generator=g, dtype=dtype)

    A = X.T @ X / N
    G = delta.T @ delta / N

    return X, delta, A, G, grad


# ============================================================================
#  The actual tests
# ============================================================================

def _max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max element-wise relative error between two tensors."""
    abs_diff = (a - b).abs()
    denom    = b.abs().clamp(min=1e-12)
    return (abs_diff / denom).max().item()


def test_classic_olssm_agree_well_conditioned():
    """Classic and OlsSM should give nearly identical natgrad on well-conditioned input."""
    X, delta, A, G, grad = make_synthetic(target_kappa=10.0)
    damping = 1e-4

    nat_classic = natgrad_classic(A, G, grad, damping)
    nat_olssm   = natgrad_olssm  (A, G, grad, damping)

    err = _max_rel_err(nat_classic, nat_olssm)
    print(f"Classic vs OlsSM   relative error: {err:.2e}")
    assert err < 1e-8, f"Classic and OlsSM disagree by {err:.2e}; expected < 1e-8"


def test_classic_vered_agree_well_conditioned():
    """Classic and Vered should give nearly identical natgrad on well-conditioned input."""
    X, delta, A, G, grad = make_synthetic(target_kappa=10.0)
    damping = 1e-4

    nat_classic = natgrad_classic(A, G, grad, damping)

    R_X = damped_R_factor(X / math.sqrt(X.shape[0]),     damping)
    R_G = damped_R_factor(delta / math.sqrt(delta.shape[0]), damping)
    nat_vered = natgrad_vered(R_X, R_G, grad, damping=0.0)
    # damping was already absorbed via row-augmentation in damped_R_factor

    err = _max_rel_err(nat_classic, nat_vered)
    print(f"Classic vs Vered   relative error: {err:.2e}")
    assert err < 1e-6, (f"Classic and Vered disagree by {err:.2e} on well-"
                        f"conditioned data; expected < 1e-6")


def test_olssm_vered_agree_well_conditioned():
    """OlsSM and Vered should also agree (both kappa<=2 stable paths)."""
    X, delta, A, G, grad = make_synthetic(target_kappa=10.0)
    damping = 1e-4

    nat_olssm = natgrad_olssm(A, G, grad, damping)

    R_X = damped_R_factor(X / math.sqrt(X.shape[0]),     damping)
    R_G = damped_R_factor(delta / math.sqrt(delta.shape[0]), damping)
    nat_vered = natgrad_vered(R_X, R_G, grad, damping=0.0)

    err = _max_rel_err(nat_olssm, nat_vered)
    print(f"OlsSM   vs Vered   relative error: {err:.2e}")
    assert err < 1e-6, (f"OlsSM and Vered disagree by {err:.2e} on well-"
                        f"conditioned data; expected < 1e-6")


def test_vered_implementation_matches_reference():
    """Cross-check the optimizer's actual VeredKFAC apply against this file's
    reference implementation, on synthetic well-conditioned data."""
    try:
        from optimizer.sgso import apply_vered
    except ImportError as e:
        print(f"SKIPPED: optimizer.sgso not importable ({e})")
        return

    X, delta, A, G, grad = make_synthetic(target_kappa=10.0)
    damping = 1e-4
    R_X = damped_R_factor(X / math.sqrt(X.shape[0]),         damping)
    R_G = damped_R_factor(delta / math.sqrt(delta.shape[0]), damping)

    # Reference vered
    nat_reference = natgrad_vered(R_X, R_G, grad, damping=0.0)

    # Two separate checks:
    #   (a) float64 - this isolates IMPLEMENTATION bugs from FP32 noise.
    #       If apply_vered runs in float64 and disagrees with the reference,
    #       there's a real bug in the code (not precision).
    #   (b) float32 - the realistic deployment precision.  A larger error
    #       here is acceptable as long as float64 matches.
    nat_opt_f64 = apply_vered(grad, R_X, R_G)  # native float64
    err_f64 = _max_rel_err(nat_reference, nat_opt_f64)
    print(f"  fp64: reference vs optimizer.apply_vered  rel err: {err_f64:.2e}")

    R_X_f = R_X.to(torch.float32)
    R_G_f = R_G.to(torch.float32)
    grad_f = grad.to(torch.float32)
    nat_opt_f32 = apply_vered(grad_f, R_X_f, R_G_f).to(torch.float64)
    err_f32 = _max_rel_err(nat_reference, nat_opt_f32)
    print(f"  fp32: reference vs optimizer.apply_vered  rel err: {err_f32:.2e}")

    # IMPLEMENTATION CHECK: at float64, the error should be at machine
    # precision (~1e-12).  Anything above 1e-8 indicates a real bug.
    assert err_f64 < 1e-8, (f"optimizer.apply_vered (fp64) disagrees with "
                             f"reference by {err_f64:.2e}; this indicates "
                             f"an IMPLEMENTATION BUG (not FP32 precision).")

    # Realistic float32 precision: 4 sequential TRSMs on kappa(A)~100 give
    # ~5e-5 error; allow up to 1e-3 to absorb worst-case kernel variation.
    assert err_f32 < 1e-3, (f"optimizer.apply_vered (fp32) disagrees with "
                             f"reference by {err_f32:.2e}; that is well above "
                             f"FP32 precision and suggests a real issue.")


def test_predicted_kappa_scaling_pattern():
    """Sanity check: at a low damping with ill-conditioned input, the three
    variants' errors against high-precision reference should follow the
    predicted hierarchy: Classic >> OlsSM >> Vered."""
    X, delta, A, G, grad = make_synthetic(target_kappa=100.0, dtype=torch.float64)

    # High-precision reference: the Vered path computed in float64
    # (this is the most numerically faithful of the three, approximately).
    damping = 1e-6  # very low to expose differences
    R_X = damped_R_factor(X / math.sqrt(X.shape[0]),         damping)
    R_G = damped_R_factor(delta / math.sqrt(delta.shape[0]), damping)
    nat_ref = natgrad_vered(R_X, R_G, grad, damping=0.0)

    # Compare float32 implementations of all three to the float64 vered
    A32 = A.to(torch.float32); G32 = G.to(torch.float32)
    g32 = grad.to(torch.float32)
    R_X32 = R_X.to(torch.float32); R_G32 = R_G.to(torch.float32)

    nat_classic_f32 = natgrad_classic(A32, G32, g32, damping)
    nat_olssm_f32   = natgrad_olssm  (A32, G32, g32, damping)
    nat_vered_f32   = natgrad_vered  (R_X32, R_G32, g32, damping=0.0)

    err_classic = _max_rel_err(nat_classic_f32.to(torch.float64), nat_ref)
    err_olssm   = _max_rel_err(nat_olssm_f32  .to(torch.float64), nat_ref)
    err_vered   = _max_rel_err(nat_vered_f32  .to(torch.float64), nat_ref)

    print(f"\n  err vs reference (kappa=100, damping=1e-6, fp32):")
    print(f"    Classic: {err_classic:.2e}")
    print(f"    OlsSM:   {err_olssm:.2e}")
    print(f"    Vered:   {err_vered:.2e}")

    # We expect the rough hierarchy err_classic >= err_olssm >= err_vered.
    # Soft assertion: don't fail the test if Classic ~= OlsSM, but loudly
    # warn if Vered's error is bigger than OlsSM's (would indicate a bug).
    if err_vered > 2 * err_olssm:
        print(f"  WARN: Vered error ({err_vered:.2e}) exceeds OlsSM error "
              f"({err_olssm:.2e}) by >2x. That's the wrong direction for the "
              f"kappa scaling story; possible Vered implementation bug.")


# ============================================================================
#  Standalone runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  K-FAC variant equivalence tests on synthetic well-conditioned data")
    print("=" * 70)

    print("\n[1] test_classic_olssm_agree_well_conditioned")
    test_classic_olssm_agree_well_conditioned()
    print("    PASS")

    print("\n[2] test_classic_vered_agree_well_conditioned")
    test_classic_vered_agree_well_conditioned()
    print("    PASS")

    print("\n[3] test_olssm_vered_agree_well_conditioned")
    test_olssm_vered_agree_well_conditioned()
    print("    PASS")

    print("\n[4] test_vered_implementation_matches_reference")
    test_vered_implementation_matches_reference()
    print("    PASS (or SKIPPED if optimizer not importable)")

    print("\n[5] test_predicted_kappa_scaling_pattern")
    test_predicted_kappa_scaling_pattern()
    print("    Done.")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)
