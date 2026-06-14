"""
benchmark/verify_vered_damping.py

End-to-end empirical check that VeredKFAC honors its `damping` constructor
argument all the way through to the cached R factors and the applied
natural gradient.  Independent of where the original hardcoded-1e-6 bug
lived; this probe will tell us whether the fix actually propagates.

Three tests per damping value lambda in {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}:

  T1. Smallest eigenvalue of cached R_X^T R_X tracks lambda.
      Construction: X is rank-deficient by design (we use a rank-1 batch
      of activations).  Then X^T X has one nonzero eigenvalue and (n-1)
      zero eigenvalues.  After damping, (X^T X + lambda I) has
      eigenvalues (||x||^2 + lambda, lambda, lambda, ..., lambda).  So
      lambda_min(R^T R) should equal lambda.
      Tolerance: |lambda_min/lambda - 1| < 1e-3.

  T2. Cached R_X^T R_X equals X_true^T X_true + lambda*I in float64.
      Compare element-wise: max |R^T R - (X^T X + lambda I)| / lambda < 1e-3.
      (Normalized by lambda so small-lambda runs aren't artificially
      penalized for floating-point near-zero comparisons.)

  T3. Applied natural gradient matches (G + lambda I)^{-1} dW (A + lambda I)^{-1}
      computed in float64 via direct linear solves.
      Tolerance: rel Frobenius err < 1e-3.

If T1 reports lambda_min ~ 1e-6 for ALL configured lambdas, the hardcoded
damping is still in force.  If T2 fails but T1 passes, the damping
reaches the factor but not the natgrad application (or vice versa).
If all three pass across decades, the damping is fully wired and
controllable.

Usage:
    python benchmark/verify_vered_damping.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from optimizer.vered_kfac import VeredKFAC, apply_vered


def make_layer_and_input(in_dim: int = 16, out_dim: int = 24, batch: int = 32,
                         seed: int = 0):
    """Single Linear layer + a rank-1 activation batch (each row is c_i * v
    for a shared direction v).  Rank-1 makes X^T X have (n-1) zero eigenvalues
    so damping is the smallest eigenvalue of R^T R."""
    torch.manual_seed(seed)
    layer = nn.Linear(in_dim, out_dim).double().float()  # init in float32
    v = torch.randn(in_dim)
    v = v / v.norm()                        # unit direction
    coeffs = torch.randn(batch).abs() + 0.5  # positive scalars
    X = coeffs[:, None] * v[None, :]         # (batch, in_dim), rank 1
    return layer, X, v


def run_one_step_and_extract(damping: float, in_dim=16, out_dim=24,
                              batch=32, device="cpu", seed: int = 0):
    """Build VeredKFAC, run forward+backward, step, then return cached R."""
    layer, X, v = make_layer_and_input(in_dim, out_dim, batch, seed=seed)
    layer = layer.to(device)
    X = X.to(device)

    # Wrap the layer in a tiny module so VeredKFAC has something to iterate
    model = nn.Sequential(layer).to(device)

    # factor_update_freq=1 -> refresh every step so we see effects immediately.
    # momentum=0 to keep the natgrad clean for T3.
    opt = VeredKFAC(model, lr=1.0, damping=damping,
                    factor_update_freq=1, momentum=0.0,
                    gamma=0.0, max_out_dim=0)

    # Synthetic target so backward produces a known dY direction.
    target = torch.randn(batch, out_dim, device=device)

    Y = model(X)
    loss = ((Y - target) ** 2).mean()
    loss.backward()

    # Save dW and dY BEFORE step (step zeros gradients).
    dW = layer.weight.grad.detach().clone()
    if layer.bias is not None and layer.bias.grad is not None:
        db = layer.bias.grad.detach().clone()
    else:
        db = None

    opt.step()

    # Cached factors after step
    factors = opt._factors.get(layer)
    R_X = factors[0].detach().clone() if factors is not None else None
    R_G = factors[1].detach().clone() if factors is not None else None

    return {
        "X": X.detach().clone(),
        "dW": dW,
        "R_X": R_X,
        "R_G": R_G,
        "layer": layer,
        "opt": opt,
        "weight_after_step": layer.weight.detach().clone(),
    }


def test_one_damping(damping: float, seed: int = 0):
    res = run_one_step_and_extract(damping, seed=seed)
    R_X, R_G = res["R_X"], res["R_G"]
    if R_X is None:
        return {"damping": damping, "error": "no R cached -- step did not refresh"}

    # ---- T1: lambda_min(R_X^T R_X) ~= damping  (using rank-1 X) -----------
    R_X_f64 = R_X.to(torch.float64)
    A_v = R_X_f64.T @ R_X_f64
    eigs = torch.linalg.eigvalsh(A_v).cpu().numpy()
    lam_min = float(eigs[0])
    lam_max = float(eigs[-1])

    # ---- T2: R_X^T R_X == X^T X + lambda*I ---------------------------------
    X_f64 = res["X"].to(torch.float64)
    XtX = X_f64.T @ X_f64
    n = XtX.shape[0]
    expected = XtX + damping * torch.eye(n, dtype=torch.float64)
    abs_err = (A_v - expected).abs()
    max_abs_err = float(abs_err.max().item())
    rel_err_T2 = max_abs_err / max(damping, 1e-30)

    # ---- T3: applied natgrad == float64 reference --------------------------
    # Use Vered's apply_vered with cached R; reference uses direct solve.
    dW_f32 = res["dW"]
    ng_vered = apply_vered(dW_f32, R_X, R_G).detach().to(torch.float64)
    # float64 reference: (G+lam I)^-1 @ dW @ (A+lam I)^-1
    R_G_f64 = R_G.to(torch.float64)
    G_plus = R_G_f64.T @ R_G_f64
    dW_f64 = dW_f32.to(torch.float64)
    tmp = torch.linalg.solve(G_plus, dW_f64)
    ng_ref = torch.linalg.solve(A_v, tmp.T).T
    rel_err_T3 = float(((ng_vered - ng_ref).norm() / ng_ref.norm().clamp(min=1e-30)).item())

    return {
        "damping":     damping,
        "lam_min":     lam_min,
        "lam_max":     lam_max,
        "T1_ratio":    lam_min / damping,           # should be ~1.0 if fix works
        "T2_rel_err":  rel_err_T2,                  # should be ~0
        "T3_rel_err":  rel_err_T3,                  # should be ~0
    }


def main():
    print("=" * 80)
    print(" VeredKFAC damping pass-through verification")
    print("=" * 80)
    print(f" Setup: 1 Linear(16->24), rank-1 batch (size 32) so lambda_min(A) ~ damping")
    print()
    print(f" {'damping':>10}  {'lam_min':>10}  {'lam_max':>10}  "
          f"{'T1: lam_min/damping':>22}  {'T2: rel_err':>12}  {'T3: rel_err':>12}  verdict")
    print(" " + "-" * 96)

    dampings = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    rows = []
    for d in dampings:
        r = test_one_damping(d)
        if "error" in r:
            print(f"  {d:.0e}  ERROR: {r['error']}")
            continue
        # Verdicts
        t1_ok = abs(r["T1_ratio"] - 1.0) < 1e-3
        t2_ok = r["T2_rel_err"] < 1e-3
        t3_ok = r["T3_rel_err"] < 1e-3
        verdict_parts = []
        verdict_parts.append("T1 OK" if t1_ok else "T1 FAIL")
        verdict_parts.append("T2 OK" if t2_ok else "T2 FAIL")
        verdict_parts.append("T3 OK" if t3_ok else "T3 FAIL")
        verdict = " | ".join(verdict_parts)
        print(f"  {r['damping']:>10.0e}  {r['lam_min']:>10.3e}  {r['lam_max']:>10.3e}  "
              f"{r['T1_ratio']:>22.6f}  {r['T2_rel_err']:>12.3e}  {r['T3_rel_err']:>12.3e}  {verdict}")
        rows.append(r)

    # Summary diagnostics ----------------------------------------------------
    print()
    print(" Interpretation:")
    if all(abs(r["T1_ratio"] - 1.0) < 1e-3 for r in rows):
        print("   T1 passes across all damping values:  the constructor's damping")
        print("   reaches the cached R factor.  The fix is working at the factor level.")
    else:
        t1_vals = [r["lam_min"] for r in rows]
        if max(t1_vals) / min(t1_vals) < 10:
            stuck_val = sum(t1_vals) / len(t1_vals)
            print(f"   T1 FAILS:  lam_min is stuck near ~{stuck_val:.2e} regardless of")
            print( "   configured damping.  Indicates a hardcoded value still in the factor")
            print( "   construction path -- the fix did not land at this point in the code.")
        else:
            print( "   T1 partially fails:  damping reaches the factor in some regimes but")
            print( "   not others.  Print the per-row table to localize.")

    if all(r["T2_rel_err"] < 1e-3 for r in rows):
        print("   T2 passes:  R^T R matches X^T X + lambda*I in float64.  The damping")
        print("   is being applied as the standard additive regularizer, not as a")
        print("   different operation (e.g. multiplicative).")
    else:
        print( "   T2 FAILS for some lambdas:  damping reaches R but not as A + lambda*I.")
        print( "   Check whether it's applied to R diagonally vs to A directly.")

    if all(r["T3_rel_err"] < 1e-3 for r in rows):
        print("   T3 passes:  Vered's natural gradient matches the float64 reference")
        print("   computed via direct solve on (A+lambda I).  Damping is honored end-to-end.")
    else:
        print( "   T3 FAILS:  cached factor is correct but applied natgrad disagrees with")
        print( "   reference.  Suggests the application path (apply_vered) has a separate")
        print( "   damping treatment from the factor construction.")
    print()
    print(" If all three pass across decades, damping is fully wired and controllable.")


if __name__ == "__main__":
    main()
