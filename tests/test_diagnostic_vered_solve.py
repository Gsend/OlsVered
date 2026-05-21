"""
Unit tests for diagnostic/vered_solve.py — inversion-free SPD solver.

Tests validate against mainstream references:
- torch.linalg.solve (general LU)
- torch.cholesky_solve (LAPACK Cholesky)
- scipy.linalg.lu_solve (LAPACK LU)
- direct inverse-then-multiply (worst-case reference)

Run with:
    python -m unittest tests.test_diagnostic_vered_solve -v
"""

import unittest

import numpy as np
import scipy.linalg
import torch

from diagnostic.vered_solve import (
    CholFactor,
    vered_apply,
    vered_decompose,
    vered_solve,
    vered_solve_batched,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

TIGHT = 1e-5
STD = 1e-4
LOOSE = 5e-2

SPD_DIMS = (4, 32, 128)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spd(d: int) -> torch.Tensor:
    """Well-conditioned random SPD matrix at dimension d."""
    torch.manual_seed(d * 7)
    A = torch.randn(d, d, dtype=torch.float64)
    return A @ A.T + torch.eye(d, dtype=torch.float64)


def make_rhs_vector(d: int) -> torch.Tensor:
    torch.manual_seed(d * 11)
    return torch.randn(d, dtype=torch.float64)


def make_rhs_batch(d: int, k: int = 5) -> torch.Tensor:
    torch.manual_seed(d * 13)
    return torch.randn(d, k, dtype=torch.float64)


# ===========================================================================
# Class 1 — agreement with mainstream solvers
# ===========================================================================

class TestVeredSolveMatchesMainstream(unittest.TestCase):

    def test_matches_torch_solve(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                r = make_rhs_vector(d)
                x_vered = vered_solve(G, r, damping=1e-10)
                x_ref = torch.linalg.solve(G, r)
                self.assertTrue(
                    torch.allclose(x_vered, x_ref, rtol=TIGHT, atol=TIGHT),
                    f"d={d}: max diff = {(x_vered - x_ref).abs().max().item():.2e}",
                )

    def test_matches_torch_cholesky_solve(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                r = make_rhs_vector(d)
                L_ref = torch.linalg.cholesky(G)
                x_ref = torch.cholesky_solve(r.unsqueeze(-1), L_ref).squeeze(-1)
                x_vered = vered_solve(G, r, damping=1e-10)
                self.assertTrue(torch.allclose(x_vered, x_ref, rtol=TIGHT, atol=TIGHT))

    def test_matches_scipy_lu_solve(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                r = make_rhs_vector(d)
                lu, piv = scipy.linalg.lu_factor(G.numpy())
                x_ref_np = scipy.linalg.lu_solve((lu, piv), r.numpy())
                x_vered = vered_solve(G, r, damping=1e-10)
                self.assertTrue(
                    np.allclose(x_vered.numpy(), x_ref_np, rtol=TIGHT, atol=TIGHT)
                )

    def test_matches_explicit_inverse(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                r = make_rhs_vector(d)
                x_inv = torch.linalg.inv(G) @ r
                x_vered = vered_solve(G, r, damping=1e-10)
                self.assertTrue(torch.allclose(x_vered, x_inv, rtol=TIGHT, atol=TIGHT))


# ===========================================================================
# Class 2 — equation satisfaction
# ===========================================================================

class TestVeredSolveSatisfiesEquation(unittest.TestCase):

    def test_residual_is_small_for_vector_rhs(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                r = make_rhs_vector(d)
                x = vered_solve(G, r, damping=1e-10)
                rel_res = ((G @ x - r).norm() / r.norm()).item()
                self.assertLess(rel_res, TIGHT, f"d={d}: rel_res = {rel_res:.2e}")

    def test_residual_is_small_for_batch_rhs(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                R = make_rhs_batch(d)
                X = vered_solve_batched(G, R, damping=1e-10)
                rel_res = ((G @ X - R).norm() / R.norm()).item()
                self.assertLess(rel_res, TIGHT)


# ===========================================================================
# Class 3 — Cholesky factor properties
# ===========================================================================

class TestCholFactorProperties(unittest.TestCase):

    def test_factor_is_lower_triangular(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                factor = vered_decompose(G, damping=1e-10)
                upper_part = torch.triu(factor.L, diagonal=1)
                self.assertLess(upper_part.abs().max().item(), 1e-12)

    def test_factor_reconstructs_input(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                factor = vered_decompose(G, damping=1e-10)
                reconstructed = factor.L @ factor.L.T
                eye = torch.eye(d, dtype=G.dtype)
                expected = G + factor.damping_used * eye
                self.assertTrue(
                    torch.allclose(reconstructed, expected, rtol=TIGHT, atol=TIGHT)
                )

    def test_factor_positive_diagonal_for_pd_input(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                factor = vered_decompose(G, damping=1e-10)
                self.assertTrue((factor.L.diag() > 0).all().item())


# ===========================================================================
# Class 4 — progressive damping retry
# ===========================================================================

class TestProgressiveDamping(unittest.TestCase):

    def test_damping_engages_on_singular_input(self):
        d = 16
        torch.manual_seed(0)
        v = torch.randn(d, dtype=torch.float64)
        G = torch.outer(v, v)  # rank 1
        r = torch.randn(d, dtype=torch.float64)
        x = vered_solve(G, r, damping=1e-6)
        self.assertTrue(torch.isfinite(x).all().item())

    def test_damping_used_field_reflects_actual_damping(self):
        d = 8
        G = torch.eye(d, dtype=torch.float64)
        factor = vered_decompose(G, damping=1e-3)
        self.assertAlmostEqual(factor.damping_used, 1e-3, delta=1e-13)
        self.assertFalse(factor.used_eigh_fallback)

    def test_eigh_fallback_engages_on_negative_definite(self):
        d = 8
        Q = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
        eigvals = -torch.ones(d, dtype=torch.float64)
        G = Q @ torch.diag(eigvals) @ Q.T
        factor = vered_decompose(G, damping=1e-6)
        self.assertTrue(factor.used_eigh_fallback)
        self.assertTrue(torch.isfinite(factor.L).all().item())

    def test_eigh_fallback_can_be_disabled(self):
        d = 8
        Q = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
        eigvals = -torch.ones(d, dtype=torch.float64)
        G = Q @ torch.diag(eigvals) @ Q.T
        with self.assertRaises(torch.linalg.LinAlgError):
            vered_decompose(G, damping=1e-6, eigh_fallback=False)


# ===========================================================================
# Class 5 — batched solve correctness
# ===========================================================================

class TestBatchedSolve(unittest.TestCase):

    def test_batched_matches_individual_solves(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                R = make_rhs_batch(d)
                X_batched = vered_solve_batched(G, R, damping=1e-10)
                X_individual = torch.stack(
                    [vered_solve(G, R[:, k], damping=1e-10) for k in range(R.shape[1])],
                    dim=1,
                )
                self.assertTrue(
                    torch.allclose(X_batched, X_individual, rtol=TIGHT, atol=TIGHT)
                )

    def test_vered_apply_reuses_factor(self):
        for d in SPD_DIMS:
            with self.subTest(d=d):
                G = make_spd(d)
                r = make_rhs_vector(d)
                factor = vered_decompose(G, damping=1e-10)
                x1 = vered_apply(factor, r)
                self.assertTrue(
                    torch.allclose(x1, torch.linalg.solve(G, r), rtol=TIGHT, atol=TIGHT)
                )
                torch.manual_seed(99 + d)
                r2 = torch.randn_like(r)
                x2 = vered_apply(factor, r2)
                self.assertTrue(
                    torch.allclose(x2, torch.linalg.solve(G, r2), rtol=TIGHT, atol=TIGHT)
                )


# ===========================================================================
# Class 6 — input validation and edge cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_rejects_non_square(self):
        with self.assertRaisesRegex(ValueError, "square"):
            vered_solve(torch.randn(4, 5), torch.randn(4))

    def test_rejects_3d_input(self):
        with self.assertRaisesRegex(ValueError, "square"):
            vered_solve(torch.randn(2, 3, 3), torch.randn(3))

    def test_handles_nan_input_gracefully(self):
        d = 8
        G = torch.eye(d, dtype=torch.float64)
        G[0, 0] = float("nan")
        x = vered_solve(G, torch.randn(d, dtype=torch.float64), damping=1e-6)
        self.assertTrue(torch.isfinite(x).all().item())

    def test_minimum_size(self):
        G = torch.tensor([[4.0]], dtype=torch.float64)
        r = torch.tensor([8.0], dtype=torch.float64)
        x = vered_solve(G, r, damping=1e-10)
        self.assertTrue(
            torch.allclose(x, torch.tensor([2.0], dtype=torch.float64),
                           rtol=TIGHT, atol=TIGHT)
        )

    def test_batched_rhs_validation(self):
        G = make_spd(8)
        r1d = torch.randn(G.shape[0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "2D"):
            vered_solve_batched(G, r1d)


# ===========================================================================
# Class 7 — dtype compatibility
# ===========================================================================

class TestDtypeCompat(unittest.TestCase):

    def test_float32_solve(self):
        torch.manual_seed(42)
        d = 32
        A = torch.randn(d, d, dtype=torch.float32)
        G = A @ A.T + torch.eye(d, dtype=torch.float32)
        r = torch.randn(d, dtype=torch.float32)
        x_vered = vered_solve(G, r, damping=1e-6)
        x_ref = torch.linalg.solve(G, r)
        self.assertTrue(torch.allclose(x_vered, x_ref, rtol=1e-3, atol=1e-3))

    def test_float64_solve(self):
        G = make_spd(64)
        r = make_rhs_vector(64)
        x_vered = vered_solve(G, r, damping=1e-10)
        x_ref = torch.linalg.solve(G, r)
        self.assertTrue(torch.allclose(x_vered, x_ref, rtol=TIGHT, atol=TIGHT))


# ===========================================================================
# Class 8 — symmetrization
# ===========================================================================

class TestSymmetrization(unittest.TestCase):

    def test_handles_slightly_asymmetric_input(self):
        G = make_spd(32)
        d = G.shape[0]
        torch.manual_seed(7)
        noise = 1e-8 * torch.randn(d, d, dtype=G.dtype)
        G_asym = G + noise
        r = torch.randn(d, dtype=G.dtype)
        x_vered = vered_solve(G_asym, r, damping=1e-10)
        G_sym = (G_asym + G_asym.T) * 0.5
        x_ref = torch.linalg.solve(G_sym, r)
        self.assertTrue(torch.allclose(x_vered, x_ref, rtol=TIGHT, atol=TIGHT))


if __name__ == "__main__":
    unittest.main(verbosity=2)
