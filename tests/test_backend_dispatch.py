"""
Unit tests for optimizer/backend.py — backend dispatch and numerical correctness.

Tests:
  P0 — lu_solve_gram matches numpy lstsq reference
  P0 — lu_inverse_gram matches numpy.linalg.inv
  P0 — eigh_f32 eigenvectors are orthonormal
  P0 — apply_kfac_eigen_f32 matches explicit numpy computation
  P1 — eigh_topk_f32 top-k eigenvalues match full EVD
  P1 — randomized_eigh_f32 approximation is close to exact EVD
  P1 — get_backend_info returns required keys
  P1 — lu_damped_inverse_f32 matches expected formula
"""

import numpy as np
import pytest
import torch

from optimizer.backend import (
    lu_solve_gram,
    lu_inverse_gram,
    lu_damped_inverse_f32,
    eigh_f32,
    eigh_topk_f32,
    apply_kfac_eigen_f32,
    randomized_eigh_f32,
    get_backend_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spd_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Generate a random symmetric positive-definite matrix of size n×n."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A @ A.T + np.eye(n) * 0.1).astype(np.float64)


def _spd_f32(n: int, seed: int = 0) -> np.ndarray:
    return _spd_matrix(n, seed).astype(np.float32)


# ---------------------------------------------------------------------------
# lu_solve_gram
# ---------------------------------------------------------------------------

class TestLuSolveGram:
    def test_vector_rhs(self):
        """lu_solve_gram(A, b) should satisfy A @ x ≈ b for vector b."""
        n = 8
        A = _spd_matrix(n)
        b = np.random.default_rng(1).standard_normal(n)
        x = lu_solve_gram(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-8, rtol=1e-6)

    def test_matrix_rhs(self):
        """lu_solve_gram(A, B) should satisfy A @ X ≈ B for matrix B."""
        n, k = 10, 4
        A = _spd_matrix(n)
        B = np.random.default_rng(2).standard_normal((n, k))
        X = lu_solve_gram(A, B)
        np.testing.assert_allclose(A @ X, B, atol=1e-8, rtol=1e-6)

    def test_matches_numpy_lstsq(self):
        """Solution should match numpy.linalg.lstsq for SPD A."""
        n, k = 12, 3
        A = _spd_matrix(n, seed=3)
        B = np.random.default_rng(3).standard_normal((n, k))

        x_ref, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        x_our = lu_solve_gram(A, B)

        np.testing.assert_allclose(x_our, x_ref, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# lu_inverse_gram
# ---------------------------------------------------------------------------

class TestLuInverseGram:
    def test_inverse_times_original_is_identity(self):
        """A @ A_inv should be approximately I."""
        n = 12
        A = _spd_matrix(n)
        A_inv = lu_inverse_gram(A)
        prod = A @ A_inv
        np.testing.assert_allclose(prod, np.eye(n), atol=1e-8, rtol=1e-6)

    def test_matches_numpy_inv(self):
        """Should match numpy.linalg.inv."""
        n = 8
        A = _spd_matrix(n, seed=5)
        ref = np.linalg.inv(A)
        ours = lu_inverse_gram(A)
        np.testing.assert_allclose(ours, ref, atol=1e-8, rtol=1e-6)


# ---------------------------------------------------------------------------
# lu_damped_inverse_f32
# ---------------------------------------------------------------------------

class TestLuDampedInverse:
    def test_damped_inverse_formula(self):
        """(A + damping*I)^{-1} should match numpy reference."""
        n = 8
        damping = 0.05
        A = _spd_f32(n, seed=7)
        result = lu_damped_inverse_f32(A, damping)

        # numpy reference in float64
        A64 = A.astype(np.float64)
        expected = np.linalg.inv(A64 + damping * np.eye(n))
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_result_shape(self):
        n = 6
        A = _spd_f32(n)
        result = lu_damped_inverse_f32(A, 0.01)
        assert result.shape == (n, n)


# ---------------------------------------------------------------------------
# eigh_f32
# ---------------------------------------------------------------------------

class TestEighF32:
    def test_eigenvectors_are_orthonormal(self):
        """Q @ Q^T should be I (orthonormal columns)."""
        n = 16
        A = _spd_f32(n)
        q, inv_lam = eigh_f32(A, damping=0.01)
        q64 = q.astype(np.float64)
        np.testing.assert_allclose(q64 @ q64.T, np.eye(n), atol=1e-5, rtol=1e-5)

    def test_eigenvalues_positive(self):
        """For SPD A with damping, 1/(λ+δ) should be finite and positive."""
        n = 12
        A = _spd_f32(n)
        _, inv_lam = eigh_f32(A, damping=0.01)
        assert np.all(inv_lam > 0), "All damped inverse eigenvalues should be positive"
        assert np.all(np.isfinite(inv_lam)), "No inf/nan in eigenvalues"

    def test_reconstruction(self):
        """Q @ diag(1/inv_lam) @ Q^T should reconstruct A + damping*I."""
        n = 8
        damping = 0.05
        A = _spd_f32(n)
        q, inv_lam = eigh_f32(A, damping=damping)
        # Reconstruct: Q @ diag(1/(1/inv_lam)) @ Q^T = Q @ diag(inv_lam_orig) @ Q^T
        lam_damped = 1.0 / inv_lam.astype(np.float64)
        A_reconstructed = q.astype(np.float64) @ np.diag(lam_damped) @ q.T.astype(np.float64)
        A_expected = A.astype(np.float64) + damping * np.eye(n)
        np.testing.assert_allclose(A_reconstructed, A_expected, atol=1e-4, rtol=1e-4)

    def test_output_shapes(self):
        """q.shape=(n,n), inv_lam.shape=(n,)."""
        n = 10
        A = _spd_f32(n)
        q, inv_lam = eigh_f32(A, damping=0.01)
        assert q.shape == (n, n)
        assert inv_lam.shape == (n,)


# ---------------------------------------------------------------------------
# apply_kfac_eigen_f32
# ---------------------------------------------------------------------------

class TestApplyKfacEigenF32:
    def test_matches_explicit_numpy(self):
        """apply_kfac_eigen_f32 should match the explicit 4-matmul numpy formula."""
        d_out, d_in = 6, 8
        damping = 0.05

        rng = np.random.default_rng(11)
        G = _spd_f32(d_out, seed=11)
        A = _spd_f32(d_in, seed=12)
        grad = rng.standard_normal((d_out, d_in)).astype(np.float32)

        q_g, inv_lam_g = eigh_f32(G, damping)
        q_a, inv_lam_a = eigh_f32(A, damping)

        # our function
        result = apply_kfac_eigen_f32(q_g, inv_lam_g, grad, q_a, inv_lam_a)

        # explicit numpy reference
        tmp = q_g.T @ grad @ q_a                                  # (d_out, d_in)
        tmp = tmp * np.outer(inv_lam_g, inv_lam_a)
        expected = q_g @ tmp @ q_a.T

        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        d_out, d_in = 5, 7
        G = _spd_f32(d_out, seed=20)
        A = _spd_f32(d_in, seed=21)
        grad = np.random.randn(d_out, d_in).astype(np.float32)

        q_g, inv_lam_g = eigh_f32(G, 0.01)
        q_a, inv_lam_a = eigh_f32(A, 0.01)
        result = apply_kfac_eigen_f32(q_g, inv_lam_g, grad, q_a, inv_lam_a)

        assert result.shape == (d_out, d_in)


# ---------------------------------------------------------------------------
# eigh_topk_f32
# ---------------------------------------------------------------------------

class TestEighTopkF32:
    def test_topk_eigenvalues_match_full(self):
        """Top-k eigenvalues from eigh_topk should match those from full eigh."""
        n, k = 16, 4
        damping = 0.01
        A = _spd_f32(n)

        # Full EVD reference
        _, inv_lam_full = eigh_f32(A, damping)
        # The last k inv_lam values correspond to largest eigenvalues (eigh ascending order)
        # inv_lam is sorted ascending in eigenvalue → first k are *largest* (smallest damp inv)
        # Actually eigh_f32 returns in ascending eigenvalue order from numpy,
        # so inv_lam[-k:] are smallest (largest λ → smallest 1/(λ+δ))
        expected_topk = np.sort(inv_lam_full)[:k]  # k smallest 1/(λ+δ) = k largest λ

        _, inv_lam_k = eigh_topk_f32(A, k=k, damping=damping)
        inv_lam_k_sorted = np.sort(inv_lam_k)

        np.testing.assert_allclose(inv_lam_k_sorted, expected_topk, atol=1e-4, rtol=1e-4)

    def test_topk_eigenvectors_shape(self):
        """q_k.shape should be (n, k)."""
        n, k = 12, 3
        A = _spd_f32(n)
        q_k, inv_lam_k = eigh_topk_f32(A, k=k, damping=0.01)
        assert q_k.shape == (n, k)
        assert inv_lam_k.shape == (k,)

    def test_topk_eigenvectors_orthonormal(self):
        """Top-k eigenvectors should be orthonormal: Q_kᵀ Q_k ≈ I_k."""
        n, k = 16, 5
        A = _spd_f32(n)
        q_k, _ = eigh_topk_f32(A, k=k, damping=0.01)
        gram_k = q_k.T @ q_k
        np.testing.assert_allclose(gram_k, np.eye(k), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# randomized_eigh_f32
# ---------------------------------------------------------------------------

class TestRandomizedEighF32:
    def test_approximation_close_to_exact(self):
        """Randomized EVD top-k eigenvalues should be close to exact top-k."""
        n, k = 20, 4
        damping = 0.01
        A = _spd_f32(n, seed=42)

        q_rand, inv_rand = randomized_eigh_f32(A, k=k, n_iter=2, damping=damping)
        _, inv_exact = eigh_topk_f32(A, k=k, damping=damping)

        # Relative error should be reasonable (< 10% for n_iter=2)
        inv_rand_s  = np.sort(inv_rand)
        inv_exact_s = np.sort(inv_exact)
        rel_err = np.abs(inv_rand_s - inv_exact_s) / (np.abs(inv_exact_s) + 1e-10)
        assert np.all(rel_err < 0.10), f"Randomized EVD too inaccurate: max rel_err={rel_err.max():.3f}"

    def test_output_shape(self):
        n, k = 14, 3
        A = _spd_f32(n)
        q, inv_lam = randomized_eigh_f32(A, k=k, n_iter=1, damping=0.01)
        assert q.shape == (n, k)
        assert inv_lam.shape == (k,)


# ---------------------------------------------------------------------------
# get_backend_info
# ---------------------------------------------------------------------------

class TestGetBackendInfo:
    def test_required_keys_present(self):
        """get_backend_info() must return rust_available, backend, version."""
        info = get_backend_info()
        assert "rust_available" in info
        assert "backend" in info
        assert "version" in info

    def test_rust_available_is_bool(self):
        info = get_backend_info()
        assert isinstance(info["rust_available"], bool)

    def test_backend_is_string(self):
        info = get_backend_info()
        assert isinstance(info["backend"], str)
        assert len(info["backend"]) > 0

    def test_version_is_string(self):
        info = get_backend_info()
        assert isinstance(info["version"], str)
