"""
Integration tests for the olssm Python module.

Tests compare Rust output against numpy/scipy reference implementations.
All tests are skipped if the module is not installed (run `maturin develop` first).

TDD: these tests define the contract and are written before implementation passes.
"""

import numpy as np
import pytest

try:
    import scipy.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import olssm
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

skip_no_module = pytest.mark.skipif(not HAS_MODULE, reason="olssm not installed — run: maturin develop")
skip_no_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")

# ---------------------------------------------------------------------------
# Reference implementations (pure numpy/scipy)
# ---------------------------------------------------------------------------

def reference_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """numpy lstsq reference (SVD-based)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def reference_modified_cholesky(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Direct Python translation of Algorithm 1 from the paper."""
    Xy = np.concatenate([X, y.reshape([-1, 1])], axis=1)
    gram_mat = Xy.T @ Xy
    U = scipy.linalg.lu(gram_mat)[2]
    inv_diagU = np.diag(1.0 / np.diag(U))
    return inv_diagU.T @ U

# ---------------------------------------------------------------------------
# Algorithm 1 — modified_cholesky
# ---------------------------------------------------------------------------

class TestModifiedCholesky:

    @skip_no_module
    @skip_no_scipy
    def test_matches_scipy_reference_small(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4)).astype(np.float64)
        y = rng.standard_normal(20).astype(np.float64)
        C_rust = olssm.modified_cholesky(X, y)
        C_py = reference_modified_cholesky(X, y)
        np.testing.assert_allclose(C_rust, C_py, rtol=1e-10, atol=1e-12)

    @skip_no_module
    @skip_no_scipy
    def test_matches_scipy_reference_larger(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 5)).astype(np.float64)
        y = rng.standard_normal(50).astype(np.float64)
        C_rust = olssm.modified_cholesky(X, y)
        C_py = reference_modified_cholesky(X, y)
        np.testing.assert_allclose(C_rust, C_py, rtol=1e-10, atol=1e-12)

    @skip_no_module
    def test_diagonal_is_unit(self):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((40, 6)).astype(np.float64)
        y = rng.standard_normal(40).astype(np.float64)
        C = olssm.modified_cholesky(X, y)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)

    @skip_no_module
    def test_output_shape(self):
        X = np.eye(5, dtype=np.float64)
        X = np.vstack([X, np.ones((1, 5))])  # 6×5
        y = np.ones(6, dtype=np.float64)
        C = olssm.modified_cholesky(X, y)
        assert C.shape == (6, 6)

    @skip_no_module
    def test_upper_triangular(self):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((15, 3)).astype(np.float64)
        y = rng.standard_normal(15).astype(np.float64)
        C = olssm.modified_cholesky(X, y)
        # Lower triangle (below diagonal) should be zero
        np.testing.assert_allclose(np.tril(C, k=-1), 0.0, atol=1e-12)

    @skip_no_module
    def test_dimension_mismatch_raises(self):
        X = np.eye(4, dtype=np.float64)
        y = np.ones(3, dtype=np.float64)  # wrong length
        with pytest.raises(ValueError):
            olssm.modified_cholesky(X, y)

# ---------------------------------------------------------------------------
# solve_ols — combined solver
# ---------------------------------------------------------------------------

class TestSolveOls:

    @skip_no_module
    @pytest.mark.parametrize("n,p,seed", [
        (50, 3, 100),
        (100, 10, 200),
        (200, 20, 300),
    ])
    def test_matches_lstsq(self, n, p, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p)).astype(np.float64)
        y = rng.standard_normal(n).astype(np.float64)
        beta_rust = olssm.solve_ols(X, y)
        beta_ref = reference_ols(X, y)
        np.testing.assert_allclose(beta_rust, beta_ref, rtol=1e-8, atol=1e-10)

    @skip_no_module
    def test_exact_4x2_system(self):
        X = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float64)
        y = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
        beta = olssm.solve_ols(X, y)
        np.testing.assert_allclose(beta, [2.0, 3.0], atol=1e-8)

    @skip_no_module
    def test_residuals_small(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 5)).astype(np.float64)
        true_beta = np.array([1.0, -2.0, 3.0, 0.5, -1.5])
        y = X @ true_beta + rng.standard_normal(30) * 0.01
        beta = olssm.solve_ols(X, y)
        np.testing.assert_allclose(beta, true_beta, atol=0.1)

    @skip_no_module
    def test_output_shape(self):
        X = np.random.default_rng(1).standard_normal((20, 4)).astype(np.float64)
        y = np.ones(20, dtype=np.float64)
        beta = olssm.solve_ols(X, y)
        assert beta.shape == (4,)

# ---------------------------------------------------------------------------
# Algorithm 2 — simplified_gram_schmidt
# ---------------------------------------------------------------------------

class TestSimplifiedGramSchmidt:

    @skip_no_module
    def test_orthogonality_5x3(self):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((30, 4)).astype(np.float64)
        Q = olssm.simplified_gram_schmidt(X)
        QtQ = Q.T @ Q
        off_diag = QtQ - np.diag(np.diag(QtQ))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    @skip_no_module
    def test_output_shape(self):
        X = np.random.default_rng(2).standard_normal((10, 3)).astype(np.float64)
        Q = olssm.simplified_gram_schmidt(X)
        assert Q.shape == X.shape

    @skip_no_module
    def test_orthogonality_larger(self):
        rng = np.random.default_rng(77)
        X = rng.standard_normal((50, 8)).astype(np.float64)
        Q = olssm.simplified_gram_schmidt(X)
        QtQ = Q.T @ Q
        off = QtQ - np.diag(np.diag(QtQ))
        np.testing.assert_allclose(off, 0.0, atol=1e-9)

    @skip_no_module
    def test_identity_matrix_unchanged(self):
        # Columns of I are already orthogonal — Q should equal I (up to scale)
        X = np.eye(4, dtype=np.float64)
        Q = olssm.simplified_gram_schmidt(X)
        # Each column should be a unit vector (since X cols are already orthogonal)
        for j in range(4):
            assert abs(np.dot(Q[:, j], Q[:, j]) - 1.0) < 1e-12

# ---------------------------------------------------------------------------
# Algorithm 3 — weighted_generalized_inverse
# ---------------------------------------------------------------------------

class TestWeightedGeneralizedInverse:

    @skip_no_module
    def test_identity_weight_matches_pseudoinverse(self):
        rng = np.random.default_rng(55)
        X = rng.standard_normal((10, 3)).astype(np.float64)
        W = np.eye(10, dtype=np.float64)
        G = olssm.weighted_generalized_inverse(X, W)
        # With W=I: G = (XᵀX)⁻¹Xᵀ = pinv(X) for full-rank X
        G_ref = np.linalg.pinv(X)
        np.testing.assert_allclose(G, G_ref, atol=1e-8)

    @skip_no_module
    def test_output_shape(self):
        X = np.random.default_rng(3).standard_normal((8, 3)).astype(np.float64)
        W = np.eye(8, dtype=np.float64)
        G = olssm.weighted_generalized_inverse(X, W)
        assert G.shape == (3, 8)  # (p, n)

    @skip_no_module
    def test_generalized_inverse_property(self):
        # G should satisfy: X @ G @ X ≈ X  (Moore-Penrose property 1)
        rng = np.random.default_rng(99)
        n, p = 20, 4
        X = rng.standard_normal((n, p)).astype(np.float64)
        A = rng.standard_normal((n, n)).astype(np.float64)
        W = (A @ A.T + np.eye(n)).astype(np.float64)  # SPD
        G = olssm.weighted_generalized_inverse(X, W)
        # G = (XᵀWX)⁻¹ Xᵀ W  →  G @ X = I_p  (left-inverse property)
        np.testing.assert_allclose(G @ X, np.eye(p), atol=1e-8)

    @skip_no_module
    def test_dimension_mismatch_raises(self):
        X = np.eye(4, dtype=np.float64)
        W = np.eye(3, dtype=np.float64)  # wrong size
        with pytest.raises(ValueError):
            olssm.weighted_generalized_inverse(X, W)

    @skip_no_module
    def test_weighted_ols_solution(self):
        # Verify weighted OLS: beta_w = G @ y should minimise (y-Xβ)ᵀW(y-Xβ)
        rng = np.random.default_rng(42)
        n, p = 15, 3
        X = rng.standard_normal((n, p)).astype(np.float64)
        true_beta = np.array([1.0, -2.0, 0.5])
        y = X @ true_beta + rng.standard_normal(n) * 0.01
        W = np.diag(rng.uniform(0.5, 2.0, n))  # diagonal weight matrix
        G = olssm.weighted_generalized_inverse(X, W)
        beta_w = G @ y
        np.testing.assert_allclose(beta_w, true_beta, atol=0.1)
