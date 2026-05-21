"""
Unit tests for diagnostic/inversion.py — Method N and Method K back-target inversions.

Tests validate against mainstream references:
- torch.linalg.pinv for Method N
- Joint-Gaussian conditional formula (computed independently) for Method K
- Hand-coded ground truth for axis-aligned constraints
- Identity tests for activation inverses

Run with:
    python -m unittest tests.test_diagnostic_inversion -v
"""

import unittest

import torch
import torch.nn as nn

from diagnostic.inversion import (
    invert_activation,
    invert_layer,
    kfac_a_inverse,
    naive_pinv_inverse,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

TIGHT = 1e-5
STD = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_layer(d_in: int, d_out: int, seed: int = 0):
    torch.manual_seed(seed)
    W = torch.randn(d_out, d_in, dtype=torch.float64) / (d_in ** 0.5)
    b = torch.randn(d_out, dtype=torch.float64) * 0.1
    return W, b


def random_targets(N: int, d_out: int, seed: int = 1):
    torch.manual_seed(seed)
    return torch.randn(N, d_out, dtype=torch.float64)


def random_prior(d_in: int, seed: int = 2):
    torch.manual_seed(seed)
    mu = torch.randn(d_in, dtype=torch.float64) * 0.5
    A = torch.randn(d_in, d_in, dtype=torch.float64)
    Sigma = A @ A.T / d_in + 0.1 * torch.eye(d_in, dtype=torch.float64)
    return mu, Sigma


# ===========================================================================
# Method N — Naive pseudo-inverse
# ===========================================================================

class TestNaivePinvInverse(unittest.TestCase):

    def test_matches_torch_pinv_underdetermined(self):
        for d_in, d_out in [(8, 4), (16, 8), (64, 32), (128, 64)]:
            with self.subTest(d_in=d_in, d_out=d_out):
                W, b = random_layer(d_in, d_out)
                t_pre = random_targets(N=10, d_out=d_out)
                a_hat = naive_pinv_inverse(W, b, t_pre, eps=1e-10)
                c = t_pre - b.unsqueeze(0)
                a_ref = c @ torch.linalg.pinv(W).T
                self.assertTrue(
                    torch.allclose(a_hat, a_ref, rtol=STD, atol=STD),
                    f"d_in={d_in} d_out={d_out}: "
                    f"max diff = {(a_hat - a_ref).abs().max().item():.2e}",
                )

    def test_matches_torch_pinv_overdetermined(self):
        for d_in, d_out in [(4, 8), (8, 16), (32, 64)]:
            with self.subTest(d_in=d_in, d_out=d_out):
                W, b = random_layer(d_in, d_out)
                torch.manual_seed(99)
                a_true = torch.randn(10, d_in, dtype=torch.float64)
                t_pre = a_true @ W.T + b.unsqueeze(0)
                a_hat = naive_pinv_inverse(W, b, t_pre, eps=1e-10)
                self.assertTrue(
                    torch.allclose(a_hat, a_true, rtol=STD, atol=STD),
                    f"d_in={d_in} d_out={d_out}: "
                    f"max diff = {(a_hat - a_true).abs().max().item():.2e}",
                )

    def test_recovers_exact_solution_square_W(self):
        d = 16
        W, b = random_layer(d, d, seed=42)
        torch.manual_seed(123)
        a_true = torch.randn(10, d, dtype=torch.float64)
        t_pre = a_true @ W.T + b.unsqueeze(0)
        a_hat = naive_pinv_inverse(W, b, t_pre, eps=1e-12)
        self.assertTrue(torch.allclose(a_hat, a_true, rtol=STD, atol=STD))

    def test_satisfies_constraint(self):
        d_in, d_out = 32, 16
        W, b = random_layer(d_in, d_out)
        t_pre = random_targets(N=20, d_out=d_out)
        a_hat = naive_pinv_inverse(W, b, t_pre, eps=1e-10)
        residual = a_hat @ W.T + b.unsqueeze(0) - t_pre
        rel = (residual.norm() / t_pre.norm()).item()
        self.assertLess(rel, STD, f"constraint residual = {rel:.2e}")

    def test_minimum_norm_property(self):
        d_in, d_out = 32, 8
        W, b = random_layer(d_in, d_out)
        t_pre = random_targets(N=5, d_out=d_out)
        U, S, Vt = torch.linalg.svd(W, full_matrices=True)
        null_basis = Vt[d_out:, :]  # (d_in - d_out, d_in)
        c = t_pre - b.unsqueeze(0)
        a_unbias = c @ torch.linalg.pinv(W).T
        proj_null = a_unbias @ null_basis.T
        self.assertLess(proj_null.abs().max().item(), STD)


# ===========================================================================
# Method K — K-FAC-A regularized inverse
# ===========================================================================

class TestKFACAInverse(unittest.TestCase):

    def test_axis_aligned_recovers_known_coords_exactly(self):
        d_in, d_out = 3, 2
        W = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        b = torch.zeros(d_out, dtype=torch.float64)
        mu = torch.tensor([0.0, 0.0, 7.0], dtype=torch.float64)
        Sigma = torch.eye(d_in, dtype=torch.float64)
        t_pre = torch.tensor([[5.0, 3.0]], dtype=torch.float64)
        a_hat = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        expected = torch.tensor([[5.0, 3.0, 7.0]], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(a_hat, expected, rtol=TIGHT, atol=TIGHT),
            f"got {a_hat.tolist()}, expected {expected.tolist()}",
        )

    def test_axis_aligned_uses_correlation_for_unknowns(self):
        d_in, d_out = 3, 2
        W = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        b = torch.zeros(d_out, dtype=torch.float64)
        mu = torch.zeros(d_in, dtype=torch.float64)
        Sigma = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]],
            dtype=torch.float64,
        )
        t_pre = torch.tensor([[5.0, 3.0]], dtype=torch.float64)
        a_hat = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        expected = torch.tensor([[5.0, 3.0, 2.5]], dtype=torch.float64)
        self.assertTrue(torch.allclose(a_hat, expected, rtol=TIGHT, atol=TIGHT))

    def test_recovers_mu_when_target_is_W_mu(self):
        d_in, d_out = 16, 8
        W, b = random_layer(d_in, d_out)
        mu, Sigma = random_prior(d_in)
        t_pre = mu @ W.T + b.unsqueeze(0)
        t_pre = t_pre.expand(5, -1)
        a_hat = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        expected = mu.unsqueeze(0).expand(5, -1)
        self.assertTrue(torch.allclose(a_hat, expected, rtol=STD, atol=STD))

    def test_matches_independent_conditional_formula(self):
        torch.manual_seed(7)
        d_in, d_out = 12, 6
        W, b = random_layer(d_in, d_out, seed=7)
        mu, Sigma = random_prior(d_in, seed=11)
        WSWt = W @ Sigma @ W.T
        SWt = Sigma @ W.T
        WSWt_inv = torch.linalg.inv(
            WSWt + 1e-12 * torch.eye(d_out, dtype=torch.float64)
        )
        W_mu = W @ mu
        t_pre = random_targets(N=4, d_out=d_out)
        c = t_pre - b.unsqueeze(0)
        a_ref = mu.unsqueeze(0) + (c - W_mu.unsqueeze(0)) @ WSWt_inv.T @ SWt.T
        a_hat = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        self.assertTrue(
            torch.allclose(a_hat, a_ref, rtol=STD, atol=STD),
            f"max diff = {(a_hat - a_ref).abs().max().item():.2e}",
        )

    def test_constraint_satisfied(self):
        d_in, d_out = 32, 16
        W, b = random_layer(d_in, d_out)
        mu, Sigma = random_prior(d_in)
        t_pre = random_targets(N=10, d_out=d_out)
        a_hat = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        residual = a_hat @ W.T + b.unsqueeze(0) - t_pre
        rel = (residual.norm() / t_pre.norm()).item()
        self.assertLess(rel, STD, f"constraint residual = {rel:.2e}")

    def test_recovers_exact_solution_square_W(self):
        d = 16
        W, b = random_layer(d, d, seed=42)
        mu, Sigma = random_prior(d)
        torch.manual_seed(123)
        a_true = torch.randn(5, d, dtype=torch.float64)
        t_pre = a_true @ W.T + b.unsqueeze(0)
        a_hat = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        self.assertTrue(torch.allclose(a_hat, a_true, rtol=STD, atol=STD))


# ===========================================================================
# Comparison between methods
# ===========================================================================

class TestMethodComparison(unittest.TestCase):

    def test_methods_disagree_on_random_inputs(self):
        d_in, d_out = 32, 16
        W, b = random_layer(d_in, d_out)
        mu, Sigma = random_prior(d_in)
        t_pre = random_targets(N=5, d_out=d_out)
        a_naive = naive_pinv_inverse(W, b, t_pre, eps=1e-10)
        a_kfac = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        self.assertGreater(
            (a_naive - a_kfac).abs().max().item(), 1e-3,
            "Methods agreed too closely",
        )

    def test_k_collapses_to_n_when_prior_is_zero_mean_identity(self):
        d_in, d_out = 32, 16
        W, b = random_layer(d_in, d_out)
        mu = torch.zeros(d_in, dtype=torch.float64)
        Sigma = torch.eye(d_in, dtype=torch.float64)
        t_pre = random_targets(N=5, d_out=d_out)
        a_naive = naive_pinv_inverse(W, b, t_pre, eps=1e-12)
        a_kfac = kfac_a_inverse(W, b, t_pre, mu, Sigma, sigma2=1e-12)
        self.assertTrue(
            torch.allclose(a_naive, a_kfac, rtol=STD, atol=STD),
            f"max diff = {(a_naive - a_kfac).abs().max().item():.2e}",
        )


# ===========================================================================
# Activation inverse
# ===========================================================================

class TestInvertActivation(unittest.TestCase):

    def test_none_passes_through(self):
        a_post = torch.randn(5, 8, dtype=torch.float64)
        result = invert_activation(a_post, torch.zeros_like(a_post), None)
        self.assertTrue(torch.equal(result, a_post))

    def test_tanh_inverse(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
        a_post = torch.tanh(x).unsqueeze(0)
        result = invert_activation(a_post, torch.zeros_like(a_post), nn.Tanh())
        self.assertTrue(torch.allclose(result, x.unsqueeze(0), rtol=TIGHT, atol=TIGHT))

    def test_tanh_inverse_handles_boundary(self):
        a_post = torch.tensor([[-1.0, 1.0, 0.0]], dtype=torch.float64)
        result = invert_activation(a_post, torch.zeros_like(a_post), nn.Tanh())
        self.assertTrue(torch.isfinite(result).all().item())

    def test_sigmoid_inverse(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
        a_post = torch.sigmoid(x).unsqueeze(0)
        result = invert_activation(a_post, torch.zeros_like(a_post), nn.Sigmoid())
        self.assertTrue(torch.allclose(result, x.unsqueeze(0), rtol=STD, atol=STD))

    def test_relu_inverse_live_units(self):
        x = torch.tensor([[0.5, 1.5, 2.5]], dtype=torch.float64)
        a_post = torch.relu(x)
        result = invert_activation(a_post, x, nn.ReLU())
        self.assertTrue(torch.allclose(result, x, rtol=TIGHT, atol=TIGHT))

    def test_relu_inverse_dead_units_use_forward_pre(self):
        x = torch.tensor([[-0.5, -1.5, -2.5]], dtype=torch.float64)
        a_post = torch.relu(x)
        result = invert_activation(a_post, x, nn.ReLU())
        self.assertTrue(torch.allclose(result, x, rtol=TIGHT, atol=TIGHT))

    def test_relu_inverse_mixed(self):
        x = torch.tensor([[-1.0, 0.5, -0.5, 1.5]], dtype=torch.float64)
        a_post = torch.relu(x)
        a_pre_forward = x.clone()
        a_pre_forward[a_pre_forward <= 0] = -99.0
        result = invert_activation(a_post, a_pre_forward, nn.ReLU())
        expected = torch.tensor([[-99.0, 0.5, -99.0, 1.5]], dtype=torch.float64)
        self.assertTrue(torch.allclose(result, expected, rtol=TIGHT, atol=TIGHT))

    def test_leaky_relu_inverse(self):
        slope = 0.1
        x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]], dtype=torch.float64)
        act = nn.LeakyReLU(negative_slope=slope)
        a_post = act(x)
        result = invert_activation(a_post, torch.zeros_like(x), act)
        self.assertTrue(torch.allclose(result, x, rtol=STD, atol=STD))

    def test_elu_inverse(self):
        act = nn.ELU(alpha=1.0)
        x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]], dtype=torch.float64)
        a_post = act(x)
        result = invert_activation(a_post, torch.zeros_like(x), act)
        self.assertTrue(torch.allclose(result, x, rtol=STD, atol=STD))


# ===========================================================================
# Top-level invert_layer integration
# ===========================================================================

class TestInvertLayer(unittest.TestCase):

    def test_invert_layer_linear_only_naive(self):
        d_in, d_out = 32, 16
        W, b = random_layer(d_in, d_out)
        torch.manual_seed(0)
        a_true = torch.randn(5, d_in, dtype=torch.float64)
        a_post = a_true @ W.T + b.unsqueeze(0)
        a_hat = invert_layer(
            W, b, a_post,
            a_pre_forward=torch.zeros_like(a_post),
            activation=None, method="naive", eps=1e-12,
        )
        residual = a_hat @ W.T + b.unsqueeze(0) - a_post
        self.assertLess((residual.norm() / a_post.norm()).item(), STD)

    def test_invert_layer_with_relu_and_kfac(self):
        d_in, d_out = 16, 8
        W, b = random_layer(d_in, d_out, seed=3)
        mu, Sigma = random_prior(d_in, seed=4)
        torch.manual_seed(5)
        a_in = torch.randn(10, d_in, dtype=torch.float64)
        a_pre_forward = a_in @ W.T + b.unsqueeze(0)
        a_post = torch.relu(a_pre_forward)
        a_hat = invert_layer(
            W, b, a_post,
            a_pre_forward=a_pre_forward,
            activation=nn.ReLU(), method="kfac_a",
            mu_a=mu, Sigma_a=Sigma, sigma2=1e-12,
        )
        a_post_reproduced = torch.relu(a_hat @ W.T + b.unsqueeze(0))
        self.assertTrue(
            torch.allclose(a_post_reproduced, a_post, rtol=STD, atol=STD),
            f"max diff = {(a_post_reproduced - a_post).abs().max().item():.2e}",
        )

    def test_invert_layer_unknown_method_raises(self):
        d_in, d_out = 4, 2
        W, b = random_layer(d_in, d_out)
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            invert_layer(
                W, b,
                torch.zeros(1, d_out, dtype=torch.float64),
                a_pre_forward=torch.zeros(1, d_out, dtype=torch.float64),
                activation=None, method="bogus",
            )

    def test_invert_layer_kfac_requires_prior(self):
        d_in, d_out = 4, 2
        W, b = random_layer(d_in, d_out)
        with self.assertRaisesRegex(ValueError, "mu_a"):
            invert_layer(
                W, b,
                torch.zeros(1, d_out, dtype=torch.float64),
                a_pre_forward=torch.zeros(1, d_out, dtype=torch.float64),
                activation=None, method="kfac_a",
            )


# ===========================================================================
# Input validation
# ===========================================================================

class TestInputValidation(unittest.TestCase):

    def test_naive_rejects_wrong_W_dim(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            naive_pinv_inverse(torch.randn(3, 4, 5), None, torch.randn(1, 4))

    def test_naive_rejects_mismatched_t_pre(self):
        with self.assertRaisesRegex(ValueError, "d_out"):
            naive_pinv_inverse(torch.randn(4, 8), None, torch.randn(2, 5))

    def test_kfac_rejects_wrong_Sigma_shape(self):
        with self.assertRaisesRegex(ValueError, "Sigma_a"):
            kfac_a_inverse(
                torch.randn(4, 8), None, torch.randn(2, 4),
                torch.zeros(8), torch.eye(7),
            )

    def test_kfac_rejects_wrong_mu_shape(self):
        with self.assertRaisesRegex(ValueError, "mu_a"):
            kfac_a_inverse(
                torch.randn(4, 8), None, torch.randn(2, 4),
                torch.zeros(7), torch.eye(8),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
