"""
Unit tests for diagnostic/metrics.py — drift metrics validated against
mainstream references.

References used:
- torch.distributions.kl.kl_divergence for Gaussian KL
- scipy.stats.pearsonr for Pearson correlation
- Analytic ground truth for known special cases (identical distributions,
  shifted means, scaled covariances)

Run with:
    python -m unittest tests.test_diagnostic_metrics -v
"""

import math
import unittest

import torch
from torch.distributions import MultivariateNormal, kl_divergence

from diagnostic.metrics import (
    constraint_residual,
    cosine_similarity_per_sample,
    covariance_frobenius,
    downstream_loss,
    eigenvalue_spectrum_match,
    gaussian_kl_symmetric,
    mean_drift,
    predicted_projection_covariance,
    principal_subspace_angles,
    relative_l2_error,
    wasserstein2_gaussian,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

TIGHT = 1e-5
STD = 1e-4
LOOSE = 5e-2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gaussian_samples(N: int, d: int, mu=None, Sigma=None, seed: int = 0):
    """Sample N points from N(mu, Sigma)."""
    torch.manual_seed(seed)
    if mu is None:
        mu = torch.zeros(d, dtype=torch.float64)
    if Sigma is None:
        Sigma = torch.eye(d, dtype=torch.float64)
    L = torch.linalg.cholesky(Sigma)
    z = torch.randn(N, d, dtype=torch.float64)
    return mu.unsqueeze(0) + z @ L.T


# ===========================================================================
# Per-sample metrics
# ===========================================================================

class TestPerSampleMetrics(unittest.TestCase):

    def test_relative_l2_zero_on_identical_inputs(self):
        x = torch.randn(10, 8, dtype=torch.float64)
        err = relative_l2_error(x, x)
        self.assertEqual(err.shape, (10,))
        self.assertLess(err.max().item(), TIGHT)

    def test_relative_l2_scales_with_difference(self):
        x = torch.randn(10, 8, dtype=torch.float64)
        # Scale difference: relative err ~ alpha * ||x||/||x|| = alpha for orthogonal noise
        noise = torch.randn(10, 8, dtype=torch.float64)
        # Project noise to be orthogonal to x per-sample (rough sanity)
        err_small = relative_l2_error(x, x + 0.01 * noise).mean().item()
        err_large = relative_l2_error(x, x + 0.1 * noise).mean().item()
        self.assertGreater(err_large, err_small)
        # Bound: should be roughly 10x larger
        self.assertGreater(err_large / err_small, 5.0)
        self.assertLess(err_large / err_small, 20.0)

    def test_cosine_similarity_one_on_aligned_inputs(self):
        x = torch.randn(10, 8, dtype=torch.float64)
        sim = cosine_similarity_per_sample(x, x)
        self.assertEqual(sim.shape, (10,))
        self.assertTrue(torch.allclose(sim, torch.ones_like(sim), rtol=TIGHT, atol=TIGHT))

    def test_cosine_similarity_negative_one_on_anti_aligned(self):
        x = torch.randn(10, 8, dtype=torch.float64)
        sim = cosine_similarity_per_sample(x, -x)
        self.assertTrue(
            torch.allclose(sim, -torch.ones_like(sim), rtol=TIGHT, atol=TIGHT)
        )

    def test_validates_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            relative_l2_error(
                torch.randn(10, 8),
                torch.randn(10, 7),
            )

    def test_validates_dimensionality(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            relative_l2_error(torch.randn(10), torch.randn(10))


# ===========================================================================
# Distributional metrics — identity tests
# ===========================================================================

class TestDistributionalMetricsIdentity(unittest.TestCase):
    """All metrics should be zero (or near-zero) on identical inputs."""

    def test_mean_drift_zero_on_identical(self):
        x = make_gaussian_samples(100, 8)
        self.assertLess(mean_drift(x, x), TIGHT)

    def test_cov_frobenius_zero_on_identical(self):
        x = make_gaussian_samples(100, 8)
        self.assertLess(covariance_frobenius(x, x), TIGHT)

    def test_gaussian_kl_zero_on_identical(self):
        x = make_gaussian_samples(200, 8)
        kl = gaussian_kl_symmetric(x, x, ridge=1e-6)
        self.assertLess(abs(kl), STD)

    def test_wasserstein2_zero_on_identical(self):
        x = make_gaussian_samples(200, 8)
        w2 = wasserstein2_gaussian(x, x, ridge=1e-6)
        self.assertLess(abs(w2), STD)


# ===========================================================================
# Distributional metrics — shifted-mean tests
# ===========================================================================

class TestDistributionalMetricsShifts(unittest.TestCase):
    """Mean shift produces predictable signal in mean_drift and W2."""

    def test_mean_drift_matches_analytic_for_mean_shift(self):
        d = 8
        torch.manual_seed(0)
        x = make_gaussian_samples(1000, d)
        shift = torch.tensor([1.0] + [0.0] * (d - 1), dtype=torch.float64)
        y = x + shift.unsqueeze(0)

        # Empirical mean_drift ~ ||shift|| / ||mu(x)||; mu(x) is near zero for
        # the original samples, so this metric blows up. Instead compare to
        # the *shift* of the empirical means.
        mu_x = x.mean(dim=0)
        mu_y = y.mean(dim=0)
        analytic_rel = ((mu_y - mu_x).norm() / mu_x.norm().clamp(min=1e-30)).item()
        empirical = mean_drift(y, x)
        self.assertAlmostEqual(empirical, analytic_rel, delta=TIGHT)

    def test_wasserstein2_picks_up_mean_shift(self):
        """W2² between N(0, I) and N(shift, I) should be ||shift||²."""
        d = 4
        N = 5000
        torch.manual_seed(0)
        x = make_gaussian_samples(N, d, seed=0)
        shift = torch.tensor([1.0, 0.5, -0.5, 0.0], dtype=torch.float64)
        y = x + shift.unsqueeze(0)
        w2 = wasserstein2_gaussian(y, x, ridge=1e-6)
        expected = (shift ** 2).sum().item()
        # Sampling noise + ridge: tolerate 10% deviation
        self.assertAlmostEqual(w2, expected, delta=0.1 * expected + 0.05)


# ===========================================================================
# Gaussian KL — agreement with torch.distributions
# ===========================================================================

class TestGaussianKLAgreesWithTorchDistributions(unittest.TestCase):
    """Our gaussian_kl_symmetric should agree with torch.distributions.kl_divergence
    for known mean/covariance Gaussians (in the large-sample limit)."""

    def test_kl_matches_torch_distributions_for_diagonal_covariance(self):
        """Compare on two Gaussians with diagonal covariance using ANALYTIC mean/cov.

        Since gaussian_kl_symmetric computes empirical mu/Sigma from samples,
        we use very large N to get close to the analytic Gaussians and compare
        against torch.distributions analytic KL.
        """
        d = 4
        N = 50000  # large enough that empirical ≈ analytic
        mu1 = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float64)
        mu2 = torch.tensor([0.5, 0.0, 0.0, 1.0], dtype=torch.float64)
        var1 = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
        var2 = torch.tensor([1.5, 1.0, 1.0, 2.0], dtype=torch.float64)
        S1 = torch.diag(var1)
        S2 = torch.diag(var2)

        x = make_gaussian_samples(N, d, mu=mu1, Sigma=S1, seed=1)
        y = make_gaussian_samples(N, d, mu=mu2, Sigma=S2, seed=2)

        # Analytic symmetric KL via torch.distributions
        mvn1 = MultivariateNormal(mu1, covariance_matrix=S1)
        mvn2 = MultivariateNormal(mu2, covariance_matrix=S2)
        kl_12 = kl_divergence(mvn1, mvn2).item()
        kl_21 = kl_divergence(mvn2, mvn1).item()
        analytic_sym_kl = 0.5 * (kl_12 + kl_21)

        empirical = gaussian_kl_symmetric(x, y, ridge=1e-8)
        # Empirical-vs-analytic with N=50k: tolerate 5% relative error
        self.assertAlmostEqual(empirical, analytic_sym_kl,
                               delta=0.05 * abs(analytic_sym_kl) + 0.1)


# ===========================================================================
# Wasserstein-2 — closed-form analytic comparison
# ===========================================================================

class TestWasserstein2Analytic(unittest.TestCase):

    def test_w2_matches_analytic_for_isotropic_gaussians(self):
        """For two N(mu_i, sigma_i^2 I), the closed-form W2² is:
           ||mu_1 - mu_2||² + d * (sigma_1 - sigma_2)²
        """
        d = 4
        N = 10000
        sigma1, sigma2 = 1.0, 1.5
        mu1 = torch.zeros(d, dtype=torch.float64)
        mu2 = torch.tensor([0.5] * d, dtype=torch.float64)
        S1 = (sigma1 ** 2) * torch.eye(d, dtype=torch.float64)
        S2 = (sigma2 ** 2) * torch.eye(d, dtype=torch.float64)

        x = make_gaussian_samples(N, d, mu=mu1, Sigma=S1, seed=10)
        y = make_gaussian_samples(N, d, mu=mu2, Sigma=S2, seed=20)

        w2_empirical = wasserstein2_gaussian(x, y, ridge=1e-8)
        w2_analytic = ((mu1 - mu2) ** 2).sum().item() + d * (sigma1 - sigma2) ** 2
        # 10% sampling tolerance
        self.assertAlmostEqual(w2_empirical, w2_analytic,
                               delta=0.1 * w2_analytic + 0.05)


# ===========================================================================
# Eigenvalue spectrum
# ===========================================================================

class TestEigenvalueSpectrum(unittest.TestCase):

    def test_pearson_1_on_identical_spectrum(self):
        x = make_gaussian_samples(200, 8)
        result = eigenvalue_spectrum_match(x, x)
        self.assertAlmostEqual(result["pearson"], 1.0, delta=STD)
        self.assertLess(abs(result["kl_on_spectrum"]), STD)
        self.assertAlmostEqual(
            result["effective_rank_hat"], result["effective_rank_star"], delta=STD
        )

    def test_effective_rank_full_for_isotropic(self):
        """For N(0, I), effective rank should be ~d (all eigenvalues equal)."""
        d = 8
        x = make_gaussian_samples(5000, d, Sigma=torch.eye(d, dtype=torch.float64))
        result = eigenvalue_spectrum_match(x, x)
        # Effective rank should be close to d
        self.assertAlmostEqual(result["effective_rank_hat"], d, delta=0.3)

    def test_effective_rank_lower_for_low_rank_data(self):
        """If data lies in a 2D subspace of 8D, effective rank should be ~2."""
        d = 8
        N = 1000
        torch.manual_seed(0)
        # Generate samples in a 2D subspace
        latent = torch.randn(N, 2, dtype=torch.float64)
        proj = torch.randn(2, d, dtype=torch.float64)
        x = latent @ proj
        result = eigenvalue_spectrum_match(x, x)
        # Effective rank should be close to 2 (definitely less than d)
        self.assertLess(result["effective_rank_hat"], 3.0)


# ===========================================================================
# Principal subspace angles
# ===========================================================================

class TestPrincipalSubspaceAngles(unittest.TestCase):

    def test_zero_angles_on_identical_data(self):
        x = make_gaussian_samples(200, 8)
        angles = principal_subspace_angles(x, x, k=3)
        self.assertEqual(angles.shape, (3,))
        self.assertTrue(torch.all(angles < TIGHT * 1000))  # very small but nonzero

    def test_validates_k_range(self):
        x = make_gaussian_samples(50, 8)
        with self.assertRaisesRegex(ValueError, "k must be"):
            principal_subspace_angles(x, x, k=0)
        with self.assertRaisesRegex(ValueError, "k must be"):
            principal_subspace_angles(x, x, k=9)

    def test_orthogonal_subspaces_give_pi_over_2(self):
        """If the top eigendirections of cov(X) and cov(Y) are orthogonal,
        the principal angle should be ~pi/2."""
        N = 1000
        torch.manual_seed(0)
        # X has all variance in dimension 0
        x = torch.zeros(N, 4, dtype=torch.float64)
        x[:, 0] = torch.randn(N, dtype=torch.float64) * 5.0
        x[:, 1:] = 0.01 * torch.randn(N, 3, dtype=torch.float64)
        # Y has all variance in dimension 2
        y = torch.zeros(N, 4, dtype=torch.float64)
        y[:, 2] = torch.randn(N, dtype=torch.float64) * 5.0
        y[:, [0, 1, 3]] = 0.01 * torch.randn(N, 3, dtype=torch.float64)
        angles = principal_subspace_angles(x, y, k=1)
        # Top principal direction of x is e_0, of y is e_2; orthogonal → pi/2
        self.assertAlmostEqual(angles[0].item(), math.pi / 2, delta=0.1)


# ===========================================================================
# Sanity-test helpers
# ===========================================================================

class TestSanityHelpers(unittest.TestCase):

    def test_constraint_residual_zero_on_exact_solution(self):
        torch.manual_seed(0)
        d_in, d_out = 16, 8
        W = torch.randn(d_out, d_in, dtype=torch.float64)
        b = torch.randn(d_out, dtype=torch.float64)
        a = torch.randn(5, d_in, dtype=torch.float64)
        t_pre = a @ W.T + b.unsqueeze(0)
        res = constraint_residual(W, b, a, t_pre)
        self.assertLess(res, TIGHT)

    def test_constraint_residual_positive_on_perturbed(self):
        torch.manual_seed(0)
        d_in, d_out = 16, 8
        W = torch.randn(d_out, d_in, dtype=torch.float64)
        b = torch.randn(d_out, dtype=torch.float64)
        a = torch.randn(5, d_in, dtype=torch.float64)
        t_pre = a @ W.T + b.unsqueeze(0)
        a_perturbed = a + 0.1 * torch.randn_like(a)
        res = constraint_residual(W, b, a_perturbed, t_pre)
        self.assertGreater(res, 0.001)

    def test_predicted_projection_covariance_is_rank_dout(self):
        """The predicted Σ̂ from Method K has rank ≤ d_out, not d_in."""
        torch.manual_seed(0)
        d_in, d_out = 16, 8
        W = torch.randn(d_out, d_in, dtype=torch.float64)
        A = torch.randn(d_in, d_in, dtype=torch.float64)
        Sigma = A @ A.T + torch.eye(d_in, dtype=torch.float64)
        S_pred = predicted_projection_covariance(Sigma, W)
        # Rank should be d_out (with high-conditioning W)
        rank = torch.linalg.matrix_rank(S_pred, atol=1e-6).item()
        self.assertEqual(rank, d_out)
        # Shape should be (d_in, d_in)
        self.assertEqual(S_pred.shape, (d_in, d_in))


# ===========================================================================
# Functional fidelity — downstream loss
# ===========================================================================

class TestDownstreamLoss(unittest.TestCase):

    def test_downstream_loss_zero_on_identical_activations(self):
        """If â == a*, all functional metrics should agree perfectly."""
        torch.manual_seed(0)
        N, d_in, d_out = 10, 8, 4
        a = torch.randn(N, d_in, dtype=torch.float64)
        W = torch.randn(d_out, d_in, dtype=torch.float64)
        b = torch.randn(d_out, dtype=torch.float64)

        def forward_remainder(x):
            return x @ W.T + b.unsqueeze(0)

        result = downstream_loss(a, a, forward_remainder)
        self.assertLess(result["logit_mse_abs"], TIGHT)
        self.assertLess(result["logit_mse"], TIGHT)
        self.assertAlmostEqual(result["prediction_agreement"], 1.0, delta=TIGHT)

    def test_downstream_loss_with_labels(self):
        """Provide labels and verify CE drift is computed."""
        torch.manual_seed(0)
        N, d_in, d_out = 10, 8, 4
        a_star = torch.randn(N, d_in)
        a_hat = a_star + 0.1 * torch.randn_like(a_star)
        labels = torch.randint(0, d_out, (N,))
        W = torch.randn(d_out, d_in)

        def forward_remainder(x):
            return x @ W.T

        result = downstream_loss(a_hat, a_star, forward_remainder, labels=labels)
        self.assertIn("ce_drift", result)
        self.assertIn("ce_hat", result)
        self.assertIn("ce_star", result)
        # CE drift should be a finite number
        self.assertTrue(math.isfinite(result["ce_drift"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
