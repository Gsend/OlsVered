"""
Unit tests for the mean-correction and self-consistent-prior options
added to diagnostic.multi_step.run_multi_step_chain.

These complement tests/test_diagnostic_multi_step.py with focused
coverage of:

  correct_target_mean=True
      - Method N invariance (correction shouldn't change N's outputs because
        N doesn't use a prior; the only thing the correction does is shift
        the input target, but N's formula scales linearly, so its effect on
        N is real but bounded — we just confirm the code runs)
      - With correction, K's mean_drift at chained steps is closer to zero
        than without correction
      - n_passes=2 produces different K results than n_passes=1 in general
      - n_passes=k for Method N is invariant to k (N short-circuits)
      - Validation errors fire correctly

Run with:
    python -m unittest tests.test_diagnostic_multi_step_corrections -v
"""

import unittest

import torch
import torch.nn as nn

from diagnostic.multi_step import (
    ChainStepReport,
    run_multi_step_chain,
    _shift_mean_to,
    _empirical_mean_cov,
)


TIGHT = 1e-5
STD = 1e-4


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 12)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(12, 4)

    def forward(self, x):
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


def make_batches(n, bs, d, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(bs, d) for _ in range(n)]


# ===========================================================================
# Internal helper tests
# ===========================================================================

class TestInternalHelpers(unittest.TestCase):

    def test_shift_mean_to_produces_target_mean(self):
        torch.manual_seed(0)
        x = torch.randn(100, 5, dtype=torch.float64) + 7.0
        target = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        shifted = _shift_mean_to(x, target)
        self.assertTrue(
            torch.allclose(shifted.mean(dim=0), target, rtol=TIGHT, atol=TIGHT),
            f"got mean {shifted.mean(dim=0).tolist()}, expected {target.tolist()}"
        )

    def test_shift_mean_preserves_variance(self):
        """Mean shift is additive — covariance should be unchanged."""
        torch.manual_seed(0)
        x = torch.randn(200, 5, dtype=torch.float64) * 2.5
        var_before = x.var(dim=0)
        target = torch.zeros(5, dtype=torch.float64)
        shifted = _shift_mean_to(x, target)
        var_after = shifted.var(dim=0)
        self.assertTrue(torch.allclose(var_before, var_after, rtol=TIGHT, atol=TIGHT))

    def test_empirical_mean_cov_matches_torch(self):
        """_empirical_mean_cov returns (mean, n-1 normalized covariance)."""
        torch.manual_seed(0)
        x = torch.randn(100, 6, dtype=torch.float64)
        mu, Sigma = _empirical_mean_cov(x)
        self.assertTrue(torch.allclose(mu, x.mean(dim=0), rtol=TIGHT, atol=TIGHT))
        # torch.cov uses n-1 by default
        self.assertTrue(torch.allclose(Sigma, torch.cov(x.T), rtol=TIGHT, atol=TIGHT))


# ===========================================================================
# Method N invariance to chain-passes
# ===========================================================================

class TestMethodNInvariantToChainPasses(unittest.TestCase):
    """Method N doesn't use a prior; chain_passes > 1 should not change
    its results because it short-circuits to one effective pass."""

    def test_naive_results_invariant_to_n_passes(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 16, 8)
        chain = [model.fc3, model.fc2, model.fc1]

        reports_1pass = run_multi_step_chain(
            model, chain, batches,
            methods=("naive",),
            eps=1e-6, max_samples=64, n_passes=1,
        )
        reports_3pass = run_multi_step_chain(
            model, chain, make_batches(4, 16, 8),
            methods=("naive",),
            eps=1e-6, max_samples=64, n_passes=3,
        )
        self.assertEqual(len(reports_1pass), len(reports_3pass))
        for r1, r3 in zip(reports_1pass, reports_3pass):
            self.assertEqual(r1.chain_idx, r3.chain_idx)
            self.assertAlmostEqual(r1.cov_frob, r3.cov_frob, delta=TIGHT)
            self.assertAlmostEqual(r1.mean_drift, r3.mean_drift, delta=TIGHT)


# ===========================================================================
# n_passes does change Method K's results
# ===========================================================================

class TestNPassesChangesK(unittest.TestCase):
    """With n_passes=2, Method K should produce visibly different results
    from n_passes=1 because the prior at chained steps adapts."""

    def test_kfac_differs_between_1_and_2_passes(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 16, 8)
        chain = [model.fc3, model.fc2, model.fc1]

        rep1 = run_multi_step_chain(
            model, chain, batches,
            methods=("kfac_a",),
            eps=1e-6, sigma2=1e-10, max_samples=64, n_passes=1,
        )
        rep2 = run_multi_step_chain(
            model, chain, make_batches(4, 16, 8),
            methods=("kfac_a",),
            eps=1e-6, sigma2=1e-10, max_samples=64, n_passes=2,
        )
        # Step 0 should be identical (always uses forward prior)
        r1_step0 = next(r for r in rep1 if r.chain_idx == 0)
        r2_step0 = next(r for r in rep2 if r.chain_idx == 0)
        self.assertAlmostEqual(r1_step0.cov_frob, r2_step0.cov_frob, delta=STD)

        # Some later step should differ
        any_diff = False
        for r1, r2 in zip(rep1, rep2):
            if r1.chain_idx > 0:
                if abs(r1.cov_frob - r2.cov_frob) > 1e-4:
                    any_diff = True
                    break
        self.assertTrue(
            any_diff,
            "n_passes=1 and n_passes=2 produced near-identical K results"
        )


# ===========================================================================
# correct_target_mean reduces K's mean drift at chained steps
# ===========================================================================

class TestMeanCorrectionReducesKDrift(unittest.TestCase):

    def test_mean_correction_reduces_or_holds_k_mean_drift(self):
        """With mean correction on, K's chained-step mean_drift should be
        <= without correction (in expectation; we test on a non-trivial
        chain that the original implementation drifts noticeably on)."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(6, 16, 8)
        chain = [model.fc3, model.fc2, model.fc1]

        rep_off = run_multi_step_chain(
            model, chain, batches,
            methods=("kfac_a",),
            eps=1e-6, sigma2=1e-10, max_samples=96,
            correct_target_mean=False,
        )
        rep_on = run_multi_step_chain(
            model, chain, make_batches(6, 16, 8),
            methods=("kfac_a",),
            eps=1e-6, sigma2=1e-10, max_samples=96,
            correct_target_mean=True,
        )

        # Step 0 should be near-identical (mean correction only affects
        # step >= 1's TARGETS).
        s0_off = next(r for r in rep_off if r.chain_idx == 0)
        s0_on = next(r for r in rep_on if r.chain_idx == 0)
        self.assertAlmostEqual(s0_off.mean_drift, s0_on.mean_drift, delta=STD)

        # At some later step the corrected version should have <= mean_drift.
        # Use total drift across chained steps as the aggregate signal.
        total_off = sum(r.mean_drift for r in rep_off if r.chain_idx > 0)
        total_on = sum(r.mean_drift for r in rep_on if r.chain_idx > 0)
        self.assertLessEqual(
            total_on, total_off + STD,
            f"correction did not reduce mean_drift: off={total_off:.4g}, on={total_on:.4g}"
        )


# ===========================================================================
# Report metadata
# ===========================================================================

class TestReportMetadata(unittest.TestCase):
    """ChainStepReport records the n_passes and correct_target_mean used."""

    def test_metadata_recorded(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 8, 8)
        reports = run_multi_step_chain(
            model, [model.fc3, model.fc2], batches,
            methods=("kfac_a",),
            eps=1e-6, sigma2=1e-10, max_samples=16,
            correct_target_mean=True, n_passes=2,
        )
        for r in reports:
            self.assertEqual(r.n_passes, 2)
            self.assertTrue(r.correct_target_mean)

    def test_default_metadata(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 8, 8)
        reports = run_multi_step_chain(
            model, [model.fc3], batches,
            methods=("naive",),
            max_samples=16,
        )
        for r in reports:
            self.assertEqual(r.n_passes, 1)
            self.assertFalse(r.correct_target_mean)


# ===========================================================================
# Validation
# ===========================================================================

class TestNewValidation(unittest.TestCase):

    def test_n_passes_zero_raises(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "n_passes"):
            run_multi_step_chain(
                model, [model.fc3], make_batches(1, 4, 8),
                n_passes=0,
            )

    def test_n_passes_negative_raises(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "n_passes"):
            run_multi_step_chain(
                model, [model.fc3], make_batches(1, 4, 8),
                n_passes=-1,
            )


# ===========================================================================
# Hybrid sanity test: both flags work together
# ===========================================================================

class TestHybridMode(unittest.TestCase):
    """correct_target_mean=True AND n_passes=2 should both apply."""

    def test_hybrid_runs_and_records_flags(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 16, 8)
        chain = [model.fc3, model.fc2, model.fc1]
        reports = run_multi_step_chain(
            model, chain, batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=64,
            correct_target_mean=True, n_passes=2,
        )
        # Should produce 6 reports (3 chain steps × 2 methods)
        self.assertEqual(len(reports), 6)
        # All K reports should record n_passes=2; N reports record
        # effective_passes=1 (because N short-circuits).
        for r in reports:
            self.assertTrue(r.correct_target_mean)
            if r.method == "kfac_a":
                self.assertEqual(r.n_passes, 2)
            else:
                self.assertEqual(r.n_passes, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
