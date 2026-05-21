"""
Unit tests for diagnostic/experiment.py — end-to-end driver and persistence.

Validation strategy:
- End-to-end: run on a tiny MLP and verify reports have expected structure
- Sanity: identical-input case → drift = 0 (Method N) on square invertible W
- Comparison: Method N and Method K produce different reports in general
- Persistence: JSON save/load roundtrip preserves all fields
- Functional metrics: with remaining_forward_fn provided, those fields are populated

Run with:
    python -m unittest tests.test_diagnostic_experiment -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from diagnostic.experiment import (
    LayerDriftReport,
    load_reports,
    run_drift_experiment,
    save_reports,
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
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(12, 4)

    def forward(self, x):
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


def make_batches(n_batches: int, batch_size: int, d_in: int, seed: int = 0):
    torch.manual_seed(seed)
    return [torch.randn(batch_size, d_in) for _ in range(n_batches)]


# ===========================================================================
# Basic end-to-end
# ===========================================================================

class TestRunDriftExperimentBasic(unittest.TestCase):

    def test_produces_one_report_per_layer_per_method(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(5, 8, 8)

        layers = [model.fc1, model.fc2]
        reports = run_drift_experiment(
            model, layers, batches,
            methods=("naive", "kfac_a"),
            max_samples=100, eps=1e-6, sigma2=1e-10,
            use_predicted_cov_check=False,
        )
        self.assertEqual(len(reports), 4)  # 2 layers × 2 methods
        # Order: layer0/naive, layer0/kfac_a, layer1/naive, layer1/kfac_a
        self.assertEqual(reports[0].layer_idx, 0)
        self.assertEqual(reports[0].method, "naive")
        self.assertEqual(reports[1].layer_idx, 0)
        self.assertEqual(reports[1].method, "kfac_a")
        self.assertEqual(reports[2].layer_idx, 1)
        self.assertEqual(reports[2].method, "naive")
        self.assertEqual(reports[3].layer_idx, 1)
        self.assertEqual(reports[3].method, "kfac_a")

    def test_reports_have_expected_fields(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(3, 8, 8)
        reports = run_drift_experiment(
            model, [model.fc1], batches,
            methods=("kfac_a",),
            max_samples=50, sigma2=1e-10,
            use_predicted_cov_check=False,
        )
        r = reports[0]
        self.assertEqual(r.d_in, 8)
        self.assertEqual(r.d_out, 16)
        self.assertEqual(r.method, "kfac_a")
        self.assertEqual(r.n_samples, 24)
        self.assertEqual(r.activation_name, "ReLU")
        # All numeric fields exist and are finite (or None for unspecified)
        for fld in [
            "sample_rel_err_mean", "sample_rel_err_p95",
            "cos_sim_mean", "cos_sim_p5",
            "mean_drift", "cov_frob", "gauss_kl_sym", "w2_gauss",
            "eig_pearson", "eig_kl_on_spectrum",
            "effective_rank_hat", "effective_rank_star",
            "subspace_angle_top1", "subspace_angle_top5",
            "constraint_residual", "dead_unit_fraction",
        ]:
            val = getattr(r, fld)
            self.assertTrue(
                val is None or (val == val and val != float("inf") and val != float("-inf")),
                f"field '{fld}' is non-finite: {val}",
            )

    def test_layer_labels_passed_through(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 4, 8)
        reports = run_drift_experiment(
            model, [model.fc1], batches,
            methods=("naive",),
            layer_labels=["my fancy layer"],
            max_samples=10,
            use_predicted_cov_check=False,
        )
        self.assertEqual(reports[0].layer_label, "my fancy layer")

    def test_layer_labels_length_mismatch_raises(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "layer_labels"):
            run_drift_experiment(
                model, [model.fc1, model.fc2], make_batches(1, 4, 8),
                layer_labels=["only one"],
            )


# ===========================================================================
# Sanity: Method N on square invertible W with no activation → drift = 0
# ===========================================================================

class TestSanitySquareInvertible(unittest.TestCase):

    def test_drift_zero_on_square_invertible_linear_no_activation(self):
        """With d_in == d_out and no activation function, Method N should
        recover a_in exactly: drift metrics should all be ~0."""
        torch.manual_seed(0)
        d = 8

        class IdentityishMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(d, d)

            def forward(self, x):
                return self.fc(x)

        model = IdentityishMLP()
        model.eval()
        batches = make_batches(4, 16, d)
        reports = run_drift_experiment(
            model, [model.fc], batches,
            methods=("naive",),
            eps=1e-12, max_samples=200,
            use_predicted_cov_check=False,
        )
        r = reports[0]
        self.assertLess(r.sample_rel_err_mean, STD,
                        f"sample_rel_err_mean = {r.sample_rel_err_mean}")
        self.assertLess(r.mean_drift, STD)
        self.assertLess(r.cov_frob, STD)
        self.assertLess(r.constraint_residual, STD)


# ===========================================================================
# Method comparison
# ===========================================================================

class TestMethodComparison(unittest.TestCase):

    def test_methods_produce_different_reports(self):
        """For a layer with d_in > d_out (underdetermined), Method N and
        Method K should give visibly different drift signatures."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(8, 16, 8)
        reports = run_drift_experiment(
            model, [model.fc2], batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-8, max_samples=128,
            use_predicted_cov_check=False,
        )
        r_naive, r_kfac = reports
        # The two should produce non-identical drift signatures on a
        # non-trivial (non-identity Σ_a, non-zero μ_a) prior.
        self.assertNotAlmostEqual(r_naive.mean_drift, r_kfac.mean_drift, delta=1e-6)


# ===========================================================================
# Sanity-test fields
# ===========================================================================

class TestSanityTestFields(unittest.TestCase):

    def test_constraint_residual_is_small(self):
        """The constraint W â + b ≈ t_pre should be satisfied for both methods."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 8, 8)
        reports = run_drift_experiment(
            model, [model.fc1, model.fc2], batches,
            methods=("naive", "kfac_a"),
            eps=1e-10, sigma2=1e-12, max_samples=64,
            use_predicted_cov_check=False,
        )
        for r in reports:
            self.assertLess(
                r.constraint_residual, 1e-3,
                f"layer {r.layer_idx} method {r.method}: "
                f"constraint_residual = {r.constraint_residual}",
            )

    def test_cov_predicted_match_populated_for_kfac(self):
        """When use_predicted_cov_check=True, Method K reports get the
        cov_predicted_match field populated."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(6, 16, 8)
        reports = run_drift_experiment(
            model, [model.fc1], batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=96,
            use_predicted_cov_check=True,
        )
        r_naive, r_kfac = reports
        self.assertIsNone(r_naive.cov_predicted_match)
        self.assertIsNotNone(r_kfac.cov_predicted_match)
        # The empirical projection covariance should agree with the predicted
        # closed-form. With finite samples and damping, expect some drift.
        self.assertLess(r_kfac.cov_predicted_match, 1.0)


# ===========================================================================
# Functional metrics
# ===========================================================================

class TestFunctionalMetrics(unittest.TestCase):

    def test_functional_metrics_populated_when_remaining_fn_provided(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(3, 8, 8)

        # remaining_forward_fn for fc1: run model.act1 -> fc2 -> act2 -> fc3
        def rem_fn_fc1(a):
            return model.fc3(model.act2(model.fc2(model.act1(a))))

        reports = run_drift_experiment(
            model, [model.fc1], batches,
            methods=("naive",),
            eps=1e-6, max_samples=24,
            remaining_forward_fns={model.fc1: rem_fn_fc1},
            use_predicted_cov_check=False,
        )
        r = reports[0]
        self.assertIsNotNone(r.logit_mse)
        self.assertIsNotNone(r.logit_mse_abs)
        self.assertIsNotNone(r.prediction_agreement)


# ===========================================================================
# Persistence
# ===========================================================================

class TestPersistence(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        """save_reports + load_reports should round-trip exactly."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 8, 8)
        reports = run_drift_experiment(
            model, [model.fc1], batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=16,
            use_predicted_cov_check=False,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.json"
            save_reports(reports, path, model_name="SimpleMLP", task="unit_test")
            loaded = load_reports(path)
            self.assertEqual(len(loaded), len(reports))
            for orig, ld in zip(reports, loaded):
                self.assertEqual(orig.layer_idx, ld.layer_idx)
                self.assertEqual(orig.method, ld.method)
                self.assertAlmostEqual(orig.mean_drift, ld.mean_drift, delta=TIGHT)
                self.assertAlmostEqual(orig.cov_frob, ld.cov_frob, delta=TIGHT)
                self.assertEqual(orig.activation_name, ld.activation_name)

    def test_save_includes_metadata(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(1, 4, 8)
        reports = run_drift_experiment(
            model, [model.fc1], batches,
            methods=("naive",), max_samples=4,
            use_predicted_cov_check=False,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.json"
            save_reports(
                reports, path,
                model_name="TestModel", task="my_task",
                extra={"custom_field": 42},
            )
            with open(path) as fh:
                payload = json.load(fh)
            self.assertEqual(payload["model"], "TestModel")
            self.assertEqual(payload["task"], "my_task")
            self.assertEqual(payload["custom_field"], 42)
            self.assertEqual(payload["n_reports"], 1)
            self.assertIn("timestamp", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
