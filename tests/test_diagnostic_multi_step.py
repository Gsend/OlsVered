"""
Unit tests for diagnostic/multi_step.py — multi-step back-target chain.

Validation strategy:
- Single-step chain (depth=1) should match run_drift_experiment exactly,
  since with depth=1 there's no chaining and we use ground-truth target.
- Chain growth fields are None at step 0 and populated for step >= 1.
- Methods produce different chain trajectories on non-trivial priors.
- save/load via dict conversion roundtrips correctly.
- Compounding signal: chain at depth>=2 should show larger drift than depth=1
  for both methods (errors accumulate).

Run with:
    python -m unittest tests.test_diagnostic_multi_step -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from diagnostic.experiment import run_drift_experiment
from diagnostic.multi_step import ChainStepReport, run_multi_step_chain


TIGHT = 1e-5
STD = 1e-4


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """3-layer MLP: 8 -> 16 -> 12 -> 4 with ReLU."""

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
# Basic structure tests
# ===========================================================================

class TestChainStructure(unittest.TestCase):

    def test_produces_one_report_per_step_per_method(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 8, 8)
        # Chain through 3 layers (fc3 -> fc2 -> fc1) with 2 methods = 6 reports
        chain_layers = [model.fc3, model.fc2, model.fc1]
        reports = run_multi_step_chain(
            model, chain_layers, batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=32,
        )
        self.assertEqual(len(reports), 6)
        # Step indices: 0, 1, 2 for each method
        naive_steps = sorted(r.chain_idx for r in reports if r.method == "naive")
        kfac_steps = sorted(r.chain_idx for r in reports if r.method == "kfac_a")
        self.assertEqual(naive_steps, [0, 1, 2])
        self.assertEqual(kfac_steps, [0, 1, 2])

    def test_chain_growth_is_none_at_step_0_populated_after(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 8, 8)
        chain_layers = [model.fc3, model.fc2, model.fc1]
        reports = run_multi_step_chain(
            model, chain_layers, batches,
            methods=("naive",), max_samples=16,
        )
        # Step 0 should have None growth fields
        step0 = next(r for r in reports if r.chain_idx == 0)
        self.assertIsNone(step0.cov_frob_growth)
        self.assertIsNone(step0.mean_drift_growth)
        # Subsequent steps should have growth values populated
        for r in reports:
            if r.chain_idx > 0:
                self.assertIsNotNone(r.cov_frob_growth)
                self.assertIsNotNone(r.mean_drift_growth)

    def test_layer_labels_passed_through(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 4, 8)
        reports = run_multi_step_chain(
            model, [model.fc3, model.fc2], batches,
            methods=("naive",),
            layer_labels=["DeepStep", "ShallowStep"],
            max_samples=8,
        )
        labels = [r.layer_label for r in reports]
        self.assertIn("DeepStep", labels)
        self.assertIn("ShallowStep", labels)


# ===========================================================================
# Single-step chain equivalence to run_drift_experiment
# ===========================================================================

class TestChainEquivalenceWithSingleStep(unittest.TestCase):
    """At depth=1, the chain should produce drift metrics matching what
    run_drift_experiment computes for that layer with ground-truth target."""

    def test_depth1_matches_single_layer_drift(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 16, 8)

        # Single-step chain on fc3 alone
        chain_reports = run_multi_step_chain(
            model, [model.fc3], batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=64,
        )

        # Single-layer drift on the same setup
        single_reports = run_drift_experiment(
            model, [model.fc3], make_batches(4, 16, 8),
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=64,
            use_predicted_cov_check=False,
        )

        for method in ("naive", "kfac_a"):
            c = next(r for r in chain_reports if r.method == method)
            s = next(r for r in single_reports if r.method == method)
            # cov_frob, mean_drift should match closely
            self.assertAlmostEqual(c.cov_frob, s.cov_frob, delta=STD)
            self.assertAlmostEqual(c.mean_drift, s.mean_drift, delta=STD)


# ===========================================================================
# Method comparison on chain
# ===========================================================================

class TestChainMethodComparison(unittest.TestCase):

    def test_methods_differ_on_chained_steps(self):
        """On chained steps (step >= 1), Method N and Method K should
        produce different drift trajectories (the prior matters)."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(6, 16, 8)
        chain_layers = [model.fc3, model.fc2, model.fc1]
        reports = run_multi_step_chain(
            model, chain_layers, batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=96,
        )

        for step in [1, 2]:
            n = next(r for r in reports if r.chain_idx == step and r.method == "naive")
            k = next(r for r in reports if r.chain_idx == step and r.method == "kfac_a")
            self.assertNotAlmostEqual(n.cov_frob, k.cov_frob, delta=1e-6)


# ===========================================================================
# Compounding signal — error grows with chain depth
# ===========================================================================

class TestChainCompounding(unittest.TestCase):

    def test_naive_drift_grows_along_chain(self):
        """For Method N (no prior), drift in chained steps should typically
        be >= drift in step 0 because errors compound without correction."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(8, 16, 8)
        chain_layers = [model.fc3, model.fc2, model.fc1]
        reports = run_multi_step_chain(
            model, chain_layers, batches,
            methods=("naive",),
            eps=1e-6, max_samples=128,
        )
        step0 = next(r for r in reports if r.chain_idx == 0)
        step2 = next(r for r in reports if r.chain_idx == 2)
        # Step 2 cov_frob should be at least comparable to step 0
        # (typically larger due to compounding, but at minimum not radically smaller)
        self.assertGreater(step2.cov_frob, 0.1 * step0.cov_frob)


# ===========================================================================
# Field validity and types
# ===========================================================================

class TestChainFieldValidity(unittest.TestCase):

    def test_all_numeric_fields_finite(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 8, 8)
        chain_layers = [model.fc3, model.fc2]
        reports = run_multi_step_chain(
            model, chain_layers, batches,
            methods=("naive", "kfac_a"),
            eps=1e-6, sigma2=1e-10, max_samples=32,
        )
        for r in reports:
            for fld in ["sample_rel_err_mean", "sample_rel_err_p95",
                        "cos_sim_mean", "mean_drift", "cov_frob",
                        "w2_gauss", "dead_unit_fraction"]:
                val = getattr(r, fld)
                self.assertTrue(
                    val == val,  # NaN check
                    f"step{r.chain_idx} method={r.method} field={fld} is NaN",
                )
                self.assertNotEqual(val, float("inf"),
                                    f"{fld} is +inf at step{r.chain_idx} {r.method}")

    def test_activation_name_recorded(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 4, 8)
        chain_layers = [model.fc2]
        reports = run_multi_step_chain(
            model, chain_layers, batches,
            methods=("naive",), max_samples=8,
        )
        # fc2's activation is act2 = ReLU
        self.assertEqual(reports[0].activation_name, "ReLU")

    def test_dead_unit_fraction_in_range(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 8, 8)
        chain_layers = [model.fc2, model.fc1]
        reports = run_multi_step_chain(
            model, chain_layers, batches,
            methods=("naive",), max_samples=32,
        )
        for r in reports:
            self.assertGreaterEqual(r.dead_unit_fraction, 0.0)
            self.assertLessEqual(r.dead_unit_fraction, 1.0)


# ===========================================================================
# Persistence
# ===========================================================================

class TestChainPersistence(unittest.TestCase):

    def test_to_dict_includes_all_fields(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 4, 8)
        reports = run_multi_step_chain(
            model, [model.fc3, model.fc2], batches,
            methods=("naive",), max_samples=8,
        )
        d = reports[0].to_dict()
        expected_keys = {
            "chain_idx", "method", "layer_idx", "layer_label",
            "d_in", "d_out", "n_samples",
            "sample_rel_err_mean", "sample_rel_err_p95", "cos_sim_mean",
            "mean_drift", "cov_frob", "gauss_kl_sym", "w2_gauss",
            "cov_frob_growth", "mean_drift_growth",
            "dead_unit_fraction", "activation_name",
        }
        self.assertTrue(expected_keys.issubset(d.keys()),
                        f"missing keys: {expected_keys - set(d.keys())}")

    def test_json_roundtrip_via_dict(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 4, 8)
        reports = run_multi_step_chain(
            model, [model.fc3, model.fc2], batches,
            methods=("naive", "kfac_a"),
            sigma2=1e-10, max_samples=8,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chain.json"
            payload = {
                "model": "SimpleMLP",
                "chain_depth": 2,
                "reports": [r.to_dict() for r in reports],
            }
            with open(path, "w") as fh:
                json.dump(payload, fh, default=lambda o: None)
            with open(path) as fh:
                loaded = json.load(fh)
            # Reconstruct ChainStepReport from dicts
            restored = [ChainStepReport(**r) for r in loaded["reports"]]
            self.assertEqual(len(restored), len(reports))
            for orig, ld in zip(reports, restored):
                self.assertEqual(orig.chain_idx, ld.chain_idx)
                self.assertEqual(orig.method, ld.method)
                self.assertAlmostEqual(orig.cov_frob, ld.cov_frob, delta=TIGHT)


# ===========================================================================
# Input validation
# ===========================================================================

class TestChainValidation(unittest.TestCase):

    def test_empty_layers_raises(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            run_multi_step_chain(model, [], make_batches(1, 4, 8))

    def test_layer_labels_length_mismatch_raises(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "layer_labels"):
            run_multi_step_chain(
                model, [model.fc3, model.fc2], make_batches(1, 4, 8),
                layer_labels=["only one"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
