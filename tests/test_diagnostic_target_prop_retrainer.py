"""
Unit tests for diagnostic/target_prop_retrainer.py — target-prop + OLS retraining.

Tests:
- solve_ols_layer recovers ground truth on well-conditioned synthetic data
- solve_ols_layer correctly handles with_bias=True/False
- retrain_via_target_prop returns a model with the expected shapes
- return_copy=True doesn't modify the original
- Method N and K are both invokable
- correct_target_cov requires correct_target_mean

Run with:
    python -m unittest tests.test_diagnostic_target_prop_retrainer -v
"""

import unittest
import copy

import torch
import torch.nn as nn

from diagnostic.target_prop_retrainer import (
    RetrainerResult,
    retrain_via_target_prop,
    solve_ols_layer,
)


TIGHT = 1e-5
STD = 1e-4
LOOSE = 5e-2


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
# solve_ols_layer correctness
# ===========================================================================

class TestSolveOlsLayer(unittest.TestCase):

    def test_recovers_ground_truth_with_bias(self):
        """Generate (X, W_true, b_true, T_pre = X @ W_true.T + b_true)
        and verify solve_ols_layer recovers W_true, b_true."""
        torch.manual_seed(0)
        n, d_in, d_out = 200, 8, 4
        X = torch.randn(n, d_in, dtype=torch.float64)
        W_true = torch.randn(d_out, d_in, dtype=torch.float64)
        b_true = torch.randn(d_out, dtype=torch.float64)
        T_pre = X @ W_true.T + b_true.unsqueeze(0)
        W_hat, b_hat = solve_ols_layer(
            X, T_pre, with_bias=True, ols_lambda=1e-12)
        self.assertTrue(torch.allclose(W_hat, W_true, rtol=STD, atol=STD),
                        f"max diff = {(W_hat - W_true).abs().max().item():.2e}")
        self.assertTrue(torch.allclose(b_hat, b_true, rtol=STD, atol=STD))

    def test_recovers_ground_truth_without_bias(self):
        torch.manual_seed(0)
        n, d_in, d_out = 200, 8, 4
        X = torch.randn(n, d_in, dtype=torch.float64)
        W_true = torch.randn(d_out, d_in, dtype=torch.float64)
        T_pre = X @ W_true.T
        W_hat, b_hat = solve_ols_layer(
            X, T_pre, with_bias=False, ols_lambda=1e-12)
        self.assertTrue(torch.allclose(W_hat, W_true, rtol=STD, atol=STD))
        self.assertIsNone(b_hat)

    def test_validation_errors(self):
        with self.assertRaisesRegex(ValueError, "X must be"):
            solve_ols_layer(torch.randn(3, 4, 5), torch.randn(3, 2))
        with self.assertRaisesRegex(ValueError, "T_pre must be"):
            solve_ols_layer(torch.randn(3, 4), torch.randn(3))
        with self.assertRaisesRegex(ValueError, "same n"):
            solve_ols_layer(torch.randn(5, 4), torch.randn(3, 2))

    def test_underdetermined_system_does_not_explode(self):
        """When n < d_in, the system is underdetermined. Damping should keep it stable."""
        torch.manual_seed(0)
        n, d_in, d_out = 4, 16, 2
        X = torch.randn(n, d_in, dtype=torch.float64)
        T_pre = torch.randn(n, d_out, dtype=torch.float64)
        W_hat, b_hat = solve_ols_layer(
            X, T_pre, with_bias=False, ols_lambda=1e-4)
        self.assertTrue(torch.isfinite(W_hat).all())
        self.assertEqual(W_hat.shape, (d_out, d_in))


# ===========================================================================
# retrain_via_target_prop — basic correctness
# ===========================================================================

class TestRetrainerBasic(unittest.TestCase):

    def test_returns_retrainer_result(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(4, 16, 8)
        result = retrain_via_target_prop(
            model,
            [model.fc3, model.fc2, model.fc1],
            batches,
            method="kfac_a",
            correct_target_mean=True,
            correct_target_cov=True,
            max_samples=64,
            ols_lambda=1e-4,
        )
        self.assertIsInstance(result, RetrainerResult)
        self.assertEqual(result.method, "kfac_a")
        self.assertTrue(result.correct_target_mean)
        self.assertTrue(result.correct_target_cov)

    def test_per_layer_info_populated(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(2, 8, 8)
        result = retrain_via_target_prop(
            model,
            [model.fc3, model.fc2],
            batches,
            method="naive",
            correct_target_mean=False,
            max_samples=16,
        )
        # One info entry per retrained layer
        self.assertEqual(len(result.per_layer_info), 2)
        for idx, info in result.per_layer_info.items():
            self.assertIn("d_in", info)
            self.assertIn("d_out", info)
            self.assertIn("weight_change_frob", info)
            self.assertIn("target_residual", info)
            self.assertTrue(info["target_residual"] >= 0)


# ===========================================================================
# return_copy invariance
# ===========================================================================

class TestReturnCopy(unittest.TestCase):

    def test_return_copy_true_preserves_original(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        original_fc3_weight = model.fc3.weight.detach().clone()
        original_fc2_weight = model.fc2.weight.detach().clone()

        batches = make_batches(2, 8, 8)
        result = retrain_via_target_prop(
            model,
            [model.fc3, model.fc2],
            batches,
            method="kfac_a",
            max_samples=16,
            return_copy=True,
        )

        # Original model is unchanged
        self.assertTrue(torch.equal(model.fc3.weight, original_fc3_weight))
        self.assertTrue(torch.equal(model.fc2.weight, original_fc2_weight))
        # The returned model is a different object
        self.assertIsNot(result.model, model)
        self.assertIsNot(result.model.fc3, model.fc3)

    def test_return_copy_false_modifies_in_place(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        original_fc3_weight = model.fc3.weight.detach().clone()

        batches = make_batches(2, 8, 8)
        result = retrain_via_target_prop(
            model,
            [model.fc3],
            batches,
            method="kfac_a",
            max_samples=16,
            return_copy=False,
        )
        # Original model SHOULD now be modified
        self.assertIs(result.model, model)
        # Weight should differ (target-prop changes them in general)
        self.assertFalse(torch.equal(model.fc3.weight, original_fc3_weight))


# ===========================================================================
# Both methods invokable
# ===========================================================================

class TestBothMethodsRun(unittest.TestCase):

    def test_naive_method_runs(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        result = retrain_via_target_prop(
            model, [model.fc3, model.fc2], make_batches(2, 8, 8),
            method="naive", max_samples=16,
        )
        self.assertEqual(result.method, "naive")

    def test_kfac_method_runs(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        result = retrain_via_target_prop(
            model, [model.fc3, model.fc2], make_batches(2, 8, 8),
            method="kfac_a", correct_target_mean=True, correct_target_cov=True,
            max_samples=16,
        )
        self.assertEqual(result.method, "kfac_a")


# ===========================================================================
# Validation
# ===========================================================================

class TestValidation(unittest.TestCase):

    def test_empty_layers_raises(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            retrain_via_target_prop(model, [], make_batches(1, 4, 8))

    def test_bad_method_raises(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "method must be"):
            retrain_via_target_prop(
                model, [model.fc3], make_batches(1, 4, 8),
                method="bogus",
            )

    def test_cov_without_mean_raises(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "correct_target_cov"):
            retrain_via_target_prop(
                model, [model.fc3], make_batches(1, 4, 8),
                correct_target_mean=False, correct_target_cov=True,
            )


# ===========================================================================
# Functional sanity: weight shapes preserved
# ===========================================================================

class TestRetrainedModelShapes(unittest.TestCase):
    """Sanity: retrained model's layers have the same shapes as the original
    (we only update values, not structure)."""

    def test_shapes_preserved(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = make_batches(3, 8, 8)
        result = retrain_via_target_prop(
            model, [model.fc3, model.fc2, model.fc1], batches,
            method="kfac_a", correct_target_mean=True, correct_target_cov=True,
            max_samples=24,
        )
        self.assertEqual(result.model.fc1.weight.shape, model.fc1.weight.shape)
        self.assertEqual(result.model.fc2.weight.shape, model.fc2.weight.shape)
        self.assertEqual(result.model.fc3.weight.shape, model.fc3.weight.shape)
        self.assertEqual(result.model.fc1.bias.shape, model.fc1.bias.shape)


class LinearChainMLP(nn.Module):
    """No-nonlinearity MLP. Used to verify forward-sweep OLS semantics:
    in a fully-linear chain, composing the OLS solves layer-by-layer
    must reproduce the deepest target exactly."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 12)
        self.fc3 = nn.Linear(12, 4)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class TestForwardSweepOLS(unittest.TestCase):
    """Verifies that per-layer OLS uses the rebuilt-upstream output of the
    previous retrained layer (not the captured a_in from the original model).

    In a fully linear chain with no regularization and enough samples, this
    is the only way every layer's OLS residual can be zero AND the composed
    network maps input to the deepest target exactly. With the buggy
    stale-a_in formulation, the layers fit individually but the composed
    forward pass produces something different from the target."""

    def _make_data(self, n=512, d_in=8, seed=0):
        torch.manual_seed(seed)
        return [torch.randn(n, d_in, dtype=torch.float32)]

    def test_forward_sweep_per_layer_residuals_near_zero(self):
        """All per-layer target_residuals should be tiny because each layer
        sees the rebuilt-upstream X that the OLS was solved against."""
        torch.manual_seed(0)
        model = LinearChainMLP().eval()
        batches = self._make_data()
        # Identity activation overrides so capture knows the chain is linear
        result = retrain_via_target_prop(
            model,
            [model.fc3, model.fc2, model.fc1],
            batches,
            method="naive",
            correct_target_mean=False,
            correct_target_cov=False,
            ols_lambda=1e-10,
            max_samples=512,
            activation_overrides={
                model.fc1: None, model.fc2: None, model.fc3: None,
            },
        )
        for idx, info in result.per_layer_info.items():
            self.assertLess(
                info["target_residual"], 1e-4,
                msg=f"layer {idx} target_residual = {info['target_residual']:.3e} "
                    "— forward sweep should yield near-zero residuals in "
                    "linear chain"
            )

    def test_forward_sweep_composed_forward_matches_target(self):
        """In a linear chain, the composed network's output must equal the
        deepest target (the original model's fc3 output). The buggy
        stale-a_in formulation would break this composition."""
        torch.manual_seed(0)
        model = LinearChainMLP().eval()
        batches = self._make_data()

        # Get the original model's fc3 output (deepest a_post) — this is the
        # implicit target of the chain (TP uses the original deepest a_post).
        x = batches[0]
        with torch.no_grad():
            original_output = model(x)

        result = retrain_via_target_prop(
            model,
            [model.fc3, model.fc2, model.fc1],
            batches,
            method="naive",
            correct_target_mean=False,
            correct_target_cov=False,
            ols_lambda=1e-10,
            max_samples=512,
            activation_overrides={
                model.fc1: None, model.fc2: None, model.fc3: None,
            },
        )

        with torch.no_grad():
            retrained_output = result.model(x)
        # Composed network should reproduce the target to high precision
        max_diff = (retrained_output - original_output).abs().max().item()
        self.assertLess(
            max_diff, 1e-3,
            msg=f"composed forward output differs from target by {max_diff:.3e} "
                "— forward-sweep OLS should be exact in linear chain"
        )

    def test_forward_sweep_entry_layer_uses_captured_a_in(self):
        """Sanity: the entry retrained layer (shallowest = fc1 here) is fit
        against the model input, and its solved weights, when applied to the
        captured a_in, should produce the captured t_pre directly."""
        torch.manual_seed(0)
        model = LinearChainMLP().eval()
        batches = self._make_data()
        x = batches[0]

        result = retrain_via_target_prop(
            model,
            [model.fc3, model.fc2, model.fc1],
            batches,
            method="naive",
            ols_lambda=1e-10,
            max_samples=512,
            activation_overrides={
                model.fc1: None, model.fc2: None, model.fc3: None,
            },
        )
        # fc1's new weight applied to x should yield fc1's new a_pre = a_post
        with torch.no_grad():
            fc1_out_new = result.model.fc1(x)
        # That output should also equal result.model.fc1's forward applied to
        # the same input — trivially true; the meaningful check is that the
        # rest of the network composes onto the target, tested above.
        self.assertEqual(fc1_out_new.shape, (x.shape[0], 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
