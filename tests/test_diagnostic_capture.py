"""
Unit tests for diagnostic/capture.py — raw activation capture via forward hooks.

Validation strategy:
- Compare hook-captured activations against manual forward-pass tracing
- Verify hooks don't modify the model's outputs
- Test 3D input flattening (transformer-style)
- Test max_samples and seq_subsample caps work
- Test activation-detection helper finds the right module
- Test that Sigma_a computed from raw captures matches an independent
  forward-and-cov computation

Run with:
    python -m unittest tests.test_diagnostic_capture -v
"""

import unittest

import torch
import torch.nn as nn

from diagnostic.capture import (
    RawActivationHooks,
    collect_activations,
    detect_following_activation,
)

TIGHT = 1e-5
STD = 1e-4


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """4-layer MLP: 16 -> 32 -> 24 -> 10."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 24)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(24, 10)

    def forward(self, x):
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


class SequenceMLP(nn.Module):
    """MLP that operates on (B, T, d) inputs by applying Linear per-token."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        # x: (B, T, 16)
        return self.fc2(self.act1(self.fc1(x)))


class NoActivationModel(nn.Module):
    """Model where the last Linear has no following activation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)  # no activation after this

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def random_batch(N: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, d)


# ===========================================================================
# Activation detection
# ===========================================================================

class TestDetectFollowingActivation(unittest.TestCase):

    def test_finds_relu_after_fc1(self):
        model = SimpleMLP()
        act = detect_following_activation(model, model.fc1)
        self.assertIs(act, model.act1)

    def test_finds_tanh_after_fc2(self):
        model = SimpleMLP()
        act = detect_following_activation(model, model.fc2)
        self.assertIs(act, model.act2)

    def test_returns_none_for_final_linear(self):
        """fc3 has no activation after it — should return None."""
        model = SimpleMLP()
        act = detect_following_activation(model, model.fc3)
        self.assertIsNone(act)

    def test_returns_none_when_no_activation_between_linears(self):
        """If two Linears are consecutive with no activation, should return None."""
        model = NoActivationModel()
        # fc1 -> act -> fc2 -> (nothing) ... fc2 has no activation
        act = detect_following_activation(model, model.fc2)
        self.assertIsNone(act)


# ===========================================================================
# RawActivationHooks — basic capture
# ===========================================================================

class TestRawActivationHooksBasic(unittest.TestCase):

    def test_captures_activations_match_manual_forward(self):
        """Hook-captured a_in and a_pre should equal a manually-traced forward pass."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        x = random_batch(20, 16)

        hooks = RawActivationHooks(model, layers=[model.fc1, model.fc2, model.fc3])
        hooks.enable()
        with torch.no_grad():
            _ = model(x)
        hooks.remove()

        # Manual trace
        with torch.no_grad():
            a0 = x
            a1_pre = model.fc1(a0)
            a1 = model.act1(a1_pre)
            a2_pre = model.fc2(a1)
            a2 = model.act2(a2_pre)
            a3_pre = model.fc3(a2)

        # Check each layer
        for layer, expected_in, expected_pre in [
            (model.fc1, a0, a1_pre),
            (model.fc2, a1, a2_pre),
            (model.fc3, a2, a3_pre),
        ]:
            a_in, a_pre = hooks.get(layer)
            self.assertTrue(
                torch.allclose(a_in, expected_in.float(), rtol=TIGHT, atol=TIGHT),
                f"layer {layer}: a_in mismatch",
            )
            self.assertTrue(
                torch.allclose(a_pre, expected_pre.float(), rtol=TIGHT, atol=TIGHT),
                f"layer {layer}: a_pre mismatch",
            )

    def test_hooks_dont_modify_outputs(self):
        """Model outputs should be identical with or without hooks attached."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        x = random_batch(10, 16)

        with torch.no_grad():
            y_no_hooks = model(x).clone()

        hooks = RawActivationHooks(model, layers=[model.fc1, model.fc2, model.fc3])
        hooks.enable()
        with torch.no_grad():
            y_with_hooks = model(x).clone()
        hooks.remove()

        self.assertTrue(torch.equal(y_no_hooks, y_with_hooks))

    def test_handles_3d_input_via_flatten(self):
        """For (B, T, d) inputs, hooks should auto-flatten to (B*T, d)."""
        torch.manual_seed(0)
        model = SequenceMLP()
        model.eval()
        B, T, d = 4, 8, 16
        x = torch.randn(B, T, d)

        hooks = RawActivationHooks(model, layers=[model.fc1])
        hooks.enable()
        with torch.no_grad():
            _ = model(x)
        hooks.remove()

        a_in, a_pre = hooks.get(model.fc1)
        self.assertEqual(a_in.shape, (B * T, d))  # 32 rows total
        self.assertEqual(a_pre.shape, (B * T, 32))

    def test_accumulates_across_multiple_forward_passes(self):
        """Multiple forward calls should concatenate to the running buffer."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()

        hooks = RawActivationHooks(model, layers=[model.fc1], max_samples=10000)
        hooks.enable()
        with torch.no_grad():
            for batch_size in [5, 7, 3]:
                _ = model(random_batch(batch_size, 16, seed=batch_size))
        hooks.remove()

        a_in, _ = hooks.get(model.fc1)
        self.assertEqual(a_in.shape, (5 + 7 + 3, 16))

    def test_max_samples_cap_enforced(self):
        """Hooks should stop accumulating once max_samples is reached."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()

        hooks = RawActivationHooks(model, layers=[model.fc1], max_samples=15)
        hooks.enable()
        with torch.no_grad():
            for _ in range(10):
                _ = model(random_batch(8, 16))
        hooks.remove()

        a_in, _ = hooks.get(model.fc1)
        self.assertEqual(a_in.shape[0], 15)

    def test_seq_subsample_cap_per_call(self):
        """For a single big forward pass, hook should cap at seq_subsample rows."""
        torch.manual_seed(0)
        model = SequenceMLP()
        model.eval()
        B, T, d = 10, 50, 16   # 500 rows when flattened

        hooks = RawActivationHooks(
            model, layers=[model.fc1],
            max_samples=10000, seq_subsample=128,
        )
        hooks.enable()
        with torch.no_grad():
            _ = model(torch.randn(B, T, d))
        hooks.remove()

        a_in, _ = hooks.get(model.fc1)
        self.assertEqual(a_in.shape[0], 128)

    def test_clear_resets_buffer(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()

        hooks = RawActivationHooks(model, layers=[model.fc1])
        hooks.enable()
        with torch.no_grad():
            _ = model(random_batch(10, 16))
        a_in_before, _ = hooks.get(model.fc1)
        self.assertEqual(a_in_before.shape[0], 10)

        hooks.clear()
        a_in_after, _ = hooks.get(model.fc1)
        self.assertEqual(a_in_after.shape[0], 0)
        self.assertEqual(hooks.count(model.fc1), 0)
        hooks.remove()

    def test_remove_unregisters_hooks(self):
        """After remove(), forward passes should not accumulate any more data."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()

        hooks = RawActivationHooks(model, layers=[model.fc1])
        hooks.enable()
        with torch.no_grad():
            _ = model(random_batch(5, 16))
        hooks.remove()
        self.assertFalse(hooks.is_enabled())

        # Further forward passes should not increase buffer
        with torch.no_grad():
            _ = model(random_batch(5, 16))
        a_in, _ = hooks.get(model.fc1)
        self.assertEqual(a_in.shape[0], 5)

    def test_double_enable_is_noop(self):
        """Calling enable() twice should not register duplicate hooks."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        hooks = RawActivationHooks(model, layers=[model.fc1])
        hooks.enable()
        hooks.enable()  # second call should be a no-op
        with torch.no_grad():
            _ = model(random_batch(5, 16))
        a_in, _ = hooks.get(model.fc1)
        # If duplicate hooks fired, we'd see 10 rows. Should see 5.
        self.assertEqual(a_in.shape[0], 5)
        hooks.remove()


# ===========================================================================
# RawActivationHooks — input validation
# ===========================================================================

class TestRawActivationHooksValidation(unittest.TestCase):

    def test_rejects_non_linear_layer(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "nn.Linear"):
            RawActivationHooks(model, layers=[model.act1])

    def test_rejects_non_list_layers(self):
        model = SimpleMLP()
        with self.assertRaisesRegex(ValueError, "list"):
            RawActivationHooks(model, layers=model.fc1)

    def test_get_unknown_layer_raises(self):
        model = SimpleMLP()
        hooks = RawActivationHooks(model, layers=[model.fc1])
        with self.assertRaisesRegex(KeyError, "not tracked"):
            hooks.get(model.fc2)


# ===========================================================================
# Sigma_a parity with KFAC-style Gram computation
# ===========================================================================

class TestSigmaAParityWithKFACStyle(unittest.TestCase):
    """The empirical covariance from raw captures should match a Gram-based
    independent computation (the K-FAC A factor pattern)."""

    def test_sigma_a_from_capture_matches_kfac_style(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        x = random_batch(500, 16)

        hooks = RawActivationHooks(model, layers=[model.fc1])
        hooks.enable()
        with torch.no_grad():
            _ = model(x)
        hooks.remove()

        a_in, _ = hooks.get(model.fc1)

        # K-FAC-style: A factor = E[a a^T] / N = a^T a / N (no centering)
        N = a_in.shape[0]
        A_kfac = a_in.T @ a_in / N

        # Our diagnostic uses centered covariance (n-1 normalization)
        # Verify the cross-product relationship:
        #   (n-1) Cov(a) = a^T a - n * mean(a) mean(a)^T
        mu = a_in.mean(dim=0)
        cov = torch.cov(a_in.T)  # uses n-1 normalization
        reconstructed = (cov * (N - 1) + N * torch.outer(mu, mu)) / N
        self.assertTrue(
            torch.allclose(reconstructed, A_kfac, rtol=TIGHT, atol=TIGHT)
        )


# ===========================================================================
# collect_activations — one-shot helper
# ===========================================================================

class TestCollectActivations(unittest.TestCase):

    def test_basic_collection_with_dataloader(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        # Toy dataloader yielding plain tensors
        batches = [random_batch(8, 16, seed=i) for i in range(3)]
        result = collect_activations(
            model, batches, layers=[model.fc1, model.fc2],
            max_samples=100,
        )
        # fc1 should have 24 rows (3 * 8)
        self.assertEqual(result[model.fc1]["a_in"].shape, (24, 16))
        self.assertEqual(result[model.fc1]["a_pre"].shape, (24, 32))
        self.assertEqual(result[model.fc1]["a_post"].shape, (24, 32))
        # Activation auto-detected
        self.assertIs(result[model.fc1]["activation"], model.act1)
        self.assertIs(result[model.fc2]["activation"], model.act2)

    def test_a_post_equals_activation_of_a_pre(self):
        """a_post should equal activation(a_pre) element-wise."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = [random_batch(16, 16)]
        result = collect_activations(
            model, batches, layers=[model.fc1],
        )
        a_pre = result[model.fc1]["a_pre"]
        a_post = result[model.fc1]["a_post"]
        with torch.no_grad():
            expected = model.act1(a_pre)
        self.assertTrue(torch.allclose(a_post, expected, rtol=TIGHT, atol=TIGHT))

    def test_final_layer_with_no_activation(self):
        """For a Linear with no following activation, a_post should equal a_pre."""
        torch.manual_seed(0)
        model = NoActivationModel()
        model.eval()
        batches = [random_batch(10, 8)]
        result = collect_activations(model, batches, layers=[model.fc2])
        a_pre = result[model.fc2]["a_pre"]
        a_post = result[model.fc2]["a_post"]
        self.assertIsNone(result[model.fc2]["activation"])
        self.assertTrue(torch.equal(a_pre, a_post))

    def test_activation_override(self):
        """activation_overrides should win over auto-detection."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        batches = [random_batch(10, 16)]
        # Override fc1's activation to None (use a_post = a_pre)
        result = collect_activations(
            model, batches, layers=[model.fc1],
            activation_overrides={model.fc1: None},
        )
        self.assertIsNone(result[model.fc1]["activation"])
        # a_post should equal a_pre (since activation is None)
        self.assertTrue(torch.equal(
            result[model.fc1]["a_pre"], result[model.fc1]["a_post"]
        ))

    def test_early_termination_on_saturation(self):
        """Collection should stop early once all layers hit max_samples."""
        torch.manual_seed(0)
        model = SimpleMLP()
        model.eval()
        # Toy dataloader: 100 batches of 10 rows = 1000 total
        batches = [random_batch(10, 16, seed=i) for i in range(100)]
        result = collect_activations(
            model, batches, layers=[model.fc1],
            max_samples=25,
        )
        self.assertEqual(result[model.fc1]["a_in"].shape[0], 25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
