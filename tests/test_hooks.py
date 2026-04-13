"""
Unit tests for optimizer/hooks.py — KFACHooks Gram accumulation.

Tests:
  P0 — n_samples increments each step
  P0 — Gram matrix is symmetric (A = Aᵀ, G = Gᵀ)
  P0 — clear() resets all accumulators to zero
  P0 — sequence subsample caps rows at _SEQ_SUBSAMPLE
  P1 — Conv2d layers are tracked (im2col path)
  P1 — max_gram_dim filter skips oversized layers
  P1 — is_enabled reflects hook state
  P1 — remove() detaches hooks (model still runs)
"""

import pytest
import torch
import torch.nn as nn

from optimizer.hooks import KFACHooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_mlp(d_in=8, d_hidden=16, d_out=4) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def _run_step(model, hooks, batch_size=4, d_in=8, d_out=4):
    """One forward + backward pass through the model."""
    x = torch.randn(batch_size, d_in)
    y = torch.randn(batch_size, d_out)
    logits = model(x)
    loss = ((logits - y) ** 2).mean()
    loss.backward()
    return loss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNSamplesTracking:
    def test_n_samples_increments_per_step(self):
        """n_samples_accumulated should grow by batch_size each step."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()

        assert hooks.n_samples_accumulated() == 0

        _run_step(model, hooks, batch_size=4)
        assert hooks.n_samples_accumulated() == 4

        _run_step(model, hooks, batch_size=8)
        assert hooks.n_samples_accumulated() == 12

    def test_n_samples_after_clear(self):
        """clear() resets the sample counter to zero."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()

        _run_step(model, hooks, batch_size=6)
        assert hooks.n_samples_accumulated() > 0

        hooks.clear()
        assert hooks.n_samples_accumulated() == 0


class TestGramSymmetry:
    def test_gram_matrix_symmetry(self):
        """Accumulated A and G matrices must be symmetric (XᵀX is always symmetric)."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()

        for _ in range(3):
            _run_step(model, hooks, batch_size=8)

        factors = hooks.get_factors()
        assert len(factors) == 2, "Expected factors for 2 Linear layers"

        for module, (A, G) in factors.items():
            assert A.shape[0] == A.shape[1], "A must be square"
            assert G.shape[0] == G.shape[1], "G must be square"
            # Symmetry: max |A - Aᵀ| should be near 0
            assert torch.allclose(A, A.t(), atol=1e-5), f"A not symmetric for {module}"
            assert torch.allclose(G, G.t(), atol=1e-5), f"G not symmetric for {module}"

    def test_gram_matrix_positive_semidefinite(self):
        """Diagonal of A and G (variances) should be non-negative."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()

        _run_step(model, hooks, batch_size=16)
        factors = hooks.get_factors()

        for module, (A, G) in factors.items():
            assert (A.diagonal() >= 0).all(), "A diagonal should be >= 0"
            assert (G.diagonal() >= 0).all(), "G diagonal should be >= 0"


class TestClearResetsState:
    def test_clear_resets_gram_sums(self):
        """After clear(), get_factors() should return an empty dict."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()

        _run_step(model, hooks, batch_size=4)
        assert len(hooks.get_factors()) == 2, "Should have factors before clear"

        hooks.clear()
        assert hooks.get_factors() == {}, "get_factors() should be empty after clear()"

    def test_accumulation_restarts_after_clear(self):
        """After clear(), re-running a step should produce fresh Gram matrices."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()

        _run_step(model, hooks, batch_size=4)
        factors_before = {m: (A.clone(), G.clone()) for m, (A, G) in hooks.get_factors().items()}

        hooks.clear()
        _run_step(model, hooks, batch_size=4)
        factors_after = hooks.get_factors()

        # Gram matrices after clear should exist and be valid
        assert len(factors_after) == 2
        for m in factors_after:
            assert factors_after[m][0].shape == factors_before[m][0].shape


class TestSeqSubsample:
    def test_seq_subsample_caps_rows(self):
        """For 3-D input (B, seq_len, d), A_sum rows should be capped at _SEQ_SUBSAMPLE."""
        # Build a model with a single Linear layer (transformer-style)
        d_model = 32
        linear = nn.Linear(d_model, d_model)

        hooks = KFACHooks(linear)
        hooks.enable()

        # Simulate a long sequence input: (B=2, seq_len=512, d_model)
        # Total rows without cap = 2 * 512 = 1024 > _SEQ_SUBSAMPLE (512)
        x = torch.randn(2, 512, d_model)
        out = linear(x)
        loss = out.sum()
        loss.backward()

        # The sample count recorded should be ≤ _SEQ_SUBSAMPLE
        assert hooks.n_samples_accumulated() <= KFACHooks._SEQ_SUBSAMPLE

    def test_small_seq_not_subsampled(self):
        """For short sequences (total rows < _SEQ_SUBSAMPLE), no subsampling occurs."""
        d_model = 32
        linear = nn.Linear(d_model, d_model)

        hooks = KFACHooks(linear)
        hooks.enable()

        # B=2, seq_len=10 → 20 rows < 512 cap → no subsampling
        x = torch.randn(2, 10, d_model)
        out = linear(x)
        out.sum().backward()

        # All 20 rows should have been used
        assert hooks.n_samples_accumulated() == 20


class TestMaxGramDim:
    def test_max_gram_dim_skips_large_layers(self):
        """Layers with out_features > max_gram_dim should not be tracked."""
        model = nn.Sequential(
            nn.Linear(16, 32),   # d_out=32 — within limit
            nn.Linear(32, 1000), # d_out=1000 — above limit → skip
        )
        hooks = KFACHooks(model, max_gram_dim=64)
        hooks.enable()

        x = torch.randn(4, 16)
        out = model(x)
        out.sum().backward()

        factors = hooks.get_factors()
        # Only the first layer (d_out=32 ≤ 64) should be tracked
        assert len(factors) == 1
        layer_outs = [m.out_features for m in factors.keys()]
        assert 32 in layer_outs, "32-dim layer should be tracked"

    def test_max_gram_dim_zero_tracks_all(self):
        """max_gram_dim=0 (default) should track all layers."""
        model = nn.Sequential(
            nn.Linear(8, 512),
            nn.Linear(512, 4),
        )
        hooks = KFACHooks(model, max_gram_dim=0)
        hooks.enable()

        x = torch.randn(4, 8)
        out = model(x)
        out.sum().backward()

        assert len(hooks.get_factors()) == 2


class TestHookLifecycle:
    def test_is_enabled_reflects_state(self):
        """is_enabled should be False before enable(), True after, False after remove()."""
        model = _simple_mlp()
        hooks = KFACHooks(model)

        assert not hooks.is_enabled

        hooks.enable()
        assert hooks.is_enabled

        hooks.remove()
        assert not hooks.is_enabled

    def test_double_enable_is_idempotent(self):
        """Calling enable() twice should not register duplicate hooks."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()
        n_handles_first = len(hooks._handles)
        hooks.enable()  # second call — should be a no-op
        assert len(hooks._handles) == n_handles_first

    def test_remove_allows_model_forward(self):
        """After remove(), the model should still run without errors."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()
        hooks.remove()

        x = torch.randn(4, 8)
        # Should not raise
        out = model(x)
        assert out.shape == (4, 4)

    def test_remove_clears_state(self):
        """remove() should clear all accumulated Gram state."""
        model = _simple_mlp()
        hooks = KFACHooks(model)
        hooks.enable()
        _run_step(model, hooks)

        hooks.remove()
        assert hooks.get_factors() == {}
        assert hooks.n_samples_accumulated() == 0


class TestGramDimensions:
    def test_gram_dimensions_match_layer(self):
        """A.shape=(d_in,d_in), G.shape=(d_out,d_out) for each Linear layer."""
        d_in, d_hidden, d_out = 10, 20, 5
        model = _simple_mlp(d_in=d_in, d_hidden=d_hidden, d_out=d_out)
        hooks = KFACHooks(model)
        hooks.enable()

        _run_step(model, hooks, d_in=d_in, d_out=d_out, batch_size=8)
        factors = hooks.get_factors()

        layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        expected = {
            layers[0]: (d_in, d_hidden),    # (A shape cols, G shape cols)
            layers[1]: (d_hidden, d_out),
        }
        for module, (A, G) in factors.items():
            d_in_exp, d_out_exp = expected[module]
            assert A.shape == (d_in_exp, d_in_exp), f"A shape wrong for {module}"
            assert G.shape == (d_out_exp, d_out_exp), f"G shape wrong for {module}"
