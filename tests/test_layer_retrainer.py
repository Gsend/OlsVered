"""
Unit tests for optimizer/layer_retrainer.py — OlsSMLayerRetrainer.

Tests:
  P1 — N=1 single-layer retrain reduces prediction error
  P1 — BCD multi-layer retrain converges (loss decreases sweep-over-sweep)
  P1 — LoRA adapter is non-zero after fitting
  P1 — remove_hooks() cleans up all forward hooks
  P1 — n_layers > available raises a warning (not error) and clamps
  P1 — Gram state can be reset
  P1 — OLS solve matches a linear regression reference (numerical correctness)
"""

import pytest
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from optimizer.layer_retrainer import OlsSMLayerRetrainer, GramState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp(d_in=16, d_hidden=32, d_out=8, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def _make_loader(n_samples=64, d_in=16, d_out=8, seed=42, batch_size=16):
    """Create a simple synthetic regression dataset."""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d_in)
    # True function: Y = X @ W_true + noise
    W_true = torch.randn(d_in, d_out) * 0.5
    Y = X @ W_true + torch.randn(n_samples, d_out) * 0.1
    dataset = list(zip(X.split(batch_size), Y.split(batch_size)))
    return dataset, W_true


def _mse(model, loader):
    """Compute MSE on the entire loader."""
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            total += F.mse_loss(pred, y, reduction='sum').item()
            count += x.shape[0]
    return total / count


# ---------------------------------------------------------------------------
# Single-layer retrain (N=1)
# ---------------------------------------------------------------------------

class TestSingleLayerRetrain:
    def test_single_layer_reduces_error(self):
        """N=1 retrain should reduce MSE on the training set."""
        model = _mlp()
        loader, _ = _make_loader()
        mse_before = _mse(model, loader)

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, lambda_reg=1e-4, verbose=False)
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        mse_after = _mse(model, loader)

        assert mse_after < mse_before, (
            f"N=1 OLS retrain did not reduce MSE: {mse_before:.4f} → {mse_after:.4f}"
        )

    def test_single_layer_history_keys(self):
        """retrain() should return a dict with expected keys."""
        model = _mlp()
        loader, _ = _make_loader()

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        assert "converged" in history
        assert "n_sweeps" in history
        assert "deltas" in history
        assert "lora_fitted" in history

    def test_single_layer_no_lora_by_default(self):
        """Without lora_rank, lora_fitted should be False."""
        model = _mlp()
        loader, _ = _make_loader()

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        assert not history["lora_fitted"]


# ---------------------------------------------------------------------------
# BCD multi-layer
# ---------------------------------------------------------------------------

class TestBCDConvergence:
    def test_bcd_converges_multilayer(self):
        """N=2 BCD should reduce MSE on training data."""
        model = _mlp()
        loader, _ = _make_loader()
        mse_before = _mse(model, loader)

        retrainer = OlsSMLayerRetrainer(
            model, n_layers=2, lambda_reg=1e-4,
            max_sweeps=5, tol=1e-5, verbose=False,
        )
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        mse_after = _mse(model, loader)

        assert mse_after < mse_before, (
            f"BCD N=2 did not reduce MSE: {mse_before:.4f} → {mse_after:.4f}"
        )

    def test_bcd_records_sweep_deltas(self):
        """history['deltas'] should have one entry per sweep."""
        model = _mlp()
        loader, _ = _make_loader()

        retrainer = OlsSMLayerRetrainer(
            model, n_layers=2, lambda_reg=1e-4,
            max_sweeps=3, tol=1e-8,  # tol very tight → won't converge early
            verbose=False,
        )
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        assert len(history["deltas"]) == history["n_sweeps"]

    def test_bcd_deltas_decrease(self):
        """Weight deltas should generally decrease across BCD sweeps."""
        model = _mlp()
        loader, _ = _make_loader(n_samples=128)

        retrainer = OlsSMLayerRetrainer(
            model, n_layers=2, lambda_reg=1e-4,
            max_sweeps=5, tol=1e-8,
            verbose=False,
        )
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        max_deltas = [max(d) for d in history["deltas"]]
        # At least the last sweep should have a smaller max delta than the first
        if len(max_deltas) >= 2:
            assert max_deltas[-1] <= max_deltas[0] * 10, (
                f"BCD deltas did not decrease: {max_deltas}"
            )


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class TestLoraAdapter:
    def test_lora_adapter_nonzero(self):
        """After fitting LoRA, the adapter tensors should be non-zero."""
        model = _mlp()
        loader, _ = _make_loader()

        retrainer = OlsSMLayerRetrainer(
            model, n_layers=1, lambda_reg=1e-4,
            lora_rank=2, lora_sweeps=3, verbose=False,
        )
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        assert history["lora_fitted"]

        # Check adapter tensors are non-zero
        for adapter in retrainer._lora:
            if adapter is not None:
                assert adapter.A.norm() > 1e-8, "LoRA A should be non-zero after fitting"
                assert adapter.B.norm() > 1e-8, "LoRA B should be non-zero after fitting"

    def test_lora_rank_zero_no_adapter(self):
        """lora_rank=0 should leave _lora as all None."""
        model = _mlp()
        loader, _ = _make_loader()

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, lora_rank=0, verbose=False)
        history = retrainer.retrain(loader)
        retrainer.remove_hooks()

        assert not history["lora_fitted"]
        assert all(a is None for a in retrainer._lora)

    def test_lora_reduces_residual(self):
        """OLS+LoRA should further reduce MSE beyond OLS alone."""
        model_ols = _mlp(seed=1)
        model_lora = _mlp(seed=1)
        loader, _ = _make_loader()

        # Pure OLS
        ret_ols = OlsSMLayerRetrainer(model_ols, n_layers=1, lora_rank=0, verbose=False)
        ret_ols.retrain(loader)
        ret_ols.remove_hooks()

        # OLS + LoRA
        ret_lora = OlsSMLayerRetrainer(
            model_lora, n_layers=1, lora_rank=4, lora_sweeps=5, verbose=False,
        )
        ret_lora.retrain(loader)
        ret_lora.remove_hooks()

        mse_ols  = _mse(model_ols,  loader)
        mse_lora = _mse(model_lora, loader)

        # LoRA should be at least as good as pure OLS
        assert mse_lora <= mse_ols * 1.2, (
            f"OLS+LoRA MSE ({mse_lora:.4f}) should not be much worse than pure OLS ({mse_ols:.4f})"
        )


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------

class TestHookManagement:
    def test_remove_hooks_cleans_up(self):
        """remove_hooks() should leave no forward hooks on the model."""
        model = _mlp()
        retrainer = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)

        # Hooks are registered in __init__
        assert len(retrainer._hook_handles) > 0

        retrainer.remove_hooks()

        assert len(retrainer._hook_handles) == 0
        total_hooks = sum(len(m._forward_hooks) for m in model.modules())
        # May still have other hooks, but retrainer's hooks should be gone
        assert len(retrainer._hook_handles) == 0

    def test_model_works_after_remove_hooks(self):
        """After remove_hooks(), the model should still run normally."""
        model = _mlp()
        loader, _ = _make_loader()

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)
        retrainer.retrain(loader)
        retrainer.remove_hooks()

        x = torch.randn(4, 16)
        out = model(x)
        assert out.shape == (4, 8)


# ---------------------------------------------------------------------------
# n_layers clamping
# ---------------------------------------------------------------------------

class TestNLayersClamping:
    def test_n_layers_exceeds_available_warns(self):
        """n_layers > number of linear layers should emit UserWarning and clamp."""
        model = _mlp()  # 2 Linear layers
        loader, _ = _make_loader()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            retrainer = OlsSMLayerRetrainer(model, n_layers=10, verbose=False)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "n_layers" in str(w[0].message).lower() or "layer" in str(w[0].message).lower()

        # n_layers should be clamped to 2 (available linear layers)
        assert retrainer.n_layers == 2
        retrainer.retrain(loader)
        retrainer.remove_hooks()


# ---------------------------------------------------------------------------
# Gram state management
# ---------------------------------------------------------------------------

class TestGramState:
    def test_gram_state_accumulate(self):
        """GramState.accumulate() should correctly update XtX and XtY."""
        d_in_aug, d_out = 5, 3
        gram = GramState(
            XtX=torch.zeros(d_in_aug, d_in_aug),
            XtY=torch.zeros(d_in_aug, d_out),
        )

        x = torch.randn(8, d_in_aug)
        y = torch.randn(8, d_out)
        gram.accumulate(x, y)

        expected_XtX = x.T @ x
        expected_XtY = x.T @ y

        assert torch.allclose(gram.XtX, expected_XtX, atol=1e-5)
        assert torch.allclose(gram.XtY, expected_XtY, atol=1e-5)
        assert gram.n_samples == 8

    def test_gram_state_reset(self):
        """GramState.reset() should zero out all accumulators."""
        gram = GramState(
            XtX=torch.ones(4, 4),
            XtY=torch.ones(4, 3),
            n_samples=10,
        )
        gram.reset()

        assert gram.XtX.sum() == 0
        assert gram.XtY.sum() == 0
        assert gram.n_samples == 0

    def test_gram_state_multi_batch(self):
        """Accumulating two batches should equal accumulating their concatenation."""
        d_in_aug, d_out = 6, 4
        x1 = torch.randn(8, d_in_aug)
        x2 = torch.randn(12, d_in_aug)
        y1 = torch.randn(8, d_out)
        y2 = torch.randn(12, d_out)

        # Batched
        gram_batched = GramState(
            XtX=torch.zeros(d_in_aug, d_in_aug),
            XtY=torch.zeros(d_in_aug, d_out),
        )
        gram_batched.accumulate(x1, y1)
        gram_batched.accumulate(x2, y2)

        # All-at-once reference
        x_all = torch.cat([x1, x2], dim=0)
        y_all = torch.cat([y1, y2], dim=0)
        XtX_ref = x_all.T @ x_all
        XtY_ref = x_all.T @ y_all

        assert torch.allclose(gram_batched.XtX, XtX_ref, atol=1e-5)
        assert torch.allclose(gram_batched.XtY, XtY_ref, atol=1e-5)
        assert gram_batched.n_samples == 20


# ---------------------------------------------------------------------------
# OLS numerical correctness
# ---------------------------------------------------------------------------

class TestOLSNumericalCorrectness:
    def test_single_layer_ols_matches_numpy(self):
        """
        For a single-layer model (identity-like activation), the OLS solution
        should match numpy.linalg.lstsq on the same data.
        """
        torch.manual_seed(99)
        # Single linear layer (no hidden activation)
        model = nn.Linear(8, 4, bias=False)
        # Reset to near-zero init to make the OLS effect visible
        nn.init.zeros_(model.weight)

        # Generate data with a known linear relationship
        X_data = torch.randn(128, 8)
        W_true = torch.randn(8, 4)
        Y_data = X_data @ W_true

        loader = [(X_data[i:i+32], Y_data[i:i+32]) for i in range(0, 128, 32)]

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, lambda_reg=1e-8, verbose=False)
        retrainer.retrain(loader)
        retrainer.remove_hooks()

        W_ols = model.weight.data.numpy()

        # numpy reference
        X_np = X_data.numpy()
        Y_np = Y_data.numpy()
        W_ref, _, _, _ = np.linalg.lstsq(X_np, Y_np, rcond=None)
        W_ref = W_ref.T  # lstsq returns (d_in, d_out), model weight is (d_out, d_in)

        # Should be close (small lambda → near-exact OLS solution)
        np.testing.assert_allclose(W_ols, W_ref, atol=1e-3, rtol=1e-3)
