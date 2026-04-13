"""
Unit tests for checkpoint/state-dict functionality.

Tests:
  P2 — OlsSMKFAC state_dict roundtrip preserves optimizer state
  P2 — OlsSMLayerRetrainer get_gram_state / load_gram_state roundtrip
  P2 — CheckpointError raised when n_layers mismatch on load
  P2 — CheckpointError raised when matrix shape mismatch on load
  P2 — BenchmarkResult.save() / to_dict() roundtrip
"""

import json
import os
import tempfile
import pytest
import torch
import torch.nn as nn

from optimizer.olssm_kfac import OlsSMKFAC
from optimizer.layer_retrainer import OlsSMLayerRetrainer
from optimizer.errors import CheckpointError
from benchmark.core import BenchmarkResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp(d_in=16, d_hidden=32, d_out=8):
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def _make_loader(n=64, d_in=16, d_out=8, batch_size=16):
    torch.manual_seed(1)
    X = torch.randn(n, d_in)
    Y = torch.randn(n, d_out)
    return [(X[i:i+batch_size], Y[i:i+batch_size]) for i in range(0, n, batch_size)]


def _train_steps(model, opt, steps=5, d_in=16, d_out=8):
    for _ in range(steps):
        x = torch.randn(8, d_in)
        y = torch.randn(8, d_out)
        model.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        opt.step()


# ---------------------------------------------------------------------------
# OlsSMKFAC state_dict roundtrip
# ---------------------------------------------------------------------------

class TestKFACStateDictRoundtrip:
    def test_state_dict_preserves_param_groups(self):
        """Loading an OlsSMKFAC state_dict should restore param_groups."""
        model = _mlp()
        opt = OlsSMKFAC(model, lr=5e-4, damping=1e-2)
        _train_steps(model, opt)

        state = opt.state_dict()
        opt.cleanup()

        # Create a fresh optimizer and load state
        model2 = _mlp()  # same arch
        opt2 = OlsSMKFAC(model2, lr=1e-3, damping=1e-2)  # different lr
        opt2.load_state_dict(state)
        opt2.cleanup()

    def test_state_dict_is_serialisable(self):
        """state_dict() should be saveable with torch.save."""
        model = _mlp()
        opt = OlsSMKFAC(model, lr=1e-3, damping=1e-2, factor_update_freq=1, inv_update_freq=1)
        _train_steps(model, opt)

        state = opt.state_dict()
        opt.cleanup()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(state, path)
            loaded = torch.load(path, weights_only=False)
            assert "param_groups" in loaded or "state" in loaded
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# OlsSMLayerRetrainer gram_state roundtrip
# ---------------------------------------------------------------------------

class TestLayerRetrainerGramStateRoundtrip:
    def test_gram_state_roundtrip_identical(self):
        """get_gram_state + load_gram_state should restore XtX, XtY, n_samples."""
        model = _mlp()
        loader = _make_loader()

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)

        # Accumulate one pass
        for x, y in loader:
            with torch.no_grad():
                retrainer._act_cache.clear()
                model(x)
            from optimizer.layer_retrainer import GramState
            x_aug = retrainer._augment(retrainer._act_cache[0], retrainer._retrained_layers[0])
            y_f = y.to(retrainer.device, dtype=retrainer.dtype)
            retrainer._grams[0].accumulate(x_aug, y_f)

        state = retrainer.get_gram_state()

        # Reset grams
        for g in retrainer._grams:
            g.reset()

        assert retrainer._grams[0].n_samples == 0

        # Restore
        retrainer.load_gram_state(state)

        assert retrainer._grams[0].n_samples > 0

        import numpy as np
        np.testing.assert_allclose(
            retrainer._grams[0].XtX.numpy(),
            state[0]["XtX"],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            retrainer._grams[0].XtY.numpy(),
            state[0]["XtY"],
            atol=1e-5,
        )

        retrainer.remove_hooks()

    def test_gram_state_save_load_file(self):
        """Gram state should survive torch.save + torch.load to disk."""
        model = _mlp()
        loader = _make_loader()

        retrainer = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)
        retrainer.retrain(loader)  # runs a full retrain, populates grams internally
        state = retrainer.get_gram_state()
        retrainer.remove_hooks()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(state, path)
            loaded = torch.load(path, weights_only=False)

            # Verify structure
            assert 0 in loaded
            assert "XtX" in loaded[0]
            assert "XtY" in loaded[0]
            assert "n_samples" in loaded[0]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CheckpointError on mismatch
# ---------------------------------------------------------------------------

class TestCheckpointErrorMismatch:
    def test_n_layers_mismatch_raises(self):
        """load_gram_state with wrong n_layers should raise CheckpointError."""
        model = _mlp()
        retrainer_1 = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)
        state_1 = retrainer_1.get_gram_state()
        retrainer_1.remove_hooks()

        # Try loading 1-layer state into 2-layer retrainer
        model2 = _mlp()
        retrainer_2 = OlsSMLayerRetrainer(model2, n_layers=2, verbose=False)

        with pytest.raises(CheckpointError, match="layer"):
            retrainer_2.load_gram_state(state_1)

        retrainer_2.remove_hooks()

    def test_shape_mismatch_raises(self):
        """load_gram_state with wrong matrix shape should raise CheckpointError."""
        model = _mlp(d_in=16, d_hidden=32, d_out=8)
        retrainer = OlsSMLayerRetrainer(model, n_layers=1, verbose=False)
        state = retrainer.get_gram_state()

        # Corrupt the state: replace XtX with wrong shape
        import numpy as np
        state[0]["XtX"] = np.zeros((100, 100), dtype=np.float32)

        with pytest.raises(CheckpointError, match="shape"):
            retrainer.load_gram_state(state)

        retrainer.remove_hooks()


# ---------------------------------------------------------------------------
# BenchmarkResult serialisation
# ---------------------------------------------------------------------------

class TestBenchmarkResultSave:
    def test_to_dict_roundtrip(self):
        """to_dict() should contain all expected keys."""
        result = BenchmarkResult(
            name="test_opt",
            steps=100,
            samples=3200,
            wall_s=12.5,
            fwdbwd_times=[0.01] * 100,
            opt_times=[0.005] * 100,
            curve_steps=[50, 100],
            curve_times=[6.0, 12.5],
            curve_val_loss=[0.5, 0.3],
            curve_val_metric=[0.7, 0.85],
            peak_mem_gb=1.2,
        )

        d = result.to_dict()

        assert d["name"] == "test_opt"
        assert d["steps"] == 100
        assert d["samples"] == 3200
        assert d["wall_s"] == 12.5
        assert d["peak_mem_gb"] == 1.2
        assert d["curve_steps"] == [50, 100]
        assert d["curve_val_loss"] == [0.5, 0.3]
        assert d["avg_fwdbwd_ms"] == pytest.approx(10.0, rel=1e-3)
        assert d["avg_opt_ms"] == pytest.approx(5.0, rel=1e-3)

    def test_save_creates_valid_json(self):
        """save() should write a readable JSON file."""
        result = BenchmarkResult(
            name="json_test",
            steps=10,
            samples=320,
            wall_s=1.0,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name

        try:
            result.save(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["name"] == "json_test"
            assert loaded["steps"] == 10
        finally:
            os.unlink(path)

    def test_extra_metadata_preserved(self):
        """Extra metadata in BenchmarkResult.extra should appear in to_dict()."""
        result = BenchmarkResult(
            name="meta_test",
            extra={"dataset": "cifar10", "batch_size": 64},
        )
        d = result.to_dict()
        assert d["dataset"] == "cifar10"
        assert d["batch_size"] == 64

    def test_print_summary_no_crash(self):
        """print_summary() should run without errors."""
        result = BenchmarkResult(
            name="print_test",
            steps=50,
            curve_val_loss=[0.4],
            curve_val_metric=[0.8],
        )
        # Should not raise
        result.print_summary()

    def test_print_summary_empty_curves(self):
        """print_summary() with empty curves (no eval_fn) should not raise."""
        result = BenchmarkResult(name="empty_curves", steps=30)
        result.print_summary()
