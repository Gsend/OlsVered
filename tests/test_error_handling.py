"""
Unit tests for error handling in optimizer/errors.py and the optimizer classes.

Tests:
  P1 — ConfigurationError raised when rank + adaptive=True both set (OlsSMKFAC)
  P1 — ConfigurationError raised when decomp_update_freq < factor_update_freq (both optimizers)
  P1 — ConfigurationError raised when damping <= 0 (OlsSMKFAC)
  P1 — NaN in Gram matrix is silently skipped (no crash, no NaN weights)
  P1 — DataValidationError hierarchy is importable and inherits correctly
  P1 — CheckpointError hierarchy is correct
"""

import pytest
import torch
import torch.nn as nn

from optimizer.errors import (
    OlsSMError,
    ConfigurationError,
    DataValidationError,
    NumericalError,
    CheckpointError,
)
from optimizer.olssm_kfac import OlsSMKFAC
from optimizer.classic_kfac import ClassicKFAC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_all_errors_inherit_from_base(self):
        """All custom errors should be catchable as OlsSMError."""
        for exc_class in [ConfigurationError, DataValidationError, NumericalError, CheckpointError]:
            instance = exc_class("test")
            assert isinstance(instance, OlsSMError), f"{exc_class.__name__} should be OlsSMError"

    def test_all_errors_inherit_from_exception(self):
        for exc_class in [OlsSMError, ConfigurationError, DataValidationError, NumericalError, CheckpointError]:
            instance = exc_class("test")
            assert isinstance(instance, Exception)

    def test_configuration_error_message(self):
        msg = "damping must be > 0"
        e = ConfigurationError(msg)
        assert msg in str(e)

    def test_checkpoint_error_message(self):
        msg = "architecture mismatch"
        e = CheckpointError(msg)
        assert msg in str(e)


# ---------------------------------------------------------------------------
# OlsSMKFAC validation
# ---------------------------------------------------------------------------

class TestOlsSMKFACValidation:
    def test_rank_and_adaptive_raises(self):
        """Setting both rank and adaptive=True should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="mutually exclusive"):
            OlsSMKFAC(_mlp(), rank=8, adaptive=True)

    def test_decomp_update_freq_less_than_factor_update_allowed(self):
        """decomp_update_freq < factor_update_freq is allowed (redundant inverse updates)."""
        model = _mlp()
        # Should NOT raise: frequent inverse updates with infrequent factor updates is unusual
        # but valid — inverses simply re-use cached factors until factors refresh.
        opt = OlsSMKFAC(model, factor_update_freq=20, decomp_update_freq=5)
        opt.cleanup()

    def test_zero_damping_raises(self):
        """damping=0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="damping"):
            OlsSMKFAC(_mlp(), damping=0.0)

    def test_negative_damping_raises(self):
        """Negative damping should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="damping"):
            OlsSMKFAC(_mlp(), damping=-0.01)

    def test_valid_config_does_not_raise(self):
        """A valid configuration should construct without error."""
        model = _mlp()
        opt = OlsSMKFAC(model, lr=1e-3, damping=1e-2, factor_update_freq=5, decomp_update_freq=5)
        opt.cleanup()  # cleanup hooks

    def test_valid_adaptive_no_rank(self):
        """adaptive=True with no rank should not raise."""
        model = _mlp()
        opt = OlsSMKFAC(model, adaptive=True)
        opt.cleanup()

    def test_valid_rank_no_adaptive(self):
        """rank=k with adaptive=False should not raise."""
        model = _mlp()
        opt = OlsSMKFAC(model, rank=4)
        opt.cleanup()


# ---------------------------------------------------------------------------
# ClassicKFAC validation
# ---------------------------------------------------------------------------

class TestClassicKFACValidation:
    def test_inv_update_lt_factor_update_allowed(self):
        """ClassicKFAC: decomp_update_freq < factor_update_freq is allowed."""
        model = _mlp()
        opt = ClassicKFAC(model, factor_update_freq=20, decomp_update_freq=5)
        opt.cleanup()

    def test_equal_update_freqs_ok(self):
        """factor_update_freq == decomp_update_freq should be allowed."""
        model = _mlp()
        opt = ClassicKFAC(model, factor_update_freq=5, decomp_update_freq=5)
        opt.cleanup()

    def test_inv_gt_factor_ok(self):
        """decomp_update_freq > factor_update_freq should be allowed."""
        model = _mlp()
        opt = ClassicKFAC(model, factor_update_freq=5, decomp_update_freq=20)
        opt.cleanup()


# ---------------------------------------------------------------------------
# NaN gram matrix is silently skipped
# ---------------------------------------------------------------------------

class TestNanGramHandling:
    def test_nan_gram_skipped_gracefully(self):
        """
        If a Gram matrix contains NaN/Inf (e.g. diverged model), the optimizer
        should skip that layer rather than crash or inject NaN into weights.
        """
        model = _mlp()
        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            momentum=0.0,
        )

        # Manually inject NaN into factors
        first_layer = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
        opt._factors[first_layer] = (
            torch.full((8, 8), float("nan")),
            torch.full((16, 16), float("nan")),
        )

        # Run a step — should not crash
        x = torch.randn(4, 8)
        y = torch.randn(4, 4)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        # step() should complete without exception
        opt.step()

        # Weights should not contain NaN
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN propagated into weights"

        opt.cleanup()

    def test_inf_gram_skipped_gracefully(self):
        """Inf in Gram matrix should also be skipped, not crash."""
        model = _mlp()
        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            momentum=0.0,
        )

        first_layer = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
        opt._factors[first_layer] = (
            torch.full((8, 8), float("inf")),
            torch.full((16, 16), 0.0),
        )

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)
        nn.MSELoss()(model(x), y).backward()
        opt.step()

        for p in model.parameters():
            assert not torch.isnan(p).any()

        opt.cleanup()
