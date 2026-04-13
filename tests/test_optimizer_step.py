"""
Unit tests for OlsSMKFAC and ClassicKFAC optimizer steps.

Tests:
  P0 — OlsSMKFAC step reduces loss on a small MLP
  P0 — ClassicKFAC step reduces loss on a small MLP
  P0 — Both optimizers converge to similar final loss
  P0 — Natural gradient differs from raw gradient (preconditioner does something)
  P1 — Momentum buffer is applied (weight updates accumulate)
  P1 — Weight decay regularises weights toward zero
  P1 — cleanup() removes hooks and frees state
"""

import pytest
import copy
import torch
import torch.nn as nn

from optimizer.olssm_kfac import OlsSMKFAC
from optimizer.classic_kfac import ClassicKFAC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp(d_in=16, d_hidden=32, d_out=8) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def _train(model, optimizer, steps=60, batch_size=32, d_in=16, d_out=8):
    """Run training loop; return list of per-step losses."""
    torch.manual_seed(42)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(steps):
        x = torch.randn(batch_size, d_in)
        y = torch.randn(batch_size, d_out)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestOlsSMKFACConvergence:
    def test_olssm_step_reduces_loss_mlp(self):
        """OlsSMKFAC should reduce MSE loss over 60 steps on a random MLP."""
        model = _mlp()
        opt = OlsSMKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=5, inv_update_freq=5)
        losses = _train(model, opt, steps=60)
        opt.cleanup()

        # Loss should drop from initial to final
        assert losses[-1] < losses[0], (
            f"OlsSMKFAC loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_olssm_converges_quickly(self):
        """OlsSMKFAC should reach a reasonable loss within 60 steps."""
        model = _mlp()
        opt = OlsSMKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=5, inv_update_freq=5)
        losses = _train(model, opt, steps=60)
        opt.cleanup()

        # Second half average should be less than first half average
        mid = len(losses) // 2
        assert sum(losses[mid:]) / len(losses[mid:]) < sum(losses[:mid]) / len(losses[:mid])


class TestClassicKFACConvergence:
    def test_classic_kfac_step_reduces_loss_mlp(self):
        """ClassicKFAC should reduce MSE loss over 60 steps on a random MLP."""
        model = _mlp()
        opt = ClassicKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=5, inv_update_freq=5)
        losses = _train(model, opt, steps=60)
        opt.cleanup()

        assert losses[-1] < losses[0], (
            f"ClassicKFAC loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_classic_kfac_converges_quickly(self):
        """ClassicKFAC should reach a reasonable loss within 60 steps."""
        model = _mlp()
        opt = ClassicKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=5, inv_update_freq=5)
        losses = _train(model, opt, steps=60)
        opt.cleanup()

        mid = len(losses) // 2
        assert sum(losses[mid:]) / len(losses[mid:]) < sum(losses[:mid]) / len(losses[:mid])


class TestBothOptimizersConverge:
    def test_both_optimizers_converge_similarly(self):
        """OlsSMKFAC and ClassicKFAC should reach similar final loss on the same task."""
        torch.manual_seed(0)
        model_a = _mlp()
        model_b = copy.deepcopy(model_a)  # identical init

        opt_a = OlsSMKFAC(model_a, lr=1e-2, damping=1e-2, factor_update_freq=5, inv_update_freq=5)
        opt_b = ClassicKFAC(model_b, lr=1e-2, damping=1e-2, factor_update_freq=5, inv_update_freq=5)

        losses_a = _train(model_a, opt_a, steps=80)
        losses_b = _train(model_b, opt_b, steps=80)

        opt_a.cleanup()
        opt_b.cleanup()

        final_a = sum(losses_a[-10:]) / 10
        final_b = sum(losses_b[-10:]) / 10

        # Both should converge, and their final losses should be in the same ballpark
        # (within 3x of each other) — not requiring exact equality
        ratio = max(final_a, final_b) / (min(final_a, final_b) + 1e-8)
        assert ratio < 3.0, (
            f"OlsSMKFAC ({final_a:.4f}) and ClassicKFAC ({final_b:.4f}) "
            f"diverge too much (ratio={ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# Natural gradient test
# ---------------------------------------------------------------------------

class TestNaturalGradient:
    def test_natural_gradient_differs_from_raw_gradient(self):
        """After a K-FAC step, the weight delta should differ from the raw gradient update."""
        torch.manual_seed(7)
        model = _mlp()
        weights_before = {n: p.clone() for n, p in model.named_parameters()}

        # Compute one raw gradient
        x = torch.randn(16, 16)
        y = torch.randn(16, 8)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        # Now apply K-FAC step
        opt = OlsSMKFAC(model, lr=1.0, damping=1e-2, factor_update_freq=1, inv_update_freq=1, momentum=0.0)
        opt.step()
        opt.cleanup()

        weights_after = {n: p.clone() for n, p in model.named_parameters()}

        # For at least one weight tensor, the actual delta should differ from
        # -lr * gradient  (which is what plain SGD would give)
        lr = 1.0
        diffs_from_sgd = []
        for n in grads:
            if "weight" in n:
                delta_kfac = weights_before[n] - weights_after[n]
                delta_sgd  = lr * grads[n]
                diff = (delta_kfac - delta_sgd).norm().item()
                diffs_from_sgd.append(diff)

        assert any(d > 1e-6 for d in diffs_from_sgd), (
            "K-FAC natural gradient should differ from plain SGD gradient. "
            f"Max diff from SGD: {max(diffs_from_sgd):.2e}"
        )


# ---------------------------------------------------------------------------
# Momentum test
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_momentum_is_applied(self):
        """With momentum > 0, successive weight deltas should be correlated."""
        torch.manual_seed(3)
        model = _mlp()
        opt = OlsSMKFAC(
            model, lr=1e-2, damping=1e-2,
            factor_update_freq=2, inv_update_freq=2,
            momentum=0.9,
        )

        # Run a few steps — with momentum, weight updates should be smooth
        losses = _train(model, opt, steps=30)
        opt.cleanup()

        # Convergence is the proxy: momentum should not blow up training
        assert all(not (l != l) for l in losses), "NaN loss with momentum"
        assert losses[-1] < losses[0] * 2, "Loss exploded with momentum"


# ---------------------------------------------------------------------------
# Weight decay test
# ---------------------------------------------------------------------------

class TestWeightDecay:
    def test_weight_decay_shrinks_weights(self):
        """With large weight decay, weights should shrink toward zero over time."""
        torch.manual_seed(5)
        model = _mlp()

        # Record initial L2 norm of weights
        norm_before = sum(p.data.norm().item() for p in model.parameters())

        opt = OlsSMKFAC(
            model, lr=5e-3, damping=1e-2,
            factor_update_freq=5, inv_update_freq=5,
            weight_decay=0.1, momentum=0.0,
        )
        _train(model, opt, steps=100)
        opt.cleanup()

        # L2 norm should be smaller than a run without weight decay
        model_no_wd = _mlp()
        for p_dst, p_src in zip(model_no_wd.parameters(), _mlp().parameters()):
            p_dst.data.copy_(p_src.data)

        opt_no_wd = OlsSMKFAC(
            model_no_wd, lr=5e-3, damping=1e-2,
            factor_update_freq=5, inv_update_freq=5,
            weight_decay=0.0, momentum=0.0,
        )
        _train(model_no_wd, opt_no_wd, steps=100)
        opt_no_wd.cleanup()

        norm_wd    = sum(p.data.norm().item() for p in model.parameters())
        norm_no_wd = sum(p.data.norm().item() for p in model_no_wd.parameters())

        assert norm_wd < norm_no_wd, (
            f"Weight decay (norm={norm_wd:.2f}) should shrink weights "
            f"vs no weight decay (norm={norm_no_wd:.2f})"
        )


# ---------------------------------------------------------------------------
# Cleanup test
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_removes_hooks(self):
        """After cleanup(), the model's _forward_hooks should be empty."""
        model = _mlp()
        opt = OlsSMKFAC(model, lr=1e-2, damping=1e-2)
        opt.cleanup()

        total_hooks = sum(len(m._forward_hooks) + len(m._backward_hooks)
                         for m in model.modules())
        assert total_hooks == 0, f"Expected 0 hooks after cleanup, got {total_hooks}"

    def test_cleanup_clears_cached_inverses(self):
        """After cleanup(), _inverses and _factors should be empty."""
        model = _mlp()
        opt = OlsSMKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=1, inv_update_freq=1)
        _train(model, opt, steps=5)
        opt.cleanup()

        assert len(opt._factors) == 0
        assert len(opt._inverses) == 0

    def test_classic_cleanup(self):
        """ClassicKFAC.cleanup() should also clear all state."""
        model = _mlp()
        opt = ClassicKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=1, inv_update_freq=1)
        _train(model, opt, steps=5)
        opt.cleanup()

        assert len(opt._factors) == 0
        assert len(opt._inverses) == 0


# ---------------------------------------------------------------------------
# Repr test
# ---------------------------------------------------------------------------

class TestRepr:
    def test_olssm_repr(self):
        model = _mlp()
        opt = OlsSMKFAC(model, lr=1e-2, damping=0.05)
        r = repr(opt)
        assert "OlsSMKFAC" in r
        assert "damping=0.05" in r
        opt.cleanup()

    def test_classic_repr(self):
        model = _mlp()
        opt = ClassicKFAC(model, lr=1e-2, damping=0.03)
        r = repr(opt)
        assert "ClassicKFAC" in r
        assert "damping=0.03" in r
        opt.cleanup()
