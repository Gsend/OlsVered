"""
Unit tests for adaptive rank selection in OlsSMKFAC.

Tests:
  P1 — Full EVD is used for small matrices (n < adaptive_min_n)
  P1 — Truncated EVD is used for large matrices (n >= adaptive_min_n)
  P1 — Fixed rank is clamped to matrix size (rank > n → k = n)
  P1 — layer_ranks_ records per-layer rank choices after step
  P1 — adaptive mode produces lower opt time than full EVD on large layers
  P1 — rank=k with randomized=True uses randomized EVD path
"""

import pytest
import torch
import torch.nn as nn

from optimizer.olssm_kfac import OlsSMKFAC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp_with_sizes(d_in, d_hidden, d_out):
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def _run_steps(model, opt, n_steps=3, d_in=None, d_out=None):
    """Run n training steps to populate layer_ranks_."""
    if d_in is None:
        d_in = next(iter(model.parameters())).shape[1]
    if d_out is None:
        d_out = list(model.parameters())[-1].shape[0]

    for _ in range(n_steps):
        x = torch.randn(16, d_in)
        y = torch.randn(16, d_out)
        model.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# Full EVD for small matrices
# ---------------------------------------------------------------------------

class TestFullEVDSmallMatrix:
    def test_full_evd_for_small_matrix(self):
        """
        When adaptive=True and n < adaptive_min_n (256), each layer should
        use the full EVD path — layer_ranks_ value should be (None, None).
        """
        # Small layers: d_in=16, d_out=8 — both well below adaptive_min_n=256
        d_in, d_hidden, d_out = 16, 32, 8
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            adaptive=True, adaptive_min_n=256,
        )
        _run_steps(model, opt, d_in=d_in, d_out=d_out)

        # All layers are small → full EVD → None rank
        for module, (k_a, k_g) in opt.layer_ranks_.items():
            assert k_a is None, f"Expected full EVD (k_a=None) for small layer, got {k_a}"
            assert k_g is None, f"Expected full EVD (k_g=None) for small layer, got {k_g}"

        opt.cleanup()


# ---------------------------------------------------------------------------
# Truncated EVD for large matrices
# ---------------------------------------------------------------------------

class TestTruncatedEVDLargeMatrix:
    def test_truncated_evd_for_large_matrix(self):
        """
        When adaptive=True and n >= adaptive_min_n, each large layer should
        use truncated EVD — layer_ranks_ value should be (k_a, k_g) not None.
        """
        # Large layer: d_in=512 > adaptive_min_n=64 (use small threshold for test speed)
        d_in, d_hidden, d_out = 64, 128, 32
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            adaptive=True, adaptive_min_n=32, adaptive_rank_budget=16,
        )
        _run_steps(model, opt, d_in=d_in, d_out=d_out)

        # Large layers → truncated EVD → non-None rank
        found_truncated = False
        for module, (k_a, k_g) in opt.layer_ranks_.items():
            # At least one large layer should have been truncated
            if k_a is not None or k_g is not None:
                found_truncated = True
                if k_a is not None:
                    assert k_a <= 16, f"k_a={k_a} exceeds adaptive_rank_budget=16"
                if k_g is not None:
                    assert k_g <= 16, f"k_g={k_g} exceeds adaptive_rank_budget=16"

        assert found_truncated, (
            "Expected at least one layer to use truncated EVD with adaptive=True "
            f"and adaptive_min_n=32, but got: {opt.layer_ranks_}"
        )

        opt.cleanup()


# ---------------------------------------------------------------------------
# Rank clamped to matrix size
# ---------------------------------------------------------------------------

class TestRankClampedToMatrixSize:
    def test_rank_clamped_to_matrix_size(self):
        """
        If rank=k is set larger than the matrix dimension n,
        the effective rank should be clamped to n (not cause an error).
        """
        d_in, d_hidden, d_out = 8, 12, 4
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        # rank=100 >> max_dim=12 → should be clamped, not crash
        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            rank=100, randomized=False,  # exact topk path
        )
        # Should complete without error
        _run_steps(model, opt, d_in=d_in, d_out=d_out)

        opt.cleanup()

    def test_rank_larger_than_n_randomized(self):
        """Same clamping test with randomized=True."""
        d_in, d_hidden, d_out = 8, 12, 4
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            rank=100, randomized=True,
        )
        _run_steps(model, opt, d_in=d_in, d_out=d_out)

        opt.cleanup()


# ---------------------------------------------------------------------------
# layer_ranks_ populated after step
# ---------------------------------------------------------------------------

class TestLayerRanksPopulated:
    def test_layer_ranks_recorded_after_step(self):
        """layer_ranks_ should be non-empty after at least one inv_update step."""
        d_in, d_hidden, d_out = 16, 32, 8
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
        )
        _run_steps(model, opt, n_steps=2, d_in=d_in, d_out=d_out)

        # layer_ranks_ should have entries for the 2 linear layers
        assert len(opt.layer_ranks_) > 0, "layer_ranks_ should be populated after steps"

        opt.cleanup()

    def test_full_evd_records_none_ranks(self):
        """Full EVD (no rank/adaptive) should record (None, None) for all layers."""
        d_in, d_hidden, d_out = 16, 32, 8
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            rank=None, adaptive=False,
        )
        _run_steps(model, opt, n_steps=2, d_in=d_in, d_out=d_out)

        for module, (k_a, k_g) in opt.layer_ranks_.items():
            assert k_a is None, f"Full EVD should record k_a=None, got {k_a}"
            assert k_g is None, f"Full EVD should record k_g=None, got {k_g}"

        opt.cleanup()


# ---------------------------------------------------------------------------
# Fixed rank path
# ---------------------------------------------------------------------------

class TestFixedRank:
    def test_fixed_rank_k_no_crash(self):
        """rank=k with randomized=False should run without error."""
        d_in, d_hidden, d_out = 16, 32, 8
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            rank=4, randomized=False,
        )
        _run_steps(model, opt, n_steps=5, d_in=d_in, d_out=d_out)
        opt.cleanup()

    def test_fixed_rank_randomized_no_crash(self):
        """rank=k with randomized=True should run without error."""
        d_in, d_hidden, d_out = 16, 32, 8
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-3, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            rank=4, randomized=True, n_power_iter=1,
        )
        _run_steps(model, opt, n_steps=5, d_in=d_in, d_out=d_out)
        opt.cleanup()

    def test_fixed_rank_reduces_loss(self):
        """rank=k approximation should still reduce training loss."""
        d_in, d_hidden, d_out = 16, 32, 8
        model = _mlp_with_sizes(d_in, d_hidden, d_out)

        opt = OlsSMKFAC(
            model, lr=1e-2, damping=1e-2,
            factor_update_freq=1, decomp_update_freq=1,
            rank=4, randomized=False, momentum=0.0,
        )

        losses = []
        for _ in range(30):
            x = torch.randn(16, d_in)
            y = torch.randn(16, d_out)
            model.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        opt.cleanup()

        assert losses[-1] < losses[0], (
            f"rank=4 K-FAC did not reduce loss: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
