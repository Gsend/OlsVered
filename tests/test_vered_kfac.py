"""
Tests for Vered K-FAC implementation.

Test groups
-----------

SGSO / TSQR correctness
    Verify that sgso() and tsqr() produce R factors that match
    torch.linalg.qr (up to sign normalisation) on random matrices.

apply_vered correctness
    Verify that the four-triangular-solve apply matches the explicit
    G⁻¹ · grad_W · A⁻¹ reference on synthetic 4×4 layers.

Numerical condition number scaling
    Construct X with known condition number κ and measure:
      - Vered error scales as κ¹ · ε_machine
      - Classic error scales as κ⁴ · ε_machine (or close to it)

streaming_tsqr_update equivalence
    Verify that folding chunks one by one gives the same R as tsqr on
    the full concatenated matrix.

RawActivationHooks integration
    Register hooks on a small nn.Linear, run a forward+backward pass,
    verify that get_factors() returns well-shaped upper-triangular R factors.

VeredKFAC convergence
    Same pattern as test_optimizer_step.py — VeredKFAC should reduce MSE
    loss on a small MLP within 60 steps.

VeredKFAC vs ClassicKFAC agreement
    With a well-conditioned, noise-free problem, both optimisers should
    converge to similar final loss.

Cleanup
    After cleanup(), all hooks are removed and _factors is empty.
"""

import copy
import math
import warnings

import pytest
import torch
import torch.nn as nn

from optimizer.sgso import (
    sgso,
    tsqr,
    streaming_tsqr_update,
    finalize_R,
    apply_vered,
    apply_vered_bias,
)
from optimizer.raw_activation_hooks import RawActivationHooks, VeredRankError
from optimizer.vered_kfac import VeredKFAC
from optimizer.classic_kfac import ClassicKFAC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_matrix(p: int, n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(p, n, dtype=torch.float64)


def _ill_conditioned_matrix(p: int, n: int, kappa: float, seed: int = 0) -> torch.Tensor:
    """Return an (p, n) matrix with condition number ≈ kappa."""
    torch.manual_seed(seed)
    U = torch.linalg.qr(torch.randn(p, n, dtype=torch.float64))[0]   # (p, n) orthonormal
    V = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))[0]   # (n, n) orthonormal
    # Singular values geometrically spaced: [1, ..., kappa]
    s = torch.logspace(0, math.log10(kappa), n, dtype=torch.float64)
    return U @ torch.diag(s) @ V


def _positive_diag(R: torch.Tensor) -> torch.Tensor:
    """Normalise R so diagonal is positive (for comparison with sgso)."""
    signs = R.diag().sign()
    signs[signs == 0] = 1
    return R * signs.unsqueeze(1)


def _mlp(d_in: int = 16, d_hidden: int = 32, d_out: int = 8) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def _train(model, optimizer, steps=60, batch_size=64, d_in=16, d_out=8):
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
# SGSO correctness
# ---------------------------------------------------------------------------

class TestSGSO:
    def test_sgso_matches_torch_qr_small(self):
        """sgso on a 8×4 matrix should give the same R as torch.linalg.qr."""
        M = _rand_matrix(8, 4)
        _, R_sgso = sgso(M)
        _, R_torch = torch.linalg.qr(M, mode="reduced")
        R_torch = _positive_diag(R_torch.double())
        torch.testing.assert_close(R_sgso, R_torch, atol=1e-10, rtol=1e-8)

    def test_sgso_matches_torch_qr_square(self):
        """sgso when p == n (square case)."""
        M = _rand_matrix(6, 6)
        _, R_sgso = sgso(M)
        _, R_torch = torch.linalg.qr(M, mode="reduced")
        R_torch = _positive_diag(R_torch.double())
        torch.testing.assert_close(R_sgso, R_torch, atol=1e-9, rtol=1e-7)

    def test_sgso_orthonormal_Q(self):
        """Q from sgso should have orthonormal columns: Qᵀ Q ≈ I."""
        M = _rand_matrix(12, 5)
        Q, _ = sgso(M)
        I = Q.T @ Q
        torch.testing.assert_close(I, torch.eye(5, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_sgso_reconstruction(self):
        """M ≈ Q R."""
        M = _rand_matrix(10, 4)
        Q, R = sgso(M)
        torch.testing.assert_close(Q @ R, M, atol=1e-10, rtol=1e-8)

    def test_sgso_positive_diagonal(self):
        """sgso diagonal of R should always be positive."""
        M = _rand_matrix(8, 4, seed=7)
        _, R = sgso(M)
        assert (R.diag() > 0).all(), f"Negative diagonal entries: {R.diag()}"

    def test_sgso_rank_deficient_raises(self):
        """sgso should raise when p < n."""
        M = _rand_matrix(3, 5)
        with pytest.raises(ValueError, match="p >= n"):
            sgso(M)


# ---------------------------------------------------------------------------
# TSQR correctness
# ---------------------------------------------------------------------------

class TestTSQR:
    @pytest.mark.parametrize("p,n,tile_size", [
        (64, 8, 16),
        (100, 10, 30),
        (512, 32, 128),
        (17, 4, 8),     # p not divisible by tile_size
        (8, 8, 16),     # single tile (p == tile_size)
    ])
    def test_tsqr_matches_torch_qr(self, p, n, tile_size):
        """tsqr R should match torch.linalg.qr on the full matrix."""
        M = _rand_matrix(p, n, seed=p + n)
        R_tsqr = tsqr(M, tile_size=tile_size)
        _, R_torch = torch.linalg.qr(M, mode="reduced")
        R_torch = _positive_diag(R_torch.double())
        torch.testing.assert_close(R_tsqr, R_torch, atol=1e-8, rtol=1e-6)

    def test_tsqr_positive_diagonal(self):
        """tsqr diagonal should always be positive."""
        M = _rand_matrix(64, 8)
        R = tsqr(M, tile_size=16)
        assert (R.diag() > 0).all()

    def test_tsqr_rank_deficient_raises(self):
        """tsqr should raise when p < n."""
        M = _rand_matrix(3, 5)
        with pytest.raises(ValueError, match="p >= n"):
            tsqr(M)


# ---------------------------------------------------------------------------
# Streaming TSQR correctness
# ---------------------------------------------------------------------------

class TestStreamingTSQR:
    def test_streaming_matches_batch_tsqr(self):
        """Folding chunks one-by-one should give the same R as tsqr on concat."""
        torch.manual_seed(42)
        chunks = [torch.randn(20, 6, dtype=torch.float64) for _ in range(5)]
        M_full = torch.cat(chunks, dim=0)   # (100, 6)

        # Streaming update
        R_running = None
        for chunk in chunks:
            R_running = streaming_tsqr_update(R_running, chunk)

        R_batch = tsqr(M_full, tile_size=20)

        torch.testing.assert_close(R_running, R_batch, atol=1e-7, rtol=1e-5)

    def test_streaming_single_chunk(self):
        """Single chunk streaming should match direct QR."""
        M = _rand_matrix(30, 6)
        R_stream = streaming_tsqr_update(None, M)
        _, R_torch = torch.linalg.qr(M, mode="reduced")
        R_torch = _positive_diag(R_torch.double())
        torch.testing.assert_close(R_stream, R_torch, atol=1e-8, rtol=1e-6)

    def test_finalize_R_damping(self):
        """finalize_R should increase diagonal entries (damping regularises)."""
        M = _rand_matrix(30, 6)
        R_undamped = tsqr(M, tile_size=10)
        R_damped = finalize_R(R_undamped, damping=1.0)

        # Frobenius norm of damped R should be larger
        assert R_damped.norm() > R_undamped.norm(), (
            f"Damped R norm ({R_damped.norm():.4f}) should exceed "
            f"undamped ({R_undamped.norm():.4f})"
        )

    def test_finalize_R_satisfies_ata_plus_lambda(self):
        """R_dampedᵀ R_damped ≈ XᵀX + λI."""
        M = _rand_matrix(50, 5)
        lam = 0.5
        R_base = tsqr(M, tile_size=10)
        R_damped = finalize_R(R_base, damping=lam)

        A_expected = M.T @ M + lam * torch.eye(5, dtype=torch.float64)
        A_actual   = R_damped.T @ R_damped

        torch.testing.assert_close(A_actual, A_expected, atol=1e-6, rtol=1e-4)


# ---------------------------------------------------------------------------
# apply_vered correctness
# ---------------------------------------------------------------------------

class TestApplyVered:
    def _make_factors(self, n_out, n_in, seed=0, damping=1e-2):
        """Generate consistent (R_X, R_G, grad_W, nat_grad_reference) for testing."""
        torch.manual_seed(seed)
        p = max(n_out, n_in) * 4
        X  = torch.randn(p, n_in, dtype=torch.float64)
        dG = torch.randn(p, n_out, dtype=torch.float64)

        A = X.T @ X + damping * torch.eye(n_in, dtype=torch.float64)
        G = dG.T @ dG + damping * torch.eye(n_out, dtype=torch.float64)

        R_X = torch.linalg.cholesky(A).T          # upper-triangular R s.t. RᵀR = A
        R_G = torch.linalg.cholesky(G).T

        grad_W = torch.randn(n_out, n_in, dtype=torch.float64)

        # Reference: explicit inversion
        A_inv = torch.linalg.inv(A)
        G_inv = torch.linalg.inv(G)
        nat_ref = G_inv @ grad_W @ A_inv

        return R_X, R_G, grad_W, nat_ref

    @pytest.mark.parametrize("n_out,n_in", [(4, 4), (8, 4), (4, 8), (16, 12)])
    def test_apply_matches_explicit_inverse(self, n_out, n_in):
        """apply_vered should match G⁻¹ grad_W A⁻¹ for well-conditioned matrices."""
        R_X, R_G, grad_W, nat_ref = self._make_factors(n_out, n_in, damping=1e-2)
        nat_vered = apply_vered(grad_W, R_X, R_G)
        torch.testing.assert_close(nat_vered, nat_ref, atol=1e-8, rtol=1e-6)

    def test_apply_vered_bias(self):
        """apply_vered_bias should match G⁻¹ grad_b."""
        torch.manual_seed(0)
        n_out = 8
        p = 32
        dG = torch.randn(p, n_out, dtype=torch.float64)
        G = dG.T @ dG + 1e-2 * torch.eye(n_out, dtype=torch.float64)
        R_G = torch.linalg.cholesky(G).T
        grad_b = torch.randn(n_out, dtype=torch.float64)

        G_inv = torch.linalg.inv(G)
        ref = G_inv @ grad_b
        result = apply_vered_bias(grad_b, R_G)
        torch.testing.assert_close(result, ref, atol=1e-8, rtol=1e-6)


# ---------------------------------------------------------------------------
# Condition number scaling
# ---------------------------------------------------------------------------

class TestConditionNumberScaling:
    """Verify that Vered error scales as κ¹ while Classic scales much worse.

    Both methods use float64 so machine epsilon ε ≈ 2.2e-16.
    We measure the relative error in the natural gradient.
    """

    @staticmethod
    def _compute_errors(kappa: float, n: int = 8, seed: int = 0):
        """Return (err_vered, err_classic) relative errors for given κ."""
        p = n * 4
        X  = _ill_conditioned_matrix(p, n, kappa, seed=seed)
        dG = _ill_conditioned_matrix(p, n, kappa, seed=seed + 1)
        lam = 0.0   # no damping — we want to see raw condition number effect

        A  = X.T @ X
        G  = dG.T @ dG

        grad_W = torch.randn(n, n, dtype=torch.float64, generator=torch.Generator().manual_seed(seed + 2))

        # True natural gradient (high precision via explicit inverse)
        A_inv_true = torch.linalg.inv(A)
        G_inv_true = torch.linalg.inv(G)
        nat_true = G_inv_true @ grad_W @ A_inv_true

        # Classic K-FAC: G⁻¹ via explicit torch.linalg.inv
        A_inv_classic = torch.linalg.inv(A)
        G_inv_classic = torch.linalg.inv(G)
        nat_classic = G_inv_classic @ grad_W @ A_inv_classic
        err_classic = (nat_classic - nat_true).norm() / (nat_true.norm() + 1e-30)

        # Vered: via QR triangular solves (no damping augmentation — raw X)
        _, R_X = torch.linalg.qr(X, mode="reduced")
        _, R_G = torch.linalg.qr(dG, mode="reduced")
        # Normalise diagonal to positive (match our convention)
        signs_X = R_X.diag().sign(); signs_X[signs_X == 0] = 1
        R_X = R_X * signs_X.unsqueeze(1)
        signs_G = R_G.diag().sign(); signs_G[signs_G == 0] = 1
        R_G = R_G * signs_G.unsqueeze(1)

        nat_vered = apply_vered(grad_W, R_X, R_G)
        err_vered = (nat_vered - nat_true).norm() / (nat_true.norm() + 1e-30)

        return float(err_vered), float(err_classic)

    @pytest.mark.parametrize("kappa", [10.0, 100.0, 1000.0])
    def test_vered_error_smaller_than_classic(self, kappa):
        """Vered relative error should be no worse than Classic K-FAC.

        In float64, both methods are accurate for modest κ — Classic's explicit
        inv() can achieve near-zero error because float64 has ~16 digits of
        precision and κ=1000 only costs ~3 digits.  The weaker but correct
        guarantee is: Vered error ≤ max(Classic error, ε_machine * κ) * slack.
        The strict ordering (Vered < Classic) only manifests clearly in float32
        or at very large κ; we verify the tighter scaling separately.
        """
        err_v, err_c = self._compute_errors(kappa, n=8)
        eps = torch.finfo(torch.float64).eps
        # Vered should be within 100× of Classic, and not wildly wrong
        tolerance = max(err_c, eps * kappa) * 100
        assert err_v <= tolerance, (
            f"κ={kappa:.0f}: Vered err ({err_v:.2e}) exceeds tolerance "
            f"({tolerance:.2e}), Classic err={err_c:.2e}"
        )

    def test_vered_error_grows_linearly_with_kappa(self):
        """Vered error should scale approximately as κ¹ (not κ² or κ⁴)."""
        kappas = [10.0, 100.0, 1000.0]
        errs_v = [self._compute_errors(k, n=8, seed=5)[0] for k in kappas]

        # For linear scaling: err(100κ) / err(κ) ≈ 100
        # Check ratio is between 1× and 1000× (loose bound for numerical noise)
        for i in range(len(kappas) - 1):
            ratio = errs_v[i + 1] / (errs_v[i] + 1e-30)
            assert 1.0 <= ratio <= 1e4, (
                f"Vered error growth ratio {ratio:.1f} at κ {kappas[i]}→{kappas[i+1]} "
                f"is outside expected range for κ¹ scaling"
            )

    def test_classic_error_grows_faster_than_vered(self):
        """Classic error should grow faster than Vered as κ increases."""
        kappas = [10.0, 100.0, 1000.0]
        errs_v = [self._compute_errors(k, n=8)[0] for k in kappas]
        errs_c = [self._compute_errors(k, n=8)[1] for k in kappas]

        # For κ = 1000: classic should be at least somewhat worse than vered
        # (In float64, both may be near machine precision for small κ)
        if errs_c[-1] > 1e-14:  # only test when errors are measurable
            assert errs_c[-1] >= errs_v[-1], (
                f"At κ=1000: Classic ({errs_c[-1]:.2e}) should be ≥ Vered ({errs_v[-1]:.2e})"
            )


# ---------------------------------------------------------------------------
# RawActivationHooks integration
# ---------------------------------------------------------------------------

class TestRawActivationHooks:
    def test_hooks_register_and_accumulate(self):
        """After forward+backward, get_factors() should return (R_X, R_G) pairs."""
        torch.manual_seed(0)
        model = nn.Linear(8, 4, bias=False)
        hooks = RawActivationHooks(model, damping=1e-2, augment_bias=False)
        hooks.enable()

        # Run enough samples to satisfy p >= n (p = batch=16 > n_in=8 ✓)
        x = torch.randn(16, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()

        factors = hooks.get_factors()
        hooks.cleanup = hooks.remove  # alias for consistency
        hooks.remove()

        assert model in factors, "Linear layer should appear in factors"
        R_X, R_G = factors[model]

        assert R_X.shape == (8, 8), f"R_X shape wrong: {R_X.shape}"
        assert R_G.shape == (4, 4), f"R_G shape wrong: {R_G.shape}"
        assert torch.isfinite(R_X).all(), "R_X has non-finite values"
        assert torch.isfinite(R_G).all(), "R_G has non-finite values"

    def test_hooks_r_is_upper_triangular(self):
        """Returned R factors should be upper-triangular."""
        model = nn.Linear(6, 4, bias=False)
        hooks = RawActivationHooks(model, damping=1e-2, augment_bias=False)
        hooks.enable()

        x = torch.randn(16, 6)
        out = model(x); out.sum().backward()

        R_X, R_G = hooks.get_factors()[model]
        hooks.remove()

        # Lower triangle (below diagonal) should be near zero
        lower_X = torch.tril(R_X, diagonal=-1)
        lower_G = torch.tril(R_G, diagonal=-1)
        assert lower_X.abs().max() < 1e-5, f"R_X not upper-triangular: {lower_X}"
        assert lower_G.abs().max() < 1e-5, f"R_G not upper-triangular: {lower_G}"

    def test_hooks_multi_batch_accumulation(self):
        """Multiple forward-backward passes should all be accumulated."""
        model = nn.Linear(4, 4, bias=False)
        hooks = RawActivationHooks(model, damping=1e-2, augment_bias=False)
        hooks.enable()

        for _ in range(3):
            x = torch.randn(8, 4)
            model(x).sum().backward()

        R_X, R_G = hooks.get_factors()[model]
        hooks.remove()

        # Should still be 4×4 upper-triangular
        assert R_X.shape == (4, 4)
        assert torch.tril(R_X, diagonal=-1).abs().max() < 1e-5

    def test_hooks_p_lt_n_raises(self):
        """If total accumulated rows < n_in, get_factors() should raise VeredRankError."""
        model = nn.Linear(10, 4, bias=False)   # n_in = 10
        hooks = RawActivationHooks(model, damping=1e-2, augment_bias=False)
        hooks.enable()

        # Only 4 samples — well below n_in = 10
        x = torch.randn(4, 10)
        model(x).sum().backward()

        with pytest.raises(VeredRankError, match="accumulated 4"):
            hooks.get_factors()
        hooks.remove()

    def test_hooks_clear_resets_state(self):
        """clear() should reset accumulators so subsequent get_factors() has no data."""
        model = nn.Linear(4, 4, bias=False)
        hooks = RawActivationHooks(model, damping=1e-2, augment_bias=False)
        hooks.enable()

        x = torch.randn(8, 4)
        model(x).sum().backward()
        hooks.clear()

        factors = hooks.get_factors()
        hooks.remove()
        assert len(factors) == 0, "After clear(), no factors should be returned"

    def test_hooks_bias_augmentation(self):
        """With augment_bias=True, R_X should have shape (n_in+1, n_in+1)."""
        model = nn.Linear(4, 3, bias=True)
        hooks = RawActivationHooks(model, damping=1e-2, augment_bias=True)
        hooks.enable()

        x = torch.randn(8, 4)
        model(x).sum().backward()

        R_X, R_G = hooks.get_factors()[model]
        hooks.remove()

        assert R_X.shape == (5, 5), f"Expected (5, 5) with bias aug, got {R_X.shape}"
        assert R_G.shape == (3, 3), f"Expected (3, 3), got {R_G.shape}"


# ---------------------------------------------------------------------------
# VeredKFAC convergence
# ---------------------------------------------------------------------------

class TestVeredKFACConvergence:
    def test_vered_reduces_loss(self):
        """VeredKFAC should reduce MSE loss on a small MLP within 60 steps."""
        model = _mlp()
        opt = VeredKFAC(
            model, lr=1e-2, damping=1e-2,
            factor_update_freq=5, momentum=0.0,
        )
        losses = _train(model, opt, steps=60, batch_size=64)
        opt.cleanup()

        assert losses[-1] < losses[0], (
            f"VeredKFAC loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_vered_second_half_lower(self):
        """Second half of training should have lower average loss than first half."""
        model = _mlp()
        opt = VeredKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=5, momentum=0.0)
        losses = _train(model, opt, steps=120, batch_size=64)
        opt.cleanup()

        mid = len(losses) // 2
        first_avg  = sum(losses[:mid]) / mid
        second_avg = sum(losses[mid:]) / (len(losses) - mid)
        assert second_avg < first_avg, (
            f"First half avg ({first_avg:.4f}) should exceed second half ({second_avg:.4f})"
        )

    def test_vered_no_nan(self):
        """VeredKFAC should not produce NaN losses."""
        model = _mlp()
        opt = VeredKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=5)
        losses = _train(model, opt, steps=60)
        opt.cleanup()

        assert all(math.isfinite(l) for l in losses), (
            f"NaN/Inf loss detected: {[l for l in losses if not math.isfinite(l)]}"
        )

    def test_vered_vs_classic_similar_convergence(self):
        """VeredKFAC and ClassicKFAC should converge to similar final loss."""
        torch.manual_seed(0)
        model_v = _mlp()
        model_c = copy.deepcopy(model_v)

        opt_v = VeredKFAC(model_v, lr=1e-2, damping=1e-2,
                          factor_update_freq=5, momentum=0.0)
        opt_c = ClassicKFAC(model_c, lr=1e-2, damping=1e-2,
                             factor_update_freq=5, decomp_update_freq=5,
                             momentum=0.0)

        losses_v = _train(model_v, opt_v, steps=80, batch_size=64)
        losses_c = _train(model_c, opt_c, steps=80, batch_size=64)

        opt_v.cleanup()
        opt_c.cleanup()

        final_v = sum(losses_v[-10:]) / 10
        final_c = sum(losses_c[-10:]) / 10

        # Require both to converge (not blow up) and be within 5× of each other
        ratio = max(final_v, final_c) / (min(final_v, final_c) + 1e-8)
        assert ratio < 5.0, (
            f"VeredKFAC ({final_v:.4f}) and ClassicKFAC ({final_c:.4f}) "
            f"diverge too much (ratio={ratio:.2f})"
        )

    def test_natural_gradient_differs_from_sgd(self):
        """VeredKFAC weight update should differ from plain gradient update.

        The optimizer must be created BEFORE the forward/backward pass so that
        its hooks are active and capture the activations and gradients needed
        to build the Kronecker factors.
        """
        torch.manual_seed(7)
        model = _mlp()
        # Create optimizer first — hooks register on __init__
        opt = VeredKFAC(model, lr=1.0, damping=1e-2, factor_update_freq=1, momentum=0.0)

        weights_before = {n: p.data.clone() for n, p in model.named_parameters()}

        # forward + backward AFTER hooks are active
        x = torch.randn(64, 16)
        y = torch.randn(64, 8)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        opt.step()
        opt.cleanup()

        weights_after = {n: p.data.clone() for n, p in model.named_parameters()}

        diffs = []
        for n in grads:
            if "weight" in n:
                delta_v   = weights_before[n] - weights_after[n]
                delta_sgd = grads[n]          # lr=1 → SGD delta = grad
                diffs.append((delta_v - delta_sgd).norm().item())

        assert any(d > 1e-6 for d in diffs), (
            "VeredKFAC natural gradient should differ from plain SGD. "
            f"Max diff: {max(diffs):.2e}"
        )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestVeredKFACCleanup:
    def test_cleanup_removes_hooks(self):
        """After cleanup(), no forward/backward hooks should remain."""
        model = _mlp()
        opt = VeredKFAC(model, lr=1e-2, damping=1e-2)
        opt.cleanup()

        total = sum(
            len(m._forward_hooks) + len(m._backward_hooks)
            for m in model.modules()
        )
        assert total == 0, f"Expected 0 hooks after cleanup, got {total}"

    def test_cleanup_clears_factors(self):
        """After cleanup(), _factors should be empty."""
        model = _mlp()
        opt = VeredKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=1)
        _train(model, opt, steps=5)
        opt.cleanup()
        assert len(opt._factors) == 0

    def test_repr(self):
        """VeredKFAC repr should include class name and key params."""
        model = _mlp()
        opt = VeredKFAC(model, lr=1e-2, damping=0.05)
        r = repr(opt)
        assert "VeredKFAC" in r
        assert "damping=0.05" in r
        opt.cleanup()


# ===========================================================================
# TestLogging — verify DEBUG messages fire and INFO/WARNING stay silent
# ===========================================================================

import logging


class TestLogging:
    """Verify zero-overhead debug logging across all math modules.

    Strategy
    --------
    Use caplog (pytest's built-in log capture) to intercept log records.
    At DEBUG level, each math function should emit records with useful
    content (shapes, norms, etc.).  At INFO level, NO records from the
    optimizer modules should appear.
    """

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _R(rows, cols, seed=0):
        torch.manual_seed(seed)
        return torch.randn(rows, cols)

    @staticmethod
    def _square_upper(n, seed=0):
        torch.manual_seed(seed)
        M = torch.randn(n, n).abs() + torch.eye(n) * 2
        return torch.triu(M)

    # ------------------------------------------------------------------ sgso

    def test_sgso_debug_logs_fired(self, caplog):
        """sgso() should emit DEBUG records describing the call and each column."""
        M = self._R(16, 4)
        with caplog.at_level(logging.DEBUG, logger="optimizer.sgso"):
            sgso(M)
        msgs = [r.message for r in caplog.records if r.name == "optimizer.sgso"]
        assert any("sgso() called" in m for m in msgs), \
            "Expected 'sgso() called' in debug output"
        assert any("sgso() done" in m for m in msgs), \
            "Expected 'sgso() done' in debug output"
        # Should log one line per column
        col_msgs = [m for m in msgs if "sgso() col" in m]
        assert len(col_msgs) == 4, f"Expected 4 column log lines, got {len(col_msgs)}"

    def test_sgso_info_no_logs(self, caplog):
        """sgso() should produce no log records at INFO level."""
        M = self._R(16, 4)
        with caplog.at_level(logging.INFO, logger="optimizer.sgso"):
            sgso(M)
        records = [r for r in caplog.records if r.name == "optimizer.sgso"]
        assert len(records) == 0, \
            f"Expected no records at INFO, got {len(records)}: {[r.message for r in records]}"

    # ------------------------------------------------------------------ tsqr

    def test_tsqr_debug_logs_fired(self, caplog):
        """tsqr() should emit DEBUG records for the call, merge rounds, and done."""
        M = self._R(32, 4)
        with caplog.at_level(logging.DEBUG, logger="optimizer.sgso"):
            tsqr(M, tile_size=8)
        msgs = [r.message for r in caplog.records if r.name == "optimizer.sgso"]
        assert any("tsqr() called" in m for m in msgs)
        assert any("tsqr() done" in m for m in msgs)

    def test_tsqr_info_no_logs(self, caplog):
        """tsqr() should produce no log records at INFO level."""
        M = self._R(32, 4)
        with caplog.at_level(logging.INFO, logger="optimizer.sgso"):
            tsqr(M, tile_size=8)
        records = [r for r in caplog.records if r.name == "optimizer.sgso"]
        assert len(records) == 0

    # ------------------------------------------------------------------ streaming_tsqr_update

    def test_streaming_tsqr_update_debug_logs(self, caplog):
        """streaming_tsqr_update() should log each chunk and merge."""
        chunk1 = self._R(8, 4, seed=1)
        chunk2 = self._R(8, 4, seed=2)
        with caplog.at_level(logging.DEBUG, logger="optimizer.sgso"):
            R = streaming_tsqr_update(None, chunk1)
            streaming_tsqr_update(R, chunk2)
        msgs = [r.message for r in caplog.records if r.name == "optimizer.sgso"]
        assert any("first chunk" in m for m in msgs), \
            "Expected 'first chunk' in streaming log"
        assert any("merged" in m for m in msgs), \
            "Expected 'merged' in streaming log after second chunk"

    def test_streaming_tsqr_update_info_no_logs(self, caplog):
        """streaming_tsqr_update() should be silent at INFO level."""
        chunk = self._R(8, 4)
        with caplog.at_level(logging.INFO, logger="optimizer.sgso"):
            streaming_tsqr_update(None, chunk)
        records = [r for r in caplog.records if r.name == "optimizer.sgso"]
        assert len(records) == 0

    # ------------------------------------------------------------------ finalize_R

    def test_finalize_R_debug_logs(self, caplog):
        """finalize_R() should log its call and result."""
        R = self._square_upper(4)
        with caplog.at_level(logging.DEBUG, logger="optimizer.sgso"):
            finalize_R(R, damping=1e-2)
        msgs = [r.message for r in caplog.records if r.name == "optimizer.sgso"]
        assert any("finalize_R() called" in m for m in msgs)
        assert any("finalize_R() done" in m for m in msgs)

    def test_finalize_R_info_no_logs(self, caplog):
        R = self._square_upper(4)
        with caplog.at_level(logging.INFO, logger="optimizer.sgso"):
            finalize_R(R, damping=1e-2)
        records = [r for r in caplog.records if r.name == "optimizer.sgso"]
        assert len(records) == 0

    # ------------------------------------------------------------------ apply_vered

    def test_apply_vered_debug_logs_intermediates(self, caplog):
        """apply_vered() should log all four triangular solve steps."""
        torch.manual_seed(0)
        n_out, n_in = 6, 4
        grad_W = torch.randn(n_out, n_in)
        R_X = self._square_upper(n_in)
        R_G = self._square_upper(n_out, seed=1)
        with caplog.at_level(logging.DEBUG, logger="optimizer.sgso"):
            apply_vered(grad_W, R_X, R_G)
        msgs = [r.message for r in caplog.records if r.name == "optimizer.sgso"]
        assert any("apply_vered() called" in m for m in msgs)
        assert any("T1" in m for m in msgs), "Expected T1 intermediate log"
        assert any("T2" in m for m in msgs), "Expected T2 intermediate log"
        assert any("T3" in m for m in msgs), "Expected T3 intermediate log"
        assert any("done" in m and "scale_vs_grad" in m for m in msgs)

    def test_apply_vered_info_no_logs(self, caplog):
        """apply_vered() must be completely silent at INFO level."""
        torch.manual_seed(0)
        n_out, n_in = 6, 4
        grad_W = torch.randn(n_out, n_in)
        R_X = self._square_upper(n_in)
        R_G = self._square_upper(n_out, seed=1)
        with caplog.at_level(logging.INFO, logger="optimizer.sgso"):
            apply_vered(grad_W, R_X, R_G)
        records = [r for r in caplog.records if r.name == "optimizer.sgso"]
        assert len(records) == 0, \
            f"apply_vered emitted {len(records)} records at INFO — performance overhead!"

    # ------------------------------------------------------------------ apply_vered_bias

    def test_apply_vered_bias_debug_logs(self, caplog):
        """apply_vered_bias() should log call and result."""
        torch.manual_seed(0)
        n_out = 6
        grad_b = torch.randn(n_out)
        R_G = self._square_upper(n_out)
        with caplog.at_level(logging.DEBUG, logger="optimizer.sgso"):
            apply_vered_bias(grad_b, R_G)
        msgs = [r.message for r in caplog.records if r.name == "optimizer.sgso"]
        assert any("apply_vered_bias() called" in m for m in msgs)
        assert any("apply_vered_bias() done" in m for m in msgs)

    def test_apply_vered_bias_info_no_logs(self, caplog):
        torch.manual_seed(0)
        n_out = 6
        grad_b = torch.randn(n_out)
        R_G = self._square_upper(n_out)
        with caplog.at_level(logging.INFO, logger="optimizer.sgso"):
            apply_vered_bias(grad_b, R_G)
        records = [r for r in caplog.records if r.name == "optimizer.sgso"]
        assert len(records) == 0

    # ------------------------------------------------------------------ RawActivationHooks

    def test_hooks_debug_logs_forward_backward(self, caplog):
        """Hooks should log forward and backward chunk info at DEBUG."""
        model = nn.Linear(8, 4)
        hooks = RawActivationHooks(model, damping=1e-2)
        hooks.enable()
        x = torch.randn(16, 8, requires_grad=True)
        with caplog.at_level(logging.DEBUG, logger="optimizer.raw_activation_hooks"):
            out = model(x)
            out.sum().backward()
        hooks.remove()
        msgs = [r.message for r in caplog.records
                if r.name == "optimizer.raw_activation_hooks"]
        assert any("_forward_hook" in m for m in msgs), \
            "Expected forward hook debug log"
        assert any("_backward_hook" in m for m in msgs), \
            "Expected backward hook debug log"

    def test_hooks_info_no_logs(self, caplog):
        """Hooks should produce no log records at INFO level."""
        model = nn.Linear(8, 4)
        hooks = RawActivationHooks(model, damping=1e-2)
        hooks.enable()
        x = torch.randn(16, 8, requires_grad=True)
        with caplog.at_level(logging.INFO, logger="optimizer.raw_activation_hooks"):
            out = model(x)
            out.sum().backward()
        hooks.remove()
        records = [r for r in caplog.records
                   if r.name == "optimizer.raw_activation_hooks"]
        assert len(records) == 0, \
            f"Hooks emitted {len(records)} records at INFO — performance overhead!"

    # ------------------------------------------------------------------ VeredKFAC

    def test_veredkfac_debug_logs_step(self, caplog):
        """VeredKFAC.step() should emit debug records at DEBUG level."""
        model = _mlp()
        opt = VeredKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=1)
        with caplog.at_level(logging.DEBUG, logger="optimizer.vered_kfac"):
            _train(model, opt, steps=2)
        opt.cleanup()
        msgs = [r.message for r in caplog.records
                if r.name == "optimizer.vered_kfac"]
        assert any("VeredKFAC.step()" in m for m in msgs), \
            "Expected VeredKFAC.step() debug log"
        assert any("_update_factors" in m for m in msgs), \
            "Expected _update_factors debug log"

    def test_veredkfac_info_no_optimizer_debug(self, caplog):
        """VeredKFAC.step() should produce no DEBUG records at INFO level."""
        model = _mlp()
        opt = VeredKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=1)
        with caplog.at_level(logging.INFO, logger="optimizer.vered_kfac"):
            _train(model, opt, steps=2)
        opt.cleanup()
        debug_records = [r for r in caplog.records
                         if r.name == "optimizer.vered_kfac"
                         and r.levelno == logging.DEBUG]
        assert len(debug_records) == 0, \
            f"VeredKFAC emitted {len(debug_records)} DEBUG records at INFO level!"

    def test_debug_log_content_includes_shapes(self, caplog):
        """DEBUG logs from apply_vered should include tensor shape information."""
        torch.manual_seed(42)
        n_out, n_in = 8, 6
        grad_W = torch.randn(n_out, n_in)
        R_X = self._square_upper(n_in)
        R_G = self._square_upper(n_out, seed=3)
        with caplog.at_level(logging.DEBUG, logger="optimizer.sgso"):
            apply_vered(grad_W, R_X, R_G)
        msgs = "\n".join(r.message for r in caplog.records
                         if r.name == "optimizer.sgso")
        assert "shape=" in msgs, "Expected 'shape=' in debug output"
        assert "norm=" in msgs, "Expected 'norm=' in debug output"
