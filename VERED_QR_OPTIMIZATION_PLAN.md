# Vered QR GPU Optimization Plan

Goal: take VeredKFAC from ~794 ms/step to ~250-350 ms/step (2.5-3× speedup) without changing numerical output. With the bf16 result in hand (see motivation below), closing the wall-time gap is now the single biggest blocker between this codebase and a publishable paper.

## Why this matters NOW (bf16 result, 2026-05-30)

Recent benchmark `benchmark/kfac_bf16_compare.py`:

| Cell        | fp32 ppl | bf16 ppl | degradation |
|-------------|---------:|---------:|------------:|
| Classic     |      922 |     1997 |       +1075 |
| Vered       |      921 |      940 |         +19 |
| Vered+WGSO  |     ~898 |   (TBD)  |           — |

The κ¹ vs κ⁴ stability prediction finally manifests at bf16 ε ≈ 4e-3. This is the headline finding that justifies a paper. The blocker: Vered currently runs at 13.9 min / 1000 steps vs Classic's 3.8 min (~3.6× slower). Reviewer-1's first comment will be "just run Classic 3.6× longer in fp32." Closing the wall-time gap turns that objection into a non-issue.

After this optimization, Vered's per-step time should drop to ~270-320 ms, near Classic's. The bf16 comparison then becomes:
- Same per-step wall time
- Same fp32 final ppl
- Vered: graceful bf16 degradation; Classic: collapse

That is a publishable claim. Today's "Vered survives bf16 but is 3.6× slower" is not.

## What's actually slow

Empirically: VeredKFAC is **launch-bound, not compute-bound**. Per training step it issues ~128 cuSOLVER QR launches (32 layers × 2 hooks × 2 ops). Each launch has 10-50 µs of fixed driver+cuSOLVER overhead, so 128 launches × 30 µs ≈ 4 ms of pure overhead per step before any actual math runs. The actual QR compute on the GPU finishes in microseconds.

The fix: reduce 128 launches to ~6 by batching. cuSOLVER's `geqrf` family supports batched mode for tensors of shape `(batch, p, n)`. PyTorch exposes this via `torch.linalg.qr` when given a 3D input.

## Phase 1: Build the verification harness FIRST (before any optimization)

The order matters: write the verification before changing the implementation.

**Target file**: `tests/test_vered_qr_equivalence.py`

**What it does**: runs the baseline Vered implementation and a candidate optimized version on identical inputs (same seeds, same activations, same gradients), then compares:

1. **R-factor equivalence per layer.** After each step, snapshot every layer's R-factor. Compare to baseline. Tolerance: max-relative-error < 1e-5 in fp32, < 1e-12 in fp64. R-factors should be bit-identical up to QR sign ambiguity (the sign of each row of R can flip if the corresponding Q column flips). Account for this by comparing `|R|` element-wise or canonicalizing the sign convention.

2. **Natural-gradient direction equivalence.** After applying the K-FAC update at each step, compare the resulting `ΔW` per layer. Tolerance: cosine-similarity between baseline and optimized `ΔW` > 0.9999 in fp32. A drop below that indicates real numerical divergence, not just sign flips.

3. **End-to-end loss equivalence.** Run 100 steps with both implementations, compare the loss curve. Differences should stay within ~0.1%.

4. **Final-ppl equivalence at full training.** Run a 1000-step micro-grid on both implementations. Final-ppl distributions should overlap within ~5 ppl.

**Acceptance criterion**: all four checks pass before declaring the optimization safe for grid searches.

This is ~150-200 lines of Python and runs in 30-60 minutes. Build it before touching the optimizer code.

## Phase 2: Implement Task 4 (Batched QR)

**Strategy**: change `RawActivationHooks` to bucket layers by `(p, n)` dimensions, accumulate per-bucket activation chunks, and call `torch.linalg.qr` once per bucket per backward pass instead of once per layer.

`optimizer/raw_activation_hooks.py`:
```python
# Before:
for layer in self.layers:
    R_new = self._streaming_qr_update(layer.activations_chunk, layer.R)

# After (sketch):
buckets = self._group_layers_by_dim()  # dict: (p, n) -> list[layer_idx]
for (p, n), layer_indices in buckets.items():
    stacked = torch.stack([self.layers[i].activations_chunk for i in layer_indices], dim=0)
    Q_batch, R_batch = torch.linalg.qr(stacked, mode='reduced')
    for k, idx in enumerate(layer_indices):
        self.layers[idx].R_new = R_batch[k]
```

`optimizer/sgso.py`:
```python
def streaming_tsqr_update_batched(R_old_batch, R_new_batch, gamma):
    # (n_layers, n, n) upper-triangular each
    stacked = torch.cat([
        torch.sqrt(gamma) * R_old_batch,
        torch.sqrt(1 - gamma) * R_new_batch,
    ], dim=1)  # (n_layers, 2n, n)
    _, R_combined = torch.linalg.qr(stacked, mode='reduced')
    return R_combined
```

**Buckets for SmallGPT (12 blocks, d_model=384, d_ff=1536)**:
- 24 attention Q/K projections at `(p, n) = (S·B, 384)` → 1 bucket
- 24 attention V/output projections, same shape → bucket continues
- 12 FFN-input layers at `(p, n) = (S·B, 1536)` → 1 bucket
- 12 FFN-output layers at `(p, n) = (S·B, 1536)` → continues bucket

~2-3 effective buckets, each containing 12-48 layers. Net launch reduction: 128 → ~6.

**Estimated effort**: 2-3 days.

**Estimated speedup**: 2.5-3× per-step. 794 ms → ~270-320 ms. Brings Vered near Classic parity.

## Phase 3: Verify equivalence

Run the harness from Phase 1 against the batched implementation:

1. **R-factor equivalence**: fp32 tolerance 1e-5. If it fails, the likely cause is numerical instability in the batched stack — the `[√γ·R_old; √(1-γ)·R_new]` form can have higher condition number than per-layer counterparts.

2. **Natural-gradient direction equivalence**: cosine > 0.9999 should hold trivially if R-factors are equivalent. If R passes but `ΔW` fails, look at `apply_vered` in `optimizer/sgso.py`.

3. **Loss curve equivalence over 100 steps**: should match within 0.1%. If they diverge by more, suspect floating-point non-associativity in the stacked QR.

4. **Final-ppl equivalence at 1000 steps**: run one full SmallGPT training with each implementation at champion (γ=0.9, mom=0.7, λ=1e-4, lr=2e-3, freq=20). Acceptable: within ~5 ppl.

## Phase 4: Performance benchmark

1. **Per-step wall-time**: 50 forward+backward+step iterations, average. Baseline vs optimized.
2. **GPU utilization profiling**: `nsys` to confirm launch overhead has dropped. Timeline should show ~6 large QR ops per step instead of 128 small ones.
3. **End-to-end 1000-step delta**: divide elapsed time. Should reflect per-step speedup minus a small fixed overhead.

If per-step speedup < 2×, fall back to Phase 5.

## Phase 5: Fallback — Task 3 (CUDA Graphs)

If batched QR doesn't deliver, capture the entire per-step QR sequence as a CUDA Graph and replay it as a single dispatch. Requires fixed input shapes throughout training:
- Disable random subsampling (or fix seed per step so shape is deterministic)
- Pre-pad activations if minibatch size varies

CUDA Graphs typically deliver 5-10× launch-overhead reduction but are fragile to shape changes.

**Estimated effort**: 3-5 days. **Estimated speedup**: 3-5× per-step.

## Phase 6: Lock in and rerun bf16 comparison

Once verified equivalent and fast:

1. **Re-run champion (γ=0.9, mom=0.7, lr=2e-3, λ=1e-4, freq=20)** at 1000 steps. Compare final ppl to existing 921. Match within ~5 ppl.
2. **Re-run `kfac_bf16_compare.py`** with the optimized Vered. The "Vered survives bf16" headline now comes with matched wall time, killing the reviewer-1 objection.
3. **Multi-seed sweep on all three bf16 cells** (n=3-5 seeds × {classic, vered, vered+wgso}). Cheap once Vered is 3× faster.
4. **Multi-size sweep** at one cell to show the gap scales: SmallGPT, MediumGPT (~50M), check whether bf16 Classic collapse persists.

These are the missing items 1, 3, 4 from the "for a paper" checklist.

## Total budget

| Phase | Duration | Notes |
|---|---|---|
| 1: Verification harness | 1 day | Write before optimizing |
| 2: Batched QR implementation | 2-3 days | Core work |
| 3: Verification | 1 day | Run harness + interpret |
| 4: Performance benchmark | 0.5 day | Measure speedup |
| 5: Fallback (only if needed) | 3-5 days | CUDA Graphs |
| 6: Lock-in + bf16 multi-seed | 2-3 days | Confirm + extend to paper-grade |
| **Total** | **~1.5-2 weeks** | Without fallback |

After this, every Vered grid is ~2.5-3× cheaper, multi-seed bf16 becomes affordable, and the paper has a defensible per-step wall-time comparison.

## Risks worth flagging

1. **Batched QR may have different numerical conditioning.** Most likely the verification catches it; less likely the difference is small in fp32 but accumulates over many steps. Mitigation: verify at multiple training depths (100, 500, 1000 steps).

2. **The 2.5-3× estimate is unmeasured.** Real number could be 2× (less impressive but still useful) or 4× (bigger payoff). Won't know without prototyping.

3. **bf16 path needs re-verification.** The bf16 monkey-patch in `kfac_bf16_compare.py` intercepts `streaming_tsqr_update` per call. After batching, the patch needs to operate at bucket granularity, not per-layer. Plan for a Phase 2.5: rebuild the bf16 quantization at the batched API surface.

4. **Code complexity.** Bucketing logic adds branches to a hot path. Keep a `--use-baseline-qr` flag that runs the un-batched path for debugging and equivalence regressions.

## Recommended immediate next step

Write the verification harness (`tests/test_vered_qr_equivalence.py`) first. ~150 lines. Tests current implementation against itself (a no-op check) confirming the harness runs and equivalence checks behave correctly. Safety net before any optimizer code changes.

After that, Phase 2 implementation can proceed with confidence that any regression will be caught — and the bf16 headline becomes paper-ready by Phase 6.
