# Vered QR GPU Optimization Plan (with Noise Verification)

Goal: take VeredKFAC from ~794 ms/step to ~250-350 ms/step (a 2.5-3× speedup) without changing the numerical output. The investment buys back ~50-65% wall time on every subsequent Vered experiment, which more than pays for itself after 2-3 grid runs.

The plan builds on Tasks 3 (CUDA Graphs) and 4 (Batched QR) from `PERFORMANCE_IMPROVEMENT_TASKS.md`. Task 4 is the primary path because it's more robust to dynamic shapes; Task 3 is a fallback if Task 4's batching constraints become unworkable.

## What's actually slow

Empirically: VeredKFAC is *launch-bound*, not compute-bound. Per training step it issues ~128 cuSOLVER QR launches (32 layers × 2 hooks × 2 ops). Each launch has 10-50 µs of fixed driver+cuSOLVER overhead, so 128 launches × 30 µs ≈ 4 ms of pure overhead per step before any actual math runs. The actual QR compute on the GPU finishes in microseconds.

The fix: reduce 128 launches to ~4 by batching. cuSOLVER's `gels`/`geqrf` family supports batched mode for tensors of shape `(batch, p, n)` where `batch` is the number of independent QR problems. PyTorch exposes this via `torch.linalg.qr` when given a 3D input.

## Phase 1: Build the verification harness FIRST (before any optimization)

The order matters: write the verification before changing the implementation. That way the moment you have a faster version, you can immediately check whether it produces the same output.

**Target file**: `tests/test_vered_qr_equivalence.py`

**What it does**: runs the baseline Vered implementation and a candidate optimized version on identical inputs (same seeds, same activations, same gradients), then compares:

1. **R-factor equivalence per layer.** After each step, snapshot every layer's R-factor. Compare to baseline. Tolerance: max-relative-error < 1e-5 in fp32, < 1e-12 in fp64. R-factors should be bit-identical up to QR sign ambiguity (the sign of each row of R can flip if the corresponding Q column flips). Account for this by comparing `|R|` element-wise instead of `R` directly, or by canonicalizing the sign convention.

2. **Natural-gradient direction equivalence.** After applying the K-FAC update at each step, compare the resulting `ΔW` per layer. Tolerance: cosine-similarity between baseline and optimized `ΔW` should be > 0.9999 in fp32. A drop below that indicates real numerical divergence, not just sign flips.

3. **End-to-end loss equivalence.** Run 100 steps with both implementations, compare the loss curve. Differences should stay within ~0.1% — anything bigger is a real change in dynamics.

4. **Final-ppl equivalence at full training.** Run a 1000-step micro-grid (10 cells × ~5 min each) on both implementations. The final-ppl distributions should overlap. If the optimized version produces systematically different final ppl, we've changed the algorithm.

**Acceptance criterion**: all four checks pass before declaring the optimization safe to use for grid searches.

This is ~150-200 lines of Python and runs in 30-60 minutes on the user's hardware. Build it before touching the optimizer code.

## Phase 2: Implement Task 4 (Batched QR)

**Strategy**: change `RawActivationHooks` to bucket layers by `(p, n)` dimensions, accumulate per-bucket activation chunks, and call `torch.linalg.qr` once per bucket per backward pass instead of once per layer.

**Key code changes**:

`optimizer/raw_activation_hooks.py`:
```python
# Before:
for layer in self.layers:
    R_new = self._streaming_qr_update(layer.activations_chunk, layer.R)

# After (sketch):
buckets = self._group_layers_by_dim()  # dict: (p, n) -> list[layer_idx]
for (p, n), layer_indices in buckets.items():
    # Stack chunks: (n_layers_in_bucket, p, n)
    stacked = torch.stack([self.layers[i].activations_chunk for i in layer_indices], dim=0)
    # Single batched QR call
    Q_batch, R_batch = torch.linalg.qr(stacked, mode='reduced')
    # Distribute results back per layer
    for k, idx in enumerate(layer_indices):
        self.layers[idx].R_new = R_batch[k]
```

`optimizer/sgso.py`:
```python
# streaming_tsqr_update is called once per layer.  Need a batched analog:
def streaming_tsqr_update_batched(R_old_batch, R_new_batch, gamma):
    # R_old_batch, R_new_batch: (n_layers, n, n)  upper-triangular
    # Stack and re-QR per layer in the batch
    stacked = torch.cat([
        torch.sqrt(gamma) * R_old_batch,
        torch.sqrt(1 - gamma) * R_new_batch,
    ], dim=1)  # (n_layers, 2n, n)
    _, R_combined = torch.linalg.qr(stacked, mode='reduced')
    return R_combined  # (n_layers, n, n)
```

**Buckets for SmallGPT (4 layers, d=256, d_ff=1024)**:
- 8 attention projection layers at `(p, n) = (S × B, 256)` → 1 bucket of 8 layers
- 8 attention output projections, same shape → bucket continues
- 4 FFN-input layers at `(p, n) = (S × B, 1024)` → 1 bucket of 4
- 4 FFN-output layers at `(p, n) = (S × B, 1024)` → continues bucket

Probably 2-3 effective buckets, each containing 4-12 layers. Net launch reduction: 128 → ~6.

**Estimated effort**: 2-3 days of focused work.

**Estimated speedup**: 2.5-3× per-step. Vered's 794 ms/step → ~270-320 ms/step. Brings it close to Classic's per-step time.

## Phase 3: Verify equivalence

Run the verification harness from Phase 1 against the new batched implementation:

1. **R-factor equivalence**: should pass at fp32 tolerance 1e-5. If it fails, the most likely cause is numerical instability in the batched stack (the `[√γ·R_old; √(1-γ)·R_new]` form can have higher condition number than per-layer counterparts). Diagnose with the per-layer breakdown printed by the harness.

2. **Natural-gradient direction equivalence**: cosine-similarity > 0.9999 in fp32 should hold trivially if R-factors are equivalent. If R passes but `ΔW` fails, the discrepancy is in the apply phase (`apply_vered`), not the QR — investigate `optimizer/sgso.py:apply_vered`.

3. **Loss curve equivalence over 100 steps**: should match within 0.1%. If they diverge by more, the batched version has a real numerical artifact. Most likely cause: floating-point non-associativity in the stacked QR — the same algorithm in batched form sums in a different order than in per-layer form, accumulating rounding errors differently.

4. **Final-ppl equivalence at 5000 steps**: run one full SmallGPT training with each implementation at the current best config (γ=0.9, mom=0.3, λ=1e-4, clip=300). Compare final ppl. Acceptable: within ~5 ppl. If they differ by 10+ ppl, do not use the batched version for further grids without further investigation.

## Phase 4: Performance benchmark

Once equivalence is verified:

1. **Per-step wall-time measurement**: 50 forward+backward+step iterations, average. Compare baseline vs optimized.
2. **GPU utilization profiling**: use `nvprof` or `nsys` to confirm launch overhead has dropped. Visually: the timeline should show ~6 large QR ops per step instead of 128 small ones.
3. **End-to-end training-time delta**: run one 1000-step training with each, divide elapsed time. Should reflect the per-step speedup minus a small fixed overhead.

If the per-step speedup is < 2×, the batched approach isn't yielding enough — fall back to Phase 5.

## Phase 5: Fallback path — Task 3 (CUDA Graphs)

If batched QR doesn't deliver, capture the entire per-step QR sequence as a CUDA Graph and replay it as a single dispatch. This requires fixed input shapes throughout training, which means:
- Disable random subsampling (or fix the seed per step so shape is deterministic)
- Pre-pad activations to a fixed size if minibatch size varies

CUDA Graphs typically deliver 5-10× launch-overhead reduction but are fragile to shape changes.

**Estimated effort**: 3-5 days of work.

**Estimated speedup**: 3-5× per-step.

The risk profile is higher (fragile to shape changes) but the upside is also higher. Use only if Task 4 disappoints.

## Phase 6: Lock in and rerun the comparison

Once the optimization is verified equivalent and verified fast:

1. **Re-run the current Vered tuned config** (γ=0.9, mom=0.3, λ=1e-4, clip=300) at full 5000 steps. Compare final ppl to the existing 618 number. Should match within ~5 ppl.
2. **Re-run one Classic comparison cell** as a sanity check (Classic's per-step time shouldn't change since the optimization only touches Vered).
3. **Then** the planned grid extensions (Vered mom-axis re-sweep, multi-arch tests, etc.) become 2.5-3× cheaper. Every overnight run buys 2-3× more data.

## Total budget

| Phase | Duration | Notes |
|---|---|---|
| 1: Verification harness | 1 day | Write before optimizing |
| 2: Batched QR implementation | 2-3 days | Core work |
| 3: Verification | 1 day | Run harness + interpret |
| 4: Performance benchmark | 0.5 day | Measure speedup |
| 5: Fallback (only if needed) | 3-5 days | CUDA Graphs |
| 6: Lock-in re-run | 1 day | Confirm equivalence on full training |
| **Total** | **~1 week** | Without fallback |

After this, every Vered grid is ~2.5-3× cheaper. The 14-hour Vered grid becomes ~5 hours; 24-hour grids become 8-9 hours. The entire remaining test plan (mom re-sweep, multi-seed, multi-arch with Vered) goes from ~50 hours to ~17-20 hours of compute.

## Risks worth flagging

1. **Batched QR may have different numerical conditioning than per-layer QR.** PyTorch's QR routine is internally LAPACK-based and stable, but stacking matrices changes the effective conditioning. Most likely the verification will catch this; less likely is that the difference is small in fp32 but accumulates over 5000 steps. Mitigation: run verification at multiple training depths (100, 500, 5000 steps), not just at one point.

2. **The 2.5-3× estimate is mine, not measured.** The plan's original 3-5× was unmeasured too. Real number could be 2× (less impressive but still very useful) or 4× (bigger payoff). Won't know without prototyping.

3. **The optimization might affect the empirical comparison vs Classic.** Classic also uses cuSOLVER under the hood; if Classic gains a similar speedup from any incidental kernel-fusion that Phase 2 enables (it shouldn't, but verify), our 270 ms baseline for Classic could shift, changing relative comparison.

4. **Code complexity.** Bucketing logic adds branches to a hot path. Worth keeping a `--use-baseline-qr` flag that runs the un-batched path for debugging.

## Recommended immediate next step

Write the verification harness (`tests/test_vered_qr_equivalence.py`) first. ~150 lines. It tests the *current* implementation against itself (a no-op check), confirming the harness runs and the equivalence checks behave correctly. That gives a safety net before any optimizer code changes. After that, Task 4 implementation can proceed with confidence that any regression will be caught.
