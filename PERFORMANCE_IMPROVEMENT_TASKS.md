# Performance Improvement Tasks

*Created: 2026-04-30. Last updated: 2026-05-02.*

Backlog of optimisation work for the K-FAC variants. Roughly ordered by
estimated effort × impact. None of these change numerical behaviour;
they're all pure speed wins. Verify with the stability benchmark by
checking that final perplexity is unchanged within noise after each fix.

## Tasks at a glance

| # | Task | Variant | Est. speedup | Effort |
|---|---|---|---|---|
| 1 | Switch OlsSMKFAC apply path to EVD-everywhere | OlsSMKFAC | 1.5-2x | trivial (one line) |
| 2 | Use randomised EVD on small layers too | OlsSMKFAC | 1.2-1.5x more | small |
| 3 | CUDA Graphs for VeredKFAC TSQR | VeredKFAC | 3-5x | medium |
| 4 | Batched QR across layers | VeredKFAC | 3-5x (alternative to #3) | medium-large |
| 5 | Batched apply across layers | All variants | 1.3-1.5x | large |
| 6 | Custom fused tensor-core triangular solve | OlsSMKFAC, VeredKFAC | 2x | research project |

---

## Task 1: Switch OlsSMKFAC to EVD-everywhere apply path

### Problem

OlsSMKFAC's wall-time disadvantage vs ClassicKFAC is the
`cholesky_solve` (TRSM) apply step — TRSM doesn't use Tensor Cores
and runs ~5x slower per FLOP than equivalent GEMM. See
`KFAC_VARIANTS_MATH.md` section 8 for the full analysis.

Empirical: `gpu_benchmark.py` historical and stability benchmark
output show OlsSMKFAC at ~245 ms/step vs ClassicKFAC at ~225 ms/step
on SmallGPT, even with `decomp_update_freq=20` matched.

### Insight

OlsSMKFAC already has TWO apply paths in `_update_inverses()`:

```python
if self._use_lu(n_a) and self._use_lu(n_g):
    # Cholesky-solve path - 2 TRSM per apply (no tensor cores)
else:
    # EVD path - 4 GEMM per apply (tensor cores!)
```

The Cholesky path fires for layers ≤ `lu_max_dim` (default 4096), so
on SmallGPT every layer hits it. **Forcing the EVD path everywhere
gets tensor cores back without losing the κ(A)·ε numerical guarantee**
(EVD also stores Q + λ separately and never forms an explicit inverse).

### The fix

One line change in `make_optimizers()`:

```python
kfac_opt = OlsSMKFAC(
    ...,
    lu_max_dim=0,   # force EVD path on all layer sizes
)
```

### Wall-time arithmetic at n=1024 on RTX 3080

| Path | Decomp / 20 steps | Apply / step | Apply rate | Per-step total |
|---|---|---|---|---|
| Cholesky-solve (current) | 0.018 GFLOPs | 0.5 GFLOPs | TRSM ~12 TFLOPS | **~42 ms** |
| EVD (proposed) | 0.5 GFLOPs | 1.0 GFLOPs | GEMM ~80 TFLOPS | **~13 ms** |

Net per-step speedup: ~3x. Total OlsSMKFAC speedup ~1.5-2x once you
account for forward/backward dominating the rest of the step.

### Numerical guarantee

EVD path stores `(Q, 1/(λ+δ))` and applies as
`Q [Q^T grad Q · diag] Q^T`. No explicit inverse formed, so the same
κ(A)·ε forward error bound as the Cholesky-solve path. The math doc's
stability hierarchy κ(X)⁴ > κ(X)² ≥ κ(X) is preserved.

### Success criteria

- `OlsSMKFAC median_step_ms` drops from ~245 to ~150-180 ms
- Phase 2 final perplexity unchanged within ±2%
- No new divergence at the same LRs Phase 1 declared stable

### Files

- `benchmark/stability_benchmark.py` — `make_optimizers()` (one line)
- `optimizer/olssm_kfac.py` — verify EVD path supports the same dtypes/devices
- `KFAC_VARIANTS_MATH.md` — update once empirically confirmed

---

## Task 2: Force randomised EVD on small layers too

### Problem

Even with task #1, the EVD decomposition is O(n³) and fires every
`decomp_update_freq` steps. For n=1024 that's ~10 GFLOPs per layer per
decomp window — non-trivial.

### Insight

`OlsSMKFAC` has an `adaptive=True` mode that uses the Halko-Martinsson-
Tropp randomised EVD with rank budget k. Cost drops to O(k·n²) instead of
O(n³). Currently gated behind `adaptive_min_n=4096` so it only fires on
large layers; small layers get full EVD.

### The fix

Lower the gate so randomised EVD applies to all layers:

```python
kfac_opt = OlsSMKFAC(
    ...,
    adaptive=True,
    adaptive_min_n=64,           # was 4096
    adaptive_rank_budget=128,    # k = 128 covers most curvature
    lu_max_dim=0,                # from task #1
)
```

Decomposition cost at n=1024, k=128: ~16x cheaper than full EVD.

### Risk

- Rank=128 may miss some curvature directions on layers where the
  effective rank is much higher. Could degrade convergence.
- The rank budget should be tuned empirically per task. Try k ∈ {64, 128,
  256}.

### Success criteria

- Decomp time reduced; per-step apply time unchanged
- Phase 2 final perplexity within ±5% of full-EVD baseline
- Total wall time another 10-20% lower than task #1 alone

---

## Task 3: CUDA Graphs for VeredKFAC TSQR (originally for tomorrow)

### Problem

VeredKFAC keeps the GPU largely idle while the CPU is busy. After
fixing the sample-budget asymmetry (`_SEQ_SUBSAMPLE = 512` in
`RawActivationHooks`), per-step time improved but GPU still hits maybe
30% utilisation.

### Root cause

VeredKFAC fires forward and backward hooks on every K-FAC-covered
Linear layer, every step. Inside each hook, `streaming_tsqr_update`
runs **two** `torch.linalg.qr` calls — leaf QR on the new chunk + merge
QR on `[running_R; R_new]`. For SmallGPT:

```
32 K-FAC-covered Linear layers × 2 hooks (fwd + bwd) × 2 QR ops/hook
= 128 cuSOLVER QR launches per training step
```

cuSOLVER QR has high per-launch overhead — at our matrix sizes
(`(512, 1024)` and `(2048, 1024)`), the actual compute finishes in
microseconds but launch latency dominates:

- CPU: busy enqueueing 128 launches per step
- GPU: bursts through each QR faster than the next launch arrives → idle
  between bursts

Subsampling reduced the FLOP count but not the launch count.

### The fix

Wrap the per-step QR sequence in a `torch.cuda.CUDAGraph`. The first
training step captures the full sequence of cuSOLVER launches; subsequent
steps replay the graph as one launch.

- Pros: 5-10x speedup for launch-bound kernels; no algorithm change.
- Cons: CUDAGraph capture is fragile — requires fixed input shapes and
  static memory pools. Data-dependent control flow (e.g. the per-batch
  `torch.randperm` for subsampling) breaks the graph.
- Workarounds: pre-generated index pools (sample once, reuse), or
  fixed-shape padding.
- Risk: every model architecture change requires a graph re-capture.

### Suggested investigation order

1. **Profile first.** Use `torch.profiler` (or nsys) on a Phase 2 probe
   of VeredKFAC. Confirm the launch-bound hypothesis: cuSOLVER QR total
   compute time should be much smaller than total wall time, and CPU time
   in `cuLaunchKernel` should be substantial.
2. **Try CUDA Graphs.** Wrap the training step in capture/replay. To
   simplify, temporarily disable the `torch.randperm` subsampling (or
   replace with a pre-generated rotating index buffer). If the speedup is
   real (3-5x), commit to making the subsample-friendly version work.

### Success criteria

- VeredKFAC `median_step_ms` within 20% of OlsSMKFAC's
- `nvidia-smi dmon` shows >70% SM utilisation during a Phase 2 probe
- Phase 2 final perplexity unchanged within noise

### Files

- `optimizer/raw_activation_hooks.py` — per-layer QR fires here
- `optimizer/sgso.py` — `streaming_tsqr_update`
- `optimizer/vered_kfac.py` — orchestration; per-step capture would go here

---

## Task 4: Batched QR (alternative to CUDA Graphs)

### Approach

Stack per-layer chunks into a single tensor and call `torch.linalg.qr`
once on a batched `(n_layers, p, n)` input. cuSOLVER's batched API uses
a single dispatch for all layers.

- Pros: launch count drops from 128 to ~4 per step; no static-shape
  constraint.
- Cons: requires layers to have **the same dimensions** to batch — for
  SmallGPT we have layers of dim 256 (Q/K/V/Out) and 1024 (FFN). Need to
  bucket by dim and do one batched QR per bucket.

### Implementation sketch

1. In `RawActivationHooks.__init__`, group `_linear_layers` by their
   `in_features` / `out_features` dimension. Build per-bucket index maps.
2. Hooks no longer call `streaming_tsqr_update` immediately. Instead they
   append `(module, chunk)` to a per-bucket queue.
3. After `loss.backward()` returns, iterate buckets and call
   `streaming_tsqr_update_batched(running_Rs, chunks)` once per bucket.
4. `streaming_tsqr_update_batched` stacks chunks along a new batch dim,
   calls `torch.linalg.qr` once, then unstacks the results.

Choose this over CUDA Graphs if the per-step capture proves too brittle.

---

## Task 5: Batched apply across layers (all variants)

### Problem

Each variant's apply step runs a sequence of matmuls per layer. For
SmallGPT with 32 K-FAC-covered layers, that's ~64 individual GEMM (or
TRSM) launches per step. Even though each is fast, dispatch adds up.

### The fix

Collect `(grad, decomp_state)` for all layers with matching shapes and
do batched matmul via `torch.bmm`. Same trick as task #4 but applied to
the apply step instead of the decomposition.

### Practical issues

- Shape diversity is higher than the QR case (need to match BOTH input
  and output dims). On SmallGPT: maybe 4-6 unique shapes total.
- Each variant has a different apply structure (Classic: 2 GEMMs, OlsSM
  EVD: 4 GEMMs, Vered: 4 TRSMs). Implementation differs per variant.
- Estimated 30-50% speedup if done well.

Lower priority than #1-3 because it's complex per-variant work; should
follow naturally from solving #3 (the CUDA Graph approach also batches
the launches).

---

## Task 6: Custom fused tensor-core triangular solve

### What

A custom CUDA kernel that does `Q^T @ X @ Q · outer_scale` (the EVD
apply) in one launch instead of three, using tensor cores throughout.
Or equivalently for VeredKFAC, a fused `R^-T @ X @ R^-1` kernel.

### Why not yet

- Research project — requires CUDA C++ and intimate cuBLAS knowledge
- Estimated 2x speedup
- Tasks #1-5 likely close most of the gap with much less effort
- NVIDIA may eventually ship a structured-sparse tensor-core variant
  for triangular solve in Hopper/Blackwell, making this obsolete

Park indefinitely. Revisit if all other paths are exhausted and the
remaining gap is worth a multi-month investment.

---

## Out of scope (don't do these)

- **Replacing TSQR with Cholesky of XᵀX in VeredKFAC** — defeats the
  κ(X)¹ stability guarantee that's the whole point of VeredKFAC.
- **Replacing EVD with `torch.linalg.inv` in OlsSMKFAC** — defeats the
  κ(X)² stability advantage; you'd just become Classic.
- **Multi-GPU / distributed** — different problem family; revisit only
  after the single-GPU performance is acceptable.
- **Mixed-precision (FP16) for any decomposition step** — TRSM and QR
  are numerically sensitive in low precision; defeats the entire
  numerical-stability story.

---

## Verification protocol

After any change in this file:

1. Run `bash run_stability.ps1` (or manual python invocation).
2. Compare to baseline (pre-change) `stability_phase2_runs.json`:
   - `final_ppl` for each variant must be within ±2% of baseline
   - `median_step_ms` should be lower (that's the point)
   - `condition_summary` should be unchanged within noise
3. Update `KFAC_VARIANTS_MATH.md` cost table with new measurements
4. Mark the task as **DONE** in this file with the measured speedup
