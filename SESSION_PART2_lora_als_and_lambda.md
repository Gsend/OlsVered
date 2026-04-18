# Session Part 2 — retrain_lora_als Rewrite + BCD Lambda Scaling

## Context
Project: OlsSMLayerRetrainer in /optimizer/layer_retrainer.py
Benchmark: /benchmark/retrainer_benchmark.py
Continuing from Part 1. Known working results: OLS N=1 = 84.98%, ALS-LoRA N=1 = 85.21%.

---

## Problem 1: retrain_lora_als Was Producing ~49% Accuracy

### Root cause
The original `retrain_lora_als` had its own accumulation loop that diverged from
`_single_layer_retrain` in subtle ways, causing catastrophic weight updates (‖ΔW‖≈147).

### Fix: Complete rewrite — delegate to retrain(), then SVD compress ΔW
```python
def retrain_lora_als(self, dataloader, target_fn=None):
    # Step 1: snapshot weights before
    W_before = [l.weight.data.clone() for l in self._retrained_layers]
    b_before = [l.bias.data.clone() if l.bias is not None else None ...]

    # Step 2: run standard OLS/BCD via retrain() — the proven code path
    #         lora_rank=0 trick: suppress _fit_lora_residual during delegation
    saved_lora_rank = self.lora_rank
    self.lora_rank = 0
    try:
        history = self.retrain(dataloader, target_fn)
    finally:
        self.lora_rank = saved_lora_rank

    # Step 3: extract ΔW = W_after - W_before, restore original weights
    for i, layer in enumerate(self._retrained_layers):
        dW = (W_after - W_before[i]).cpu().numpy()
        layer.weight.data.copy_(W_before[i])   # restore

        # Step 4: truncated SVD of ΔW → rank-r LoRA factors B, A
        U, S, Vt = np.linalg.svd(dW, full_matrices=False)
        sqS = np.sqrt(S[:r_eff])
        B = U[:, :r_eff] * sqS          # (d_out, r)
        A = Vt[:r_eff, :] * sqS[:, None] # (r, d_in)

        # Step 5: apply W += B@A, apply full bias delta separately
        layer.weight.data.add_(B_t @ A_t)
        if bias: layer.bias.data.add_(b_after - b_before[i])
```

### Key design decisions
- **lora_rank=0 trick**: zeroing `self.lora_rank` before calling `retrain()` prevents
  `_single_layer_retrain` and BCD from calling `_fit_lora_residual` (the random-init ALS stage).
  Restored in `finally` block.
- **residual_mode = (n_layers == 1)**: matches run_ols convention.
- **SVD energy capture**: printed as `rank-r capture=XX.X%` for diagnostics.

---

## Problem 2: BCD Divergence for N≥2 Layers

### Observed output (original, λ=1e-4, residual_mode=True for all N)
```
[BCD sweep 1/5]  max|ΔW| = 4.80e+02  (L0:..  L1:4.80e+02)
[BCD sweep 2/5]  max|ΔW| = 2.95e+02  (L0:..  L1:2.95e+02)
[BCD sweep 3/5]  max|ΔW| = 1.67e+04  (L0:..  L1:1.67e+04)
→ Accuracy ~53%
```

### Root cause 1: residual_mode=True hardcoded in run_lora_als for all N
`residual_mode=True` with N>1 BCD causes oscillation because each layer solves for ΔW
independently, but the residuals are computed from the same stale activations.
**Fix**: `residual_mode = (n_layers == 1)` in both `run_ols` and `run_lora_als`.

### Root cause 2: λ=1e-4 too small for N>1 BCD
With λ=1e-4 and a near-rank-deficient XtX (due to Tanh pooler saturation after L0 update),
the near-zero eigenvalue (σ_min ≈ 6e-4, estimated from empirical data) is not damped:
- λ=1e-4 → L1 blows to 480
- λ=1e-3 (10× scaling) → L1=207 sweep 1, then 44, then 809 (still diverged)
- λ=0.03 (300× scaling) → L1=277 sweep 1 (improved but still diverged at sweep 3)

### Fix attempted: lambda scaling `300^(N-1)` capped at 1.0
```python
lambda_eff = min(lambda_reg * (300.0 ** max(0, n_layers - 1)), 1.0)
# N=1 → 1e-4 (unchanged)
# N=2 → 0.03
# N=4 → 1.0 (capped)
```
Applied in both `run_ols` and `run_lora_als` in retrainer_benchmark.py.

**Result**: Lambda scaling alone was insufficient — L1 still diverged at sweep 2-3
because the root cause was not purely conditioning but BCD oscillation dynamics.
(See Part 3 for the real fix.)

---

## Files Modified in Part 2
- `/optimizer/layer_retrainer.py`: complete `retrain_lora_als()` rewrite
- `/benchmark/retrainer_benchmark.py`:
  - `run_lora_als`: added `residual_mode = (n_layers == 1)`, lambda scaling
  - `run_ols`: added `residual_mode = (actual_n == 1)`, lambda scaling
