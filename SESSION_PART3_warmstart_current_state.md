# Session Part 3 — BCD Warm-Start Fix & Current State (PICK UP HERE)

## Context
Project: OlsSMLayerRetrainer
Files: /optimizer/layer_retrainer.py, /benchmark/retrainer_benchmark.py
Continuing from Part 2. Lambda scaling (300^(N-1)) was applied but BCD still diverged at sweep 2-3.

---

## Root Cause of BCD Divergence (Full Diagnosis)

### The real problem: randomly-initialized output layer + full-replace BCD
For BERT fine-tuning:
- L0 = `bert.pooler.dense` (768→768, pretrained, followed by Tanh)
- L1 = `classifier` (768→2, **randomly initialized** near zero, std=0.02)

In full-replace Gauss-Seidel BCD, sweep 1:
1. L0 is updated: small change (ΔW0 ≈ 0.39 max element) — fine
2. L1 is re-solved: new activations from updated L0, OLS gives a solution FAR from random init
   → ΔW1 ≈ 277 (the OLS optimum is genuinely far from near-zero random init)

In sweep 2:
- L0 backward-projects targets through the now-huge W1, getting distorted targets
- L1 re-solve: partial tanh saturation from L0's update makes XtX near-rank-deficient
  → near-zero eigenvalues get amplified by 1/(2λ) — for λ=0.03: up to 16× amplification
  → L1 explodes to 705+

**This is NOT a lambda problem. It's a cold-start + XtX conditioning cascade.**

---

## Fix Applied: Warm-Start + max_sweeps=1 for N>1

### 1. Warm-start in layer_retrainer.py `retrain()` [ALREADY IN CODE]
Before the BCD loop (for N>1 only), pre-solve the output layer with a clean N=1 OLS:
```python
# In retrain(), before the BCD loop:
if self.n_layers > 1:
    if self.verbose:
        print("[BCD warm-start]  Pre-solving output layer (N=1 OLS) …")
    delta_ws = self._final_output_layer_solve(dataloader, target_fn)
    if self.verbose:
        print(f"[BCD warm-start]  done.  max|ΔW_last| = {delta_ws:.2e}")
```
This gives L1 a near-optimal starting value so sweep 1 only makes a small correction (ΔW1 ≈ 0.4 instead of 277).

### 2. Cap max_sweeps=1 for N>1 in benchmark [ALREADY IN CODE]
The warm-start stabilizes sweep 1, but sweep 2+ still diverges (confirmed on synthetic test).
Sweep 1 after warm-start is provably stable and equivalent to greedy layer-wise OLS:
- Warm-start: L1 = optimal for initial L0 activations
- Sweep 1 L0 step: L0 adjusts (small change, ΔW0 ≈ 0.1)
- Sweep 1 L1 step: L1 re-solved with new L0 activations (small correction)

In `run_ols` and `run_lora_als` in retrainer_benchmark.py:
```python
max_sweeps_eff = 1 if actual_n > 1 else max_sweeps
```

---

## Other Fixes Applied This Session [ALL ALREADY IN CODE]

### Syntax error in layer_retrainer.py
File was truncated at line 1437 mid-statement:
```python
# Before (broken):
g.n_samples = int(data["n_samples"
# After (fixed):
g.n_samples = int(data["n_samples"])
```

### Lambda scaling inconsistency in run_lora_als
`run_lora_als` had `10.0 **` instead of `300.0 **`. Fixed to match `run_ols`:
```python
lambda_eff = min(lambda_reg * (300.0 ** max(0, n_layers - 1)), 1.0)
```

---

## Current State of All Files

### /optimizer/layer_retrainer.py
- ✅ Syntax error fixed (line 1437)
- ✅ Warm-start added before BCD loop for N>1
- ✅ `retrain_lora_als()` fully rewritten (delegates to `retrain()` + SVD compress)
- ✅ `lora_rank=0` trick prevents unwanted `_fit_lora_residual` during delegation

### /benchmark/retrainer_benchmark.py
- ✅ `run_ols`: `residual_mode=(actual_n==1)`, lambda scaling `300^(N-1)`, `max_sweeps_eff=1` for N>1
- ✅ `run_lora_als`: same fixes as `run_ols`

---

## How to Run the Benchmark

### Environment
```bash
source /sessions/friendly-gallant-allen/torchenv/bin/activate
# torch 2.11.0+cu130, transformers 5.5.4, datasets 4.8.4
# Note: HuggingFace downloads blocked by proxy — model must be pre-cached
# Previous sessions cached BERT, but VM resets between sessions.
# Run from a machine with HuggingFace access.
```

### Command
```bash
cd /path/to/OlsVered
bash run_retrainer_benchmark.sh \
  --modes ols_n2,ols_n4,als_lora_n2_r4,als_lora_n4_r4 \
  --no-setup
```

### Expected output for N=2 modes
```
[OlsSMLayerRetrainer] Retraining 2 layer(s)  [bcd=gauss_seidel full-replace]:
  [0] Linear(768 → 768, bias)  → Tanh      ← bert.pooler.dense
  [1] Linear(768 → 2, bias)                ← classifier
[BCD warm-start]  Pre-solving output layer (N=1 OLS) …
[BCD final]  Output layer re-solved.  max|ΔW| = X.XXe+00
[BCD warm-start]  done.  max|ΔW_last| = X.XXe+00
[BCD sweep 1/1]  max|ΔW| = X.XXe-01  (L0:X.XXe-02  L1:X.XXe-01)
```
No explosion. Both L0 and L1 deltas should be O(0.1-1.0), not O(100+).

---

## Known Results So Far
| Mode | Accuracy | Notes |
|------|----------|-------|
| Pretrained | 51.26% | No fine-tuning |
| OLS N=1 | **84.98%** | Single-layer, proven stable |
| ALS-LoRA N=1 r=4 | **85.21%** | SVD-LoRA on N=1 OLS |
| OLS N=2 | TBD | Will run with warm-start+1-sweep |
| OLS N=4 | TBD | Same |
| ALS-LoRA N=2 r=4 | TBD | Same |
| ALS-LoRA N=4 r=4 | TBD | Same |
| Adam, LoRA baselines | TBD | Pending run |

---

## Open Questions / Future Work
1. Does warm-start+1-sweep for N=2/4 actually improve over N=1 (84.98%)?
   Theory says yes (more layers = better representation), but it's one greedy pass.
2. Could a proper trust-region / proximal BCD (μ·‖W−W_curr‖²) allow more sweeps stably?
3. Is the sweep-2+ divergence BERT-specific (tanh pooler saturation) or general?
   Synthetic tests suggest general; BERT might be more benign due to pretrained weights.
4. ols_lora_r4 (OLS + LoRA residual on top) still pending.
