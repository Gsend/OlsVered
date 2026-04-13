# OlsveredKFAC Session Context
*Saved: 2026-04-13*

---

## Project Overview
OlsveredKFAC is a K-FAC (Kronecker-Factored Approximate Curvature) optimizer using a randomized EVD backend. It is benchmarked against ClassicKFAC and Adam across four tasks: Large MLP (MNIST), CIFAR-10 MLP, BERT fine-tuning (SST-2), and SmallGPT from scratch (WikiText-2).

**Repo root:** `C:/Users/nat79/OlsVered/`
**Key files:**
- `optimizer/hooks.py` — forward/backward hooks, KFAC-Reduce subsampling
- `optimizer/olsvered_kfac.py` — main optimizer
- `optimizer/classic_kfac.py` — baseline
- `benchmark/gpu_benchmark.py` — all 4 benchmark tasks
- `run_benchmark.sh` — launcher with CLI args
- `benchmark/results/` — JSON + CSV results, PNG plots

---

## Hardware
**RTX PRO 6000 Blackwell, 96 GB VRAM, 221 GB RAM, 28 vCPU**
- Required `pip install torch --index-url https://download.pytorch.org/whl/cu128` (Blackwell sm_100 support)

---

## All Code Changes Made This Session

### 1. `optimizer/hooks.py` — KFAC-Reduce for transformers
**Problem:** BERT hooks were taking ~1500ms/step because B×seq_len rows (65,536) were used for Gram outer products.
**Fix:** Added `_SEQ_SUBSAMPLE = 512` — randomly subsample rows in forward/backward hooks for 3D tensors.
**Also added:** `max_gram_dim` parameter — skip K-FAC hooks on layers where `out_features > max_gram_dim`. Used to exclude the LM head (out=50,257) in the transformer task, whose G matrix would be 50,257×50,257 ≈ 10 GB.

```python
# In KFACHooks.__init__:
def __init__(self, model: nn.Module, max_gram_dim: int = 0):
    # max_gram_dim > 0: skip layers with out_dim > threshold (prevents OOM on LM head)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if max_gram_dim > 0:
                out_dim = module.out_features if isinstance(module, nn.Linear) else module.out_channels
                if out_dim > max_gram_dim:
                    continue
            self._linear_layers.append(module)
```

### 2. `optimizer/olsvered_kfac.py`
- Added `max_gram_dim: int = 0` parameter, passed to `KFACHooks(model, max_gram_dim=max_gram_dim)`
- Added `kfac_state_dict()` and `load_kfac_state_dict()` methods for checkpoint warm-start

### 3. `optimizer/classic_kfac.py`
- Added `max_gram_dim: int = 0` parameter, passed to `KFACHooks`

### 4. `benchmark/gpu_benchmark.py`

#### Transformer task — dual optimizer fix (CRITICAL)
OlsveredKFAC only registers `nn.Linear`/`nn.Conv2d` in param groups and only updates those in `step()`. `nn.Embedding` layers (tok_emb, pos_emb) were never updated → model stuck at ppl=9999.

**Fix:** Dual optimizer — K-FAC for Linear layers, AdamW for embeddings + LayerNorm + excluded LM head:
```python
_KFAC_MAX_DIM = 4096
kfac_covered_ids = set()
for mod in model.modules():
    if isinstance(mod, (nn.Linear, nn.Conv2d)):
        out_dim = mod.out_features if isinstance(mod, nn.Linear) else mod.out_channels
        if out_dim <= _KFAC_MAX_DIM:
            for p in mod.parameters():
                kfac_covered_ids.add(id(p))
other_params = [p for p in model.parameters() if id(p) not in kfac_covered_ids]
emb_opt = torch.optim.AdamW(other_params, lr=cfg['lr'], weight_decay=0.01)
```
Training loop uses `model.zero_grad()` (not `opt.zero_grad()`) to clear ALL params, then calls both `opt.step()` and `emb_opt.step()`.

#### Transformer task — LR optimization CLI args (new)
```
--lr-ols-transformer LR    Override OlsveredKFAC lr (default 3e-3)
--lr-cls-transformer LR    Override ClassicKFAC lr (default 3e-3)
--lr-sweep-transformer     Auto-sweep [1e-3, 3e-3, 5e-3, 8e-3] for each K-FAC optimizer
```

#### BERT task changes
- Configs reordered: Adam → OlsveredKFAC → ClassicKFAC
- OlsveredKFAC BERT params: `lr=3e-3, damping=3e-3, factor_update_freq=20, inv_update_freq=5, adaptive=True, adaptive_min_n=256, adaptive_rank_budget=128, momentum=0.0, grad_clip=1.0, gamma=0.95`
- Threshold-triggered LR decay at `val_acc >= 0.91`: replaces scheduler with `CosineAnnealingLR(T_max=remaining_steps)`, drops damping to `2e-4`
- Checkpoint save/load: files `bert_ckpt_{name}_{model|opt|kfac|meta}.pt`
- `max_steps_bert` default: 8000 → 5000

#### Transformer model (SmallGPT)
- 4 transformer blocks, d_model=256, n_heads=4, d_ff=1024, vocab=50,257, seq_len=128
- ~16M total params; ~787K params per block; ~12.9M in embedding table (weight-tied with LM head)
- K-FAC covers 32 param tensors (all Linear layers except LM head); AdamW covers 20 (embeddings, LayerNorm, LM head)

### 5. `run_benchmark.sh`
Added CLI args:
```
--lr-ols-transformer LR
--lr-cls-transformer LR
--lr-sweep-transformer
--steps-transformer N   (already existed)
```

---

## Benchmark Results (Current)

### Task 1: SmallGPT / WikiText-2 (from scratch)
| Optimizer | Wall | Final PPL | Opt ms |
|-----------|------|-----------|--------|
| Adam | 3.2 min | 1,079 | 0.4 |
| OlsveredKFAC | 6.3 min | **682** | 45.1 |
| ClassicKFAC | 5.2 min | 721 | 31.0 |

**K-FAC wins at equal wall time:** at 3 min, OlsveredKFAC ppl=863 vs Adam ppl=1,079.
**LR optimization pending** — `--lr-sweep-transformer` implemented but not yet run. Suspected optimal: OlsveredKFAC ~5e-3, ClassicKFAC ~4e-3.

### Task 2: BERT Fine-tuning / SST-2
| Optimizer | Wall | Final Acc | Notes |
|-----------|------|-----------|-------|
| Adam | 5.0 min | **93.0%** | Canonical recipe, peak 93.12% |
| OlsveredKFAC | ~7 min | ~91% | No result JSON (from session log) |
| ClassicKFAC | 55.4 min | 85.9% | ⚠ STALE — pre-hooks-fix run (221ms/step) |

**Adam wins on fine-tuning.** ClassicKFAC needs re-run with fixed hooks (expected ~15ms/step).
**Cold-start BERT:** `rm benchmark/results/bert_ckpt_*.pt`

### Task 3: CIFAR-10 MLP
| Optimizer | Final Acc |
|-----------|-----------|
| Adam | 57.9% |
| OlsveredKFAC | 55.4% |
| ClassicKFAC | 51.4% |

### Task 4: Large MLP (MNIST)
| Optimizer | Final Acc |
|-----------|-----------|
| Adam | **98.7%** |
| OlsveredKFAC | 97.6% |
| ClassicKFAC | 97.7% |

---

## Key Findings

1. **K-FAC shines on from-scratch training** (rough landscape) — confirmed on SmallGPT.
2. **Adam wins on fine-tuning** (smooth landscape near pre-trained weights) — confirmed on BERT.
3. **K-FAC step cost:** OlsveredKFAC 45ms vs Adam 0.4ms (100×). Break-even requires ~2-3× better loss-per-sample.
4. **Embedding layers must use AdamW** when training from scratch — K-FAC only covers Linear/Conv2d.
5. **LM head with vocab_size output must be excluded from K-FAC** (`max_gram_dim=4096`) to avoid OOM (50,257×50,257 G matrix ≈ 10 GB).

---

## Pending Tasks

1. **Run LR sweep for transformer task:**
   ```bash
   bash run_benchmark.sh --task transformer --lr-sweep-transformer
   ```

2. **Re-run BERT ClassicKFAC** with fixed hooks (delete stale checkpoint first):
   ```bash
   rm benchmark/results/bert_ckpt_classickfac_*.pt
   bash run_benchmark.sh --task bert --skip adam,olsveredkfac
   ```

3. **Re-run BERT OlsveredKFAC** clean to get a result JSON:
   ```bash
   rm benchmark/results/bert_ckpt_olsveredkfac_*.pt
   bash run_benchmark.sh --task bert --skip adam,classickfac
   ```

4. **Run transformer with optimized LRs** (after sweep):
   ```bash
   bash run_benchmark.sh --task transformer \
       --lr-ols-transformer <best_from_sweep> \
       --lr-cls-transformer <best_from_sweep>
   ```

---

## Architecture Notes

### Why K-FAC doesn't support nn.Embedding
- Embedding lookup = Linear with one-hot input → A matrix is `diag(token_frequencies)` (vocab×vocab diagonal)
- G matrix is d_model×d_model (manageable), but A would be 50,257×50,257 even as sparse diagonal
- The natural gradient reduces to frequency-weighted scaling per token row — which AdamW already approximates
- All K-FAC LM papers use AdamW for embeddings; Linear layers only for K-FAC

### SmallGPT parameter breakdown
- 4 blocks × ~787K params/block = ~3.15M transformer params
- Embedding table: 50,257 × 256 = 12.87M params (weight-tied with LM head)
- Total: ~16.05M params
- K-FAC covers: Q/K/V/Out (4×65K) + FFN fc1 (262K) + FFN fc2 (262K) = 8 Linear layers × 4 blocks = 32 tensors

### VRAM requirements
| Task | Min VRAM |
|------|----------|
| MLP/CIFAR | 4 GB |
| SmallGPT transformer | 10 GB |
| BERT Adam | 6 GB |
| BERT OlsveredKFAC B=512 | 16 GB |
| BERT ClassicKFAC B=512 | 32 GB |
| Full benchmark | 32 GB |

---

## Run Commands

```bash
# Full benchmark (all tasks)
bash run_benchmark.sh --task all

# Single task
bash run_benchmark.sh --task transformer
bash run_benchmark.sh --task bert
bash run_benchmark.sh --task cifar
bash run_benchmark.sh --task mlp

# Skip optimizers
bash run_benchmark.sh --task transformer --skip adam
bash run_benchmark.sh --task bert --skip classickfac

# LR sweep for transformer
bash run_benchmark.sh --task transformer --lr-sweep-transformer

# Manual LR override
bash run_benchmark.sh --task transformer \
    --lr-ols-transformer 5e-3 \
    --lr-cls-transformer 5e-3

# Cold-start (delete checkpoints)
rm benchmark/results/bert_ckpt_*.pt
```
