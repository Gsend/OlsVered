# Session Context: revive-kfac
*Saved: 2026-04-22*

---

## Project Overview
**OlsVeredKFAC** — K-FAC optimizer with randomized EVD backend, benchmarked against ClassicKFAC and Adam across four tasks.

**Repo root:** `C:/Users/nat79/OlsVered/`
**Key files:**
- `optimizer/hooks.py` — forward/backward hooks, KFAC-Reduce subsampling
- `optimizer/olssm_kfac.py` — main optimizer (named OlsVeredKFAC in results)
- `optimizer/classic_kfac.py` — baseline
- `benchmark/gpu_benchmark.py` — all 4 benchmark tasks (**modified this session**)
- `run_benchmark.sh` — launcher with CLI args
- `benchmark/results/` — JSON + CSV results, PNG plots

---

## Results State at Start of Session

### Known results (from result JSONs)
| Task | Adam | OlsVeredKFAC | ClassicKFAC |
|------|------|--------------|-------------|
| MLP/MNIST | **98.7%** | 97.6% | 97.7% |
| CIFAR-10 | **57.9%** | 53.7% | 51.4% |
| Transformer/WikiText-2 | ppl=1079 | ppl=368 | ppl=366 |
| BERT/SST-2 | **93.0%** | ❌ MISSING | ⚠ STALE |

### Gaps identified
1. **`bert_olsveredkfac_result.json` — entirely missing.** Run reached ~91% in a prior session but crashed before saving JSON.
2. **`bert_classickfac_result.json` — stale.** Was run at 221ms/step (pre-hooks-fix). With fixed hooks expected ~15ms/step. Final acc 85.9% is not trustworthy for comparison.
3. All other results came from a prior pod — hardware consistency across tasks is broken.

### Decision: re-run all tasks on a single fresh pod
Mixed-pod results make comparisons meaningless. Run `--task all` in one shot.

---

## Transformer LR Sweep Results (already in results folder)
Best LR from sweep:
- OlsVeredKFAC: lr=0.008 → ppl=368
- ClassicKFAC: lr=0.008 → ppl=366

Use `--lr-ols-transformer 0.008 --lr-cls-transformer 0.008` for the definitive run.

---

## Code Change Made This Session

### `benchmark/gpu_benchmark.py` — hardware provenance stamping

**1. New `get_hardware_info()` function** (added before GPU utilities section):
Collects: `gpu_name`, `gpu_vram_gb`, `gpu_count`, `cuda_version`, `gpu_sm` (e.g. `sm_120`),
`gpu_driver`, `cpu`, `cpu_cores`, `ram_gb`, `torch_version`, `python_version`.

**2. Module-level `HW_INFO: dict = {}`** — populated once in `main()` after `get_device()`.
Also prints a one-liner at startup:
```
Hardware: NVIDIA RTX PRO 6000 | driver 570.xx | CUDA 12.8 | torch 2.x.x
```

**3. `save_result_incremental()` patched** — merges `"hw": HW_INFO` into every result dict
before writing JSON (shallow copy, caller's dict unchanged).

Every `*_result.json` from the next run will have a `"hw"` block at the top level.

---

## Pod Requirements (for full `--task all` run)

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| **VRAM** | **32 GB** | ClassicKFAC BERT peak = 25.7 GB |
| **CUDA** | ≥ 11.8 | run_benchmark.sh auto-installs matching PyTorch wheel |
| **RAM** | 32 GB | BERT tokenization + 2 DataLoader workers |
| **CPU** | 4+ cores | 2 workers per task |
| **Disk** | ~5 GB free | BERT ~440 MB, datasets ~400 MB, Rust ~1.5 GB, venv ~3 GB |
| **Internet** | Required | HuggingFace downloads (BERT, SST-2, WikiText-2) on first run |
| **tmux** | Recommended | Script uses it by default; ~90 min total run |

**Suitable pod GPUs:** A100 80GB, H100 80GB, RTX 6000 Ada (48 GB — tight for ClassicKFAC BERT), RTX PRO 6000 Blackwell 96 GB (used previously).

**Per-task VRAM peaks (from actual results):**
| Task | Adam | OlsVeredKFAC | ClassicKFAC |
|------|------|--------------|-------------|
| MLP | 0.8 GB | 1.0 GB | 1.1 GB |
| CIFAR-10 | 0.8 GB | 1.0 GB | 1.1 GB |
| Transformer | 7.6 GB | 7.6 GB | 7.6 GB |
| BERT | 3.1 GB | ~16 GB | **25.7 GB** |

---

## Full Run Command (copy-paste ready)

```bash
# From OlsVered repo root on the remote pod
# Clear stale BERT checkpoints first
rm -f benchmark/results/bert_ckpt_*.pt

# Run all 4 tasks with best known transformer LRs
bash run_benchmark.sh --task all \
  --lr-ols-transformer 0.008 \
  --lr-cls-transformer 0.008
```

- Runs inside tmux session named `benchmark`
- If SSH drops: `tmux attach -t benchmark`
- Results saved incrementally to `benchmark/results/` after each optimizer completes
- Each result JSON will now include a `"hw"` block with full pod hardware info

**Expected wall time (~90 min total):**
- MLP: ~6 min
- CIFAR: ~10 min
- Transformer: ~20 min
- BERT ×3: ~45–60 min (with fixed hooks at ~15 ms/step)

---

## Key Architecture Notes (from SESSION_CONTEXT.md)

### K-FAC doesn't support nn.Embedding
- A matrix would be 50,257×50,257 even as sparse diagonal → use AdamW for embeddings
- Transformer task uses dual optimizer: K-FAC for Linear layers, AdamW for embeddings + LayerNorm + LM head

### LM head excluded from K-FAC
- `max_gram_dim=4096` skips layers with out_features > 4096
- LM head (out=50,257): G matrix would be 50,257×50,257 ≈ 10 GB → OOM

### SmallGPT (transformer task)
- 4 blocks, d_model=256, n_heads=4, d_ff=1024, vocab=50,257, seq_len=128
- ~16M total params; K-FAC covers 32 Linear tensors, AdamW covers 20

---

## Open Questions
1. Will clean re-run confirm OlsVeredKFAC BERT ~91%? (crashed before JSON in prior session)
2. Will fixed-hooks ClassicKFAC BERT improve over stale 85.9%?
3. Are transformer results (ppl≈368 vs 366 OlsVeredKFAC vs ClassicKFAC) meaningful or within noise?
