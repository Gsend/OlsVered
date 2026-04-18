# Session Part 1 — OLS Theory & Efficiency Discussion

## Context
Project: OlsSMLayerRetrainer — closed-form OLS-based weight retraining for BERT fine-tuning.
Model: bert-base-uncased, Task: SST-2 (binary sentiment), Benchmark: retrainer_benchmark.py

---

## Key Q&A Covered

### What is the added benefit of ols_lora over ols?
- Pure OLS (N=1) replaces the classifier head with the exact OLS solution.
- OLS+LoRA adds a low-rank residual adapter (B@A, rank r) on top of the OLS weights.
- The LoRA adapter captures non-linear structure that the linear OLS solve misses.
- For the SST-2 classifier head (768→2), OLS already saturates most of the linear signal,
  so ols_lora gives marginal gain at rank 4.

### How much more efficient is OLS vs normal closed-form OLS?
The retrainer uses **streaming accumulation** of XᵀX and XᵀY:
- Memory: O(d²) for the accumulator vs O(N·d) to store X explicitly
- For BERT 768-dim layer: accumulator = 768² × 4B = **2.4 MB** vs storing X = N×768×4B
  - For N=67K samples: X = 67000 × 768 × 4B ≈ **206 MB** (just for one layer)
  - For sequence-level: N_tokens = 67K × 128 = 8.5M rows → **26 GB**
- One forward pass through data, O(d³) solve — same math as "normal OLS", better memory.

### Which of the 3 standard OLS algorithms is most efficient at runtime?
| Method | Memory | Runtime | Streaming? |
|--------|--------|---------|-----------|
| Normal equations (XᵀX solve) | O(d²) | O(Nd² + d³) | ✓ |
| QR decomposition | O(Nd) | O(Nd²) | ✗ |
| SVD | O(Nd) | O(Nd² + d³) | ✗ |

- **Normal equations** are the only one amenable to streaming accumulation.
- For N >> d (typical in BERT fine-tuning), normal equations are fastest in wall time too.
- QR/SVD require materialising the full X matrix — infeasible for large N.

### Are there use cases where OLS-style weight correction beats Adam and KFAC?
Yes — cases where you need **exact, one-shot, data-efficient** updates:
1. **Model editing (ROME/MEMIT)**: inject a single fact into weights, no iterative training.
2. **Continual learning (EWC-style)**: closed-form update that respects Fisher constraint.
3. **Federated learning**: each client sends XᵀX + XᵀY (not raw data), server aggregates.
4. **Few-shot adaptation**: with very few samples, Adam overfits; OLS with regularisation is stable.
5. **Knowledge distillation target fitting**: fit teacher logits in one pass.

KFAC approximates the Fisher with Kronecker products — still requires iterative updates and
is sensitive to the Kronecker factorisation assumption. OLS gives the exact minimum of the
quadratic approximation in one shot.

---

## Benchmark Results Known at This Point
| Mode | Accuracy |
|------|----------|
| Pretrained (no FT) | 51.26% |
| OLS N=1 (classifier head only) | **84.98%** |
| ALS-LoRA N=1 r=4 | **85.21%** |

All other modes (N=2, N=4, adam, lora_r4, etc.) pending.
