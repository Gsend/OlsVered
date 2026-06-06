# Per-layer operations: Classic / Vered / Vered+WGSO / SINGD on GPT-2 medium

Reference architecture: **GPT-2 medium** (24 transformer blocks, d_model = 1024, n_heads = 16, d_ff = 4096, vocab = 50257, seq_len = 1024).

All four optimizers in this comparison preconditrolloperate identically with respect to **which layers they cover**: they hook only `nn.Linear` (and `nn.Conv2d`) modules, with an `out_features ≤ 4096` filter to exclude the LM head. Non-Linear modules (embeddings, LayerNorms) are updated by a separate AdamW optimizer in our codebase. The differences are entirely in **what numerical operation is performed inside the hook and the apply step**.

---

## 1. Layer coverage

| Layer type | Count in GPT-2 medium | Classic | Vered | Vered+WGSO | SINGD | Fallback (all 4) |
|---|---:|:---:|:---:|:---:|:---:|---|
| Token embedding (`nn.Embedding 50257 × 1024`) | 1 | — | — | — | — | **AdamW** |
| Position embedding (`nn.Embedding 1024 × 1024`) | 1 | — | — | — | — | **AdamW** |
| LayerNorm | 49 (2/block + final) | — | — | — | — | **AdamW** |
| Q proj (`Linear 1024 → 1024`) | 24 | ✓ | ✓ | ✓ | ✓ | — |
| K proj (`Linear 1024 → 1024`) | 24 | ✓ | ✓ | ✓ | ✓ | — |
| V proj (`Linear 1024 → 1024`) | 24 | ✓ | ✓ | ✓ | ✓ | — |
| Attention output (`Linear 1024 → 1024`) | 24 | ✓ | ✓ | ✓ | ✓ | — |
| FFN1 (`Linear 1024 → 4096`) | 24 | ✓ | ✓ | ✓ | ✓ | — |
| FFN2 (`Linear 4096 → 1024`) | 24 | ✓ | ✓ | ✓ | ✓ | — |
| **LM head** (`Linear 1024 → 50257`) | 1 | — | — | — | — | **AdamW** (out > 4096 = `KFAC_MAX_DIM`) |
| Bias terms (where present) | varies | bias-only via apply_vered_bias | bias-only via apply_vered_bias | as Vered | included in K via bias-augmented column | — |

**Total K-FAC-covered Linear layers per forward pass: 144** (24 blocks × 6 layers/block).

---

## 2. Per-step operations on a K-FAC-covered Linear layer `W ∈ ℝ^(n_out × n_in)`

This is the heart of the comparison — what happens on **every** training step for **every** preconditioned Linear layer, ordered by phase. `x` denotes the layer's input activation (after row-subsampling to ≤ 2048 rows), `δ` denotes the back-propagated output gradient, `λ` denotes Tikhonov damping, `T = 20` is the factor-update frequency.

### 2.1 Hook phase — runs every step

| Step | Classic K-FAC | Vered K-FAC | Vered + WGSO | SINGD-Dense |
|---|---|---|---|---|
| Subsample `x`, `δ` | rows → 2048 | rows → 2048 | rows → 2048 | rows → all (no subsample) |
| Per-row pre-transform | none | none | `x ← x · √(1/(‖x‖²+α·median))` (WGSO) and same for `δ` | none |
| Bias augmentation | none (bias treated separately) | none (bias treated separately) | none | append `1` column → input becomes `[x, 1]` |
| **Forward hook accumulator update** | `A_sum += xᵀx` (single GEMM) | `R_X ← QR([R_X ; x]).R` (leaf QR + merge QR) | same as Vered, on the WGSO-weighted `x` | `H_K ← H_K + K.from_inner(aᵀ)` (matmul into structured matrix) |
| **Backward hook accumulator update** | `G_sum += δᵀδ` (single GEMM) | `R_G ← QR([R_G ; δ]).R` | same as Vered, on the WGSO-weighted `δ` | `H_C ← H_C + C.from_inner(gᵀ)` |
| Memory carried between steps per layer | `A_sum (n_in × n_in)` + `G_sum (n_out × n_out)` = `2 (n_in² + n_out²)` | `R_X (n_in × n_in)` + `R_G (n_out × n_out)` upper-triangular = `2 (n_in² + n_out²)` | same as Vered | `K, C` (factors) + `m_K, m_C` (Riemannian momentum) + `H_K, H_C` (accumulators) = **3 (n_in² + n_out²)** |
| Cost per layer | 2 GEMMs (matmul) | 2 QR + 2 cat (Householder, slower than GEMM at equal FLOPs but tensor-core-poor) | identical FLOPs to Vered plus per-row weight computation (norm + median + sqrt) | 2 GEMMs (matmul; tensor-core friendly) |

### 2.2 Factor refresh — runs every `T = 20` steps

| Step | Classic K-FAC | Vered K-FAC | Vered + WGSO | SINGD-Dense |
|---|---|---|---|---|
| Compute current factor | `A = A_sum / n`, `G = G_sum / n` | `R_X_norm = R_X / √n` (per-sample normalization) | same as Vered | (already maintained incrementally) |
| EMA blend with previous | `A ← γA_old + (1-γ)A_new` (linear) | `R_X ← QR([√γ·R_X_old ; √(1-γ)·R_X_new]).R` (exact augmented-QR blend) | same as Vered | implicit in Riemannian momentum |
| Damping | `A_damped = A + λI` (explicit ridge add) | `R_X_damped = QR([R_X ; √λ·I_n]).R` (ridge augmentation, no Gram formed) | same as Vered | folded into `m_K` update: `m_K += λ · KᵀK` |
| **Invert (or factor-and-solve representation)** | `A_inv = torch.linalg.inv(A_damped)` and same for `G` — produces full inverse matrices `(n_in × n_in)` and `(n_out × n_out)` | (no inversion — `R_X_damped` and `R_G_damped` stored directly as upper-triangular) | (no inversion) | **no inversion** — multiplicative update: `m_K ← α₁·m_K + β·(H_K + damping·KᵀK − I)`, then `K ← K - β₁·K·m_K` (truncated Expm) |
| Cost per layer per refresh | 2 explicit matrix inverses (cubic in factor dim) | 2 augmented QRs + 2 ridge-QRs (cubic in factor dim; same FLOPs as inversion but stable) | same as Vered | matmuls only (cubic in factor dim, but GEMM-only and tensor-core friendly) |

### 2.3 Apply phase — runs every step

| Step | Classic K-FAC | Vered K-FAC | Vered + WGSO | SINGD-Dense |
|---|---|---|---|---|
| Compute natural-gradient direction | `nat_grad = G_inv @ grad_W @ A_inv` (two GEMMs) | 4 triangular solves: `T1 = R_Gᵀ \ grad_W`, `T2 = R_G \ T1`, `T3 = R_Xᵀ \ T2ᵀ`, `T4 = R_X \ T3`; `nat_grad = T4ᵀ` | same as Vered (the WGSO weighting is already baked into `R_X`, `R_G`) | `nat_grad = K @ Kᵀ @ grad_W @ C @ Cᵀ` (four GEMMs, no solves) |
| Cost per layer | 2 GEMMs | 4 triangular solves (each between matmul and back-sub in cost) | same as Vered | 4 GEMMs (all tensor-core friendly) |
| Numerical-error scaling | `O(κ(X)² · ε)` per Kronecker side (Higham 2002 Th. 14.7, 20.3) | `O(κ(X) · ε)` per side (Higham 2002 Th. 19.10) | same as Vered with `κ(W·X) ≤ κ(X)` (van der Sluis 1969) | empirically stable at bf16; structural argument via Riemannian metric (Lin et al. 2024) |
| Momentum/grad-clip | applied per layer after `nat_grad` | applied per layer after `nat_grad` | applied per layer after `nat_grad` | applied per layer after `nat_grad` |

---

## 3. Layers handled by AdamW (all four optimizers)

These get standard AdamW with `lr = EMB_LR`, `weight_decay = 0.01`, `β1 = 0.9`, `β2 = 0.999`. No second-order machinery touches them.

| Layer | Parameter shape | Notes |
|---|---|---|
| Token embedding | `(vocab=50257, d_model=1024)` ≈ 51.5M params | Would dominate K-FAC memory if covered; AdamW is sufficient for embedding tables |
| Position embedding | `(seq_len=1024, d_model=1024)` ≈ 1M | Same rationale |
| LayerNorm `weight, bias` | `(1024,)` each × 49 = 100K params | Affine-only; second-order curvature has limited value at this dim |
| **LM head** | `(d_model=1024, vocab=50257)` ≈ 51.5M params | `out_features = 50257 > KFAC_MAX_DIM = 4096`; if covered, `G ∈ ℝ^(50257 × 50257)` would be ~10 GB and OOM |

Roughly **104 M of GPT-2 medium's 345 M params** are handled by AdamW. The other ~241 M (covered by K-FAC) sit in the 144 attention + FFN Linear layers.

---

## 4. Memory cost per K-FAC-covered layer

For a Linear layer `(n_in × n_out)`:

| Method | Persistent state per layer | Working memory at refresh |
|---|---|---|
| Classic K-FAC | `A_sum` + `G_sum` + `A_inv` + `G_inv` = **4 (n_in² + n_out²) · 4 B** | `A_damped`, `G_damped`, `A_inv`, `G_inv` simultaneously alive during `inv()` call |
| Vered K-FAC | `R_X` + `R_G` + cached damped `R_X` + cached damped `R_G` = **4 (n_in² + n_out²) · 4 B** (upper-triangular, but stored as dense) | augmented matrices `[R; √λ·I]` of shape `(2n × n)` temporarily |
| Vered + WGSO | identical to Vered | identical to Vered, plus per-row weight vector (cheap) |
| SINGD-Dense | `K` + `C` + `m_K` + `m_C` + `H_K` + `H_C` = **6 (n_in² + n_out²) · 4 B** (default; `m_K, m_C` only present if `alpha1 ≠ 0`) | factor product `K @ Kᵀ` etc. as intermediates |

GPT-2 medium covered-layer total (144 layers, mixture of `n=1024` and `n=4096`):

| | Per-layer state at `n=1024` | Per-layer state at `n=4096` (FFN1/FFN2) | Total K-FAC state |
|---|---:|---:|---:|
| Classic | 16 MB | 256 MB | **~6.5 GB** |
| Vered / WGSO | 16 MB | 256 MB | **~6.5 GB** |
| SINGD-Dense | 24 MB | 384 MB | **~9.7 GB** |

(SINGD's structured variants — Diagonal, Toeplitz, Block30Diagonal, Rank-1 Triangular — reduce these to O(n) instead of O(n²); we use SINGD-Dense as the apples-to-apples comparator.)

---

## 5. Numerical primitives count per step (per K-FAC-covered layer)

| Primitive | Classic | Vered | Vered + WGSO | SINGD-Dense |
|---|:---:|:---:|:---:|:---:|
| GEMM (matmul) | 2 hook + 2 apply = **4** | 0 hook + 0 apply = **0** | 0 | 2 hook + 4 apply = **6** |
| `torch.linalg.qr` | 0 | 4 (2 hook leaf + 2 hook merge) | 4 | 0 |
| `torch.linalg.solve_triangular` | 0 | **4** (apply) | 4 | 0 |
| `torch.linalg.inv` | 2 (at refresh, every T steps) | 0 | 0 | 0 |
| Other ops | bias-vector solve / matmul | bias-vector triangular solves | bias-vector triangular solves + WGSO weight ops | none extra |

cuSOLVER tensor-core utilization:
- GEMM and matmul-based ops: ~120 TFLOPs theoretical at bf16 on Ampere/Hopper
- QR (cuSOLVER `geqrf`): fp32-only path; bf16 emulated via quantize-and-upcast round-trip → ~30 TFLOPs effective
- Triangular solve: partially GEMM-fused; effective throughput between QR and pure GEMM

This is the underlying reason SINGD is fastest in bf16 (everything matmul, tensor-core friendly) and Vered's QR-route incurs a ~3× wall-time penalty.

---

## 6. Numerical-stability summary

| Method | Forms `XᵀX`? | Explicit matrix inverse? | Error bound | bf16 behavior in our sweep |
|---|:---:|:---:|---|---|
| Classic K-FAC | **Yes** | **Yes** (`torch.linalg.inv`) | `O(κ(X)² · ε)` | Collapses: 1891 ± 196 ppl (small), 1876 ± 131 (medium) |
| Vered K-FAC | No | No | `O(κ(X) · ε)` | Preserved: 914 ± 15 ppl (small), 811 ± 15 (medium) |
| Vered + WGSO | No | No | `O(κ(W·X) · ε)`, `κ(W·X) ≤ κ(X)` | Comparable to Vered: 935 ± 20 (small), 836 ± 18 (medium) |
| SINGD-Dense | No | No | Structural / Riemannian argument | Comparable to Vered: 950 ± 2 (small), 846 ± 20 (medium) |

The empirical bf16 perplexity ranking matches the numerical-stability ranking exactly: methods that never form `XᵀX` and never explicitly invert preserve their fp32 performance; the one that does both collapses.
