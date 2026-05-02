# Mathematical comparison of the three K-FAC variants

*ClassicKFAC vs OlsSMKFAC vs VeredKFAC.  Reference doc for the
stability benchmark.*

## 1. The shared K-FAC framework

Second-order optimisers want to apply the **natural gradient** update:

$$
\Delta W = -\eta\,F^{-1}\nabla L
$$

where $F$ is the Fisher information matrix.  For a fully-connected layer
$y = W x$, the per-layer Fisher block factorises (Martens & Grosse 2015) as
the Kronecker product

$$
F \approx A \otimes G,
\qquad
A = \mathbb{E}[xx^\top],
\qquad
G = \mathbb{E}[\delta \delta^\top]
$$

where $x \in \mathbb{R}^{n_{\text{in}}}$ is the layer input and
$\delta \in \mathbb{R}^{n_{\text{out}}}$ is the gradient at its output.
Using the Kronecker identity
$(A \otimes G)^{-1} \mathrm{vec}(\nabla W) = \mathrm{vec}(G^{-1}\nabla W A^{-1})$,
the K-FAC update collapses to:

$$
\boxed{\;\Delta W \;=\; -\eta\, G^{-1}\, \nabla L_W\, A^{-1}\;}
$$

Three matrices are involved per layer per step:

1. The current weight gradient $\nabla L_W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$.
2. The input-side Kronecker factor $A \in \mathbb{R}^{n_{\text{in}} \times n_{\text{in}}}$ (symmetric PSD).
3. The output-side Kronecker factor $G \in \mathbb{R}^{n_{\text{out}} \times n_{\text{out}}}$ (symmetric PSD).

**The three variants of K-FAC differ in exactly one thing: how they apply
$A^{-1}$ and $G^{-1}$ to $\nabla L_W$.**  Everything else — the layer
hooks, the EMA over batches, the damping, the momentum buffer — is identical.

## 2. Where ill-conditioning enters

Both $A$ and $G$ are computed as **sample second moments** from minibatch
data.  Concretely, $A = X^\top X / N$ where $X \in \mathbb{R}^{N \times n_{\text{in}}}$
is the matrix whose rows are individual input activations.

This matters numerically because forming the Gram matrix **squares the
condition number**:

$$
\kappa(A) \;=\; \kappa(X^\top X) \;=\; \kappa(X)^2
$$

So even before any inversion takes place, $A$ is fundamentally a worse-
conditioned matrix than $X$.  If $X$ has condition $10^3$ (typical for
deep-network activations), $A$ has condition $10^6$.

The variants differ in **how much further** they amplify this in the
inversion step:

| Variant       | Inversion strategy            | Error scaling   |
|---------------|-------------------------------|-----------------|
| ClassicKFAC   | explicit `inv(A+λI)`          | $\kappa(X)^4 \cdot \varepsilon$ |
| OlsSMKFAC     | `cholesky_solve(·, chol(A+λI))` | $\kappa(X)^2 \cdot \varepsilon$ |
| VeredKFAC     | QR of $X$ directly, never form $A$ | $\kappa(X)^1 \cdot \varepsilon$ |

The square-root improvements at each step are the entire numerical
motivation for the OlsSM and Vered designs.

## 3. ClassicKFAC — explicit inverse via `torch.linalg.inv`

### Algorithm

```
1.  A_damped ← A + λI
2.  G_damped ← G + λI
3.  A_inv ← inv(A_damped)        # via LAPACK getrf+getri  (LU + back-solve)
4.  G_inv ← inv(G_damped)
5.  for each step:
        ΔW ← G_inv  @  ∇L_W  @  A_inv     # two GEMM kernels
        W  ← W - η ΔW
```

`torch.linalg.inv` calls `cuSOLVER getrf` (LU factorisation with partial
pivoting) followed by `cuSOLVER getri` (back-substitution against the
identity) to form the explicit inverse matrix.

### Numerical analysis

Computing $M^{-1}$ explicitly via LU on a matrix with condition $\kappa(M)$
produces a result whose **forward relative error** is bounded by
$\kappa(M)^2 \cdot \varepsilon_{\text{mach}}$ in the worst case.  (Higham,
*Accuracy and Stability of Numerical Algorithms*, ch. 9 — explicit inverse
is unstable in this sense.)

For Classic K-FAC:

$$
\text{err}\,\big[\Delta W\big] \;\sim\; \kappa(A)^2 \cdot \varepsilon
\;=\; \kappa(X)^{2 \cdot 2} \cdot \varepsilon
\;=\; \kappa(X)^4 \cdot \varepsilon
$$

That is, the natural-gradient update has elements whose error grows as
the **fourth power** of the input activation matrix's condition number.

### Where this hurts in practice

If $\kappa(X) = 10^3$ (common for ReLU activations after a few layers),
Classic's inverse can have elements with error $\sim 10^{12} \cdot
\varepsilon \approx 10^{12} \cdot 10^{-7} = 10^5$ — much larger than 1.
Without aggressive damping, the natural gradient is essentially garbage.

## 4. OlsSMKFAC — Cholesky decomposition + triangular solve

### Algorithm

```
1.  L_A ← cholesky(A + λI)         # lower-triangular: L_A L_A^T = A+λI
2.  L_G ← cholesky(G + λI)
3.  for each step:
        C  ← cholesky_solve(∇L_W,    L_G)    # solves (G+λI) C  = ∇L_W
        ΔW ← cholesky_solve(C^T,     L_A).T  # solves (A+λI) ΔW^T = C^T
        W  ← W - η ΔW
```

Two key changes from Classic:

1. Cholesky is used instead of LU.  For an $n \times n$ symmetric PSD
   matrix, Cholesky costs roughly $\tfrac{1}{3}n^3$ FLOPs vs LU's
   $\tfrac{2}{3}n^3$ — half the work.  And Cholesky needs no pivoting.
2. The inverse is **never formed explicitly**.  Each step solves two
   triangular systems via `torch.cholesky_solve`, which calls cuBLAS
   `TRSM` under the hood.

### Numerical analysis

Triangular-solve is **backward stable**: the computed solution exactly
solves a slightly perturbed system $(M + \delta M) x = b$ with
$\|\delta M\| / \|M\| = O(\varepsilon)$.  The forward error is therefore
$\kappa(M) \cdot \varepsilon$, **not** $\kappa(M)^2 \cdot \varepsilon$
(Higham again — the square only appears when you explicitly form $M^{-1}$).

For OlsSM K-FAC:

$$
\text{err}\,\big[\Delta W\big] \;\sim\; \kappa(A) \cdot \varepsilon
\;=\; \kappa(X)^2 \cdot \varepsilon
$$

Compared to Classic's $\kappa(X)^4 \cdot \varepsilon$, this is the
**square-root improvement**.  At $\kappa(X) = 10^3$ that's
$10^6 \cdot \varepsilon \approx 10^{-1}$ — manageable, vs Classic's $10^5$.

### Trade-offs

- **Per-step compute**: cuBLAS GEMM is more aggressively optimised than
  cuBLAS TRSM at the matrix sizes we use.  Empirically OlsSM is ~10-20%
  slower per step on GPU than Classic, even at equal decomposition
  frequency.  The numerical advantage costs runtime.
- **Memory**: identical to Classic — store one $n \times n$ matrix per
  factor (the Cholesky factor instead of the inverse).

### Code reference

`optimizer/olssm_kfac.py:_decompose_lu` (despite the name, the routine
runs Cholesky), and the apply path in `step()` that uses
`torch.cholesky_solve`.  There is also a fallback EVD path for layers
larger than `lu_max_dim` (default 4096), which gives the same numerical
guarantee via eigendecomposition.

## 5. VeredKFAC — QR of raw activations, never form the Gram matrix

### Algorithm

The crucial insight: if $X$ has QR factorisation $X = Q\,R$ with $Q$
orthonormal and $R$ upper-triangular, then

$$
A = X^\top X = R^\top Q^\top Q R = R^\top R
$$

so $R$ encodes everything about $A$ — but $R$ has condition $\kappa(R) =
\kappa(X)$, not $\kappa(X)^2$.  We can solve any system involving $A$
using just $R$ via two triangular solves, never forming $A$ itself.

```
1.  R_X ← qr(X).R           # streaming TSQR over batches
                            # never materialises A = X^T X
2.  R_G ← qr(δ).R           # same for output gradients δ
3.  Damp:  R_X ← stack[R_X ; sqrt(λ)·I],  qr again → R_X_damped
           similarly for R_G_damped
4.  for each step:
        # Apply (R_G^T R_G)^{-1} = R_G^{-1} R_G^{-T} from the left
        # Apply (R_X^T R_X)^{-1} from the right
        # Each application = two TRSMs (one with R, one with R^T)
        u   ← solve_triangular(R_G^T,  ∇L_W)         # R_G^T u = ∇L_W
        v   ← solve_triangular(R_G,    u)            # R_G v   = u
        w   ← solve_triangular(R_X^T,  v^T)          # R_X^T w = v^T
        ΔW^T ← solve_triangular(R_X,    w)
        W   ← W - η ΔW
```

This needs four triangular solves per step instead of two GEMMs — but
each solve is on the **un-squared** condition-number matrix.

### Numerical analysis

QR of $X$ directly is backward stable in $X$:

$$
\text{err}\,\big[\Delta W\big] \;\sim\; \kappa(X) \cdot \varepsilon
$$

Compared to Classic's $\kappa(X)^4$, this is a **double square-root
improvement**.  At $\kappa(X) = 10^3$ the worst-case error is $10^3
\cdot \varepsilon \approx 10^{-4}$ — essentially negligible.

### Why this is non-trivial in practice

Three things make QR-based K-FAC harder than it sounds:

1. **You need $X$ raw**, not just its Gram.  This means the hooks have to
   collect raw activations (potentially huge: batch × seq_len × d_model
   for transformers) instead of the much smaller A matrix.  Solved with
   *streaming TSQR*: each minibatch chunk is QR'd, then the running R is
   merged with the new R via another QR, keeping memory at $O(n^2)$ per
   layer — same as the Gram-based variants.
2. **Damping** has to be applied via *ridge augmentation* of the rows
   (stack $\sqrt{\lambda} I$ underneath $X$ before QR), not by adding
   $\lambda I$ to a Gram matrix.  Equivalent in effect, different in
   implementation.
3. **The $p \geq n$ requirement**: QR produces a full-rank R only when
   the chunk has at least as many rows as columns.  A small minibatch on
   a wide layer can violate this.  The implementation falls back to
   Classic K-FAC for offending layers and logs a warning.

### Cost trade-offs

- **Per-step compute**: four TRSMs vs two GEMMs.  Each TRSM is also
  smaller (triangular work is half), so wall-clock cost is comparable
  but typically slightly higher.
- **Memory**: $O(n^2)$ R-factors per layer — same as Classic and OlsSM.
- **QR launches**: each hook fires *two* `torch.linalg.qr` calls (leaf
  QR + merge QR) every step.  cuSOLVER QR has high per-launch overhead.
  This is the main wall-time disadvantage on GPU and is what
  `TASK_vered_gpu_saturation.md` plans to address with CUDA Graphs or
  batched QR.

### Code reference

`optimizer/vered_kfac.py` for the optimiser, `optimizer/raw_activation_hooks.py`
for the streaming TSQR implementation, `optimizer/sgso.py:streaming_tsqr_update`
for the actual chunk-merge QR.

## 6. The role of damping (λ)

All three variants regularise the matrix-to-be-inverted with **Tikhonov
damping**: $A \to A + \lambda I$, $G \to G + \lambda I$.  This:

1. **Bounds the condition number from above**: after damping, the
   smallest eigenvalue of $A + \lambda I$ is at least $\lambda$, so the
   inverse can amplify by at most $1/\lambda$ in any direction.
2. **Prevents division-by-zero** from rank-deficient batches.  No matter
   the rank of the data, the damped matrix is full-rank.
3. **Is NOT what gives the variants their stability hierarchy.** All
   three use damping equally.  The hierarchy comes from how each
   amplifies the *remaining* condition number after damping.

VeredKFAC implements damping via row-augmentation, the others via
diagonal addition — mathematically equivalent.

## 7. Cost summary at a glance

For a single $n \times n$ Kronecker factor, per K-FAC update window
(amortised across `factor_update_freq` steps):

| Variant       | Decomposition cost      | Per-step apply cost           | Memory      |
|---------------|-------------------------|-------------------------------|-------------|
| ClassicKFAC   | $\tfrac{2}{3}n^3$ LU + $\tfrac{4}{3}n^3$ inv-from-LU $\approx 2n^3$ | $2 \cdot n^2 \cdot d$ via GEMM | $O(n^2)$ |
| OlsSMKFAC     | $\tfrac{1}{3}n^3$ Cholesky **only** | $2 \cdot n^2 \cdot d$ via TRSM | $O(n^2)$ |
| VeredKFAC     | $O(n^2 \cdot p)$ TSQR per chunk | $4 \cdot n^2 \cdot d$ TRSM | $O(n^2)$ |

(where $d$ is the other dimension of $\nabla L_W$, and $p$ is the
per-chunk row count fed into TSQR — capped at 512 by `_SEQ_SUBSAMPLE`).

**By raw FLOPs, OlsSMKFAC should win:**

- Decomposition: Cholesky's $\tfrac{1}{3}n^3$ is **6× fewer FLOPs** than LU + explicit inverse's $2n^3$.
- Per-step apply: same FLOP count as Classic ($2 n^2 d$), just spent in TRSM instead of GEMM.

So in pure arithmetic accounting, OlsSMKFAC should be 5-10% faster overall
(Cholesky saves at decomposition steps, apply steps are equal). **But empirically
on GPU, OlsSMKFAC ends up 5-15% slower.** Section 8 explains why.

## 8. Theory vs reality: why Cholesky doesn't actually win on GPU

This is the most counter-intuitive piece of the whole framework. The FLOP
count on paper says OlsSMKFAC should be faster than ClassicKFAC. The
benchmark (and `gpu_benchmark.py` historical data) shows the opposite —
~80 ms/step for OlsSMKFAC vs ~50 ms/step for Classic on SmallGPT, and even
after equalising `decomp_update_freq` the gap is ~10-20%. There are five
compounding reasons.

### 8.1. GEMM has tensor cores, TRSM does not

cuBLAS GEMM (`@` operator, used by ClassicKFAC's apply step) is the
single most aggressively optimised kernel in the entire CUDA ecosystem.
On RTX 3080 Ampere, GEMM dispatches to **Tensor Cores** — special
hardware units that do a 4×4 matrix multiply per cycle in FP16/TF32 with
FP32 accumulation. Peak throughput: ~119 TFLOPS in TF32 mode for the
3080.

cuBLAS TRSM (`torch.cholesky_solve`, used by OlsSMKFAC's apply step)
**cannot use Tensor Cores.** The triangular structure forces sequential
back-substitution: row $i$ depends on rows $0..i-1$. Tensor cores need
all 16 input elements ready in parallel, which TRSM's data dependency
graph forbids. So TRSM falls back to standard FP32 ALU throughput —
roughly **8× slower** per FLOP than tensor-core GEMM.

Net: even though TRSM and GEMM have the same FLOP count for our apply
step, GEMM finishes ~3-5× faster in wall time at our matrix sizes.

### 8.2. TRSM has limited parallelism

GEMM `C = A @ B` decomposes perfectly into independent dot products
($C[i,j] = \sum_k A[i,k] \cdot B[k,j]$) — every output element can be
computed in parallel. The GPU's thousands of CUDA cores all stay busy.

TRSM `solve T x = b` for triangular $T$ has a **wavefront** dependency
pattern: row 0 must finish before row 1 can start, row 1 before row 2,
etc. The compiler can pipeline this somewhat (and cuBLAS does some
clever blocking), but there is a fundamental serialisation that GEMM
doesn't have. SM utilisation during TRSM is typically 30-50% vs GEMM's
~95%.

### 8.3. Two TRSMs vs two GEMMs — but each TRSM has more overhead

ClassicKFAC's apply: `G_inv @ ∇L_W @ A_inv` is **two GEMM calls** with
no transposes between them.

OlsSMKFAC's apply:
```
C   = cholesky_solve(∇L_W,  L_G)   # 1st TRSM, output (n_out, n_in)
ΔW  = cholesky_solve(C^T,   L_A).T # 2nd TRSM on a transposed input
```

Note the `C^T` and `.T` — these are not free. PyTorch makes the transpose
implicit (just adjusts strides), but cuBLAS TRSM internally performs a
copy to a contiguous layout because the kernel needs row-major access.
Each `cholesky_solve` call therefore launches: setup → memcopy for
transpose → TRSM kernel → result. Three GPU operations vs GEMM's one.

### 8.4. Decomposition is amortised, so the FLOP win there matters less

OlsSMKFAC's decomposition is genuinely faster (Cholesky 6× cheaper than
LU+inv). But this fires only every `decomp_update_freq` steps — default
20 in the stability benchmark. Wall-time savings:

```
Classic decomp: ~5ms × every 20 steps = 0.25 ms/step amortised
OlsSM decomp:   ~1ms × every 20 steps = 0.05 ms/step amortised
Saving:         0.20 ms/step           ← negligible vs apply gap
```

The apply step happens *every* step, so a 30 ms apply gap dwarfs a 0.2 ms
decomp gap. Cholesky's FLOP advantage is in the wrong place to matter.

### 8.5. PyTorch dispatch overhead

`torch.cholesky_solve` has more Python-side overhead than `@`:
- Validates that the input is triangular (it isn't, it's the Cholesky factor — but the validation runs anyway)
- Allocates output tensor
- Calls cuSOLVER's dispatcher (one extra layer vs cuBLAS direct)

For small matrices ($n \leq 1024$ as in SmallGPT), this overhead is a
non-trivial fraction of the actual compute. At larger sizes (e.g., BERT's
$n=3072$ FFN), the overhead becomes negligible relative to the kernel
time and the GEMM/TRSM gap shrinks.

### Summary: why the math lies

The FLOP argument predicts OlsSMKFAC is faster because it counts
operations as if they all execute at the same rate. On GPU they don't:

| Operation | Hardware path on RTX 3080 | Effective rate (TF32) |
|---|---|---|
| GEMM (large) | Tensor Cores | ~119 TFLOPS |
| GEMM (small ≤1024) | Tensor Cores w/ overhead | ~50 TFLOPS |
| TRSM | Standard ALUs | ~10-15 TFLOPS |
| Cholesky factor | cuSOLVER (mostly dot products) | ~30 TFLOPS |
| Matrix inverse | cuSOLVER LU + back-sub | ~20 TFLOPS |

OlsSMKFAC trades a 6× FLOP saving (Cholesky vs LU+inv) for a 3-5× kernel
slowdown (TRSM vs GEMM). The slowdown wins because it happens every step,
not just at decomposition.

### When this changes

OlsSMKFAC's wall-time deficit shrinks or reverses in three regimes:

1. **Larger matrices** ($n \geq 4096$): tensor-core advantage on GEMM
   doesn't grow as fast as FLOP count once you're already kernel-bound.
   On a BERT-large or LLaMA layer, OlsSMKFAC could plausibly equal or
   beat Classic.
2. **CPU execution**: no tensor cores, so GEMM and TRSM go through the
   same SIMD path. Cholesky's FLOP saving translates more directly to
   wall time.
3. **Future GPU generations**: NVIDIA has been adding "structured sparse"
   tensor-core paths in Hopper/Blackwell. If TRSM ever gets a tensor-core
   variant, OlsSMKFAC's apply cost would drop to GEMM levels and the FLOP
   advantage at decomposition would dominate.

The numerical-stability benefit of OlsSMKFAC ($\kappa(X)^2$ vs $\kappa(X)^4$)
is the *real* value proposition — wall-time speedup over Classic is unlikely
on current consumer GPUs and shouldn't be the headline claim.

## 9. When each variant should win

Based on the math:

- **Low LR + well-conditioned data + stable workloads** → all three
  produce essentially identical updates.  Pick whichever is fastest
  (probably Classic).
- **High LR or rough loss landscape** → OlsSM and Vered should
  outperform Classic because their updates have less inversion noise to
  amplify across many steps.
- **Heavy damping required (λ ≥ 1e-2) for stability** → Classic with
  high damping nearly catches up to OlsSM with low damping, because high
  damping bounds the condition number anyway.  The whole point of
  OlsSM/Vered is to enable *lower* damping → more faithful curvature
  estimate.
- **Pre-training a small model from scratch** (rough landscape, large
  natural-gradient steps) → Vered's $\kappa(X)^1$ scaling matters most.

## 10. What the stability benchmark is actually measuring

The benchmark in `benchmark/stability_benchmark.py` quantifies these
predictions on the SmallGPT / WikiText-2 task.  Phase 1 finds each
variant's max stable LR (the LR at which training does not catastrophically
diverge); Phase 2 runs full convergence at that LR.  If the math holds,
expect:

$$
\text{LR}_{\max}^{\text{Vered}} \;>\; \text{LR}_{\max}^{\text{OlsSM}} \;>\; \text{LR}_{\max}^{\text{Classic}}
$$

with ratios roughly tracking the $\kappa(X)^k$ exponent differences (i.e.
$\sqrt{\kappa(X)}$ between adjacent tiers).  The condition-number tracker
inside the benchmark records the actual $\kappa(A)$ and $\kappa(G)$ over
training so the predicted-vs-observed comparison can be done numerically,
not just by theory.

## References

- Martens & Grosse (2015), *Optimizing Neural Networks with Kronecker-
  factored Approximate Curvature*, ICML.  The original K-FAC paper.
- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., ch.
  9 (matrix inversion) and ch. 19 (QR factorisation).  The numerical
  analysis behind the $\kappa^k$ scaling for each method.
- Halko, Martinsson & Tropp (2011), *Finding Structure with Randomness*.
  Source of the randomised EVD used in OlsSMKFAC's fallback path.
- Demmel et al., *Communication-avoiding parallel and sequential QR
  factorizations*.  TSQR.
