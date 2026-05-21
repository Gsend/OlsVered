# TP+OLS for LoRA Training — Experiment Protocol

**Objective.** Test whether the target-propagation + closed-form OLS machinery can fit **LoRA adapters** in closed form (no gradient descent), by recognizing that a per-layer LoRA fit under a propagated target is a **reduced-rank regression (RRR)** problem — solvable as one inversion-free OLS solve plus one truncated SVD.

**Relation to prior findings.** The architecture-coverage round established: (a) single-pass per-layer OLS ≫ iterated; (b) **feature-level distillation is exact and closed-form** across MLP/CNN; (c) label-retrain through non-invertible ops (pool, attention) hits a wall. LoRA-TP inherits these: it should be strongest as a **closed-form distillation/adaptation** of a frozen base, weakest as from-scratch label retraining.

---

## 1. Core idea (math)

A LoRA layer freezes the base weight `W0 ∈ R^{d_out×d_in}` and learns a low-rank correction:

```
W_eff = W0 + s · B A ,   A ∈ R^{r×d_in},  B ∈ R^{d_out×r},  s = α/r,  r ≪ min(d_in,d_out)
```

Target propagation gives this layer an input `X ∈ R^{n×d_in}` (rebuilt-upstream activations) and a **pre-activation target** `T ∈ R^{n×d_out}` (after activation inversion). The per-layer fit is

```
minimize_{A,B}  || X (W0 + s B A)^T − T ||_F^2  (+ λ‖·‖)
```

Subtract the frozen base's contribution to get the **residual target**

```
T' = T − X W0^T
```

so the problem becomes a **rank-r constrained least squares**:

```
minimize_{A,B}  || X (s B A)^T − T' ||_F^2 ,   rank(BA) ≤ r .
```

**Closed form (reduced-rank regression, Izenman 1975).**
1. Unconstrained OLS on the residual target: `M = argmin_W ||X W^T − T'||^2` via the existing inversion-free `solve_ols_layer` (vered_solve). `M ∈ R^{d_out×d_in}`.
2. Fitted values `Ŷ = X M^T ∈ R^{n×d_out}`. Take the top-r right singular subspace of `Ŷ` (= top-r eigenvectors `V_r ∈ R^{d_out×r}` of the small `d_out×d_out` matrix `Ŷ^T Ŷ`).
3. The optimal rank-r weight is `W_r = V_r V_r^T M`. Factor it into LoRA terms:
   ```
   B = V_r / s ,        A = V_r^T M        ⇒   s B A = V_r V_r^T M = W_r .
   ```
   (Any invertible r×r rebalancing between A and B is equivalent; pick the balanced/`s`-normalized split.)

Both steps are inversion-free and cheap: one OLS solve (already in the codebase) and one eigendecomposition of a `d_out×d_out` Gram (or randomized SVD for large `d_out`). **No backprop, one pass per layer.**

**Why this is the right object.** Gradient LoRA minimizes the same per-layer objective by SGD over many steps; RRR gives its global optimum directly when the target `T'` is fixed. Under target propagation the per-layer targets *are* fixed within a pass — exactly the regime where the project already showed closed-form OLS wins.

---

## 2. Hypotheses

| ID | Statement | Falsifier |
|----|-----------|-----------|
| L1 | Closed-form RRR LoRA matches/exceeds the *per-layer* fit quality of gradient LoRA at equal rank (lower residual `‖X(sBA)^T−T'‖`). | RRR per-layer residual ≥ gradient-LoRA residual at matched r. |
| L2 | **Distillation** into LoRA adapters (teacher per-layer activations as targets) recovers near-teacher accuracy in a single closed-form pass. | Closed-form LoRA distill acc ≪ teacher at moderate r (e.g. r=8 on a 256-wide MLP). |
| L3 | RRR-LoRA quality rises monotonically with r and saturates near the full-OLS (rank-∞) accuracy. | Non-monotone, or large gap to full-OLS that never closes as r→min(d_in,d_out). |
| L4 | The single-pass ≫ iterated finding holds for LoRA-TP too (iterating the LoRA chain drifts/degrades). | Iterated LoRA-TP strictly improves test acc over single pass across ≥2 architectures. |
| L5 | From-scratch **label** retraining via LoRA-TP underperforms distillation by the same margin seen for full-weight TP (the bottleneck is target quality, not the rank constraint). | Label LoRA-TP ≈ distill LoRA-TP, i.e. the rank constraint, not the chain, dominates. |

---

## 3. Method variants

| Variant | Base `W0` | Target `T` | Fit |
|---------|-----------|------------|-----|
| **V0 full-OLS (control)** | n/a (overwrite W) | per-layer | existing `solve_ols_layer` (rank-∞) |
| **V1 RRR-LoRA distill** | frozen pretrained | teacher per-layer activations | residualize + RRR-SVD |
| **V2 RRR-LoRA label-retrain** | frozen random / pretrained | GT one-hot, chain-back-prop | residualize + RRR-SVD per chain layer |
| **V3 gradient-LoRA (baseline)** | frozen | end-task loss | AdamW on A,B (standard PEFT) |
| **V4 RRR-warmstart + grad** | frozen | — | V1/V2 init, then a few SGD steps |

`s`, `r`, `λ` swept. V4 tests whether the closed-form solution is a good *initializer* even when it is not the final answer.

---

## 4. Framework changes

| Component | Change |
|-----------|--------|
| `diagnostic/lora.py` (new) | `LoRALinear(W0_frozen, r, alpha)`; `solve_lora_layer(X, T, W0, r, alpha, lam)` → returns `(A, B, residual)` via residualize + RRR-SVD (uses `vered_solve` for step 1; `torch.linalg.eigh` on `Ŷ^TŶ` or randomized SVD for step 2). |
| `target_prop_retrainer.py` | Add `lora_rank` / `lora_alpha` kwargs; when set, the per-layer solve calls `solve_lora_layer` and writes A,B instead of overwriting W. Forward sweep unchanged (rebuilt-upstream). |
| `capture.py` | No change (LoRALinear forward is still `x → (W0 + sBA) x`; hook captures input/output as before). |
| runners | `diagnostic/runners/lora/` — `lora_mlp_mnist.py` (V0–V4 on the MnistMLP / DeepMLP), reusing existing MNIST + GT-target helpers. |

---

## 5. Experiments

**Primary testbed = TinyTransformer (E8).** LoRA's native habitat is the
attention/MLP projections (Q/K/V/O, fc_in, fc_out); E8 already proved those Linears are
OLS-distillable to 100% (the rank-∞ ceiling). **All substantive runs are ADAPTATION:**
freeze a base trained on task A, fit low-rank adapters to hit **task-B** targets.
Self-distillation over the same frozen base is degenerate — the residual target
`T' = T − X W0^T ≈ 0`, so the optimal adapter is ~zero at any rank (a flat,
meaningless curve). LR0 alone uses a trivial standalone layer (it is a math unit test).

| # | Setup | Battery | Tests |
|---|-------|---------|-------|
| **LR0** | Math unit test: one standalone `Linear` (e.g. a transformer `Wq` or random `Linear(64,64)`), `r = min(d_in,d_out)` (full rank). | residual vs V0 full-OLS, both metrics | RRR reduces to OLS at full rank (match ⇒ implementation correct). HARD GATE. |
| **LR1** | **Transformer adaptation (headline).** Base = TinyTransformer trained on task A (majority), all base weights frozen; teacher = TinyTransformer trained on task B; fit LoRA adapters on Q/K/V/O+MLP (also a Q,V-only variant) so frozen-base+adapter reproduces the task-B teacher's per-layer activations. Sweep `r ∈ {1,2,4,8,16,32}`. Task B = mild shift (threshold `>= V//4` or label-flip) first for a clean curve, then `pointer` as stretch. | task-B acc-vs-r, per-layer residual-vs-r, bounds (base-A-on-B lower, task-B teacher upper) | L1, L2, L3 |
| **LR2** | Gradient-LoRA baseline at matched `r`, data budget (AdamW on A,B). | acc, wall-time, #steps to match RRR | L1, practical value of the closed form |
| **LR3** | Label LoRA-TP from random base, GT one-hot chain. Sweep r. | acc, residual profile | L5 |
| **LR4** | Iterate LR1/LR3 for 10 passes. | acc trajectory | L4 |
| **LR5** | Gradient-LoRA baseline (V3) + warmstart (V4) at matched r. | acc, wall-time, #steps to match RRR | L1, practical value of closed form |
| **LR6** (stretch) | RRR-LoRA on the E5 ResNet / E8 transformer Linears (Q/K/V/O), distillation only. | per-layer residual, acc | does closed-form LoRA distill survive conv/attention stacks |

**Common hyperparameters:** seed=42; n_samples 16,384; `λ=1e-4`; `α=r` (so `s=1`) unless swept; eval on held-out test split. Single seed unless a result is borderline.

**Primary artifacts:** acc-vs-rank curve (RRR vs gradient-LoRA vs full-OLS ceiling); per-layer residual-vs-rank table; wall-time comparison.

---

## 6. Decision rules

| Outcome | Action |
|---------|--------|
| L1+L2 hold (RRR ≈ gradient at the per-layer fit, distill ≈ teacher) | Closed-form LoRA distillation is viable → promote to a named capability; write up. |
| L1 holds, L2 fails (good per-layer fit, poor task acc) | Per-layer optimality ≠ task optimality through the chain → quantify the gap; test V4 warmstart. |
| L3 fails (non-monotone in r) | Bug in RRR projection or metric (RRR is provably monotone in r) → halt, debug LR0. |
| L5 shows rank constraint dominates | LoRA is the bottleneck, not TP → report; lower the priority of label-retrain LoRA. |

**Escalate to user** if: RRR ≠ OLS at full rank (LR0 fails); or gradient-LoRA beats RRR per-layer residual (contradicts the closed-form optimum — implies a metric/whitening error worth understanding).

---

## 7. Team structure (mirrors SESSION_PART8)

| Role | Owns | Does not own |
|------|------|--------------|
| **TeamLeader** | hypotheses L1–L5, RRR-vs-OLS go/no-go (LR0 gate), acc-vs-rank synthesis, plan updates | implementation, debugging |
| **Scientist** | `lora.py`, runner, gradient-LoRA baseline, battery execution, error triage (≤2 fix attempts then escalate) | hypothesis verdicts, scope/queue changes |

Handoff + findings-log conventions identical to the arch-coverage protocol (`lora_findings.md`, one row per experiment).

---

## 8. Open questions

| ID | Question |
|----|----------|
| LQ1 | Should the rank-r truncation use the plain output metric (`Ŷ^TŶ`) or the **whitened** metric (project in `X^TX`-norm)? Whitened RRR is the textbook optimum; verify empirically at LR0. |
| LQ2 | One shared adapter per layer vs separate adapters per attention projection (Q/K/V/O) for LR6. |
| LQ3 | Does `s = α/r` scaling interact with the residual target magnitude (mean-correction)? Tie to the depth-4 moment-match crossover already found. |
| LQ4 | For adaptation (LR2): targets from the new-task teacher's activations, or chain-back-prop of new-task labels onto the frozen base? (Default: teacher activations — the established distillation strength.) |
| LQ5 | Worth comparing RRR-LoRA against the `OlsSMKFAC`/ALS recommender factorization machinery already in the repo (both are low-rank closed-form)? |

---

## 9. Out of scope

Quantization/QLoRA; multi-adapter routing; production PEFT integration; LLM-scale bases (use ≤256-wide MLP layers and the E8 tiny-transformer Linears so SVDs stay on small `d_out×d_out` matrices). Multi-seed only on borderline outcomes.
