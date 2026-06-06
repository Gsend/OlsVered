# Publication Positioning + Draft Abstract

**Purpose.** Honest novelty assessment against prior art (lit search 2026-05-21) and a
draft abstract, so we know which claims survive review *before* writing the full paper.

---

## 1. Bottom line

Frame the paper as an **empirical characterization study + a unifying account**, NOT a
new training algorithm. Every individual mechanism we use has close prior art (see §3).
What is defensible and, as far as the search found, not already done in one place:

1. A **systematic cross-architecture map** (MLP, deep MLP, LayerNorm-MLP, CNN, ResNet,
   U-Net, transformer) of *where closed-form per-layer OLS works*, with a single sharp,
   reproducible boundary:
   - **Feature-level distillation is near-exact and closed-form everywhere** (one
     OLS/layer, no SGD): residuals ~1e-3 and student≈teacher for every architecture,
     including additive skips, concat skips, and softmax attention.
   - **Label chain-retraining is bounded by the most information-destroying
     non-invertible op on the path** — max-pool (CNN/ResNet/U-Net) and softmax attention
     (transformer) — quantified by a clean accuracy "wall" (e.g. transformer pointer
     task: distill 99.98% vs label-retrain 57.6%, a 42-pt wall; majority task only 22 pt,
     showing the wall size tracks how much the task *needs* the non-invertible op).
2. A **unifying explanation**: per-layer OLS retraining is block-coordinate / Gauss-Seidel
   descent; **distillation has a fixed objective (teacher activations) → it converges and
   is exact; label chain-retraining has a moving objective (recomputed targets) → it
   drifts** (single-pass ≫ iterated, observed in every experiment). This connects TP, ALS,
   and the LoRA-RRR factorization under one lens.
3. An honest **negative result**: the activation-covariance (K-FAC-A) prior on the layer
   inverse shows *no consistent advantage* over the naive min-norm pseudo-inverse across
   architectures — useful because structured priors are often assumed to help.

These three are study/insight contributions. They fit **TMLR or a workshop**, where
correctness + interest (not novelty/SOTA) is the bar. They do NOT clear a top-conference
"new method" bar on their own.

---

## 2. The claim sentence (use this, it survives the lit check)

> "We do not introduce a new training primitive. We give a systematic, reproducible
> characterization of closed-form per-layer least-squares training across seven
> architectures, establishing that (i) it distills any tested architecture near-exactly
> in a single pass, while (ii) label-driven chain retraining is fundamentally bounded by
> the least-invertible operation on the path (pooling, attention), and we explain both
> with a fixed- vs moving-objective (block-coordinate) account."

---

## 3. Prior-art map (what collides — be explicit in Related Work)

| Our ingredient | Closest prior art | Verdict |
|----------------|-------------------|---------|
| Target propagation with a **regularized layer inverse** | Roulet & Harchaoui, *Target Propagation via Regularized Inversion* (arXiv:2112.01453); Lee et al., *Difference Target Propagation* (arXiv:1412.7525); Meulemans et al., *A Theoretical Framework for TP* (arXiv:2006.14331, TP↔Gauss-Newton) | **Not novel.** Our inverse is a regularized TP inverse; cite and differentiate. |
| **Closed-form / backprop-free layerwise** training | NoProp (arXiv:2503.24322); *Closed-Form Feedback-Free Learning with Forward Projection* (arXiv:2501.16476); Extreme Learning Machines | **Crowded, active (2025).** Position as characterization, not "first closed-form layerwise." |
| **Closed-form feature distillation via least squares** | FitNets (arXiv:1412.6550); *Knowledge Distillation from Few Samples* — least-squares 1×1 alignment of student to teacher features (OpenReview HkgDTiCctQ) | **Already exists.** "We distill via OLS" is NOT novel by itself. Our angle = *cross-architecture exactness* + the retrain contrast. |
| **LoRA via reduced-rank regression / SVD** | PiSSA (arXiv:2404.02948, NeurIPS'24); update-approximation init (arXiv:2411.19557); dual SVD-LoRA (arXiv:2505.14367); Izenman RRR (1975) | **Adjacent.** PiSSA = SVD of *weights* for init; ours = RRR on *data-driven targets*. Distinct mechanism but must differentiate carefully. |
| **K-FAC activation covariance as inverse prior** | Martens & Grosse, K-FAC (2015) — used for natural-gradient optimization, not as a TP inverse prior | **Most novel ingredient, but our data refutes its benefit.** Report as honest negative unless a regime where it wins is found. |

**Reviewer risk #1:** Roulet & Harchaoui (regularized-inversion TP) is the nearest neighbor
to the method itself — read it closely and state precisely what we add (the
cross-architecture exact-distillation/retrain-wall characterization, not the inverse).

---

## 4. Draft abstract (v0)

> Training neural networks by propagating *targets* and solving each layer in closed form
> (ordinary least squares) is an old idea, but it is unclear which architectures it can
> actually fit and why. We give a systematic, single-seed-reproducible study across seven
> architectures — plain and deep MLPs, a LayerNorm MLP, a CNN, a residual network, a
> U-Net, and a small transformer — using one per-layer least-squares solver under target
> propagation. We find a sharp and consistent dividing line. When the targets are a
> teacher's own per-layer activations (feature distillation), a *single* closed-form pass
> reproduces the teacher near-exactly on every architecture, including those with additive
> skips, concatenation skips, and softmax attention (per-layer relative residuals ~1e-3).
> When the targets are derived from labels and propagated through the network (chain
> retraining), accuracy is bounded by the least-invertible operation on the path: max
> pooling and softmax attention impose large, measurable "walls" (e.g. on a task that
> requires content-based attention, distillation reaches 99.98% while label retraining
> reaches 57.6%), whereas the same network distills perfectly. We explain both phenomena
> with a block-coordinate-descent account: distillation optimizes a *fixed* objective and
> converges, while chain retraining optimizes a *moving* objective and drifts (single-pass
> consistently beats iteration). We further show that a data-covariance (K-FAC) prior on
> the layer inverse gives no consistent advantage over the naive minimum-norm inverse, and
> we connect the per-layer solve to reduced-rank regression (LoRA) and alternating least
> squares. Our results delineate exactly when closed-form least-squares training is a
> viable, fast alternative to gradient descent (distillation/compression and warm-starting)
> and when it is not (label training through non-invertible operators).

(Trim to venue length; this is ~250 words.)

---

## 5. Contribution bullets (for the intro)

- A reproducible cross-architecture benchmark + code for closed-form per-layer LS training
  (7 architectures, standardized battery: train / distill / retrain / residual profile).
- The **distillation-exact / retrain-walled** dichotomy, with the wall size shown to track
  task dependence on the non-invertible operator (majority 22 pt vs pointer 42 pt).
- The **fixed- vs moving-objective** (block-coordinate) explanation tying together the
  single-pass≫iterated drift seen in all experiments.
- Honest negatives: K-FAC-prior inverse ≈ naive; moment-matching harmful past depth ~4.
- Connections: per-layer LS ≡ RRR (LoRA) and ALS; closed-form LoRA adaptation + OLS
  warm-start as the practically useful regimes (pending WP-C/WP-D results).

---

## 5b. Limitations to disclose (do not bury these)

- **The synthetic task's *label* shift is low-rank, so LoRA-rank conclusions are limited.**
  On the pointer task, end-to-end label LoRA reaches the teacher at **rank 1–2** (LR2b),
  while *activation-matching* distillation needs **rank ≈32–64** (LR1/LR2a). This is a clean
  and real dichotomy (matching activations is high-rank; matching a 2-class label is
  low-rank), but the *absolute* rank numbers are an artifact of a binary synthetic task with
  a ~100% ceiling. **Any claim about "how much rank a task needs" must be validated on a
  harder/real task** (lower achievable ceiling, multi-class, real data) before it can be
  stated as a finding. Until then, present the rank sweep as a *capacity-vs-objective*
  illustration, not a measurement of intrinsic task rank.
- **The LoRA base is random in LR1/LR2** — the adapter carries the entire transform, which
  is why near-full rank is needed there. The realistic pretrained-base→new-task setting
  (LR3) is the one whose rank/forgetting numbers should be cited; the random-base runs are a
  method stress test, not the LoRA use case.
- **Forgetting is only measured with the adapter merged in** (single combined model); the
  frozen base trivially recovers the original task once the adapter is detached.

---

## 6. What to add before submission (by venue)

- **Workshop / TMLR (recommended):** add ≥3 seeds on the headline numbers; finish WP-C
  (LoRA-RRR vs gradient LoRA) and WP-D (warm-start break-even) to give a *practical* hook;
  write the Related Work honestly against §3. Current MNIST/CIFAR/synthetic scope is
  acceptable here.
- **Top conference (only if a win appears):** one realistic setting (a real small
  transformer or a standard KD/compression benchmark) **plus** a concrete advantage —
  e.g. warm-start beating random+SGD in *total* compute, or closed-form LoRA competitive
  with gradient LoRA at lower cost. Without a win, do not target this tier.

---

## 7. Housekeeping

- Cite Vered Madar's inversion-free OLS solver as the numerical backend.
- Release code (already structured), pin seeds/configs, include the findings logs as an
  appendix table.
- Author/affiliation + an advisor co-author will help with venue credibility and the
  required rigor (seeds, baselines).
