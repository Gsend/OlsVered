# Architecture Coverage — Running Findings Log

Each row = one completed experiment. Updated by TeamLeader after each battery run.

---

## E1 — DeepMLP-10L, MNIST Classification

**Date:** 2026-05-19  
**Status:** GREEN (battery complete, all JSON + PNG artifacts saved)  
**Wall time:** ~6 min (CUDA, 15 epochs train + full battery)

### Numbers

| Battery item | Result |
|---|---|
| B-train | 98.23% (15 epochs, AdamW + cosine LR, 730K params) |
| B-distill (trained start, own output as target) | 95.37% |
| B-retrain-rand (random init, GT one-hot, kfac+mom) | 29.19% |
| B-retrain-rand (random init, GT one-hot, **naive+no-mom**) | **88.71%** ⚠️ |
| B-retrain-rand (random init, GT one-hot, kfac+no-mom) | 88.60% |
| B-retrain-rand (random init, GT one-hot, naive+mom-match) | 11.35% |

### Residual Profile Shape

- **B-distill:** monotonically decreasing deep→shallow (fc10=0.152 → fc1=0.739). Deeper layers fit well; noise accumulates as back-prop propagates toward input.  
- **B-retrain-rand:** U-shaped (fc10=0.595, minimum ~fc6=0.34, fc1=0.587). Middle layers easiest to fit from random init; both extremes are harder.

### B-iter (10 iterations, random init, kfac+mom)

Trajectory: 9.82% → **29.19%** → 11.35% → 11.35% (flat for iters 3–10)

Iter 1 lifts accuracy, iter 2 collapses it, then stagnation. Classic drift.

### B-subset (trained start, GT targets, layer sweep)

| Layers retrained | Acc | Delta |
|---|---|---|
| fc10 only | 98.21% | −0.02% |
| fc10+fc9 | 98.23% | 0.00% |
| fc10..fc8 | 98.05% | −0.18% |
| all 10 | 95.55% | −2.68% |

Monotonically decreasing — retaining more layers always degrades from trained start.

### Hypothesis Updates

| H | Assessment | Evidence |
|---|---|---|
| H1 (smooth depth scaling) | **YELLOW** — needs E0 comparison at matched settings. Single-pass kfac+mom: 29.19% from random. No-mom: 88.7% — not catastrophic. Cannot assess "smooth vs catastrophic" without 4-layer baseline. | E1 B-retrain-rand |
| H3 (K-FAC lift larger on structured) | **YELLOW** — K≈N in no-mom condition (88.60% vs 88.71%). K beats N in mom-match condition (29.19% vs 11.35%). Gap is mom-dependent, not purely architectural. | E1 B-K-vs-N |
| H5 (iteration drift) | **CONFIRMED** ✓ — iter 1 lifts, iter 2 collapses to near-random (11.35%), stagnation thereafter. Very clean. | E1 B-iter |
| H6 (subset sweep shape) | **CONFIRMED** ✓ — monotonically decreasing accuracy with more layers retrained from trained start. fc10-only ≈ baseline; all-10 = −2.68%. | E1 B-subset |

### Surprising Finding (open question added)

**Moment-matching reverses sign for 10-layer MLP.** With no-mom, both naive and kfac reach ~88.7% from random init. With mom-match, accuracy collapses to 11–29%. This is the opposite of the phase1 4-layer finding where mom-match helped. Likely cause: with 10 layers, 9 sequential moment-match corrections compound and over-constrain the target distribution, destroying signal rather than stabilizing it.

**New open question Q6:** At what chain depth does moment-matching switch from beneficial to harmful? Is the crossover between 4 and 10 layers?

---

## E2 — LNMLP-10L (MLP + LayerNorm+ReLU), MNIST Classification

**Date:** 2026-05-20  
**Status:** GREEN (battery complete, all JSON + PNG artifacts saved)  
**Wall time:** ~1.5 min (CUDA, skip-train; trained weights from prior run)

### Numbers

| Battery item | Result |
|---|---|
| B-train | 98.57% (15 epochs, AdamW + cosine LR) |
| B-distill (trained start, own output as target) | 97.12% |
| B-retrain-rand (kfac_a + no-mom) | **88.16%** |
| B-retrain-rand (naive + no-mom) | 84.80% |
| B-retrain-rand (naive + mom-match) | 74.11% |
| B-retrain-rand (kfac_a + mom-match) | 74.03% |

### Residual Profile Shape

- **B-distill:** fc1=0.593 → fc10=0.071 — monotone decreasing, deep-first. *Lower residuals than E1/plain-ReLU* (E1 fc1=0.739). LNReLU inversion is more exact than plain-ReLU mask inversion.
- **B-retrain-rand:** fc1=0.583 → fc9=0.328, fc10=0.411 — roughly decreasing from shallowest, with fc10 (output, no activation) higher than fc9.

### B-iter (10 iterations, random init, kfac+mom-match)

Trajectory: 9.43% → **74.03%** → 73.94% → 68.87% → 66.58% → 59.19% → 52.41% → 44.09% → 34.16% → 28.53% → **25.72%**

Iter 1 gives large jump to 74%, then MONOTONE DECLINE every subsequent iteration. This is distinct from E1's pattern (collapse-then-stagnation after iter 2). E2 never collapses to near-random but steadily degrades.

### B-subset (trained start, GT targets, layer sweep)

| Layers retrained | Acc | Delta |
|---|---|---|
| fc10 only | 98.51% | −0.06% |
| fc10+fc9 | 98.48% | −0.09% |
| fc10..fc8 | 98.50% | −0.07% |
| fc10..fc7 | 98.44% | −0.13% |
| fc10..fc6 | 98.37% | −0.20% |
| fc10..fc5 | 98.28% | −0.29% |
| fc10..fc4 | 97.82% | −0.75% |
| fc10..fc3 | 97.50% | −1.07% |
| fc10..fc2 | 97.25% | −1.32% |
| all 10 | 96.27% | −2.30% |

Extremely smooth monotone degradation. LNReLU does not increase sensitivity in the B-subset sweep compared to E1 (−2.30% vs E1's −2.68%).

### Hypothesis Updates

| H | Assessment | Evidence |
|---|---|---|
| H1 (smooth depth scaling) | **CONFIRMED** for MLP family — kfac+no-mom gives 88.16% vs E1's 88.6%. LNReLU does not break the chain. | E2 B-K-vs-N |
| H3 (K-FAC lift larger on structured) | **STRONGER** — K > N in no-mom condition (88.16% vs 84.80%), unlike E1 where K≈N. LNReLU's covariance structure benefits K-FAC more. | E2 B-K-vs-N |
| H4 (LNReLU invertibility) | **CONFIRMED** ✓ — LNReLU inversion is *more* accurate than ReLU mask: E2 distill residuals (fc1=0.593) lower than E1 (fc1=0.739). Exact LN inverse formula beats mask fallback. | E2 B-residual-profile |
| H5 (iteration drift) | **CONFIRMED** ✓ but pattern differs — E2 shows monotone decline vs E1's collapse-stagnation. Both drift, but LNReLU drift is gradual rather than catastrophic. | E2 B-iter |
| H6 (subset sweep shape) | **CONFIRMED** ✓ — monotone, very smooth (−2.30% over all 10 layers). LNReLU is equally benign as plain ReLU for subset-sweep degradation. | E2 B-subset |

### New Finding

**E2 B-iter monotone decline vs E1 collapse-stagnation.** LNReLU's exact LayerNorm inverse provides better gradient signal in the first iteration (74% vs E1's 29%), and the subsequent decline is gradual rather than collapsing. Hypothesis: LayerNorm normalizes activations, reducing the drift magnitude per iteration.

**New open question Q7:** Does LayerNorm prevent the collapse because it normalizes the drifting targets, or is the difference purely due to better first-pass invertibility (higher 1-shot accuracy leaves less room for improvement)?

---

## Q7 RESOLUTION — LayerNorm drift, matched-start (e2_q7_ln_drift.py)

**Date:** 2026-05-20
**Answer: NEITHER hypothesis. The B-iter gap was a moment-match × architecture
confound, not a LayerNorm stabilization effect.**

The original B-iter used kfac+mom for BOTH models. But mom-match wrecks
plain-ReLU at depth 10 far more than LN-ReLU (E1 kfac+mom=0.29 vs no-mom=0.89).
Running both models under BOTH conditions from random init (12 iters):

Full 4-config artifact (12 iters, drop = iter1 - final):

| config | iter1 | final | DROP | act-RMS i0->fin | diverged |
|--------|-------|-------|------|-----------------|----------|
| deepmlp no-mom | 0.886 | 0.834 | +0.052 | 0.089 -> 2100 | -- |
| deepmlp mom    | 0.292 | 0.114 | +0.178 | 0.089 -> 0.47 | -- |
| lnmlp no-mom   | 0.882 | 0.377 | +0.505 | 0.664 -> 1.22 | iter 8 |
| lnmlp mom      | 0.740 | 0.161 | +0.580 | 0.664 -> 1.19 | -- |

**By DROP (the right metric), LN is the LESS stable architecture in BOTH
conditions** (drop ~0.5-0.58 vs plain's 0.05-0.18). LN "looked better" in the
original B-iter only because it STARTED higher (0.74 vs 0.29) — mom-match does
not wreck LN's first pass the way it wrecks plain-ReLU's. The original reading
confused "ends at a higher absolute number" with "more stable." So LN does NOT
prevent collapse; it just started higher (mom-tolerance on the first pass).
At matched no-mom start LN drops 10x more than plain and diverges to NaN at
iter 8.

**Mechanism punchline (RMS column):** LayerNorm DOES bound activation scale
(~1.2 vs plain's runaway to 2100), but the model with bounded scale is the one
whose accuracy collapses. Bounding activation scale is ORTHOGONAL to
TP-iteration accuracy stability: plain MLP lets scale explode 4 orders of
magnitude yet holds accuracy (argmax scale-invariant); LN pins the scale but
loses half its accuracy. The thing LayerNorm controls is not the thing that
protects accuracy under iteration.

**Mechanism note:** accuracy drift and activation-scale drift are DECOUPLED.
- deepmlp no-mom: activation RMS explodes 0.09 -> 922 -> 2100 yet accuracy
  holds (argmax is scale-invariant).
- deepmlp mom: RMS stays bounded ~0.47 but accuracy collapses to chance.
So bounded activation scale neither implies nor is implied by accuracy stability.

**Hypothesis updates:**
| H | Assessment | Evidence |
|---|---|---|
| H5 (iteration drift) | CONFIRMED + nuanced | all configs drift; collapse vs gradual is driven by mom-match, not architecture |
| "LN stabilizes drift" (was Q7) | REFUTED | LN diverges under no-mom iteration; plain MLP is more robust |

Open Q7b: the LN+no-mom NaN divergence (near-zero-variance activation) is a
numerical-stability issue worth a guard if LN architectures are revisited.

---

## E3 — LeNet CNN, MNIST Classification

**Date:** 2026-05-20  
**Status:** GREEN (battery complete, all JSON + PNG artifacts saved)  
**Wall time:** ~9 min (CUDA, 15 epochs train + full battery)

### Numbers

| Battery item | Result |
|---|---|
| B-train | 99.22% (15 epochs) |
| B-distill (trained start, own output as target) | **28.0%** ⚠️ |
| B-retrain-rand (kfac_a + mom-match) | 9.93% (≈ random) |
| B-retrain-rand (naive + no-mom) | 7.46% (BELOW random!) |
| B-retrain-rand (naive + mom-match) | 10.97% |
| B-retrain-rand (kfac_a + no-mom) | 9.55% |

### Residual Profile Shape

- **B-distill:** fc3=0.00027 (perfect fit) → fc2=0.200 → fc1=0.223 → conv2=**0.820** → conv1=**0.705**. The conv layer residuals are catastrophically high. Target back-propagation through MaxPool is the bottleneck.
- **B-retrain-rand:** fc3=0.378 → fc2=0.236 → fc1=0.162 → conv2=0.563 → conv1=0.552.

### B-iter (10 iterations, random init, kfac+mom-match)

Trajectory: 9.80% → 9.93% → 13.80% → 8.78% → 15.37% → 17.81% → 14.63% → 9.80% → 9.80% → 11.35% → **17.64%**

Oscillates near random (9.8%–17.8%). No sustained improvement across 10 iterations. Complete chain failure.

### B-subset (trained start, GT targets, layer sweep deepest-first)

| Layers retrained | Acc | Delta |
|---|---|---|
| fc3 only | 98.52% | −0.70% |
| fc3+fc2 | 96.84% | −2.38% |
| fc3+fc2+fc1 | **35.13%** | −64.09% ⚠️ |
| + conv2 | 12.94% | −86.28% |
| all 5 | 10.56% | −88.66% |

Sharp collapse when fc1 is retrained — its target is back-propagated from pool1 (output of MaxPool), requiring approximate inversion through pool1 (nearest-neighbor upsample). The pool-inversion error accumulates and destroys the chain.

### Hypothesis Updates

| H | Assessment | Evidence |
|---|---|---|
| H1 (smooth depth scaling) | **RED for CNN** — TP fails catastrophically through MaxPool. From random init, best result is ≈9.82% random. MLP chain works at depth 10; CNN chain does not through pooling. | E3 B-retrain-rand, B-iter |
| H3 (K-FAC lift larger on structured) | **N/A** — all methods near random, no meaningful lift for any variant. | E3 B-K-vs-N |
| H5 (iteration drift) | **DIFFERENT REGIME** — oscillation near random, not drift-after-lift. The chain never achieves meaningful signal to drift from. | E3 B-iter |
| H6 (subset sweep shape) | **SHARP, NOT SMOOTH** — FC-only retrain is smooth (96.84%); adding fc1 (which requires pool inversion) causes 64% drop. Convolutiona+pooling architecture creates a phase transition, not a gradient. | E3 B-subset |

### Key Insight

**MaxPool is the chain-breaker.** The nearest-neighbor upsample approximation for MaxPool inversion introduces too much error (conv layer residuals 0.70–0.82). The FC layers alone work well (B-subset fc3+fc2 = 96.84%). The conv target back-propagation through the pool must be addressed (e.g., by storing argmax indices from the forward pass) for E3-class architectures to work.

**New open question Q8:** Can MaxPool inversion be made exact using stored argmax indices? Would this fix the chain for LeNet-class networks?

---

## E4 — MLP Autoencoder, MNIST Reconstruction

**Date:** 2026-05-20  
**Status:** GREEN (battery complete, all JSON + PNG artifacts saved)  
**Wall time:** ~12 min (CUDA, 20 epochs train + full battery)

### Numbers (MSE; lower = better)

| Battery item | Result |
|---|---|
| B-train | MSE 73.64 (20 epochs, AdamW + cosine LR) |
| B-distill (trained start, own output as target) | MSE 99.84 ⚠️ (worse than trained!) |
| B-retrain-rand (naive + no-mom) | MSE 212.16 (from random MSE=798.25) |
| B-retrain-rand (kfac_a + no-mom) | MSE 228.73 |
| B-retrain-rand (naive + mom-match) | MSE 382.52 |
| B-retrain-rand (kfac_a + mom-match) | MSE 355.33 |

### Residual Profile Shape

- **B-distill:** fc1=0.292, fc2=0.352, fc3=0.311, fc4=0.407, fc5=0.507, fc6=0.188 — no clear monotone pattern. fc5 has highest residual (bottleneck layer, 64-dim).
- **B-retrain-rand:** all residuals 0.37–0.67, highest at fc6 (output decoder layer).

### B-iter (10 iterations, random init, kfac+mom-match)

Trajectory: MSE 798.25 → **355.33** → 387.02 → 398.59 → 404.16 → 405.84 → 406.60 → 406.48 → 402.29 → 396.81 → **388.58**

Iter 1 gives large MSE reduction, then MONOTONE INCREASE (degradation) for iters 2–7, slight plateau. Same drift pattern as E2.

### B-subset (trained start, GT targets = input images, layer sweep)

| Layers retrained | MSE | Delta |
|---|---|---|
| fc6 only | 71.28 | −2.36 (slight improvement) |
| fc6+fc5 | 75.26 | +1.62 |
| fc6+fc5+fc4 | 76.75 | +3.11 |
| fc6+fc5+fc4+fc3 | 81.81 | +8.17 |
| +fc2 | 103.12 | **+29.48** ⚠️ |
| all 6 | 131.49 | +57.85 |

Sharp degradation when fc2 is added (encoder layers). Decoder-only retrain is benign; including encoder causes exponential MSE increase.

### Hypothesis Updates

| H | Assessment | Evidence |
|---|---|---|
| H1 (smooth depth scaling) | **YELLOW** — single-pass from random gives MSE=212 (naive/no-mom) vs trained=73.64. Substantial but not catastrophic gap. Continuous targets work differently from classification. | E4 B-retrain-rand |
| H3 (K-FAC lift larger on structured) | **REVERSED** — naive_no_mom (MSE=212) beats kfac_a_no_mom (MSE=229). K-FAC HURTS for autoencoder targets. Likely because the autoencoder output covariance is dominated by reconstruction artifacts, not activations. | E4 B-K-vs-N |
| H5 (iteration drift) | **CONFIRMED** ✓ — same pattern as E2: lift on iter 1, monotone degradation after. | E4 B-iter |
| H6 (subset sweep shape) | **CONFIRMED** for decoder-only; encoder layers cause sharp jump. Different breakpoint than E3 (no pooling, purely depth-related). | E4 B-subset |

### Key Insight

**Distillation from trained start DEGRADES for autoencoders (MSE=99.84 vs trained=73.64).** This is unique to E4 — classification experiments show improvement from trained start (E1: 95.37%, E2: 97.12%). For autoencoders, the self-distillation target (model's own reconstruction) introduces compounding reconstruction error: the model is asked to reproduce its own outputs (imperfect), not the original inputs. This creates a target that is systematically biased away from the ground truth.

**New open question Q9:** For autoencoder TP, should distillation use the original training inputs (as B-retrain-rand does) rather than the model's own reconstruction? If so, "B-distill" as currently defined is inappropriate for generative/reconstruction models.

---

## E3 RESOLUTION — conv diagnostics (e3_conv_diag.py, options 1-5)

**Date:** 2026-05-20
**Resolves Q8** (can MaxPool inversion be fixed?) and reframes the E3 RED.

The original E3 RED was caused by **two stacked bugs**, not a fundamental
CNN/TP incompatibility:

1. **Wrong max-pool inverse.** Nearest-neighbor upsample = min-norm
   pseudo-inverse, demands spatially-uniform 2x2 blocks (unachievable by conv).
   Fix: switch-aware max-unpool via stored argmax (`F.max_unpool2d`) — target
   at argmax, forward pre-pool value elsewhere.
2. **Stale-a_in forward sweep.** The CNN path (e3_lenet.py) never received the
   forward-sweep fix the MLP got — fit each layer against captured *original*
   activations, not the rebuilt-upstream output. Fix: re-propagate through
   rebuilt layers between OLS solves.

Effect on random-init retrain: 0.10 → 0.10 (bug1 only) → **0.52** (both fixed).

### Decisive characterization (the dividing line is target type, not conv)

| Path | conv2 res | acc |
|---|---|---|
| Opt1 round-trip (feature-level, student=teacher) | 2e-4 | **0.9922** |
| Opt5 feature-distill (RANDOM init, teacher activations) | 2e-4 | **0.9922** |
| Opt4a chain-back-prop distill from trained (logits) | 0.55 | 0.6516 |
| Opt4b chain-back-prop GT from trained | 0.55 | 0.4827 |
| Opt2 chain-back-prop GT from random | 0.38 | 0.5221 |

### Conclusion

- **Conv OLS + im2col machinery is fully sound** (Opt1/Opt5 recover the teacher
  EXACTLY from random init, all residuals ~1e-4 incl conv).
- The framework **DISTILLS CNNs perfectly** (feature-level targets = teacher's
  per-layer activations → exact recovery, closed-form, one OLS/layer).
- The framework **CANNOT retrain CNNs from labels** (chain-back-prop through
  max-pool caps ~0.5-0.65). Max-pool inversion is information-destroying; even
  switch-aware unpool can't recover what the FC chain pre-distorted (~20%).

This extends the MLP "distillation >> chain-back-prop retraining" finding;
pooling amplifies the gap. **Q8 answered:** switch-aware unpool *helps*
(0.10→0.52 with the stale-a_in fix) but does NOT make max-pool inversion exact
enough for full chain-back-prop retraining. Feature-level distillation is the
viable CNN use case.

**H3 (K-FAC > naive on structured): trending REFUTED** — no consistent K>N
advantage on conv.

---

## E5 — MiniResNet (BN-free, identity skips), CIFAR-10 Classification

**Date:** 2026-05-21
**Status:** GREEN (train + distill + retrain complete; CUDA)
**Q1 resolution:** decompose y = x + f(x); residual-branch target
t_f = ReLU⁻¹(t_out) − x_block_input; recurse input target through f's first
conv; identity skips only; max-pool reuses E3 switch-unpool; GAP inverted by
spatial broadcast. BN omitted (no norm-inversion primitive).

### Numbers

| Battery item | Result |
|---|---|
| B-train (30 ep, AdamW+cosine, width 32) | **71.24%** |
| B-distill (feature-level, random init) | **71.25%** (Δ +0.01 vs teacher) |
| B-retrain-rand (GT one-hot, kfac, skip-decomp chain) | **33.06%** (from 8.87% random, +24.2) |

### Residual Profile

- **B-distill:** all layers ~1e-3 (conv0=1.6e-3, blocks ~8e-4, fc=1e-3) → near-exact.
- **B-retrain-rand:** 0.39–0.82, no clean monotone shape; b1_conv2=0.82 and
  conv0=0.79 worst (skip subtraction + GAP broadcast inject error at the
  block-2→block-1 and input ends).

### Conclusion

- **Additive identity skips do NOT break feature distillation** — closed-form
  one-OLS-per-layer recovers the teacher exactly (71.25% vs 71.24%), same as the
  plain CNN. The skip is just replayed in the forward sweep.
- **Label chain-retrain partially works** (8.87%→33.06%) but lands well below
  distillation, consistent with the CNN pooling wall. The skip decomposition +
  GAP-broadcast + 6-deep chain accumulate residual; max-pool inversion (×2) is
  again the dominant lossy step.
- **H6 holds** (distill ≫ chain-retrain). H1 (smooth depth scaling) consistent.

---

## E6 — SmallUNet (BN-free 2-level U-Net), MNIST denoising autoencoder

**Date:** 2026-05-21
**Status:** GREEN (train + distill + retrain complete; CUDA)
**Architecture:** 1→16→32→32→16→1 channel progression; two MaxPool/MaxUnpool pairs;
channel-concat skips (dec2_in=64ch, dec1_in=32ch); final conv 16→1 (no activation).
Images padded 28→32 (2px border) for two clean 2x downsamples.

### Numbers

| Battery item | Result |
|---|---|
| B-train (20 ep, AdamW+cosine, lr=1e-3) | MSE = 0.0070 (from random 0.0698) |
| B-distill (feature-level, random init) | MSE = 0.0070 (Δ +0.0000 vs teacher) |
| B-retrain-rand (GT reconstruction, kfac, concat-split chain) | MSE = 0.0135 (from 0.0698 random, −0.0563) |

### Residual Profile

- **B-distill:** enc layers ~1.7e-4 to 7e-4 (near-exact); dec layers ~4e-3 to 1e-2
  (slightly worse — concat feeds are not in the student's causal path). Final conv 3.4e-3.
  All residuals <<1 → exact recovery of all 9 conv layers.
- **B-retrain-rand:** 0.15–0.40 across all layers; no clean monotone. final_conv=0.40 worst
  (first in chain, target is noisy reconstruction image directly). Decoder layers 0.15–0.32.

### Conclusion

- **Channel-concat skips are FULLY COMPATIBLE with feature-level OLS distillation** (Gate B1 PASS:
  teacher MSE=0.0070, distill MSE=0.0070, Δ=0.0000). The concat is simply replayed in the
  rebuilt-upstream forward sweep — no special handling needed beyond tracking enc1_post/enc2_post.
- **Chain back-prop retrain partially works** (0.0698→0.0135, 5.2x improvement) but lands 1.9x
  away from teacher (0.0135 vs 0.0070). Eight conv layers + two pool inversions + two concat
  decompositions accumulate sufficient chain noise to prevent full recovery. Consistent with E3/E5.
- **MaxUnpool adjoint** (_unpool_adjoint: gather at argmax) correctly inverts the decoder unpool
  step. _switch_unpool correctly inverts the encoder MaxPool. Both are needed and correct.
- **Concat-skip target decomposition**: dec_in split along channel axis → two targets summed at
  encoder feature (same pattern as E5 additive skip but over channel axis, not spatial).

---

## Q10 RESOLUTION — Pointer task isolates attention wall cleanly

**Date:** 2026-05-21
**Resolves Q10** (E8 retrain-probe confounded by task easiness)

The original E8 retrain-probe used the "majority" task, which random attention can partially solve
(uniform averaging suffices). Q10 asked: does a task REQUIRING content-based attention show a much
stronger wall?

**Result: yes. Pointer task exposes the full 42.4 pp attention wall.**

### Setup changes

- New `make_pointer_dataset` in tiny_transformer.py: token[0]=position pointer p; label=1 iff
  token[p] >= vocab//2. Requires content-based attention; uniform averaging gives ~50%.
- Required d_model=128 (64 was insufficient — model stuck at 0.59 with d_model=64). Root cause:
  positional encodings initialized with std=0.02 were too similar to distinguish positions;
  bumped to std=0.1, d_model=128 → convergence at epoch 14, final 99.98%.
- Per-task model sizes in e8_transformer.py: majority=64, pointer=128.

### Numbers

| Battery item | Majority task | Pointer task |
|---|---|---|
| B-train | 100.00% | 99.98% |
| B-distill | 100.00% (Δ=0.00) | 99.98% (Δ=0.00) |
| B-retrain-probe (attention-opaque) | 77.42% (wall=22.6 pp) | **57.63%** (wall=**42.4 pp**) |

All per-layer distill residuals <1e-3 (near-exact for all 13 Linears in pointer arch).

### Conclusion

- **Pointer task: full attention wall = 42.4 pp gap (99.98% → 57.63%).** The fitted
  {head, b1.Wo, b1.fc_in, b1.fc_out} are insufficient — without Q/K/V (both blocks), the model
  cannot route the pointer token's information to the output. The 57.63% is essentially at the
  label-independent baseline (random attention can't fetch position p).
- **Majority task wall is 22.6 pp (100% → 77%)** — confounded because random attention still
  passes majority statistics through; the pointer task cleanly isolates the V/K/Q dependency.
- **Q10 resolved.** The attention wall is real and severe (42.4 pp) once the task forces
  content-based routing. Prior 77% (majority) underestimated the wall by 2x.

---

## E8 — TinyTransformer (2-block pre-LN), synthetic majority sequence

**Date:** 2026-05-21
**Status:** GREEN for train+distill; retrain-probe rerun pending (CPU-device fix applied)
**Q2 resolution:** softmax attention treated as OPAQUE/non-invertible (like
max-pool). Q/K/V/O are OLS-fit Linears; attention + LN replayed forward, never
inverted. Distill copies teacher non-Linear params (embed/pos/LN affines) so OLS
isolates the 13 Linears.

### Numbers

| Battery item | Result |
|---|---|
| B-train (20 ep, AdamW+cosine) | **100.00%** (majority task is easy) |
| B-distill (feature-level, all 13 Linears) | **100.00%** (Δ +0.00 vs teacher) |
| B-retrain-probe (attention-opaque, label GT) | **77.42%** (from 44.47% random, +32.95) |

### Residual Profile

- **B-distill:** every Linear ~3e-4 to 1.7e-3 (Q/K/V slightly worse than O/MLP/head,
  ~1.6e-3). Near-exact recovery through both blocks.
- **B-retrain-probe:** fitted only top-block {Wo, fc_in, fc_out}; Q/K/V (both
  blocks), the entire block-0, AND the head were left random (head was
  propagated *through* for the target, never directly fit).

### Conclusion

- **Attention + LayerNorm are fully OLS-distillable in closed form** — with the
  residual stream matched at the input, fitting Q/K/V exactly reproduces the
  teacher's attention pattern, so ctx/Wo/MLP all fit exactly → 100% recovery,
  one pass. Transformer analogue of E3's exact conv round-trip. **Decisive.**
- **Attention wall is real but PARTIAL on this task (100% → 77%, 23-pt gap).**
  The gap is the part requiring Q/K/V / lower-block retraining, which label
  back-prop cannot reach through softmax. BUT the result is **confounded by task
  easiness**: majority is solvable by near-uniform averaging, which random
  attention already approximates, so features reaching the top MLP stay
  informative and one retrained projection layer recovers most accuracy.
  ⚠️ Do NOT read 77% as a clean wall measure. **Q10 opened:** rerun the probe on
  an attention-REQUIRING task (e.g. pointer/induction "fetch token at position
  k") where random attention destroys the signal — only then does the probe
  isolate the wall.
- Side observation: TP drove the top layers to satisfy targets defined *through*
  a random head and still hit 77% — last-layer fit is not even required here.

---
