# SESSION PART 7 — Phase 1 findings and Phase 2/3 test plan

This document closes out Phase 1 and lays out a concrete test plan for
Phase 2 and Phase 3. The central question driving Phase 1 was: **can K-FAC-A
regularized inversion beat naive pseudo-inverse for target-prop retraining of
a small MLP?** Across the experiments we ran, the picture is more nuanced than
"K beats N" or "N beats K". Two qualitatively different tasks have been
conflated under the label "retraining":

1. **Distillation** — reproducing the original model's intermediate behavior
   from a frozen reference, with the original's per-layer outputs as targets.
2. **Retraining** — recovering accuracy on a task using only GT labels (no
   intermediate teacher targets), starting from any initial weights.

Phase 1 results show that the framework solves (1) essentially perfectly, but
(2) is still open. Phase 2 and Phase 3 are designed to test (2) at meaningful
scale, on representation-rich models where the K-FAC-A prior actually has
something to bite on.

---

## 1. Phase 1 — what the experiments actually showed

### 1.1 Single-layer inversion drift

`phase1_mlp_mnist.py` ran each MNIST-MLP layer in isolation: pick a single
layer, perturb the post-activation target, invert through the activation, OLS-
solve, measure drift in cov_frob / mean / Wasserstein-2 / functional
acc-change. K wins 4/4 on distributional drift; the two methods are
indistinguishable functionally. Plausible explanation: the constraint
`X @ W^T + b = target_post` is identical for both N and K up to numerical
noise — the difference shows only in *distribution* of the recovered weights,
not in the per-batch pre-activation residual. **This was the cleanest
"K-prior helps" signal we got, but it's also the least task-relevant.**

### 1.2 Multi-step chain (4 layers, no GT)

`phase1_mlp_mnist.py --chain` propagates a single perturbation through 4
inversion+OLS steps. K wins ~2× on cov_frob at depth 4; functionally, the
naive variant drifts a noticeable amount further. Mean correction +
moment-matching narrows the gap on both methods. Mean correction alone is
*worse than nothing* in some configs (it shifts the activation distribution
without re-coloring it). **The K prior helps in multi-step but not by a
factor that would justify it as a standalone contribution.**

### 1.3 Multi-config retraining sweep

`phase1_retrain_via_targets.py`, on 16k MNIST-train samples, 5 iterations:

| config                | final acc | delta vs baseline |
|-----------------------|-----------|-------------------|
| baseline              |   0.9772  |                   |
| naive, no correction  |   0.2832  |   −0.6940         |
| naive, mean only      |   0.1243  |   −0.8529         |
| naive, moment match   |   ~0.85   |   ~−0.13          |
| kfac, no correction   |   ~0.78   |   ~−0.19          |
| kfac, mean only       |   ~0.74   |   ~−0.24          |
| kfac, moment match    |   0.9080  |   −0.0692         |

`kfac + moment_match` is the clear winner at 90.8%. **But every multi-pass
variant degrades from its single-pass value** — iter accuracies go
0.91→0.90→0.87→0.80→…, which is exactly the divergence the user identified
("the iterations are fundamentally wrong"). The iteration bug is that at
iter k>1 the chain back-prop runs through the already-updated work model, so
fresh noise compounds rather than self-correcting toward a fixed point.
**Single-pass is the meaningful number here.**

### 1.4 Final-layer refit and forward sweep

`phase1_final_layer_refit.py`, same 16k train samples:

| variant                                          |  acc   |
|--------------------------------------------------|--------|
| baseline (original cached model)                 | 0.9772 |
| TP retrain (kfac + moment match, single pass)    | 0.9080 |
| TP retrain + fc4-only OLS refit against original | 0.9345 |
| TP retrain + forward sweep (fc1→fc4 vs original) | 0.9772 |

The forward sweep achieves **exact** baseline recovery. Each layer's
target_residual on the sweep is ~2e−4 (i.e. a perfect linear map exists
between work-model intermediate activations and original-model intermediate
activations, given 16k samples).

This is the moment the distillation/retraining distinction crystallized:

> Forward sweep uses the ORIGINAL model's intermediate outputs as the per-
> layer target. That isn't "retraining" — it's distilling the frozen
> reference into a new set of weights. With ample samples and a linear OLS
> head per layer, this collapses to: "find weights that make the new model's
> internal representation match the old model's." It's a function-
> reproduction problem, and we just confirmed the framework solves it.

The numbers we don't yet have are the ones that matter for the original
research question: **how high can the framework go using GT labels only?**

### 1.5 The iteration-divergence bug

Independent of (1)-(4): multi-pass `retrain_via_target_prop` diverges because
the chain back-prop is computed against the work-model's current state at
each iteration. As earlier layers shift under OLS, the targets handed to
later layers shift too, producing a feedback loop. The fix is conceptually
small (freeze the target chain after iter 1, or back-prop targets only
through the original model) but has not been merged. **All retraining
numbers above use n_iterations=1 implicitly or as the best-iter point.**

---

## 2. Two hypotheses to test next

The Phase 1 experiments answer one question and reframe another:

| hypothesis                                       | Phase 1 status                              |
|--------------------------------------------------|---------------------------------------------|
| H-D (distillation)                               | confirmed for 4-layer MLP at 16k samples    |
| H-R (retraining)                                 | open — no isolated GT-label test run yet    |

**H-D — Distillation hypothesis.**
*"The framework can reproduce a frozen reference model's behavior via per-
layer OLS, even without GT labels, as long as intermediate targets come from
the reference."* This is what the forward sweep demonstrated. K-FAC-A prior
is not necessary here because the linear system is well-determined with
enough samples; the prior matters most when (a) samples are scarce or (b)
the per-layer linear map doesn't exist exactly and damping needs to be
principled.

**H-R — Retraining hypothesis.**
*"The framework can recover task accuracy starting from damaged or random
initial weights, given only GT labels as the deepest-layer target, by
propagating GT-derived targets through the chain and OLS-fitting each
layer."* This has **not** been measured. Until it is, we can't claim the
target-prop+OLS framework is a viable retrainer (as opposed to a viable
distiller).

The newly-added `diagnostic/runners/phase1_gt_retrain.py` is the test for
H-R on the MLP-MNIST regime. It builds one-hot logit targets ±margin from
the labels, runs the TP retrainer with those as the deepest a_post, and
compares:

- starting model (cached, or damaged via --noise-std, or random init)
- TP retrain with GT targets
- TP retrain + fc4 refit against GT (a sanity check that the head alone
  isn't the bottleneck)
- fc4-only OLS refit against GT on the *starting* model (control: what does
  a final-layer linear probe alone do without any TP propagation?)

The phase1_gt_retrain runs are the bridge between Phase 1 distillation
findings and the Phase 2/3 plan. If H-R fails on the MLP (i.e. GT-target TP
retrain converges to substantially below baseline accuracy and no better
than a final-layer probe), that's important to know before investing in
BERT-scale infrastructure.

---

## 3. Phase 2 — single BERT block

### 3.1 Goal

Test H-D and H-R on a single transformer block in isolation. The block is
representation-rich (multi-head attention + FFN + LayerNorms) and is the
unit of retraining most analogous to existing OlsSMLayerRetrainer's BCD
loop, but it lives inside a real pretrained model so the activations have
real structure.

### 3.2 Setup

- Backbone: pretrained `bert-base-uncased` from HuggingFace.
- Task: SST-2 (GLUE) — small, fast eval, accuracy is the headline metric.
- Block under test: BERT layer index L (start with L=11, the last
  encoder block; later sweep L=8, 4, 2 to see how depth changes things).
- Inputs: contextualized hidden states feeding into block L (extracted
  once, cached on disk).
- Reference targets: outputs of block L from the original frozen BERT (for
  H-D) and SST-2 GT labels (for H-R).

### 3.3 H-D test (distillation, block-level)

For each of the four linear sub-layers inside the block (Q, K, V, attention
output, FFN-in, FFN-out — six total, but Q/K/V can be jointly fit since
they share inputs), run target-prop+OLS in both methods (N and K), with and
without moment matching, with the reference block's per-sublayer outputs as
the OLS target. Measure:

- per-sublayer target_residual
- post-block output drift vs the original block's output
- end-to-end SST-2 accuracy (block in question is the only one replaced;
  rest of BERT frozen)
- sample efficiency: residual + acc at {1k, 4k, 16k, 64k} training
  examples worth of hidden states

**Pass criterion for H-D at the block level:** with 16k hidden states, the
retrained block reaches end-to-end accuracy within 0.5 absolute points of
the frozen reference. This is the analog of "fc4 sweep recovered exact
baseline" but at a single transformer block.

**Where K should help:** the FFN inner activation has very non-Gaussian /
low-rank structure; the activation covariance Σ_a is more informative there
than for a fully-connected ReLU MLP, so the K prior should yield a real
sample-efficiency gain over N at the 1k-4k sample budget. **This is the
primary signal we want from Phase 2.** If K and N are within 0.1 acc points
at every sample budget, then K-FAC-A is not earning its place in the system
and we revert to the simpler N variant.

### 3.4 H-R test (retraining, block-level)

The block doesn't directly output logits, so "GT labels only" at the block
level means: take the SST-2 classification head + LayerNorm above the
target block, freeze them, and treat the *post-block hidden state that
would maximize logit-correct minus logit-incorrect* as the deepest target.
This is constructed as `target_hidden = original_hidden + α · grad`, where
the gradient is of cross-entropy w.r.t. the post-block hidden state, and α
is sized so that the resulting hidden state produces correct predictions on
mispredicted examples (or amplifies margin on correct ones).

Then run TP+OLS as in 3.3 but with this GT-derived `target_hidden` as the
top of the propagation chain.

**Pass criterion for H-R at the block level:** retrained block (alone)
recovers within 1 acc point of the frozen baseline, *and* outperforms a
pure final-layer probe (logistic regression on frozen block input) by ≥1
absolute point.

This is the test for whether the framework can do real retraining or only
distillation, separated from the question of "is OLS-without-inversion fast
enough."

### 3.5 Sample-efficiency curves

Both H-D and H-R should be run as **curves**, not single points:
acc vs # samples at {1k, 4k, 16k, 64k, full}, with K, N, K+momentmatch,
N+momentmatch. The OlsSM/Vered story is fundamentally about sample-
efficient OLS; if the curves are indistinguishable at all sample budgets,
we have nothing to say. The expected shape is: K and N converge at high
samples; K dominates at low samples; moment-matching dominates K at
intermediate samples. If the actual shape differs, that's the finding.

### 3.6 Phase 2 deliverables

- `diagnostic/runners/phase2_bert_block.py`
- Cached SST-2 hidden states per block (one-time precompute)
- Sample-efficiency curve plots (per-block + end-to-end acc)
- JSON dump of per-config metrics
- A go/no-go decision on Phase 3 based on:
  - H-D pass at L=11 → go
  - H-D fail at L=11 → stop, debug
  - H-R pass at L=11 → go to Phase 3 retraining track
  - H-R fail at L=11 → Phase 3 is distillation-only (still useful, but
    smaller claim)

---

## 4. Phase 3 — BERT last-N layers, comparison to BCD

### 4.1 Goal

Compare target-prop+OLS retraining of the last 2 / 4 / 6 BERT blocks
against the existing `OlsSMLayerRetrainer` BCD benchmark on the same task.
This is where Vered-without-inversion can plausibly differ from standard
OLS solvers — at the multi-block scale, K-FAC-A regularization and the
inversion-free SPD solve both become non-trivial cost/quality knobs.

### 4.2 Setup

- Backbone: same `bert-base-uncased`, SST-2.
- Blocks under retrain: last 2, last 4, last 6 (sweep).
- Per-block sub-layer set: same as Phase 2.
- Methods compared:
  1. Target-prop + OLS chain (best variant from Phase 2)
  2. BCD via `OlsSMLayerRetrainer` (existing benchmark)
  3. Plain GD fine-tuning of the last N layers (control)
  4. Single-pass linear probing on the top layer (control)
- GT label use: Phase 3 is retraining, not distillation — use GT labels for
  the deepest target, not original block outputs.

### 4.3 Comparison axes

| axis                 | what we measure                         |
|----------------------|-----------------------------------------|
| accuracy             | SST-2 dev acc                           |
| sample efficiency    | acc at 1k / 4k / 16k / 64k samples      |
| wall time            | seconds per retrain, single GPU         |
| memory               | peak VRAM during retrain                |
| determinism          | std across 5 seeds                      |

### 4.4 Pass criteria

The framework is interesting in Phase 3 if **at least one** of the
following holds against the BCD baseline at matched wall time:

- ≥0.5 acc points better at any sample budget
- equal acc at ≥2× sample efficiency
- equal acc at ≥2× memory savings

If none holds, the recommender / fine-tuning use case for Vered-style OLS
is the more promising direction and Phase 3 ends here with a written
negative result.

### 4.5 Phase 3 deliverables

- `diagnostic/runners/phase3_bert_lastN.py`
- Integration of `OlsSMLayerRetrainer` as a baseline runner with identical
  data path and eval harness
- Comparison plots (acc vs samples, acc vs wall time)
- Final write-up with pass/fail call per pass criterion

---

## 5. Open issues to fix before Phase 2

These are blockers, in priority order:

1. **TP+GT on the CLEAN trained model** *(quick and important — run first).*
   `python -m diagnostic.runners.phase1_gt_retrain --noise-std 0`. The
   starting model is the cached 0.9772 MLP. The three outcomes each carry
   distinct meaning:
   - acc > 0.9772 → framework finds a strictly better optimum than backprop
     under GT signal. Strong positive for the framework as a retrainer.
   - acc ≈ 0.9772 → framework is consistent with the existing optimum, the
     minimum bar for "this can retrain."
   - acc < 0.9772 → chain-backprop noise dominates the GT signal even from
     optimal start. Framework is distillation-only on this regime.

   Also worth a `--gt-margin 20` run, since the default ±5 logit target is
   *softer* than the trained model's actual logits on confident examples.
2. **Iteration divergence in `retrain_via_target_prop`** — chain back-prop
   uses updated work-model weights at iter k>1; needs to freeze the target
   chain (or back-prop through frozen reference) before multi-pass is
   meaningful. Until fixed, all multi-pass results are misleading.
3. **GT-label retraining from damaged / random init** — `phase1_gt_retrain`
   with `--noise-std 0.05` and `--random-init`. Tells us the framework's
   "rescue range" — from how far away can TP+OLS recover a working model
   using only GT labels?
4. **K-FAC-A sample-efficiency curve on MLP-MNIST** — we have point
   estimates at 16k samples; we need 1k / 4k / 16k / 64k curves to confirm
   the prior helps at low samples in this regime before scaling to BERT.

---

## 6. Decision tree summary

```
H-D on MLP (Phase 1 §1.4)
    PASS → known: framework distills well.

H-R on MLP (phase1_gt_retrain, next run)
    PASS → proceed to Phase 2 with H-R as primary metric.
    FAIL → proceed to Phase 2 but with H-D as primary metric and H-R as
           exploratory.

H-D on BERT block (Phase 2 §3.3)
    PASS → proceed to Phase 3 with sample-efficiency curve goal.
    FAIL → stop and debug; the framework not reproducing a real
           transformer block is a serious negative.

H-R on BERT block (Phase 2 §3.4)
    PASS → Phase 3 includes both H-R and H-D tracks.
    FAIL → Phase 3 is distillation-only; reposition the story around
           sample-efficient distillation.

Phase 3 vs BCD baseline (§4.4)
    Any criterion met → write up framework as a retrainer.
    None met → write up as a negative result and pivot to ALS-for-
               recommenders track.
```

---

## 7. What this does NOT settle

- ALS for recommendation engines — orthogonal; OlsSMKFAC-style benefit
  there does not depend on these MLP/BERT results.
- The recency-weighted ALS analysis (SESSION_PART4) stands on its own.
- The retrainer EV analysis (SESSION_PART5) is the higher-level story that
  Phase 2/3 either supports or undermines, but the EV calculation itself is
  independent.

The path that closes the loop on Phase 1 → Phase 2 → Phase 3 is the path
that turns "we have a closed-form OLS without inversion" into "and here's a
real model-retraining setting where that mattered." Phase 1 told us that's
not the MLP-MNIST distillation case (any OLS does that). Phase 2 will tell
us whether it's the BERT block case. Phase 3 tells us whether it survives
contact with a serious baseline (BCD).
