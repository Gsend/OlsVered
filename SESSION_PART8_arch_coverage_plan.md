# Architecture Coverage Experiment Protocol

**Objective.** Characterize TP+OLS retraining behavior across architectures, input modalities, and output target types. Build broad intuition rather than optimize a single application.

**Status.** Phase-1 MLP-MNIST results in hand (see SESSION_PART7). This protocol scopes the next round.

---

## 1. Hypotheses

| ID  | Statement | Falsifier |
|-----|-----------|-----------|
| H1  | Single-pass TP+OLS retraining accuracy degrades smoothly with chain depth, not catastrophically. | Random-init acc drops >20 points between 4-layer and 10-layer MLP. |
| H2  | In-chain LayerNorm reduces per-layer chain-back-prop residual (stabilizes intermediate distributions). | LN-MLP residuals ≥ plain-MLP residuals at matched depth. |
| H3  | K-FAC-A prior shows larger lift over naive pseudo-inverse on architectures with structured (non-Gaussian) activations (conv, attention) than on plain MLPs. | Method-K vs Method-N gap is no larger on CNN/transformer than on MLP. |
| H4  | Continuous high-dim targets (reconstruction, dense per-pixel) carry more retraining signal than one-hot logits at matched n_samples. | Random-init recon accuracy ≤ random-init classification accuracy at matched per-output-dim sample budget. |
| H5  | The iteration-drift pattern (inner residuals collapse, deepest residual grows, acc declines) generalizes across architectures. | Drift pattern absent or reversed in any architecture tested. |
| H6  | Layer-subset sweep curve shape (last-layer-only > pairs > full-chain) generalizes across architectures. | Any architecture shows full-chain > last-layer-only on retraining. |

---

## 2. Experimental Design

### 2.1 Coverage matrix

Rows = architectural shape; columns = task / input modality. ★ = unique structural lesson; · = redundant w/ another cell.

|                       | MNIST class. | CIFAR class. | Recon/dense | Sequence |
|-----------------------|:--:|:--:|:--:|:--:|
| Deep MLP (10L)        | ★ E1 | · | ★ E4 | — |
| MLP + LayerNorm       | ★ E2 | · | · | — |
| LeNet CNN             | ★ E3 | · | · | ★ E7 |
| Small ResNet          | · | ★ E5 | — | — |
| Small U-Net           | — | — | ★ E6 | — |
| Tiny transformer      | — | — | — | ★ E8 |
| Tiny RNN/GRU          | — | — | — | (optional) E9 |

### 2.2 Experiment list

| #   | Network | Task | Input shape | Output | Key lesson |
|-----|---------|------|-------------|--------|------------|
| E0  | MnistMLP (control, done)     | MNIST classification | (1,28,28)→784 | 10-d logits | baseline |
| E1  | DeepMLP-10L                  | MNIST classification | (1,28,28)→784 | 10-d logits | depth scaling |
| E2  | MLP + LayerNorm              | MNIST classification | 784 | 10-d logits | normalization in chain |
| E3  | LeNet CNN                    | MNIST classification | (1,28,28) | 10-d logits | conv + max-pool |
| E4  | MLP autoencoder              | MNIST reconstruction | 784 | 784-d continuous | high-dim continuous target |
| E5  | ResNet-8 (CIFAR)             | CIFAR-10 classification | (3,32,32) | 10-d logits | additive skip + RGB |
| E6  | Small U-Net                  | shape-segmentation or denoising | (1,32,32) | per-pixel mask | concat skip + dense target |
| E7  | 1D-CNN                       | synthetic time-series regression | (1, T) | scalar or vector | 1D conv + sequence input |
| E8  | Tiny transformer (2 blocks)  | SST-2-tiny or synthetic seq class | (T, d_model) | binary class | attention bilinear |
| E9  | Tiny GRU                     | sequence regression | (T, d_in) | scalar | recurrence + weight sharing |

### 2.3 Standardized battery (run on every experiment)

| Battery item | Inputs | Outputs | Notes |
|--------------|--------|---------|-------|
| **B-train**  | dataset, architecture | trained weights cache, backprop val acc | reference baseline |
| **B-distill** | trained weights, deepest target = trained-model output | acc, per-layer residuals | tests H-D |
| **B-retrain-rand** | random-init weights (seed=42), deepest target = GT/label/recon | acc, per-layer residuals | tests H1, H4, H6 |
| **B-residual-profile** | result of B-distill and B-retrain-rand | per-layer residual curve (deepest→shallowest) | primary cross-arch artifact |
| **B-iter** | random-init weights, 10 iterations of full-chain TP+OLS | acc trajectory, residual trajectory per layer | tests H5 |
| **B-subset** | trained weights, target_layers ∈ {deepest only, deepest+second, ..., full chain} | acc per subset | tests H6 |
| **B-K-vs-N** | random-init, single pass, method ∈ {naive, kfac_a} × {no_mom, mom_match} | 4 acc points | tests H3 |

Each battery item writes JSON to `benchmark/results/diagnostic/arch_coverage/{exp_id}/{battery_item}.json`.

### 2.4 Common hyperparameters

| Param | Value |
|-------|-------|
| n_samples (retrain) | 16,384 (or full train if smaller) |
| seed | 42 |
| gt_margin (classification) | 5.0 |
| ols_lambda | 1e-4 |
| ols_eps (inversion) | 1e-4 |
| iterations (B-iter) | 10 |
| eval | held-out test split, full size |

### 2.5 Per-experiment framework changes

| #   | New code required |
|-----|-------------------|
| E1  | `DeepMLP(n_layers)` class. No framework changes. |
| E2  | `LNMLP` class. Add LN handling in `collect_activations` and `invert_layer`. |
| E3  | Conv layer support in capture + OLS solve (im2col path). Max-pool inversion strategy. |
| E4  | Continuous-target variant of `make_gt_logit_target` (identity / no-op for autoencoder). |
| E5  | Residual-block handling: treat block as single inversion unit OR implement implicit inversion of `y = x + f(x)`. CIFAR-10 data loader. |
| E6  | Concat-skip handling (channel-split inversion). Per-pixel target shape. |
| E7  | 1D conv layer (special case of E3 with kernel along time). |
| E8  | Multi-head attention as opaque non-invertible op; OLS on Q/K/V/O projections; LN handling. |
| E9  | RNN-cell chain unrolling for per-timestep capture. |

---

## 3. Team Structure

### 3.1 Roles

| Role | Subagent type | Owns | Does not own |
|------|---------------|------|--------------|
| **TeamLeader** | general-purpose | hypotheses, decisions, plan updates, cross-experiment synthesis, go/no-go calls | implementation, debugging |
| **Scientist**  | general-purpose | model classes, training pipelines, battery execution, error handling, raw results | hypothesis updates, scope changes, pivots |

Both agents share read access to `diagnostic/`, `benchmark/results/diagnostic/arch_coverage/`, and the running findings log.

### 3.2 TeamLeader responsibilities

1. Read latest results JSON for the most recent completed experiment.
2. Cross-reference against hypotheses; mark each falsifier as triggered / not triggered.
3. Append a row to `arch_coverage_findings.md` (running log; one row per experiment).
4. Decide next action: (a) advance to next experiment in queue, (b) re-run with modified params, (c) modify hypothesis set, (d) escalate to user.
5. Update this protocol document when hypotheses are added/dropped or when the experiment queue changes.
6. **Must not** modify code in `diagnostic/` or write any `_runner` modules — that is Scientist's domain.

### 3.3 Scientist responsibilities

1. Implement the model class for the assigned experiment (in `diagnostic/models/`).
2. Implement or extend the runner script (in `diagnostic/runners/arch_coverage/`).
3. Train the baseline via backprop; cache weights under `benchmark/weights/arch_coverage/`.
4. Run the standardized battery; produce JSON outputs under the experiment's results directory.
5. Generate residual-profile plot per experiment (PNG, saved alongside JSON).
6. On error: triage (timeout, OOM, numerical NaN, shape mismatch), attempt up to 2 fixes, then escalate.
7. **Must not** declare a hypothesis confirmed/refuted, change the experiment queue order, or modify protocol.

### 3.4 Handoff contract

```
TeamLeader -> Scientist:
  { experiment_id, network_spec, battery_items, hyperparams, seed }

Scientist -> TeamLeader:
  { experiment_id, status ∈ {complete, failed, partial},
    artifacts: [json_paths, plot_paths],
    issues: [error_messages, retried_fixes],
    wall_time_minutes }

TeamLeader -> User (on each completed experiment):
  short report ≤200 words:
    - acc numbers (B-distill, B-retrain-rand)
    - residual profile shape (1-line description)
    - hypothesis updates if any
    - next experiment queued
```

### 3.5 Communication artifacts

| Artifact | Maintained by | Path |
|----------|---------------|------|
| Running findings log | TeamLeader | `arch_coverage_findings.md` (one row per experiment) |
| Experiment queue state | TeamLeader | `arch_coverage_queue.md` |
| Raw results | Scientist | `benchmark/results/diagnostic/arch_coverage/{exp_id}/*.json` |
| Plots | Scientist | `benchmark/results/diagnostic/arch_coverage/{exp_id}/*.png` |
| Runner code | Scientist | `diagnostic/runners/arch_coverage/{exp_id}.py` |
| Model code | Scientist | `diagnostic/models/{model_name}.py` |
| This protocol | TeamLeader (revisions only) | `SESSION_PART8_arch_coverage_plan.md` |

---

## 4. Execution Phases

| Phase | Experiments | Gates / decision rules | Expected wall time |
|-------|-------------|------------------------|--------------------|
| **P-A: Free** (no framework changes) | E1, E4 | If both fail H1/H4 → pause, reconsider framework. Else: proceed. | ~1 day |
| **P-B: Small extensions** | E2, E7 | If LN changes break captures → fix capture API, retry. | ~2 days |
| **P-C: Conv layers** | E3 (+ optional re-run of E7 in 1D conv form) | If conv OLS solve numerically unstable → halt and add diagnostics. | ~3 days |
| **P-D: Skips and dense targets** | E5, E6 | Run in parallel if Scientist capacity allows. | ~4 days |
| **P-E: Attention** | E8 | Treat as Phase-2 gate; if fails, do NOT proceed to E9. | ~3 days |
| **P-F: Recurrence (optional)** | E9 | Skip if Phases A-E sufficient for intuition. | ~3 days |

Total ≤ ~16 working days under sequential execution; less under partial parallelism.

---

## 5. Decision Rules

### 5.1 Per-experiment outcome classification

| Class | Definition | Action |
|-------|------------|--------|
| GREEN | Battery completes; all hypotheses' falsifiers untriggered. | Proceed. |
| YELLOW | Battery completes; ≥1 falsifier triggered. | TeamLeader updates hypothesis; proceed with annotation. |
| RED | Battery fails (numerical, OOM, ≥2 fix attempts unsuccessful). | Halt sequence; escalate to user with diagnostics. |

### 5.2 Cross-experiment synthesis triggers

TeamLeader writes a mid-phase synthesis to `arch_coverage_findings.md` whenever:

- Two consecutive YELLOW experiments occur.
- Any RED occurs.
- An entire phase completes.
- A hypothesis is added or dropped.

### 5.3 User escalation triggers

- Any RED outcome.
- Three or more YELLOW outcomes in a single phase.
- Wall-time budget exceeded by >50% for a phase.
- Scientist requests scope expansion (e.g., new framework primitive needed beyond the per-experiment list above).

---

## 6. Deliverables

| Item | Format | Audience |
|------|--------|----------|
| Per-experiment results | JSON + PNG (per battery item) | User, future agents |
| Running findings log | `arch_coverage_findings.md` | User |
| Cross-architecture comparison plot | PNG: residual profile overlay, all experiments | User |
| Cross-architecture comparison plot | PNG: retraining acc vs distillation acc, all experiments | User |
| Final synthesis | `SESSION_PART9_arch_coverage_findings.md` (written by TeamLeader on P-F completion or early halt) | User |
| Updated hypothesis table | Section in PART9 with each H1-H6 marked confirmed/refuted/inconclusive | User |

---

## 7. Out of Scope

- Specific application deployment (BERT-on-SST-2, etc.).
- Wall-time optimization or GPU efficiency studies.
- Comparison to backprop / fine-tuning baselines beyond the cached baseline acc.
- Statistical significance testing across seeds (single seed=42 by default; multi-seed only if a YELLOW outcome demands it).
- Implementation of `OlsSMLayerRetrainer` BCD as a comparison baseline.

---

## 8. Open Questions (for TeamLeader to track)

| ID | Question | Notes |
|----|----------|-------|
| Q1 | For ResNet/U-Net: invert each residual block opaquely, or decompose into `y = x + f(x)` with implicit fixed-point? | Decide before E5. |
| Q2 | For transformer (E8): is attention treated as one opaque layer, or do we OLS-fit Q/K/V/O separately? | Decide before E8. |
| Q3 | For LayerNorm (E2): is the chain target the pre-LN or post-LN activation? | Affects how invert_activation handles LN. |
| Q4 | For autoencoder (E4): what does "GT" mean — the input itself, or an external clean target? | Default: input itself. |
| Q5 | When does it become worth multi-seed runs vs single-seed coverage? | TeamLeader call. |
