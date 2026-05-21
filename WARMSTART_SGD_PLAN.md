# OLS Warm-Start → SGD: Convergence-Speedup Experiment Protocol

**Objective.** Measure how much SGD compute an OLS initialization saves to reach a
target accuracy, **amortizing the cost of the OLS pass itself**, and report the
**break-even** point. The product is not standalone OLS accuracy — it is cheap
initialization that shortens SGD. The headline number is "total compute to target,
OLS-init vs random-init."

**Relation to prior findings.** OLS-from-labels single-pass quality is
architecture-dependent (DeepMLP ~0.89, LeNet ~0.10–0.52 through pooling, transformer
label-probe limited); OLS feature-distillation is near-exact everywhere. So warm-start
value is expected to track init quality: large for MLP / distillation, uncertain for
conv-through-pool / attention. Also note: applying full TP+GT to an *already trained*
MLP HURT it (−32.6%) — a warning that the OLS basin can be high-accuracy but fragile
for SGD (saturated units, large weights from ±margin targets).

---

## 1. Metric definitions (get these right or the result is meaningless)

- **Cumulative compute** = OLS pass cost + SGD cost, in FLOPs (preferred) or wall-clock.
  - OLS pass ≈ one forward over N capture samples + per-layer solves
    (`XtX`: `d_in²·N` to form, `~d_in³` to factor). Account for it explicitly; it is
    NOT free, and for wide layers it dominates (the `d_in²` wall).
  - SGD ≈ `3 × forward_FLOPs × batch × steps` (fwd+bwd ≈ 3× fwd).
- **Steps/epochs-to-target** at thresholds: 90%, 95%, 99% of the *random-init final*
  accuracy (and of each condition's own final).
- **Final-accuracy delta**: does SGD from the OLS basin reach a better/worse optimum?
- **Break-even accuracy**: the lowest target X at which
  `OLS_cost + warmstart_SGD_to_X < random_SGD_to_X`. If break-even is *below* the
  achievable accuracy, OLS init pays off; if *above*, it does not.
- **Init-conditioning diagnostics** (to explain *why*): dead-ReLU fraction, weight Frob
  norm, per-layer activation RMS, and first-step gradient norm at init.

---

## 2. Conditions

**Init (rows):**
| Tag | Init | Needs |
|-----|------|-------|
| R   | random (Kaiming/default) | — (baseline) |
| O-label | OLS label-retrain, single pass (existing `gt_target_retrain`) | labels only |
| O-distill | OLS feature-distill, single pass (existing feature-distill sweep) | a trained teacher |

**Architectures (columns; span the OLS-init quality range):**
- DeepMLP-10L, MNIST — OLS-label strong.
- LeNet CNN, MNIST — OLS-label weak (pooling wall), OLS-distill strong.
- TinyTransformer, synthetic — OLS-distill strong, OLS-label limited.

**SGD:** the model's standard optimizer (AdamW or SGD+momentum), same data, same batch
size across conditions; vary ONLY init and LR schedule.

---

## 3. Confounds to control (these will flip the result if ignored)

1. **LR schedule is per-condition.** Random init wants warmup + high peak LR; a warm
   start usually wants lower LR / little-to-no warmup, or the first high-LR steps erase
   the init. **Sweep peak LR × warmup length for every condition** and report each
   condition at its own best schedule. Reusing the random schedule on the OLS init is
   the #1 way to falsely measure "no gain."
2. **SGD seed noise.** Convergence speed is noisy — ≥3 SGD seeds per cell; report
   mean ± std for steps-to-target.
3. **Honest OLS amortization.** Include capture-forward + solve FLOPs on the x-axis.
4. **Matched data budget** across conditions.

---

## 4. Hypotheses

| ID | Statement | Falsifier |
|----|-----------|-----------|
| W1 | OLS-init reduces steps-to-target vs random, most on MLP, least on conv-through-pool. | No steps-to-target reduction on DeepMLP at its best LR schedule. |
| W2 | The gain shrinks/reverses if the random-init LR schedule is reused on the OLS init. | OLS-init gain identical across schedules. |
| W3 | O-distill init + short SGD beats standard KD (random + SGD on teacher targets) in total compute to matched acc. | KD reaches matched acc in ≤ the O-distill total compute. |
| W4 | Final accuracy of OLS-init+SGD ≥ random+SGD (equal or better basin). | OLS-init+SGD converges to lower final acc under all schedules (bad basin). |
| W5 | Break-even accuracy is below the achievable accuracy for MLP / distill regimes (pays off), possibly above it for conv-label (does not). | Break-even above achievable acc even for DeepMLP / distillation. |

---

## 5. Experiments

| # | Cell | Output |
|---|------|--------|
| WS0 | Sanity: R baseline reproduces each model's known final acc. | confirms harness |
| WS1 | DeepMLP: R vs O-label, per-condition LR sweep. | acc-vs-compute, steps-to-{90,95,99%}, break-even |
| WS2 | LeNet: R vs O-label vs O-distill. | tests W1 (weak label init) + W3 (distill init) |
| WS3 | TinyTransformer: R vs O-distill (and O-label probe). | distill warm-start on attention |
| WS4 | KD baseline (random + SGD on teacher soft targets) at matched budget, per arch. | W3 comparison line |
| WS5 | Init-conditioning diagnostics for every init, correlate with first-50-step loss drop. | explains W4 (fragility) |

**Common:** seed=42 for init; ≥3 SGD seeds; capture N=16,384; eval on held-out test;
log every K steps. Primary artifact: acc-vs-cumulative-FLOPs overlay + a break-even table.

---

## 6. Decision rules

| Outcome | Action |
|---------|--------|
| W1 + W5 hold for MLP/distill (break-even below achievable acc) | Warm-start has real total-compute value → write up, push to a larger model. |
| Gain only appears at tuned LR (W2) | Report the schedule sensitivity prominently — it is the practical catch. |
| W4 falsified (OLS basin gives worse final acc) | Investigate conditioning (WS5); the OLS targets/margins likely need rescaling before SGD. |
| O-label init worse than random after SGD on conv | Confirms label-init not worth it through pooling; restrict warm-start to distill/MLP. |

**Escalate** if OLS-init diverges under ALL LR schedules (init basin incompatible with
SGD — itself a finding worth reporting, not patching).

---

## 7. Code

| Component | Note |
|-----------|------|
| `diagnostic/runners/warmstart/warmstart_run.py` | builds model → applies init (R / O-label via `gt_target_retrain` / O-distill via the feature-distill sweep) → runs SGD logging acc vs cumulative FLOPs; `--arch {mlp,lenet,transformer}`, `--init {random,ols_label,ols_distill}`, LR-sweep args. |
| `diagnostic/flops.py` (new, small) | forward-FLOPs per arch + OLS solve FLOPs + SGD FLOPs accounting, so the x-axis is honest. |
| reuse | existing trainers, `gt_target_retrain`, feature-distill code (E3/E5/E8), `_eval`, data loaders. |

Findings → `warmstart_findings.md` (one row per cell); plots alongside.

---

## 8. Out of scope

Large-model scaling; optimizing the `d_in²` OLS solve; multi-GPU; exhaustive optimizer
comparison (pick one optimizer per arch and hold it fixed). Multi-seed only on the SGD
side (init is deterministic at seed=42).
