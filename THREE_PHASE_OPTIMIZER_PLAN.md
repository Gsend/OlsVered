# Three-Phase Adaptive Optimizer Plan

A concrete instance of the [composite optimizer architecture](./OPTIMIZER_SWITCHING_ARCHITECTURE.md): Adam → ClassicKFAC (or OlsSMKFAC) → VeredKFAC, with phase transitions triggered by *convergence speed* rather than fixed step counts. The phase ordering matches three loosely-distinct regimes of language-model pretraining; each phase is matched to the regime where its cost/benefit is dominant.

This document is scoped to (a) the design of this specific 3-phase configuration, (b) the convergence-speed trigger logic, and (c) a quantitative theoretical analysis of expected wall-time savings on the SmallGPT/WikiText-2 testbed.

## The phase rationale: matching optimizer to regime

Pretraining on a small transformer goes through three loosely-distinct regimes. The current Vered run gives a clean trace of where each regime starts and ends.

### Regime 1: high-loss, gradient-magnitude-dominated (steps 0–500 in current runs)

The model is essentially random. Loss is at the cap (val_ppl=9999), gradients are huge, the loss surface near current weights is far from any basin. *Any* descent direction works because the magnitude of the gradient is the dominant signal — its direction is approximate but the move is so large that fine direction quality is irrelevant.

K-FAC during this regime is paying its ~16× per-step cost to compute curvature factors that are mostly noise: the EMAs haven't accumulated enough samples to be meaningful, the damping must be large to keep the inverse stable, and the resulting natural-gradient direction is barely distinguishable from a scaled raw gradient. In your traces, all three K-FAC variants spend the first 500 steps idling at the loss cap — the curvature estimation is pure overhead.

**Right tool**: Adam. Cheap (~50 ms/step), diagonal preconditioner, no warmup state to populate. Will traverse Regime 1 in ~1000 Adam steps for ~50 s of wall time, vs ~500 K-FAC steps for ~6–7 min.

### Regime 2: descent, mid-loss, moderate anisotropy (steps 500–2500)

Loss has dropped into a meaningful range (val_ppl from ~9000 down to ~1000). Gradients are still large, but the loss surface now has structure: directions of high curvature exist, anisotropy is starting to matter. Adam's diagonal preconditioner can no longer fully exploit the gradient signal because correlated coordinates within layers need a non-diagonal preconditioner.

K-FAC starts earning its cost here. But all three K-FAC variants behave similarly in this regime, because damping must still be relatively large (e.g., λ=1e-4 to 1e-3) to keep the inverse well-conditioned — and once damping dominates the spectrum, Classic's κ⁴, OlsSM's κ², and Vered's κ¹ scaling all reduce to the same effective conditioning. Vered's per-step direction quality advantage is masked by the regularization.

**Right tool**: ClassicKFAC (or OlsSMKFAC), at full momentum. Cheaper per step than Vered (~700 ms vs ~800 ms), uses the same momentum amplification, and produces equivalent direction quality given the high damping. The cleaner conditioning of OlsSM/Vered is wasted compute in this regime.

### Regime 3: fine convergence, small gradients, low damping needed (steps 2500+)

Loss has settled into a basin (val_ppl < 1000). Gradients are small and *direction-quality dominates magnitude*: a cleaner natural-gradient direction picks the right tradeoff between coordinates that are now subtly correlated. To exploit this, damping must be *low* (1e-5 or 1e-6) so the preconditioner isn't dominated by the regularizer. This is exactly the regime where Vered's κ¹ stability advantage materializes: at low damping with large per-step κ, Classic's `κ⁴·ε` forward error becomes catastrophic but Vered's `κ¹·ε` stays well-behaved.

This is also the regime where the comparison results become slope-sensitive: the run hasn't flattened by step 5000, and the slope's decay rate determines the asymptotic gap between configs. Vered's per-step direction quality (140× cleaner per the synthetic test) compounds in this tail.

**Right tool**: VeredKFAC, mom=0.0 or low mom, low damping. Pays full per-step cost (~800 ms) but each step produces a clean direction that lets the loss continue descending where Classic would stall (or require high damping to avoid divergence, which kills its convergence).

## Phase transitions: convergence-speed triggers

Fixed step counts are a poor trigger for these transitions because the regime boundaries depend on model size, dataset, batch size, and learning rate. A model that hits Regime 2 at step 200 in a 4-block GPT might take 2000 steps in a 24-block one. The trigger needs to *measure* where the model is, not assume it.

Convergence speed — defined as the rate of change of a smoothed loss — is the natural primitive because it directly measures "is this optimizer still making progress in the current regime."

### Convergence-speed metric: definition

Let `L_smooth(t)` be an EMA-smoothed validation (or training) loss with smoothing coefficient β=0.95 (about 20-step effective window).

Define **per-step improvement velocity** at step t as:
```
v(t) = (L_smooth(t-W) - L_smooth(t)) / W
```
for window W (e.g., W=100 steps). This is loss reduction per step, regardless of phase or optimizer.

Define **per-second improvement velocity** as:
```
v_time(t) = (L_smooth(t-W) - L_smooth(t)) / wall_time_for_window
```
This accounts for the cost differential between phases — Adam's `v(t)` may be worse than K-FAC's, but `v_time(t)` is what determines whether you should keep paying for cheap-or-expensive steps.

Both metrics are useful for different purposes; the design uses both.

### Phase 1 → Phase 2 trigger: "Adam can't keep up with anisotropy"

Switch from Adam to Classic K-FAC when Adam's `v(t)` (per-step improvement) falls below a threshold AND the current loss is below a hard cap (so we don't switch prematurely while still in Regime 1).

```
trigger_1to2 = And(
    LossSlope(window=200, threshold=0.005, source="train"),   # progress slowing
    LossBelow(threshold=1500.0, source="val"),                # past Regime 1
    PhaseStepCount(min_steps=300),                            # don't switch too early
    NextOptimizerWarm(),                                      # K-FAC EMA pre-warmed
)
```

The slope threshold of 0.005 corresponds to "loss not improving by even 0.005 per step on smoothed average, over a 200-step window" — Adam has hit its diagonal-preconditioner ceiling and isn't gaining ground.

The val ppl < 1500 ensures we're past the loss-cap regime. Switching while loss is at the 9999 cap would be a false positive (no progress because loss can't be reported, not because Adam is stalling).

The minimum-300-steps floor prevents the trigger from firing on the first batch of noise.

NextOptimizerWarm is critical: K-FAC's EMAs must have absorbed enough samples to be useful. With pre-warming during the Adam phase, this is satisfied automatically once Adam has run for ~300+ steps.

### Phase 2 → Phase 3 trigger: "Classic stalled because damping needs to drop"

This is the more interesting trigger because it captures the precise condition under which Vered's κ¹ stability advantage becomes valuable.

Classic stalls in late training for one of two reasons:
1. **Damping is too low and updates are diverging.** Classic's κ⁴·ε amplification means at low damping, numerical noise corrupts the natural-gradient direction. The optimizer takes bad steps, loss oscillates or rises.
2. **Damping is too high and updates are too SGD-like.** The natural-gradient correction is washed out, the optimizer behaves like vanilla SGD-with-momentum, and progress slows.

Both are cases for switching to Vered, which can run at lower damping without (1) and so reach lower asymptotic loss.

The trigger combines slope detection with a damping-attempt-failure signal:

```
trigger_2to3 = Or(
    # Case A: Classic stalled, even with current damping
    And(
        LossSlope(window=300, threshold=0.002, source="val"),
        LossBelow(threshold=900.0, source="val"),
        PhaseStepCount(min_steps=500),
    ),
    # Case B: Damping reduction caused divergence (Classic can't go lower)
    DampingFloorReached(min_damping=1e-4),
)
```

The `LossBelow(900)` ensures we're in Regime 3 territory (the 821 floor seen in the current run is below 900, but pre-fix Vered hit 756 — the threshold should be set conservatively to switch *before* Classic would have stalled).

`DampingFloorReached` is a special trigger that watches Classic's damping schedule (if any) and fires once Classic's damping has been reduced to a value at which further reduction caused divergence. This is opt-in via a damping-schedule callback; without it, only Case A fires.

### Why convergence speed beats step count

A step-count trigger says "switch at step 1000, then step 3500." It works in expectation but is brittle: a different LR makes the regime boundaries shift; a different batch size moves them again. With each hyperparameter sweep, the step-count thresholds need re-tuning.

A convergence-speed trigger says "switch when this optimizer's marginal contribution drops below the next one's expected." It's hyperparameter-invariant in a much stronger sense: as long as the *characterization of regimes* holds (Adam fast in Regime 1, Classic mid-cost in Regime 2, Vered slow-but-deep in Regime 3), the trigger correctly times the transition regardless of LR, batch size, or model size.

The computational cost of the trigger is negligible: a smoothed-loss EMA + a windowed slope, evaluated once per validation pass.

## Theoretical run-time analysis

Empirical inputs from the current Vered runs:

- Adam per-step cost: ~50 ms (estimated from typical small-transformer Adam wall time; not measured directly in current benchmarks, but consistent with PyTorch overhead at this batch size)
- ClassicKFAC per-step cost: ~700 ms (estimated; Classic's matrix inversion uses GEMM-friendly explicit inv, similar but slightly cheaper than OlsSM's Cholesky-solve and Vered's QR)
- VeredKFAC per-step cost: ~794 ms (measured: median_step_ms in current runs)

Empirical time-in-regime split for the current pure-Vered baseline (5000 steps, 75 min):

| Regime | Approx steps | Approx wall time | % of total |
|---|---|---|---|
| 1 (val_ppl > 5000) | 0–500 | 6.7 min | 9% |
| 2 (val_ppl 5000 → 1000) | 500–2500 | 27 min | 36% |
| 3 (val_ppl < 1000, descending) | 2500–5000 | 33 min | 44% |
| Boot/eval overhead | — | ~8 min | 11% |
| **Total** | **5000** | **75 min** | **100%** |

### Phased optimizer wall-time estimate

Assumption for Regime 1: Adam needs ~1000 steps (vs Vered's 500) to cover the same loss territory, because each Adam step is "smaller" but gradients are huge and 1000 small steps in a clean direction match 500 K-FAC steps with garbage curvature. This is approximate but consistent with literature on first-order vs second-order in early training.

Assumption for Regime 2: Classic at the same momentum/LR setting matches Vered's progress per step (in this regime, damping dominates and the κ-scaling differences are masked). Per-step cost is ~12% lower.

Assumption for Regime 3: Vered's per-step cost is the same as the baseline. Quality is the same.

| Regime | Phased optimizer | Steps | Per-step ms | Wall time |
|---|---|---|---|---|
| 1 | Adam | 1000 | 50 | 50 s = 0.83 min |
| 2 | ClassicKFAC | 2000 | 700 | 1400 s = 23.3 min |
| 3 | VeredKFAC | 2500 | 794 | 1985 s = 33.1 min |
| Boot/eval | — | — | — | 8 min |
| **Total** | | **5500** | | **65.2 min** |

Wall-time savings vs. pure Vered (75 min): **9.8 min, or ~13%**.

This is the conservative estimate. Two factors make it likely *higher* in practice:

**Factor 1**: Classic with momentum=0.9 takes 10× momentum amplification, while Vered's current best is at momentum=0.0. So Classic in Regime 2 may need only ~1500 steps to reach the same progress (instead of 2000). Recomputing:

| Regime | Optimizer | Steps | ms/step | Wall |
|---|---|---|---|---|
| 1 | Adam | 1000 | 50 | 0.8 min |
| 2 | Classic mom=0.9 | 1500 | 700 | 17.5 min |
| 3 | Vered mom=0.0 | 2500 | 794 | 33.1 min |
| | | | overhead | 8 min |
| | | | **Total** | **59.4 min** |

Wall-time savings: **15.6 min, or ~21%**.

**Factor 2**: Classic→Vered transition can transfer state directly (both store Gram-equivalent factors of A, G — Classic stores them as full matrices, Vered as R-factors). A `chol(A)` or `qr(A)` at handoff converts in ~10 ms one-shot. So the Classic→Vered transition pays no warmup cost.

The Adam→Classic transition needs pre-warming (Classic's hooks accumulating during Adam phase). Pre-warming overhead during Adam phase: Adam already runs `forward + backward`, hooks add ~5–10 ms per step in extra activation snapshots and EMA updates. With 1000 Adam steps × 8 ms overhead = 8 s extra cost — negligible.

### Quality analysis (non-quantitative)

Wall time is one axis. The harder question is whether the phased optimizer reaches a *lower final ppl* than pure Vered.

**Plausibility argument for parity or improvement**:

1. Adam in Regime 1 makes *the same* progress as Vered would (both are descent methods in a regime where direction quality doesn't matter). So entering Regime 2, the model is in the same starting state.

2. Classic in Regime 2 with momentum amplification makes *more* progress per minute than Vered without momentum. So entering Regime 3, the model is at a *deeper* basin than the pure-Vered baseline would be.

3. Vered in Regime 3 then continues to refine from that deeper start, with the same per-step quality as pure Vered. Final ppl should be ≤ pure-Vered baseline.

The concrete prediction: phased optimizer reaches the same val_ppl threshold ~15–20 min faster, and final val_ppl at 5000 steps may improve by 20–50 ppl (because more time is spent in Regime 3 from a better starting point).

**Risk factors for the quality argument**:

- **Cold-start instability at the first transition.** Adam→Classic with empty K-FAC EMAs (without pre-warming): Classic's first ~200 steps produce useless updates that may briefly inflate loss. Mitigated by pre-warming.
- **Optimizer-specific basin geometry.** Adam may find a basin shape that Classic can't exploit well, or that has a worse local geometry than the basin Vered would reach from random init. This is a known and unresolved concern in the optimizer-switching literature; usually mitigated by using a low Adam LR (3e-4 or below) so the descent is gentle.
- **Classic→Vered state transfer assumes the same factor parameterization.** Both store G (output-side) and A (input-side) Gram factors, but Classic uses full-matrix EMA while Vered uses R-factor TSQR. Conversion via `qr(chol(A))` is mathematically lossless but numerical precision in fp32 may drift slightly.

### Best-case and worst-case bookends

**Best case (~25% wall savings, ~30 ppl quality gain)**:
- Pre-warming works as designed
- Classic→Vered state transfer is lossless
- Adam doesn't find a pathological basin
- Final wall time: ~55 min, final ppl: ~790 (vs current 821)

**Worst case (~5% wall savings, slightly worse quality)**:
- Cold-start at Adam→Classic costs 200 wasted Classic steps (2.3 min)
- State transfer at Classic→Vered drops Vered into a slightly suboptimal basin
- Final wall time: ~71 min, final ppl: ~830

**Median expected**: ~15–18% wall savings, neutral-to-mild quality improvement.

## Configuration shape

Following the [composite optimizer config](./OPTIMIZER_SWITCHING_ARCHITECTURE.md#configuration-format):

```yaml
optimizer:
  type: composite
  phases:

    - name: warmup_adam
      optimizer:
        type: adam
        lr: 3.0e-4
        betas: [0.9, 0.999]
      until:
        type: and
        children:
          - { type: loss_slope,      window: 200, threshold: 0.005, source: train }
          - { type: loss_below,      threshold: 1500.0, source: val }
          - { type: phase_step,      min_steps: 300 }
          - { type: next_warm }

    - name: cruise_classic
      optimizer:
        type: classic_kfac
        lr: 8.0e-3
        damping: 1.0e-4
        gamma: 0.7
        momentum: 0.9
        grad_clip: 60.0
      pre_warm_during: [warmup_adam]
      until:
        type: or
        children:
          - type: and
            children:
              - { type: loss_slope, window: 300, threshold: 0.002, source: val }
              - { type: loss_below, threshold: 900.0, source: val }
              - { type: phase_step, min_steps: 500 }
          - { type: damping_floor_reached, min_damping: 1.0e-4 }

    - name: finetune_vered
      optimizer:
        type: vered_kfac
        lr: 8.0e-3
        damping: 1.0e-5
        gamma: 0.7
        momentum: 0.0
        grad_clip: 60.0
      inherit_state_from: cruise_classic   # transfer A, G factors
      until: null   # final phase
```

## Implementation deltas vs the general framework

The general composite optimizer framework already provides most of what's needed. The 3-phase adaptive variant requires three new components:

1. **`LossSlope` trigger** — windowed slope of EMA-smoothed loss, with separate train/val sources. Not in the general v1 plan; needs to be added. Implementation: a circular buffer of recent (loss, step) pairs, slope computed on demand from buffer endpoints.

2. **`DampingFloorReached` trigger** — opt-in trigger driven by an external damping schedule. Requires the K-FAC variants to expose their current damping value as a property and to emit a `damping_was_clamped` signal when a damping reduction caused near-divergence. Adds ~30 lines to each K-FAC variant.

3. **Classic→Vered state transfer** — converts Classic's full-matrix A, G EMAs into Vered's R-factor representation via `qr(chol(A))` and `qr(chol(G))`. Implemented as a method on Classic (`export_to_vered_state(self) -> dict`) and a constructor option on Vered (`init_from_state(state)`). ~50 lines.

These three additions slot into the existing composite framework without disturbing it. The framework's pre-warming, scheduler, handoff, and state-dict round-trip mechanisms all carry over unmodified.

## Validation plan

Before declaring the phased optimizer a win, three experiments should run:

1. **Baseline**: pure VeredKFAC, current best config (γ=?, mom=0.0, λ=1e-5 — pending Stage A result). 5000 steps. Establishes the ppl/min curve to beat.

2. **Cold-handoff phased**: 3-phase composite without pre-warming, fixed step boundaries (Adam steps 0–800, Classic 800–2800, Vered 2800–5000). Same total step count. Tests whether the phasing hypothesis works *at all* before adding adaptive triggering complexity.

3. **Adaptive phased**: 3-phase composite with convergence-speed triggers and pre-warming, no fixed step boundaries. Total step count chosen to match the cold-handoff variant's wall time. Tests whether adaptive triggering improves over cold-handoff.

If (2) > (1) by a meaningful margin (≥15% wall savings or ≥20 ppl improvement), the phasing hypothesis is validated and (3) is worth running. If (2) ≈ (1), the hypothesis is wrong for this specific testbed and the framework's value is in the *general* switching capability, not in the specific Adam→Classic→Vered recipe.

## Decision rule for whether to build this

Build it if:
- Stage A of the current grid completes and Vered's best 5000-step ppl is still meaningfully behind Classic's 776, suggesting Vered needs more wall time to flatten than the budget allows
- An extended single-config Vered run at 15000 steps confirms the slope tail extends well beyond the current 5000-step truncation
- The general composite-optimizer framework (Phase 1 of the prior plan) is already in place

Don't build it if:
- Stage B of the grid surfaces a (γ, mom) for Vered that beats Classic's 776 within 5000 steps — at that point, the phased recipe's value is reduced because Vered alone is already winning
- The CPU/GPU memory overhead of pre-warming K-FAC hooks during Adam phase turns out to be more than ~10% (would need to be measured before committing)

The strongest case for building this is the *quality* argument, not the wall-time argument: matching optimizer to regime is a cleaner experiment, easier to interpret, and produces a more honest story about where each method's value lives. The 15–25% wall savings is a side benefit.
