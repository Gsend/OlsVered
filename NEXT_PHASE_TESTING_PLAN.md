# Next-Phase Testing Plan

This plan covers the three open methodological gaps in the K-FAC stability investigation, in priority order: (1) optimize grad_clip at the current Vered winner, (2) properly optimize Classic K-FAC (and OlsSM) so the comparison is fair, (3) verify generalization across model architectures.

The plan is structured so each phase produces a decisive outcome, and the next phase only proceeds if the prior result justifies it. Total budget across all three phases: roughly 35-55 hours of GPU time, plus 2-4 hours of code work (mostly already done for Phases 1 and parts of 2).

## Current state, for reference

- **Vered post-fix winner**: γ=0.3, mom=0.3, λ=1e-4, grad_clip=60, lr=8e-3 → **681 ppl** at 5000 steps on SmallGPT/WikiText-2
- **Classic best (untuned)**: γ=0.7, mom=0.9, λ=1e-4, grad_clip=60 → 776 ppl
- **OlsSM**: untested at recent settings; crashed in earlier runs
- **Multi-arch infrastructure**: exists for MLP/MNIST and CIFAR-10 ConvNet, but result files are pre-EMA-fix and pre-grid (stale)

## Phase 1: Optimize grad_clip at the Vered winner

### Goal

Establish whether the Vered winner config has unrealized headroom on the grad_clip axis. Prior gradclip benchmark (Phase 1, 1000 steps, mom=0.9, γ=0.7) found grad_clip=30 was optimal across all variants. The current winner runs at grad_clip=60 — chosen as a stability buffer at different hyperparameters. Theoretical reasoning suggests the optimum may have shifted *tighter* given the lower momentum and lower γ.

### Runs

| Step | grad_clip | Source | Wall time |
|---|---|---|---|
| 1.1 | 60 | reused (681 ppl, full-grid winner) | 0 |
| 1.2 | 30 | new — Phase 1 prior optimum | ~75 min |
| 1.3 | 15 | new | ~75 min |
| 1.4 | 8 | new | ~75 min |

Script ready: `benchmark/vered_gradclip_probe.py`. Launcher: `run_gradclip_probe.ps1`.

**Phase 1 wall time: ~3.75 hours.**

### Decision rule

| Outcome | Next action |
|---|---|
| 30 wins by ≥10 ppl | Adopt grad_clip=30. Optionally probe {20, 25, 35} to find finer optimum (~2.5 hr). |
| 15 or 8 wins by ≥10 ppl | Tighter than expected. Probe {3, 5, 10, 12} to find U-curve minimum (~3 hr). |
| All within 10 ppl | grad_clip is insensitive at this config. Stay at 60. Move directly to Phase 2. |
| 8 diverges | Lower bound is between 8-15. Probe {10, 12} to refine (~1.5 hr). |

### Phase 1 budget

3.75 hours minimum, up to ~7 hours if a refinement sweep is needed.

## Phase 2: Properly optimize Classic K-FAC

### Goal

Run Classic K-FAC through the same coordinate-descent grid that Vered received, so the head-to-head comparison is methodologically symmetric. The current 95 ppl gap (681 vs 776) is partly a search-budget asymmetry — Vered got 12 cells, Classic got 1. Phase 2 closes that asymmetry.

Theoretical expectations (to be tested):

- **Classic should perform best at high γ + moderate λ.** Classic's κ⁴·ε forward error scaling means low-γ (noisy curvature) + low-λ (under-regularized inverse) is its worst regime. Its own optimum should sit at γ ∈ {0.7, 0.9, 0.95} with λ ∈ {1e-4, 1e-3} and mom=0.9.
- **Classic should diverge or underperform at low γ + low λ.** Including the (γ=0.3, λ=1e-4) cell — Vered's winner — is the most theoretically-loaded data point of the whole investigation. If Classic blows up there, it's empirical proof of the κ⁴ instability prediction.
- **Classic's tuned best should land in 720-770 ppl range.** That's my ~60% prediction. Worse → strongest case for Vered. Better → narrower margin, more careful claim needed.

### Phase 2A: Classic full grid

A coordinate-descent grid mirroring `vered_full_grid.py` but with Classic-anchored sweep ranges. The grid order is chosen so the "obvious wins" run first and the theoretically-loaded cells run later.

**Stage A: γ sweep at (mom=0.9, λ=1e-4) — Classic's natural regime.**

| γ | Run? |
|---|---|
| 0.7 | already done (776 ppl) — reuse |
| 0.5 | new |
| 0.9 | new |
| 0.95 | new |
| 0.3 | new — *theoretically-loaded; may diverge* |

Stage A: 4 new runs, ~5 hr.

**Stage B: momentum sweep at best γ from A, λ=1e-4.**

mom ∈ {0.5, 0.7, 0.9, 0.95}. 4 cells; one or two may be reused. ~3-5 hr.

**Stage C: λ sweep at best (γ, mom).**

λ ∈ {1e-3, 1e-4, 1e-5, 1e-6}. 4 cells; one carry-over. ~3.75 hr.

**Phase 2A wall time: ~12-14 hours.**

### Phase 2B: Classic grad_clip probe

Same probe as Phase 1, but at Classic's grid winner. ~3.75 hr.

### Phase 2C (optional): OlsSM full grid

If Phases 2A+2B confirm the κ⁴ prediction, OlsSM is the κ² midpoint and completes the hierarchy claim. Same grid structure, but OlsSM-anchored ranges. ~12 hr.

This is the most cuttable phase — if the project's narrative is "Vered vs Classic", OlsSM is supporting evidence rather than headline. Defer if compute is tight.

### Decision rules within Phase 2

- **Stage A's (γ=0.3) cell diverges** → empirical κ⁴ confirmation. Big win for the math doc story. Continue grid normally; this is now an additional headline result.
- **Stage A finds Classic best at γ=0.5 or 0.7** → expected outcome. Continue to Stage B.
- **Classic's best after 2A+2B is below 700 ppl** → the gap to Vered is now ≤20 ppl; reframe the claim as "Vered is marginally better with theoretical justification" rather than "Vered substantially beats Classic". Methodologically more honest result.
- **Classic's best stays at 776** → strongest case for Vered. The grid found nothing better for Classic.

### Phase 2 budget

~16 hours minimum (2A+2B), up to ~28 hours if OlsSM is included.

## Phase 3: Multi-architecture generalization

### Goal

Test whether Vered's winning config generalizes beyond SmallGPT/WikiText-2. The architectures available in the existing infrastructure: MLP on MNIST, ConvNet on CIFAR-10, transformer on WikiText-2 (current), BERT fine-tuning on SST-2. The most informative are MLP and CIFAR — they have different layer types and convergence dynamics, so they directly test whether the K-FAC stability advantage is a transformer-specific or architecture-general property.

The existing result files for these benchmarks are stale (pre-EMA-fix, pre-grid). They need to be re-run with the tuned configs from Phases 1 and 2.

### Phase 3A: Tier-1 generalization probe (cheapest)

For each architecture (MLP, CIFAR, SmallGPT), run each variant in two configs:

- **Variant's own grid winner** (from Phase 2 for Classic and OlsSM; from existing grid for Vered)
- **Vered's overall winner config** applied unchanged to that variant (γ=0.3, mom=0.3, λ=1e-4, grad_clip=Phase-1-winner)

That's 3 archs × 3 variants × 2 configs = 18 runs.

Per-run wall times:

- MLP/MNIST: ~10-15 min
- CIFAR-10 ConvNet: ~30-60 min
- SmallGPT (already done in Phase 1+2 for variants' own winners): reuse

Total: ~6 hours of compute.

This produces a 3×3×2 table of final accuracies/perplexities. The winning narrative looks like:

| | MLP | CIFAR | SmallGPT |
|---|---|---|---|
| Vered (own best) | best | best | **best (681)** |
| Classic (own best) | second | second | second (Phase 2 result) |
| OlsSM (own best) | third | third | third (Phase 2 result) |
| Vered config applied to Classic | poor (too low γ, too low λ for Classic) | poor | poor |
| etc. | | | |

If Vered wins all three "own-best" rows, the multi-arch claim is solid.

### Phase 3B: Tier-2 fair multi-arch (if 3A is unclear)

If Phase 3A produces ambiguous results — e.g., Vered wins SmallGPT but loses MLP — the next step is to tune each variant *on each architecture*. That's grid search per (arch, variant) pair = 9 grids. Roughly 30-45 hours of compute. Skip unless 3A demands it.

### Phase 3 budget

~6 hours for Tier 1 (Phase 3A). Phase 3B only if needed and adds 30-45 hours.

## Aggregated budget

| Phase | Min hours | Max hours |
|---|---|---|
| 1 (grad_clip) | 3.75 | 7 |
| 2A (Classic grid) | 12 | 14 |
| 2B (Classic grad_clip) | 3.75 | 5 |
| 2C (OlsSM, optional) | 0 | 12 |
| 3A (multi-arch generalization) | 6 | 8 |
| 3B (fair multi-arch, if needed) | 0 | 45 |
| **Realistic total** | **~25** | **~50** |

## Decision tree across phases

```
Phase 1: grad_clip probe
  |
  +-- Vered improves -> adopt new clip; record Phase 1 result
  +-- No improvement -> Vered stays at 681 ppl
  |
Phase 2A: Classic grid (4-stage)
  |
  +-- (γ=0.3) cell of Stage A diverges -> κ⁴ prediction empirically confirmed
  |   (continue Stage B, C; this is now an additional headline result)
  |
  +-- Stage A normal -> continue
  |
  V
Phase 2B: Classic grad_clip probe
  |
  +-- Both phases done; Classic optimum known
  |
  +-- |Classic optimum - Vered optimum| < 20 ppl?
  |     YES -> reframe claim as "marginal win with theory"
  |     NO  -> continue to multi-arch with substantial-margin claim
  |
  V
Phase 3A: multi-arch generalization probe
  |
  +-- Vered wins all 3 archs -> defensible multi-arch claim, ship
  +-- Vered wins 2/3 -> document the boundary; narrow claim
  +-- Vered wins 1/3  -> claim is transformer-specific (still real but narrower)
  |
  +-- Ambiguous results -> Phase 3B (Tier 2 fair multi-arch)
```

## What's not in the plan (and why)

- **Multi-seed variance estimates.** Standard practice but expensive (3 seeds × all main configs ≈ 22 hr). Defer until the deterministic result is locked. With multi-arch already showing a robust pattern, multi-seed becomes a reviewer-bait check rather than a load-bearing finding.
- **Larger transformer scales** (8-12 layer models). Important for scale-related generalization but the existing infrastructure only has SmallGPT. Adding larger models is a separate engineering task, not a tuning task. Defer.
- **Other LM datasets** (PTB, WikiText-103). Same reason — would need new data-loading infrastructure. Defer.
- **Wall-time vs ppl Pareto frontier.** Worth analyzing once Phase 2 completes (because Classic per-step is cheaper than Vered, "ppl per minute" comparison may differ from "ppl at 5000 steps"). One-shot post-hoc analysis from existing JSONs, no new runs needed.

## Recommended execution order

1. **Tonight**: Launch Phase 1 (grad_clip probe). 3.75 hr; results by morning.
2. **Tomorrow**: Read Phase 1 results; decide refinement sweep yes/no. Then launch Phase 2A (Classic grid). 12-14 hr; results next morning.
3. **Day 3**: Launch Phase 2B (Classic grad_clip). 3.75 hr.
4. **Day 3-4**: Launch Phase 3A (multi-arch generalization probe). 6 hr.
5. **Day 5+**: Synthesize findings; decide on Phase 2C and 3B based on what's open.

This sequencing maximizes information per unit compute: Phase 1 is cheap and may unlock 10-30 ppl; Phase 2 is the largest single scientific commitment; Phase 3 closes the generalization gap. Stopping after any phase produces a defensible (if narrower) result.

The fastest path to a defensible "Vered is better" claim — if results go favorably — is Phases 1 + 2A + 2B + 3A, totaling ~26 hours. That's three days of overnight runs. Phase 2C and 3B are optional sharpening.
