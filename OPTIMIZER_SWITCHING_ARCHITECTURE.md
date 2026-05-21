# Optimizer Switching Architecture

A composite optimizer that runs multiple sub-optimizers in sequence (or, as a stretch goal, in parallel across parameter groups), with switch points driven by user-defined criteria.

The motivating use case is the Adam-warmup-then-K-FAC recipe discussed in the K-FAC stability investigation: the early phase of training is dominated by large, well-aligned gradients where K-FAC's per-step cost (~16× Adam) is wasted on a regime where any descent direction works. K-FAC's anisotropy correction earns its cost only once the model is near a basin and gradients become small, correlated, and direction-sensitive. The architecture below generalizes that pattern: any sequence of optimizers, any trigger.

## Goals

- **Declarative phase configuration.** A user supplies a list of phases and trigger conditions; the optimizer figures out the rest.
- **Clean handoff.** Switching from optimizer A to optimizer B preserves model weights and any state B can use; everything else is discarded explicitly.
- **Pre-warming.** While phase A is active, phase B's stateful machinery (e.g., K-FAC's G/A EMAs) can be warmed in the background so the handoff isn't a cold start. This is the single largest source of wall-time savings in the Adam→K-FAC case.
- **Resumable.** Mid-training checkpoint and resume must work across a phase boundary.
- **Drop-in replacement.** From the trainer's perspective, the composite looks like a single `torch.optim.Optimizer` with a standard `step()` / `zero_grad()` / `state_dict()` API.

## Non-goals (explicitly deferred)

- **Per-parameter / per-layer optimizer mixing within a single step.** Some published recipes use Adam for embeddings + K-FAC for transformer blocks. This architecture should not preclude that, but the v1 design covers only sequential phase transitions where one optimizer is "live" at a time. Mixed-per-layer is a separate composition (a "parallel composite") that would build on the same primitives.
- **Auto-tuning of switch points.** The architecture provides triggers but not adaptive trigger discovery. Hyperparameter search over switch points is a separate concern.
- **Optimizer state translation.** When switching from Adam to K-FAC, Adam's `m, v` buffers are not converted into K-FAC factors. The handoff is "drop A's state, start B's state from whatever pre-warming gave us" — no clever translation. Translation is rarely useful in practice and adds significant complexity.

## Background: what each existing optimizer needs

The existing K-FAC variants in `optimizer/` follow a common shape:

| Component | Role | When it runs |
|---|---|---|
| Forward hook | Captures activations `X` for input-side Gram `A = X^T X` | Each forward pass |
| Backward hook | Captures gradient signals `G_signal` for output-side Gram `G` | Each backward pass |
| `step()` | Builds running A/G via EMA (or augmented-QR for Vered), inverts, applies `ΔW = G⁻¹·∇L·A⁻¹` | Each optimization step |
| Internal state | A_running, G_running, possibly streaming TSQR R-factors, momentum buffer | Persists across steps |

Adam (and any first-order optimizer) follows a much simpler shape:

| Component | Role |
|---|---|
| `step()` | Reads `param.grad`, updates internal `m, v`, writes `param.data` |
| Internal state | First and second moment buffers per parameter |

The composite must accommodate both.

## High-level design

The composite optimizer has three layers, separated by responsibility.

```
+----------------------------------------------------+
|  CompositeOptimizer  (drop-in for torch.optim.*)   |
|                                                    |
|   step()  -> route to active sub-optimizer         |
|              run pre-warm side-effects             |
|              consult scheduler for transitions     |
|                                                    |
+--------------+---------------+---------------------+
               |               |
               v               v
    +------------------+  +------------------+
    | PhaseScheduler   |  | Sub-optimizers   |
    | (pure logic)     |  | (Adam, KFAC, …)  |
    +------------------+  +------------------+
```

### Layer 1: the sub-optimizer protocol

Each sub-optimizer must implement a small extension of the `torch.optim.Optimizer` protocol. The composite assumes:

- `step(closure=None)` performs the parameter update.
- `state_dict()` / `load_state_dict()` for checkpointing.
- An `attach(model)` method that registers any hooks the optimizer needs (forward, backward, etc.) and stores their handles.
- A `detach()` method that removes all hooks the optimizer registered. Idempotent.
- An optional `accumulate_only(model)` method that registers *only the data-collection hooks*, not the update logic. This is what enables pre-warming: while Adam is the live optimizer, we can call `kfac.accumulate_only(model)` so that K-FAC's G/A EMAs build up on real forward/backward passes without K-FAC's expensive `step()` running.
- A `state_is_warm` property (boolean or numeric "warmth score") so the scheduler can decide whether B is ready to take over.

For optimizers that don't have hooks (Adam, SGD), `attach`, `detach`, and `accumulate_only` are no-ops, and `state_is_warm` returns True immediately.

For K-FAC variants, `state_is_warm` returns True once the EMAs have absorbed enough samples (e.g., the running-average buffer has been updated more than `min_warmup_steps` times, or the EMA's effective sample count exceeds a threshold).

### Layer 2: the phase scheduler

A pure-logic component that holds the user's phase definitions and answers one question per `step()`:

> Given the current step count, current loss / val-loss, wall time, and warmth of the next phase's optimizer, should we transition?

The scheduler does not touch optimizers or model state. It only emits transition events. The composite acts on those events.

The scheduler holds a list of phases:

```
phases = [
    {"name": "warmup",  "optimizer": <adam>,   "until": <trigger>},
    {"name": "cruise",  "optimizer": <vered>,  "until": <trigger>},
    {"name": "finetune", "optimizer": <vered_low_lambda>, "until": None},  # final phase
]
```

The `<trigger>` is a small declarative object describing when to leave the phase. See "Trigger types" below.

The final phase has `until=None`, meaning "stay here for the rest of training".

### Layer 3: the composite optimizer

Orchestrates the lifecycle. Pseudo-flow per `step()`:

1. Increment internal step counter.
2. Ask scheduler: should we transition out of the current phase?
3. If yes:
   - Detach the current sub-optimizer's hooks.
   - Take over the new sub-optimizer; call `attach(model)`.
   - If the new optimizer was being pre-warmed (its `accumulate_only` hooks were running), promote those hooks to "live" mode rather than reattaching from scratch. This is the warmth-preservation step.
   - Emit a transition event to logs / metrics.
4. Run the live sub-optimizer's `step()`.
5. For each *future* phase whose `pre_warm` flag is True and whose pre-warm hooks aren't yet attached, call `accumulate_only(model)` so its state begins building during the current phase.

Between steps, the model has been updated by exactly one sub-optimizer; nothing about the API differs from a single optimizer.

## Trigger types

The user can compose triggers from primitives. Each trigger answers "are we done with this phase?" given context (`step`, `loss`, `val_loss`, `wall_seconds`, `phase_step`).

| Trigger type | Param | Semantics |
|---|---|---|
| `StepCount` | `n` | True once the global step ≥ `n` |
| `PhaseStepCount` | `n` | True once steps within this phase ≥ `n` |
| `WallTime` | `seconds` | True once wall time elapsed in this phase ≥ `seconds` |
| `LossBelow` | `threshold, source ∈ {train, val}` | True once smoothed loss drops below `threshold` |
| `LossPlateau` | `window, min_delta` | True if loss hasn't improved by `min_delta` over the last `window` evaluations |
| `NextOptimizerWarm` | (none) | True once `state_is_warm` of the next phase's optimizer is True. Used as a *guard*, AND-ed with another trigger so we don't switch into a cold optimizer. |
| `Manual` | (none) | Never fires from internal state. The user can flip a flag externally (e.g., from a callback). Useful for interactive runs. |
| `And(t1, t2, ...)`, `Or(t1, t2, ...)`, `Not(t)` | composition | Boolean combinators |

Common composition: `And(PhaseStepCount(800), NextOptimizerWarm())` — switch at step 800, but only if the next optimizer is warmed. Falls back gracefully if pre-warming was disabled.

## Pre-warming: the key trick

Pre-warming is what makes the Adam→K-FAC recipe efficient. Without it:

- Phase A (Adam) runs from step 0 to N. Adam updates weights normally.
- At step N, K-FAC takes over. Its G and A EMAs are empty. The first few hundred K-FAC steps spend most of their work on building meaningful curvature estimates rather than producing useful updates.

With pre-warming:

- Phase A (Adam) runs from step 0 to N. **K-FAC's forward and backward hooks are attached**, accumulating G and A EMAs on every step. K-FAC's `step()` does not run.
- Cost during phase A: one extra activation snapshot per layer per step. Memory: O(seq_len × d_in) per linear layer for the activations buffer, plus the running A and G matrices. For SmallGPT this is small (~O(MB)).
- At step N, K-FAC takes over with EMAs that have already absorbed N samples. Warm start. The handoff is invisible in the loss curve.

The composite implements this by tracking, per phase, two flags:

- `live`: whether the optimizer is the active one (its `step()` runs).
- `pre_warming`: whether its data-collection hooks are attached and accumulating.

Transition: `pre_warming=True, live=False` → `pre_warming=False, live=True`. The hooks are simply re-tagged from accumulate-only to full mode, no re-registration.

A phase declaration controls this:

```
phase = {
    "name": "kfac_main",
    "optimizer": vered,
    "pre_warm_during": ["warmup"],   # accumulate during these earlier phases
    "until": None,
}
```

## Configuration format

A YAML or Python-dict config that the composite consumes at construction time. Designed to be readable in a benchmark log and easy to mutate for sweeps.

```yaml
optimizer:
  type: composite
  phases:

    - name: warmup
      optimizer:
        type: adam
        lr: 3.0e-4
        betas: [0.9, 0.999]
        weight_decay: 0.0
      until:
        type: phase_step
        n: 1000

    - name: cruise
      optimizer:
        type: vered_kfac
        lr: 8.0e-3
        damping: 1.0e-5
        gamma: 0.7
        momentum: 0.0
        grad_clip: 60.0
      pre_warm_during: [warmup]
      until:
        type: and
        children:
          - { type: loss_plateau, window: 500, min_delta: 0.005 }
          - { type: phase_step,   n: 3000 }   # min phase length floor

    - name: finetune
      optimizer:
        type: vered_kfac
        lr: 4.0e-3
        damping: 1.0e-6
        gamma: 0.5
        momentum: 0.0
        grad_clip: 60.0
      pre_warm_during: []   # no pre-warm; same family as cruise, can inherit warmth
      inherit_state_from: cruise   # explicit state-inheritance: takes cruise's G/A EMAs
      until: null
```

`inherit_state_from` is a special case: when the next phase uses the same optimizer family as the previous, you can inherit not just the model weights but also the curvature factors. The composite copies the EMAs at handoff, so the new phase starts truly warm, not merely pre-warmed. This is most useful for damping-schedule transitions (e.g., start with λ=1e-5 to maintain stability, drop to λ=1e-6 once converging, drop to λ=1e-7 for the final fine-tune).

## Handoff sequence (state-by-state)

For a transition from phase A to phase B at step N:

```
step N-1:  A is live, B may be pre-warming or not yet touched
           +-- A.step() runs
           +-- (if pre-warming) B.hooks accumulate G/A on A's forward/backward
                 -> B.state.warmth advances

step N:    composite.step() begins
           +-- scheduler reports: A's "until" trigger fired
           +-- composite calls A.detach()
                 -> A's hooks removed; A's state retained in memory in case
                    we resume / inspect, but no longer mutated
           +-- if B.pre_warming was on:
                 -> just flip B's hooks to full mode
              else:
                 -> composite calls B.attach(model)
                    -> B's state is empty; first ~min_warmup_steps of B's
                       step() may want to act conservatively (small lr) until
                       state.is_warm goes true
           +-- B.step() runs and produces the parameter update for step N
           +-- if any later phase has pre_warm_during including phase B:
                 -> attach those phases' accumulate_only hooks now

step N+1:  B is live, A is detached, possibly C is pre-warming
```

### What does *not* transfer at handoff (default)

- Adam's `m, v` buffers are dropped when leaving Adam.
- K-FAC's G/A EMAs are dropped when leaving K-FAC, *unless* the next phase opts in via `inherit_state_from`.
- LR-schedule state is per-phase. The composite resets the LR schedule at each transition unless the user explicitly shares one.

### What always transfers

- Model parameters (`param.data`). This is the whole point.
- Global step counter.
- Wall-time accumulator.
- Random number generator state (so resume after a transition is bit-reproducible if seeds match).

## Failure modes and mitigations

This is the section to read carefully before implementing.

| Failure mode | Cause | Mitigation |
|---|---|---|
| **Cold-start instability** at handoff | New optimizer has empty state, takes a too-large step | Phase-level LR warmup: a short ramp on B's LR for the first M steps after handoff, configurable via `cold_start_lr_warmup_steps` |
| **Pre-warm hooks slow down phase A measurably** | The hooks add a forward-pass tensor copy + matmul per linear layer | Provide `pre_warm_subsample` knob; pre-warming may sample only every K steps. K=4 typically reduces overhead to <5% with negligible state-quality loss |
| **Memory exhaustion from concurrent EMAs** | If multiple later phases are pre-warming at once | Disable nested pre-warm by default. Future phases pre-warm only after the immediately-prior phase begins. |
| **Trigger never fires** | LossPlateau with too-tight `min_delta` on a noisy run | Always AND triggers with a hard `step_count` cap as a safety net. Document this in the example configs. |
| **State_dict round-trip breaks across phase boundaries** | Each sub-optimizer has different keys; the composite has to namespace them | Composite's state_dict is a dict-of-dicts: `{phase_name: sub_state_dict, "scheduler_state": {…}, "active_phase": "cruise"}`. Resume re-attaches hooks for the active phase. |
| **Pre-warming misses gradient-clipping** | If A clips its gradients before backward, B's accumulated G_signal sees clipped grads, not raw ones | Document this. Recommend clipping in B's `step()` only, not in A's. For Adam this is fine because Adam doesn't typically clip. |
| **Hook double-registration** if `attach` called twice | The composite must enforce idempotency | `attach` records its handle list; calling twice raises. `detach` clears the list. |
| **Wrong order of `step()` and pre-warm hooks** | If pre-warm hooks register *after* the step's forward already ran, they miss this step's data | Composite attaches pre-warm hooks at *phase entry* (during the transition), not lazily at first use |
| **Loss-based triggers compute on a stale value** | If `val_loss` is evaluated only every K steps, the trigger fires K steps late | Document evaluation cadence as a known limitation; allow trigger to consume EMA-smoothed train_loss as a faster-but-noisier proxy |

## Test plan

The composite must be testable independently of the model. Recommended tests, in order of cheapness:

1. **Pure scheduler logic.** Construct triggers with synthetic step/loss histories, verify transition timing matches expected. No PyTorch needed.
2. **Sub-optimizer protocol conformance.** For each existing optimizer (Adam, ClassicKFAC, OlsSMKFAC, VeredKFAC), verify it implements `attach`, `detach`, `accumulate_only`, `state_is_warm`. Hook count before attach == 0; after attach == known number; after detach == 0.
3. **Hook idempotency.** Calling `attach` twice raises. Calling `detach` twice is a no-op.
4. **Pre-warm correctness.** Run K-FAC in pre-warm mode for N steps with a frozen model. Then put it live. Compare its EMA state to a K-FAC that ran live for the same N steps with the same model frozen. Should match exactly (same hooks, same order).
5. **Handoff parameter equivalence.** Run Adam-only for 100 steps, snapshot weights. Run composite (Adam → K-FAC at step 100) and verify weights at step 100 match Adam-only's weights at step 100.
6. **State-dict round-trip across handoff.** Save composite at step 90 (in Adam phase), at step 100 (transition step), at step 110 (K-FAC phase). Each loadable, each resumable, each yields bit-reproducible continuation given same seeds.
7. **Wall-time benchmark.** Composite vs K-FAC-only on a small SmallGPT run. Confirm the predicted ~15-20% wall-time reduction with no quality regression. This is the integration test that justifies the architecture's existence.
8. **Failure-injection tests.** A trigger that never fires (verify safety cap kicks in). A pre-warm hook that raises (verify graceful fallback to cold start with warning). A sub-optimizer's `state_is_warm` returning False forever (verify guard composition handles it).

## Implementation phases (incremental delivery)

This architecture is large enough to warrant phased delivery. Each phase below is independently useful.

**Phase 1: scheduler-only composite (no pre-warming).** Implement the sub-optimizer protocol minimally (no `accumulate_only`), the scheduler with `StepCount`, `PhaseStepCount`, and `WallTime` triggers, and the composite with cold handoff. Adam→K-FAC will work but the K-FAC phase starts cold. Buys the basic capability and is fully testable. ~3-5 days of work.

**Phase 2: pre-warming.** Add `accumulate_only` to KFAC variants, the `pre_warm_during` field to phase config, the hook re-tagging at handoff. This is where the wall-time win actually materializes. ~2-3 days.

**Phase 3: state inheritance for same-family transitions.** Add `inherit_state_from` for KFAC→KFAC handoffs (e.g., damping schedules). ~1 day.

**Phase 4: loss-based triggers and YAML config.** Add `LossBelow`, `LossPlateau`, the YAML loader. Mostly user-facing polish. ~1-2 days.

**Phase 5: integration with `run_probe`.** Make the existing benchmark harness accept a composite-optimizer config. Run the Adam→K-FAC experiment as the validation case. ~1 day.

Total: ~2 working weeks for the full feature. Phases 1 and 5 alone (~4 days) suffice to test the core hypothesis.

## Open design questions

These are decisions to make before implementation; flagged here so they're not buried.

1. **Where does `model` get passed to the composite?** PyTorch optimizers traditionally take `params`, not the whole model. But `attach` needs the model to find layers to hook. Options: (a) pass `model` explicitly to the composite at construction; (b) reconstruct module references by walking `param.parent` (fragile); (c) require sub-optimizers to receive `model` via a side-channel `attach(model)` call after construction. I lean toward (a) — explicit > clever.

2. **Pre-warming budget.** Is pre-warming always-on for opted-in phases, or capped to a max number of accumulations (in case phase A runs much longer than expected)? Lean: cap optional, default uncapped.

3. **What if two phases pre-warm the same sub-optimizer twice?** E.g., user accidentally lists the same optimizer in two phases. Lean: composite checks for object identity and rejects duplicate phases at construction.

4. **Logging granularity.** Per-step phase tag in the trace, per-transition events, or both? Lean: both. Per-step is `current_phase` in the metrics row. Per-transition is a one-time `transition` event with from/to/step/wall_time.

5. **Does the composite expose `param_groups` like a normal optimizer?** Yes, but it's the active sub-optimizer's `param_groups`. Mutates on transition. This is needed for LR-schedule libraries that read `param_groups[0]['lr']` directly.

6. **Should `inherit_state_from` deep-copy or reference?** Reference saves memory but means the prior phase's EMA continues to be updated if its hooks aren't fully detached. Deep-copy is safer. Lean: deep-copy with a knob to override.

7. **Behavior on resume mid-transition.** If the run was killed *during* the transition step, what state are we in? Lean: scheduler emits transition events atomically — the state_dict captures either pre-transition or post-transition state, never mid-transition.

## Why this architecture, and not simpler alternatives

A few less-flexible alternatives are worth naming and dismissing.

- **Just write a script that runs Adam, then runs K-FAC as a separate program.** Doesn't preserve in-memory state (model has to be saved/loaded). Doesn't allow pre-warming. Rules out future per-layer composition. Fine for one-off experiments; not architecturally a foundation.

- **Hard-code Adam-then-KFAC in a subclass of K-FAC.** Works for exactly that one case. Adding a third phase, or a different first optimizer, requires more subclassing. The combinatorics get unmanageable.

- **Use a callback-based scheduler in the trainer instead of in the optimizer.** This pushes composition into the training loop. Acceptable, but every benchmark and every trainer has to know about the composition logic. The optimizer-side encapsulation here keeps callers ignorant: a composite is just an optimizer.

The composite is the right abstraction because (a) the unit of composition is *optimizer state*, not *training-loop state*, and (b) pre-warming requires hook ownership, which only an optimizer has cleanly.

## Out of scope but worth flagging for v2

- **Per-parameter-group composition** (Adam on embeddings, K-FAC on transformer blocks). Build on the same protocol but with a "parallel composite" that routes parameter groups to different sub-optimizers each step.
- **Adaptive trigger learning.** Use a meta-controller that learns when to switch based on training dynamics. Research-grade.
- **Curvature transfer between K-FAC variants.** Translating Classic's full-matrix factors to Vered's QR R-factors. Possibly free quality wins, but requires real math work.
- **Differentiable phase boundaries.** Soft transitions where for K steps we blend `(1-α)·A_grad + α·B_grad`. Avoids handoff discontinuities entirely. Sensible for very LR-sensitive models.

---

The plan above is sized so that phases 1 and 5 (cold-handoff composite + benchmark integration) are deliverable in roughly 4 working days and would validate or reject the Adam→K-FAC hypothesis cheaply, without committing to the full pre-warming machinery upfront.
