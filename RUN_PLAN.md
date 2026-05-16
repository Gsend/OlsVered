# Run Plan: Three-Phase K-FAC Investigation Closeout

Concrete, copy-pasteable command sequences for executing
[NEXT_PHASE_TESTING_PLAN.md](./NEXT_PHASE_TESTING_PLAN.md). Each phase
has a launch command, expected wall time, what to look for in the output,
and a decision rule for what to do next.

All commands are PowerShell; run from `C:\Users\Admin\OlsVered`.

## Status check before starting

Before launching anything, verify the prerequisites:

```powershell
# 1. Vered grid winner exists (681 ppl)
Get-ChildItem benchmark\results\vered_grid_g0.30_m0.30_l1e-04.json

# 2. The four scripts you'll be running exist
Get-ChildItem benchmark\vered_gradclip_probe.py
Get-ChildItem benchmark\classic_full_grid.py
Get-ChildItem benchmark\multi_arch_probe.py

# 3. Python env activates and CUDA is visible
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA GeForce RTX 3080 Laptop GPU`.

## Phase 1 — grad_clip optimization (~3.75 hours)

### Launch

```powershell
.\run_gradclip_probe.ps1
```

### What it does

Sweeps `grad_clip in {30, 15, 8}` at the Vered winner config
(γ=0.3, mom=0.3, λ=1e-4, lr=8e-3). The grad_clip=60 reference (681 ppl)
is reused from the existing winner JSON.

Three new runs, ~75 min each. Output JSONs:
`benchmark/results/vered_clip_c{30,15,8}.json`.

### What to look for

The probe ends with one of four diagnoses, printed in the final summary:

| Output line | Meaning | Next action |
|---|---|---|
| `grad_clip=30 beats grad_clip=60 by N ppl` (N ≥ 10) | Phase 1 prior optimum confirmed at new operating point. | Optionally probe {20, 25, 35} for refinement (~2.5 hr). |
| `grad_clip=15` or `=8 wins by ≥10 ppl` | Tighter than expected. | Probe {3, 5, 10, 12} to find U-curve minimum (~3 hr). |
| `All clip values within N ppl: grad_clip is insensitive` | No gain available from this axis. | Move directly to Phase 2. |
| `grad_clip=8 DIVERGED` | Lower bound is between 8 and 15. | Probe {10, 12} to refine (~1.5 hr). |

### Decision

If Phase 1 produces a new winner, the new value becomes the canonical
grad_clip for Phase 2 and Phase 3. Update `BASE["grad_clip"]` in
`benchmark/classic_full_grid.py` to the new value before launching Phase 2:

```python
# in classic_full_grid.py, line ~70
BASE = {
    "variant":   "ClassicKFAC",
    "kfac_lr":   8e-3,
    "grad_clip": 60.0,   # <-- update to Phase 1 winner if applicable
    ...
}
```

If Phase 1 produces no improvement, leave `grad_clip=60` and move on.

## Phase 2 — Classic K-FAC optimization (~14-16 hours)

### Launch

```powershell
.\run_classic_grid.ps1
```

### What it does

Three-stage coordinate-descent grid for ClassicKFAC, mirroring
`vered_full_grid.py` but with Classic-anchored sweep ranges:

- **Stage A** (γ sweep at mom=0.9, λ=1e-4): γ ∈ {0.7, 0.9, 0.5, 0.95, 0.3}.
  The γ=0.3 cell is the κ⁴ stability test — Classic should diverge or
  strongly underperform if the math doc's prediction holds.
- **Stage B** (mom sweep at best γ): mom ∈ {0.9, 0.7, 0.5, 0.95}.
- **Stage C** (λ sweep at best (γ, mom)): λ ∈ {1e-4, 1e-3, 1e-5, 1e-6}.

12 cells, ~75 min each. Output JSONs:
`benchmark/results/classic_grid_*.json`.

### What to look for

The grid prints summaries after each stage. The final summary ends with
a verdict based on Classic's tuned best:

| Output line | Meaning | Next action |
|---|---|---|
| `Classic overall best: NNN ppl ... Vered vs Classic gap: X ppl` (X<20) | Reframe headline as marginal win. | Multi-arch probe still useful; expectations dampened. |
| `Vered vs Classic gap: 20-90 ppl` | Expected outcome; both wins are substantive. | Continue to Phase 3. |
| `Classic overall best ≥ 776` | No tuning gain over default; strongest case for Vered. | Continue to Phase 3 confidently. |
| `Classic at gamma=0.3 DIVERGED ... kappa^4 prediction empirically confirmed` | Headline result. | Note in math doc; keep all three phases' findings. |

### Decision

After Phase 2, you have each variant's best config on SmallGPT:

| Variant | Best config (after Phase 2) |
|---|---|
| VeredKFAC | γ=0.3, mom=0.3, λ=1e-4 (or whatever Phase 1 gradclip probe surfaces) |
| ClassicKFAC | filled in by `classic_grid_*.json` winner |
| OlsSMKFAC | not yet covered (Phase 2C optional) |

Record the Classic winner's hyperparameters explicitly. You'll need them
for Phase 3.

## Phase 3 — Multi-architecture generalization (~2-4 hours, with caveats)

### Pre-launch: update training_benchmark.py hardcoded configs

`benchmark/multi_arch_probe.py` subprocess-calls
`benchmark/training_benchmark.py`, which has *hardcoded* hyperparameters
per variant. For a properly fair Tier 1 comparison, update those values to
match the Phase 2 grid winners.

Edit `benchmark/training_benchmark.py` near the top, find the per-method
config dicts, and update:

```python
# ClassicKFAC config -- update with Stage 2 winner's (gamma, mom, lambda, grad_clip)
ClassicKFAC(model, lr=..., damping=<phase2_winner>, momentum=<phase2_winner>,
            grad_clip=<phase2_winner>, gamma=<phase2_winner>)

# OlsSMKFAC config -- only if you ran Phase 2C; otherwise leave default
OlsSMKFAC(model, lr=..., damping=..., momentum=..., grad_clip=..., gamma=..., adaptive=True)

# VeredKFAC config -- update with vered_full_grid.py winner
VeredKFAC(model, lr=8e-3, damping=1e-4, momentum=0.3, grad_clip=60.0, gamma=0.3)
```

Note: the LR for MLP/CIFAR may need to differ from the SmallGPT LR. Use
the existing values as a starting point and only update the (γ, mom, λ,
grad_clip) tuple from the SmallGPT grid.

If you skip this update, the probe still runs but with stale hardcoded
configs (call this "Tier 0" — useful baseline but doesn't cleanly test
the Phase 2 results).

### Launch

```powershell
.\run_multi_arch_probe.ps1
```

Defaults: probes MNIST, CIFAR, and SmallGPT for all three K-FAC variants.

For a faster first pass that skips CIFAR (its ConvNet runs are the
slowest):

```powershell
.\run_multi_arch_probe.ps1 -Archs mnist,smallgpt
```

For only one variant (e.g., to debug):

```powershell
.\run_multi_arch_probe.ps1 -Optimizers VeredKFAC
```

### What to look for

The probe prints a per-architecture × per-variant table:

```
  Variant         mnist                   cifar                   smallgpt
  ----------------------------------------------------------------------------
  ClassicKFAC     val_acc=0.9XXX          val_acc=0.8XXX          val_ppl=NNN.N
  OlsSMKFAC       val_acc=0.9XXX          val_acc=0.8XXX          val_ppl=NNN.N
  VeredKFAC       val_acc=0.9XXX          val_acc=0.8XXX          val_ppl=NNN.N
```

For MNIST and CIFAR, higher `val_acc` is better. For SmallGPT, lower
`val_ppl` is better.

### Decision

| Outcome | Implication |
|---|---|
| Vered wins all 3 architectures | Multi-arch claim is solid. Proceed to write-up. |
| Vered wins 2/3 | Document the architecture where it loses; narrow the claim. |
| Vered wins 1/3 (only SmallGPT) | Win is transformer-specific. Still real but narrower. |
| Vered loses all 3 | Surprising; investigate (likely a bug in the probe — verify the hardcoded configs were updated correctly). |
| Results are very close | Run Tier 2 (per-arch tuning) per `NEXT_PHASE_TESTING_PLAN.md` Phase 3B. |

## Aggregate timing

Three-night execution plan:

```
Night 1:  Phase 1 (grad_clip probe)            ~3.75 hr
          + optional refinement sweep          +2.5 hr if needed

Night 2:  Phase 2 (Classic grid)               ~14 hr
          + Phase 2B (Classic gradclip probe)  +3.75 hr (run after Phase 2 if time)

Night 3:  Phase 3 (multi-arch probe)           ~2-4 hr
          (preceded by manual update of training_benchmark.py configs)

Total compute: ~25-30 hours.
Total elapsed: ~3 days.
```

## Resume support

All four scripts (`vered_gradclip_probe.py`, `classic_full_grid.py`,
`multi_arch_probe.py`, plus the existing `vered_full_grid.py`) skip any
cell whose JSON output already exists. If a run is interrupted (Ctrl-C,
power loss, GPU error), just re-launch the same script — completed
cells are preserved.

If you want to *force* a re-run of a specific cell, delete its JSON
first:

```powershell
# Example: re-run grad_clip=30
Remove-Item benchmark\results\vered_clip_c30.json
.\run_gradclip_probe.ps1
```

## Where each output lives

```
benchmark/results/
├── vered_grid_*.json              <- Vered full grid (already done)
├── vered_clip_c*.json             <- Phase 1 grad_clip probe
├── classic_grid_*.json            <- Phase 2 Classic grid
├── multi_arch_probe_summary.json  <- Phase 3 unified summary
├── training_results.json          <- Phase 3 raw (training_benchmark.py)
└── (existing transformer/MLP/cifar JSONs from prior runs -- mostly stale)
```

## What to do if a phase produces unexpected results

Three failure modes worth being prepared for:

1. **Phase 1 grad_clip=8 diverges immediately.** Then the per-step
   gradient norm is regularly above 8 even at the basin. Run the refinement
   probe at {10, 12, 15, 20} to find the true lower bound.

2. **Phase 2 Classic grid blows up at multiple cells beyond just γ=0.3.**
   May indicate Classic with `lr=8e-3` is unstable for it specifically.
   Try `lr=4e-3` or `lr=2e-3` for Classic. The original `gpu_benchmark.py`
   uses `lr=7e-3` for Classic on transformer; that's a sensible alt.
   Edit `BASE["kfac_lr"]` in `classic_full_grid.py`.

3. **Phase 3 probe runs but reports `(missing)` for some cells.** Check
   the `training_results.json` raw output to see what was produced. The
   probe's variant-name fuzzy matching may not have matched correctly.
   Adjust the `optimizer_name` strings in the probe if needed.

If any of these happen, the resume support means the existing successful
runs are preserved and only the affected cells need re-running.
