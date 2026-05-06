# Session Context: K-FAC Stability + Gradient-Clip Benchmark

*Saved 2026-05-04. This is the session that ran the gradclip benchmark and
produced the first empirical Vered-wins-on-convergence result.*

---

## What this session built

A second-generation benchmark for comparing the three K-FAC variants
(`ClassicKFAC`, `OlsSMKFAC`, `VeredKFAC`) on the SmallGPT / WikiText-2
pre-training task, with three orthogonal experimental axes:

| Axis | Constant | Purpose |
|---|---|---|
| LR sweep (`stability_benchmark.py`) | grad_clip, damping | Find max stable LR per variant |
| Grad-clip frontier (`gradclip_benchmark.py`) | LR, momentum, damping | Find max stable clip per variant |
| Future damping ablation (planned) | LR, clip | Find min stable damping per variant |

The gradclip benchmark was the focus and now has working Phase 1 (frontier
sweep) + Phase 2 (5000-step convergence) with per-variant damping, per-variant
clip overrides, output tagging for parallel configs, and crash-resistant
incremental save.

## The empirical Vered-wins-on-convergence finding

**Phase 2 at LR=0.008, mom=0.9, dampings (Classic 1.5e-4, OlsSM 1.5e-4, Vered 1e-5):**

| Variant | clip | final_ppl @ 5000 steps | wall (min) |
|---|---|---|---|
| ClassicKFAC | 120 | 1153 | 22.0 |
| OlsSMKFAC | 120 | 1152 | 23.5 |
| **VeredKFAC** | **60** | **937 (-19%)** | **60.2** |

Plus the condition-number tracking showed Vered's κ(G) stays at ~1.0 throughout
training while Classic/OlsSM see κ(G) ≈ 130 average / 10⁴-10⁵ peak. That's the
first empirical confirmation of the κ-scaling story from `KFAC_VARIANTS_MATH.md`.

Trade-off: Vered is 2.7× slower per step.

## Open hypotheses about why Vered's wins are inconsistent

The Vered convergence result was clean at one config but Vered loses at low
clip (3-30). Three explanations being investigated:

1. **`_SEQ_SUBSAMPLE = 512` breaks `p ≥ n` for FFN layers** — confirmed by
   warning logs. SmallGPT FFN layers have `n_out=1024`; first factor update
   has only 512 rows accumulated; layer falls back to Classic K-FAC for that
   window. **Fix queued: bump to 2048.**

2. **Sample-noise amplification at low damping** dominates Vered's stability
   gain. Vered at λ=1e-5 has 5e-6 inverse magnitude → noisy small-eigenvalue
   directions amplify, hurting convergence even though stability is fine.
   **Mitigation tried: bumped Vered damping from 1e-5 to 5e-5.**

3. **Possible bug**: Vered's 4-TRSM apply path or streaming TSQR's R-factor
   normalization. Apply formula audited and looks correct
   (`G⁻¹ ∇L A⁻¹` via R_G/R_X solves). Streaming TSQR uses `_positive_diagonal_R`
   for sign consistency but should be unit-tested.

## Performance reality check

| Variant | step_ms | FLOP per step | Effective rate (TF32 on RTX 3080) |
|---|---|---|---|
| Classic | ~225 | 2.1n³ | ~80 TFLOPS GEMM (Tensor Cores) |
| OlsSM | ~250 | 4.0n³ | ~12 TFLOPS TRSM (no Tensor Cores) |
| Vered | ~645 | ~12n³ (current), ~4.3n³ (optimized) | TRSM + 128 cuSOLVER QR launches/step |

Vered's per-step cost is launch-bound (128 cuSOLVER QR calls per step).
Theoretical FLOP minimum is 2× Classic — cannot beat Classic per step on
current GPU hardware. Wins must come from convergence-per-step, not
wall-time-per-step.

## Files added or significantly modified this session

```
benchmark/
  gradclip_benchmark.py            ← new, the grad-clip frontier sweep
  stability_benchmark.py           ← refactored: per-variant damping, val-cap
                                     detector, lazy probe-budget exit
optimizer/
  olssm_kfac.py                    ← progressive damping retry + EVD-projection
                                     fallback for non-PSD Cholesky
  raw_activation_hooks.py          ← _SEQ_SUBSAMPLE param (currently 512)
  vered_kfac.py                    ← max_seq_rows passthrough

run_gradclip.ps1                   ← Windows launcher
KFAC_VARIANTS_MATH.md              ← updated with section 8 (TRSM vs GEMM)
PERFORMANCE_IMPROVEMENT_TASKS.md   ← consolidated speedup backlog
PLAN_dual_boot_linux.md            ← linux migration plan
diagnose_shutdown.ps1              ← BSOD post-mortem helper
Gilad_Senderovich_CV_2026.docx     ← updated CV
```

## Three things tested in current session that need follow-through

### A. Unit-test Vered vs Classic on well-conditioned synthetic data
If both variants don't agree on `ΔW` for well-conditioned `(A, G, ∇L)`, there's
a bug in Vered. Test should live in `tests/test_kfac_equivalence.py`.

### B. Bump `_SEQ_SUBSAMPLE = 2048`
Eliminates `p < n` warnings for FFN layers, gives Vered true preconditioning
from step 1. Per-step cost goes up ~4× because TSQR processes 4× more rows,
but warning goes away and convergence may improve.

### C. Low-clip + high-LR config
A regime where clip dominates magnitude so direction quality decides
convergence. Suggested: `LR=0.05, mom=0.9, clip ∈ {0.3, 1.0, 3.0}`. If Vered
wins here cleanly, κ¹ direction-quality advantage is real.

## Known config drift across runs

- `KFAC_LR` was 8e-3 → 8e-2 → 8e-3 across iterations
- `VARIANT_DAMPINGS` history:
  - (1e-3, 2e-4, 1e-4) — first run
  - (1.5e-4, 1.5e-4, 5e-6) — when LR was bumped to 8e-2
  - (1.5e-4, 1.5e-4, 1e-5) — current
- `GRAD_CLIP` was 10 → 100 → 1000 → 100 across iterations

Each variant's "correct damping" is task-and-LR dependent; the gradclip
benchmark JSON now records `variant_dampings` so resume logic can detect changes.

## Open / pending decisions

- Should Classic damping also drop (currently 1.5e-4, was 1e-3)? Probably yes,
  but reduces fairness of comparison if changed mid-experiment.
- Phase 2 step budget: 5000 currently. Long-run (10000) experiment queued
  via `--phase2-tag long --phase2-steps 10000`.
- `lambda_bcd_scale` flag for the retrainer benchmark — to test whether
  `lambda_eff = 1.0` cap is what's holding N=4/N=8 retrainer at 87.3%.

## Quick-reference launch commands (Windows / PowerShell)

```powershell
# Standard Phase 1 + Phase 2 at current config
.\run_gradclip.ps1

# Just Phase 2 at custom clips, with output tag
python benchmark\gradclip_benchmark.py --phase 2 `
  --phase2-clips "ClassicKFAC=60,OlsSMKFAC=60,VeredKFAC=120" `
  --phase2-tag matched

# Long-run convergence at clip=120
python benchmark\gradclip_benchmark.py --phase 2 `
  --phase2-clips "ClassicKFAC=120,OlsSMKFAC=120,VeredKFAC=120" `
  --phase2-steps 10000 --phase2-tag long

# Power settings before any long run
powercfg -change -standby-timeout-ac 0
powercfg -change -hibernate-timeout-ac 0
powercfg -change -monitor-timeout-ac 0
```

## Off-topic but still relevant artifacts produced

- `Gilad_Senderovich_CV_2026.docx` — updated CV (Odysight.AI current,
  Rafael 2019-21, OlsVered project section, "highly creative and strongly
  success-oriented" framing).
- `PLAN_dual_boot_linux.md` — Linux migration plan (BSOD'd during overnight
  run; root cause `KMODE_EXCEPTION_NOT_HANDLED` with `STATUS_ILLEGAL_INSTRUCTION`
  in kernel mode, almost certainly NVIDIA driver under sustained CUDA).
