# Archived screen results — cosine schedule, T_max = max_steps − warmup

Archived on 2026-05-12.

## What's in here

32 JSON files from the first pass of the 2D (momentum × learning rate)
screening grid, run via `benchmark/vered_2d_mom_lr_screen.py`.  Filenames
follow `vered_2dscreen_m{mom}_lr{lr}_s{steps}.json` — note the **absence**
of the `_const` suffix that newer runs carry.

## Why these are archived rather than deleted

The runs in this folder used `run_probe`'s **default** `lr_schedule`,
which at the time was the only option:

```
warmup = 200
cosine_steps = max(1, max_steps - warmup)
scheduler = SequentialLR(...,
    LinearLR(start_factor=0.1, end_factor=1.0, total_iters=warmup),
    CosineAnnealingLR(T_max=cosine_steps, eta_min=kfac_lr * 0.0015),
)
```

With `max_steps = 1000`, this gives `T_max = 800` for the cosine phase.
At step 469 the effective LR is already at ~75% of its peak; by step
1000 it has decayed to roughly `eta_min`.  That makes the cell labeled
`lr=8e-3` train at LR ranging from 8e-4 → 8e-3 → 1.2e-5 over the run,
not at a constant 8e-3.

The problem this caused: comparing screen results (1000 steps,
T_max=800) to the historical 5000-step Vered reference
(`benchmark/results/vered_clip_g0.90_c300.json`, which used
T_max=4800) is apples-to-oranges.  At step 469 the historical run was
at LR≈7.94e-3 while the screen was at LR≈5.97e-3 for the nominally
identical cell, which is the entire reason the screen's top-5 cells
landed at ~1800-2000 ppl at step 1000 while history showed ~1000 ppl
at the same step.

## What replaced these

After 2026-05-12 the screen runs with `lr_schedule="constant_warmup"`:
linear warmup (200 steps) → constant at `kfac_lr` for the remainder.
"lr=X" now means "trains at LR=X after warmup" regardless of run
length.  New JSONs land at the same `results/` folder with a `_const`
suffix in the filename so they cannot be confused with files in this
archive.

Relevant code: `benchmark/stability_benchmark.py::run_probe` (the
`lr_schedule` argument and the scheduler block immediately following).

## Can these still be used?

For comparing optimizer behavior under aggressive cosine decay, yes.
For predicting 5000-step refinement ranking or for any sweep where
"lr" should mean a single scalar LR, no — use the `_const` files.
