# Claude Code Work Plan — Q10, E6, LoRA-TP

**Audience.** A Claude Code agent with no prior conversation context. Read this
top to bottom, then read the referenced source files before writing code.

**Environment.** Run on the user's machine (CUDA available; `.venv` active).
All commands are `python -m diagnostic.runners.<...>`. There is no internet for
package installs; everything needed is already installed. CIFAR-10 and MNIST are
already in `data/`.

**Repo.** `C:\Users\Admin\OlsVered`. The relevant subtree is `diagnostic/`
(framework) and `diagnostic/runners/arch_coverage/` (experiments). Results go to
`benchmark/results/diagnostic/arch_coverage/<exp>/`, weights to
`benchmark/weights/arch_coverage/`.

**Context in one paragraph.** This project tests a target-propagation + closed-form
OLS retrainer (no backprop) across architectures. Established result: *feature-level
distillation* (fit each layer's weights by OLS to reproduce a teacher's captured
per-layer pre-activations) is **exact and closed-form everywhere** (MLP, CNN, ResNet,
transformer). *Label chain-retraining* (back-propagate a one-hot/label target down the
layers, OLS-fitting each) works for plain MLPs but is **blocked by non-invertible ops**
— max-pool (E3, E5) and softmax attention (E8). These three work packages close the
remaining gaps. Do them in order; each has a hard acceptance gate.

---

## 0. Read first (do not skip)

| File | Why |
|------|-----|
| `diagnostic/target_prop_retrainer.py` | `solve_ols_layer(X, T_pre, with_bias, ols_lambda)` — the OLS primitive. Also the forward-sweep retrainer. |
| `diagnostic/inversion.py` | `invert_layer(W, b, a_out_post, a_pre_forward, activation, method, mu_a, Sigma_a, eps, sigma2)` and `invert_activation(a_post, a_pre_forward, activation)`. **Read the activation semantics carefully** (see Gotchas). |
| `diagnostic/capture.py` | `collect_activations(...)`; conv im2col handling. |
| `diagnostic/runners/arch_coverage/e3_conv_diag.py` | Reference for conv im2col + `F.fold` + switch-aware max-unpool + rebuilt-upstream forward sweep. Copy these patterns. |
| `diagnostic/runners/arch_coverage/e5_resnet.py` | Reference for `_im2col`, `_fold_to_spatial`, `_switch_unpool`, `_conv_pre_2d`, additive-skip decomposition, GAP inverse. |
| `diagnostic/runners/arch_coverage/e8_transformer.py` | Reference for transformer capture, feature-distill of Linears, attention-opaque probe, `_ln_inverse`. |
| `LORA_TP_OLS_PLAN.md` | Full math + experiment design for WP-C. |
| `arch_coverage_findings.md`, `arch_coverage_queue.md` | Log results here when done (one row per run). |

---

## 1. Shared conventions & gotchas (these caused real bugs — obey them)

1. **Device/dtype discipline.** Activation captures are CPU; model weights may be
   on CUDA. Do *all* OLS/inversion math in **CPU + float64**. Extract weights with
   `layer.weight.detach().double().cpu()` (and `.reshape(C_out, -1)` for conv).
   Write back with `.to(layer.weight.device, layer.weight.dtype)`. A single tensor
   left on CUDA in an OLS/inversion call throws a device-mismatch RuntimeError.

2. **`invert_layer` activation semantics.** `a_out_post` is the **post-activation**
   target. `activation` is the activation that *follows* this layer in the forward
   pass. **Pass `activation=None` for any layer with no following activation** —
   final heads, and projections that feed a residual add (e.g. transformer `Wo`,
   `fc_out`, ResNet residual-branch `conv2`). `a_pre_forward` must have the same
   `d_out` as `a_out_post` and is only used for the ReLU/GELU dead-unit mask. Passing
   a non-`None` activation with a wrong-dim `a_pre_forward` is a bug (cost us a crash
   on `fc_out`: it has *no* activation; the ReLU is on `fc_in`).

3. **Rebuilt-upstream forward sweep.** When fitting weights, each layer's OLS input
   `X` must be the output of the *just-rebuilt* previous layer, not the captured
   `a_in`. Only the entry layer uses captured `a_in` / raw model input. (Both E5 and
   E3 do this; copy it.)

4. **Conv as linear.** im2col via `F.unfold` → `(B*Hout*Wout, C_in*kH*kW)`; weight is
   `(C_out, C_in*kH*kW)`. To turn an im2col-space input target back into a spatial
   target, fold with overlap-count normalization — see `_fold_to_spatial` in
   `e5_resnet.py`.

5. **Max-pool inverse = switch-aware unpool** (`F.max_unpool2d` with stored argmax
   indices), never nearest-neighbor broadcast. See `_switch_unpool`.

6. **Moment correction OFF by default** for nets deeper than ~4 layers (mom-match is
   harmful past depth 4). Single pass only (`n_iterations=1`); iteration always drifts.

7. **Reproducibility.** `seed=42`. Eval on the held-out test split. Save JSON per
   battery item; print a one-line residual summary per layer.

8. **Sandbox cannot run torch** — only the user's machine can. Write code, then run it
   yourself via the shell on the machine; debug option-by-option.

---

## 2. WP-A — Q10: isolate the attention wall with an attention-REQUIRING task

**Why.** E8's retrain-probe gave 77% on the "majority" task, but that is *confounded*:
majority is solvable by uniform averaging, which random attention approximates, so the
reachable top-block layers recover most accuracy even with `Q/K/V` frozen random. We
need a task where **random attention destroys the signal**, so the probe actually
measures the wall.

**Task: pointer / induction.** `make_pointer_dataset(n, seq_len, vocab, seed)` in
`diagnostic/models/tiny_transformer.py`:
- `token[:, 0]` = a pointer `p` drawn uniformly from `[1, seq_len-1]`.
- `token[:, 1:]` = payload tokens drawn uniformly from `[0, vocab)`.
- label = `1` if `token[p] >= vocab//2` else `0` (i.e. fetch the token at the
  data-dependent position the pointer names, then threshold it).
This **requires** content-based attention (read pointer, attend to position `p`);
averaging cannot solve it.

**Changes.**
- Add `make_pointer_dataset` next to `make_majority_dataset`.
- Add `--task {majority,pointer}` to `e8_transformer.py`; thread it through
  `_datasets()`, weight filename (`tiny_transformer_<task>.pt`), and results subdir
  (`e8/<task>/`). Default `majority` (preserves prior numbers).
- Add `--fit-head` to the retrain probe: also do a direct last-layer OLS on the head
  (currently the head is only propagated *through*, never fit). Report acc both with
  and without, so the reachable-set is the canonical one.

**Run & acceptance.**
```
python -m diagnostic.runners.arch_coverage.e8_transformer --task pointer --option train
python -m diagnostic.runners.arch_coverage.e8_transformer --task pointer --option distill
python -m diagnostic.runners.arch_coverage.e8_transformer --task pointer --option retrain
```
- **Gate A1:** B-train pointer acc **> 0.90**. If not, the task is unlearnable by this
  tiny model — increase `depth`/`d_model` or simplify (e.g. pointer in `[1,4]`), and
  re-confirm before proceeding. (A bad task invalidates the whole probe.)
- **Gate A2:** B-distill pointer acc ≈ teacher (residuals ~1e-3). Must stay exact —
  distillation is architecture-agnostic; if it drops, there is a capture/forward bug.
- **Expected result:** retrain-probe acc **near chance (~0.5)** and far below the
  majority-task 77%. That gap (vs distill ≈ teacher) is the clean attention wall.
- Log an updated E8 row in `arch_coverage_findings.md` comparing majority vs pointer;
  mark Q10 resolved in `arch_coverage_queue.md`.

---

## 3. WP-B — E6: small U-Net, dense per-pixel target, concat skips

**Why.** Last untested structural primitive: **channel-concat skip** + a **dense
continuous per-pixel target**. Tests H4 (continuous dense targets carry retraining
signal) on a skip architecture.

**Task.** MNIST denoising autoencoder. Input = clean image + Gaussian noise
(`std≈0.5` on normalized pixels); target = clean image. Metric = reconstruction **MSE**
(not accuracy). Reuse `_load_mnist` from `phase1_mlp_mnist.py`. Resize/pad MNIST to
`(1,32,32)` so two 2× downsamples are clean, or keep 28×28 with one downsample — your
choice; document it.

**Model.** `diagnostic/models/unet.py`, BN-free (consistent with E5):
- Encoder: `enc1 = conv→ReLU→conv→ReLU` (C=16) → `MaxPool2d(2, return_indices)` →
  `enc2 = conv→ReLU→conv→ReLU` (C=32) → bottleneck.
- Decoder: `MaxUnpool2d` (use the encoder's stored indices) → **concat** the encoder
  skip feature on the channel axis → `dec = conv→ReLU→conv→ReLU` → ... → final
  `conv(→1 channel)` (no activation; continuous output).
- Provide `forward_with_state` returning every conv pre-activation, both pool index
  sets, the skip features, and the concat inputs (mirror E5's `forward_with_state`).

**Q-for-E6 (concat-skip handling), analogous to E5's Q1:**
- **Distill:** no special handling — each conv is OLS-fit to its captured
  pre-activation given the rebuilt-upstream input; the concat is replayed in the
  forward sweep. Expect exact.
- **Retrain:** the decoder conv input is `[upsampled ; skip]` along channels. Inverting
  it yields an input target; **split it back along the channel axis** into a target for
  the upsampled-decoder branch and a target for the encoder-skip feature. The skip-branch
  target then *also* becomes a target for the encoder feature it came from (two targets
  meet at the encoder feature — sum them, analogous to E5's additive case). Use
  `MaxUnpool2d` with the encoder's stored indices for the decoder upsample inverse
  (its forward-preserving adjoint), and `_fold_to_spatial` for the conv inverses.

**Battery & acceptance.**
```
python -m diagnostic.runners.arch_coverage.e6_unet --option train
python -m diagnostic.runners.arch_coverage.e6_unet --option distill
python -m diagnostic.runners.arch_coverage.e6_unet --option retrain
```
- **Gate B1:** B-distill MSE ≈ teacher MSE (near-exact recovery). If distill is not
  near-exact, halt — it contradicts the cross-arch finding and means a capture/concat
  bug. Debug distill before retrain.
- **Expected:** retrain MSE between random and distill (continuous dense targets should
  carry *more* signal than one-hot, per H4 — compare to E4 autoencoder numbers in the
  findings log). Report random / distill / retrain MSE.
- Log an E6 row; update the queue and the H4 status.

---

## 4. WP-C — LoRA-TP via reduced-rank regression

**Why.** A per-layer LoRA fit under target propagation is exactly reduced-rank
regression, solvable closed-form (one OLS + one truncated SVD), inversion-free. See
`LORA_TP_OLS_PLAN.md` for full math. This WP delivers the primitive + the correctness
gate + the first distillation experiment.

**Deliverable 1 — `diagnostic/lora.py`:**
`solve_lora_layer(X, T, W0, b0, r, alpha, *, lam=1e-4, metric="whitened") -> (A, B, residual)`
- `s = alpha / r`.
- Residual target: `T' = T - (X @ W0.T + b0)`  (base weight+bias frozen).
- Full OLS on `T'`: reuse `solve_ols_layer(X, T', with_bias=False, ols_lambda=lam)` → `M` (`d_out×d_in`).
- Fitted values `Yhat = X @ M.T`.
- Rank-r subspace:
  - `metric="output"`: `V_r` = top-r eigenvectors of `Yhat.T @ Yhat` (`d_out×d_out`) via `torch.linalg.eigh`.
  - `metric="whitened"` (default, textbook RRR optimum): top-r right-singular subspace of
    `Yhat` in the `X.T X` metric — implement and document precisely; verify it equals
    `output` at full rank (both must reduce to `M`).
- `W_r = V_r @ V_r.T @ M`; then `B = V_r / s` (`d_out×r`), `A = V_r.T @ M` (`r×d_in`),
  so `s * B @ A == W_r`.
- Return residual `‖X (s B A).T - T'‖ / ‖T'‖`.
- Add a minimal `LoRALinear(W0, b0, r, alpha)` module whose forward is
  `x @ (W0 + s B A).T + b0`.

**Deliverable 2 — `diagnostic/runners/lora/lora_transformer.py`:**
The substantive experiments run on the **TinyTransformer** (E8) — LoRA's native
habitat (the Q/K/V/O + MLP projections are exactly the adapter targets), and we already
proved its Linears are OLS-distillable to 100%, which is the rank-∞ ceiling.

> **Why adaptation, not self-distillation.** Fitting LoRA adapters over a frozen base
> *into that same base's own targets* is degenerate: the residual target
> `T' = T - (X W0^T + b0) ≈ 0`, so the optimal adapter is ~zero at any rank — a flat,
> meaningless curve. LoRA is meaningful only when the targets come from a **different**
> task than the frozen base was trained on. So all rank sweeps below are adaptation.

- **LR0 (correctness gate, architecture-agnostic):** a small standalone unit test —
  take one Linear (a transformer block's `Wq` is fine, or a random `Linear(64,64)` on
  random `X`), set `r = min(d_in, d_out)` (full rank), and assert `solve_lora_layer`
  residual ≈ full-OLS residual (`torch.allclose`, atol ~1e-6) for **both** metrics.
  **If LR0 fails, STOP and report** — the RRR projection/metric is wrong; nothing
  downstream is valid. (Keep this cheap; it does not need a full model.)

- **LR1 (transformer LoRA adaptation, the headline):**
  1. Base = TinyTransformer trained on **task A** (majority); freeze ALL base weights.
  2. Teacher = TinyTransformer trained on **task B** (pointer); capture its per-layer
     pre-activation targets via target propagation / `forward_with_state` on task-B data.
  3. Inject LoRA adapters on the projections (start with all of Q/K/V/O/fc_in/fc_out;
     also report a Q,V-only variant — the common LoRA placement). Fit each adapter with
     `solve_lora_layer` (RRR) in a rebuilt-upstream forward sweep so the *frozen* base +
     adapter reproduces the task-B teacher's activations.
  4. Sweep `r ∈ {1,2,4,8,16,32}`. Report task-B test acc vs `r`, per-layer residual vs
     `r`, and two reference lines: base-A acc on task B (no adaptation, lower bound) and
     full task-B teacher acc (rank-∞ upper bound).

- **LR1-baseline (gradient LoRA, for context):** standard AdamW LoRA on task B at
  matched `r` and matched data budget; report acc + wall-time so the closed-form RRR
  number has a baseline.

> **Heads-up on task pairing.** majority→pointer is a genuine distribution shift that
> *requires new attention behavior*, so low-rank adapters on Q/K/V may be unable to
> synthesize the pointer attention pattern — that is a real finding (a LoRA-rank limit),
> not a bug. If you want a cleaner monotone acc-vs-r curve first, use a milder shift as
> task B (e.g. majority with a shifted threshold `>= vocab//4`, or label-flipped
> majority) and treat pointer as the stretch case. Run the mild shift first to validate
> the curve shape, then pointer.

**Acceptance.**
```
python -m diagnostic.runners.lora.lora_transformer --exp lr0
python -m diagnostic.runners.lora.lora_transformer --exp lr1            # mild-shift task B
python -m diagnostic.runners.lora.lora_transformer --exp lr1 --task-b pointer
```
- **Gate C1 (LR0):** RRR == full-OLS at full rank, both metrics. Hard gate.
- **Expected (LR1, mild shift):** task-B acc rises monotonically with `r`, from the
  base-A lower bound toward the task-B teacher upper bound. Non-monotone ⇒ projection
  bug (RRR is provably monotone in `r`) ⇒ debug LR0.
- **Expected (LR1, pointer):** may plateau below the teacher even at high `r` if low-rank
  adapters cannot create the required content-based attention — report the plateau and
  the rank at which it occurs; do not treat a plateau as a failure.
- Create `lora_findings.md` (one row per experiment); record both acc-vs-rank curves.

---

## 4b. WP-D — OLS warm-start → SGD convergence speedup

**Why.** The practical payoff question: does an OLS init shorten SGD enough to beat
random-init SGD in *total* compute (OLS pass cost included)? Full protocol in
`WARMSTART_SGD_PLAN.md` — read it before coding.

**Deliverables.**
- `diagnostic/flops.py` — honest FLOPs accounting (forward per arch, OLS solve `~d_in³`,
  SGD `≈3×fwd`), so the x-axis amortizes the OLS pass.
- `diagnostic/runners/warmstart/warmstart_run.py` — `--arch {mlp,lenet,transformer}`,
  `--init {random,ols_label,ols_distill}`, LR-sweep args. Applies the init (reuse
  `gt_target_retrain` for ols_label, the E3/E5/E8 feature-distill code for ols_distill),
  then runs SGD logging accuracy vs cumulative FLOPs.

**The make-or-break confound (do not skip):** sweep peak-LR × warmup length **per init
condition** and report each at its own best schedule. Reusing the random-init schedule
on an OLS init is the #1 way to falsely measure "no gain" — the first high-LR steps can
erase the init. Use ≥3 SGD seeds per cell (convergence speed is noisy).

**Run & acceptance.**
```
python -m diagnostic.runners.warmstart.warmstart_run --arch mlp        --init random
python -m diagnostic.runners.warmstart.warmstart_run --arch mlp        --init ols_label
python -m diagnostic.runners.warmstart.warmstart_run --arch lenet      --init ols_distill
python -m diagnostic.runners.warmstart.warmstart_run --arch transformer --init ols_distill
```
- **Gate D1 (WS0 sanity):** the `random` baseline must reproduce each model's known
  final accuracy, else the SGD harness is wrong.
- **Headline outputs per arch×init:** steps-to-{90,95,99%} of the random-init final
  acc, final-acc delta, and the **break-even accuracy** (lowest target where
  OLS_cost + warmstart_SGD < random_SGD). Plot accuracy vs cumulative FLOPs.
- **Expected:** clear step savings on DeepMLP and on the distill inits; uncertain or
  negative on conv-through-pool label init. A *negative* result at a tuned schedule is a
  valid finding — report it, don't hide it.
- Log to `warmstart_findings.md` (one row per cell).

---

## 5. Execution order, gates, escalation

Order: **WP-A → WP-B → WP-C → WP-D** (A and B share the arch-coverage machinery and warm
you up on the gotchas; C is standalone; D depends on A/B/C existing because it reuses the
label-retrain and feature-distill inits). Within each WP run `train → distill →
retrain/lr` and confirm the distill/LR0/WS0 gate before the harder step.

**Escalate to the user (stop, report diagnostics) if:**
- WP-A Gate A1 fails (task unlearnable) after one redesign attempt.
- Any **distill** step is not near-exact (A2, B1) — this contradicts a strong, repeatedly
  confirmed finding and signals a framework/capture bug worth understanding, not patching.
- WP-C LR0 fails (RRR ≠ full-OLS at full rank).
- WP-D: OLS-init SGD diverges under ALL swept LR schedules (init basin incompatible with
  SGD — report as a finding, do not patch).
- Any battery dies with a numerical NaN/instability after 2 fix attempts.

**Logging discipline.** Only write a findings row *after* a successful run with real
numbers — never pre-fill expected results. Append to `arch_coverage_findings.md`
(E6, Q10/E8 update), `lora_findings.md` (LoRA), and `warmstart_findings.md` (WP-D);
update statuses in `arch_coverage_queue.md`. Keep rows in the existing scientific,
minimal-text format.
