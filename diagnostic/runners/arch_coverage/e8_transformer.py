"""
diagnostic/runners/arch_coverage/e8_transformer.py
===================================================

E8 — tiny 2-block pre-LN transformer on a synthetic sequence task. Probes
attention (the bilinear softmax) and in-chain LayerNorm.

Battery (run via --option):

  train     B-train.  Train TinyTransformer by backprop on the synthetic
            "majority" task; cache weights.

  distill   B-distill (feature-level) of the Q/K/V/O + MLP + head Linears.
            The student copies the teacher's NON-Linear parameters (token +
            positional embedding and all LayerNorm affines) — those are not
            objects the OLS framework fits — then OLS-fits every Linear, in a
            rebuilt-upstream forward sweep, to reproduce the teacher's
            captured pre-activation. With the residual stream matched at the
            input, each Linear fit is exactly achievable; the softmax
            attention and LN are replayed in the forward. Tests whether
            attention + LN break closed-form OLS distillation (they should
            not). Transformer analogue of E3's exact conv round-trip.

  retrain   B-retrain-probe (attention-opaque, Q2). From RANDOM init, the GT
            label target is back-propagated ONLY through the invertible path:
            head <- LN_f^{-1} <- mean-broadcast <- residual <- top-block MLP
            (fc_out, fc_in) <- residual <- Wo. Softmax attention is opaque, so
            Q/K/V receive NO label-derived target and stay random; the lower
            block is unreachable (its output feeds the top block's attention).
            This quantifies the attention WALL for label-retraining — the
            transformer analogue of the max-pool wall in E3.

Usage:
    python -m diagnostic.runners.arch_coverage.e8_transformer --option train
    python -m diagnostic.runners.arch_coverage.e8_transformer --option distill
    python -m diagnostic.runners.arch_coverage.e8_transformer --option retrain
    python -m diagnostic.runners.arch_coverage.e8_transformer --option all
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.models.tiny_transformer import (
    TinyTransformer, make_majority_dataset, make_pointer_dataset,
)
from diagnostic.multi_step import _empirical_mean_cov
from diagnostic.runners.phase1_gt_retrain import make_gt_logit_target
from diagnostic.target_prop_retrainer import solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"

# These are set per-run from --task in main(); defaults preserved for majority.
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e8" / "majority"
TRANSFORMER_WEIGHTS = WEIGHTS_DIR / "tiny_transformer_majority.pt"

VOCAB = 32
SEQ_LEN = 16
N_HEADS = 4
DEPTH = 2
N_CLASSES = 2
# per-task model sizes (majority uses smaller model; pointer needs more capacity)
_D_MODEL = {"majority": 64,  "pointer": 128}
_D_FF    = {"majority": 128, "pointer": 256}
D_MODEL = _D_MODEL["majority"]   # kept for backward-compat JSON logging
D_FF    = _D_FF["majority"]

N_TRAIN = 20_000
N_TEST = 4_000
N_CAPTURE = 2_000          # sequences used for OLS capture (-> N*T token rows)
SEED = 42
GT_MARGIN = 5.0
OLS_LAMBDA = 1e-4
OLS_EPS = 1e-4
TRAIN_EPOCHS = 20


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


_CURRENT_TASK = "majority"   # set in main() before battery calls

def _make_model(device, task=None):
    t = task or _CURRENT_TASK
    d = _D_MODEL[t]; ff = _D_FF[t]
    return TinyTransformer(vocab=VOCAB, seq_len=SEQ_LEN, d_model=d,
                           n_heads=N_HEADS, d_ff=ff, depth=DEPTH,
                           num_classes=N_CLASSES).to(device)


@torch.no_grad()
def _eval(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for tok, y in loader:
        tok, y = tok.to(device), y.to(device)
        pred = model(tok).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


def _datasets(task: str = "majority"):
    if task == "pointer":
        tr_tok, tr_y = make_pointer_dataset(N_TRAIN, SEQ_LEN, VOCAB, seed=SEED)
        te_tok, te_y = make_pointer_dataset(N_TEST, SEQ_LEN, VOCAB, seed=SEED + 1)
    else:
        tr_tok, tr_y = make_majority_dataset(N_TRAIN, SEQ_LEN, VOCAB, seed=SEED)
        te_tok, te_y = make_majority_dataset(N_TEST, SEQ_LEN, VOCAB, seed=SEED + 1)
    return (TensorDataset(tr_tok, tr_y), TensorDataset(te_tok, te_y),
            tr_tok[:N_CAPTURE].clone(), tr_y[:N_CAPTURE].clone())


def _train(model, train_loader, test_loader, *, epochs, device, lr=3e-4):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for tok, y in train_loader:
            tok, y = tok.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(tok), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        acc = _eval(model, test_loader, device)
        print(f"  [train] epoch {epoch + 1:2d}/{epochs}  test_acc={acc:.4f}")
    return _eval(model, test_loader, device)


def _flat(t):
    """(B,T,d) -> (B*T, d)."""
    return t.reshape(-1, t.shape[-1])


def _fit_linear(layer, X, target, *, lam=OLS_LAMBDA):
    """OLS-fit a Linear to `target` given input X (both 2D). Returns residual."""
    X = X.double().cpu()
    target = target.double().cpu()
    n = min(X.shape[0], target.shape[0])
    W, b = solve_ols_layer(X[:n], target[:n], with_bias=(layer.bias is not None), ols_lambda=lam)
    pred = X[:n] @ W.T + (b.unsqueeze(0) if b is not None else 0)
    res = float((pred - target[:n]).norm() / target[:n].norm().clamp(min=1e-30))
    with torch.no_grad():
        layer.weight.copy_(W.to(layer.weight.device, layer.weight.dtype))
        if layer.bias is not None and b is not None:
            layer.bias.copy_(b.to(layer.bias.device, layer.bias.dtype))
    return res


# ===========================================================================
# B-train
# ===========================================================================

def battery_train(train_set, test_loader, *, device, epochs,
                  weights_path=None, results_dir=None, task="majority", lr=3e-4):
    weights_path = weights_path or TRANSFORMER_WEIGHTS
    results_dir = results_dir or RESULTS_DIR
    print(f"\n[E8] === B-train: TinyTransformer on synthetic {task} ===")
    torch.manual_seed(SEED)
    model = _make_model(device, task)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    acc = _train(model, train_loader, test_loader, epochs=epochs, device=device, lr=lr)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"  trained acc = {acc:.4f}  -> {weights_path.name}")
    _save_json(results_dir / "b_train.json",
               {"battery": "B-train", "exp_id": "e8", "task": task, "seed": SEED,
                "epochs": epochs, "lr": lr, "trained_acc": acc,
                "config": {"vocab": VOCAB, "seq_len": SEQ_LEN,
                           "d_model": _D_MODEL[task], "d_ff": _D_FF[task],
                           "n_heads": N_HEADS, "depth": DEPTH}})
    return model


# ===========================================================================
# B-distill — feature-level OLS distillation of all Linears
# ===========================================================================

def battery_distill(teacher, cap_tokens, test_loader, *, device,
                    results_dir=None, task="majority"):
    results_dir = results_dir or RESULTS_DIR
    print(f"\n[E8] === B-distill: feature-level OLS distillation [{task}] ===")
    teacher_acc = _eval(teacher, test_loader, device)
    print(f"  teacher acc = {teacher_acc:.4f}")

    teacher.eval()
    with torch.no_grad():
        st = teacher.forward_with_state(cap_tokens.to(device))
    tgt = {k: v.detach().cpu() for k, v in st.items()}

    # student: random Linears, but copy teacher's NON-Linear params
    torch.manual_seed(SEED)
    student = _make_model(device, task)
    rand_acc = _eval(student, test_loader, device)
    with torch.no_grad():
        student.embed.weight.copy_(teacher.embed.weight)
        student.pos.copy_(teacher.pos)
        student.ln_f.weight.copy_(teacher.ln_f.weight)
        student.ln_f.bias.copy_(teacher.ln_f.bias)
        for sblk, tblk in zip(student.blocks, teacher.blocks):
            for ln_s, ln_t in ((sblk.ln1, tblk.ln1), (sblk.ln2, tblk.ln2)):
                ln_s.weight.copy_(ln_t.weight)
                ln_s.bias.copy_(ln_t.bias)
    print(f"  random-init student acc = {rand_acc:.4f}  (non-Linear params copied)")

    info: Dict[str, dict] = {}
    relu = nn.ReLU()
    tok = cap_tokens.to(device)
    with torch.no_grad():
        h = student.embed(tok) + student.pos.unsqueeze(0)
        for bi, blk in enumerate(student.blocks):
            z = blk.ln1(h)
            info[f"b{bi}.Wq"] = {"target_residual": _fit_linear(blk.Wq, _flat(z), _flat(tgt[f"b{bi}.q_pre"]))}
            info[f"b{bi}.Wk"] = {"target_residual": _fit_linear(blk.Wk, _flat(z), _flat(tgt[f"b{bi}.k_pre"]))}
            info[f"b{bi}.Wv"] = {"target_residual": _fit_linear(blk.Wv, _flat(z), _flat(tgt[f"b{bi}.v_pre"]))}
            ctx, _ = blk.attention(z)
            info[f"b{bi}.Wo"] = {"target_residual": _fit_linear(blk.Wo, _flat(ctx), _flat(tgt[f"b{bi}.o_pre"]))}
            h = h + blk.Wo(ctx)
            z2 = blk.ln2(h)
            info[f"b{bi}.fc_in"] = {"target_residual": _fit_linear(blk.fc_in, _flat(z2), _flat(tgt[f"b{bi}.fcin_pre"]))}
            fcin_post = relu(blk.fc_in(z2))
            info[f"b{bi}.fc_out"] = {"target_residual": _fit_linear(blk.fc_out, _flat(fcin_post), _flat(tgt[f"b{bi}.fcout_pre"]))}
            h = h + blk.fc_out(fcin_post)
        pooled = student.ln_f(h).mean(dim=1)
        info["head"] = {"target_residual": _fit_linear(student.head, pooled, tgt["logits"])}

    acc = _eval(student, test_loader, device)
    print(f"  feature-distill student acc = {acc:.4f}  (delta vs teacher {acc - teacher_acc:+.4f})")
    names = student.linear_layer_names_deepest_first()
    for nm in names:
        r = info.get(nm, {}).get("target_residual")
        print(f"    {nm:<10s}: residual={r:.3g}" if r is not None else f"    {nm}: --")

    result = {
        "battery": "B-distill-feature", "exp_id": "e8", "task": task,
        "teacher_acc": teacher_acc, "random_init_acc": rand_acc,
        "student_acc": acc, "delta_vs_teacher": acc - teacher_acc,
        "per_layer_residuals": {nm: info.get(nm, {}).get("target_residual") for nm in names},
        "note": "non-Linear params (embed, pos, LayerNorm affines) copied from teacher; only Linears OLS-fit.",
    }
    _save_json(results_dir / "b_distill.json", result)
    return result


# ===========================================================================
# B-retrain-probe — attention-opaque label retrain (the attention wall)
# ===========================================================================

def _ln_inverse(y_target, x_forward, ln: nn.LayerNorm):
    """Invert a standalone LayerNorm per row, using x_forward's per-row stats.
       y = (x-mu)/std*gamma + beta  =>  x = (y-beta)/gamma * std + mu."""
    mu = x_forward.mean(dim=-1, keepdim=True)
    var = ((x_forward - mu) ** 2).mean(dim=-1, keepdim=True)
    std = (var + ln.eps).sqrt()
    gamma = ln.weight.detach().double().cpu()
    beta = ln.bias.detach().double().cpu()
    safe_gamma = torch.where(gamma.abs() < 1e-6, torch.ones_like(gamma), gamma)
    return (y_target - beta) / safe_gamma * std + mu


def battery_retrain(cap_tokens, cap_labels, test_loader, *, device, method="kfac_a",
                    fit_head: bool = False, results_dir=None, task="majority"):
    results_dir = results_dir or RESULTS_DIR
    print(f"\n[E8] === B-retrain-probe: attention-opaque label retrain [{task}] ===")
    torch.manual_seed(SEED)
    model = _make_model(device, task)
    rand_acc = _eval(model, test_loader, device)
    print(f"  random-init acc = {rand_acc:.4f}")

    n = min(N_CAPTURE, cap_labels.shape[0])
    tok = cap_tokens[:n].to(device)
    gt = make_gt_logit_target(cap_labels[:n], n_classes=N_CLASSES, margin=GT_MARGIN).double()

    top = model.depth - 1
    blk = model.blocks[top]
    relu = nn.ReLU()
    info: Dict[str, dict] = {}
    fitted: List[str] = []
    t0 = time.time()

    # ---- forward h states (model frozen during back-prop) ----
    model.eval()
    with torch.no_grad():
        h = model.embed(tok) + model.pos.unsqueeze(0)
        h_in_blocks = []
        for b in model.blocks:
            h_in_blocks.append(h)                       # input to block b
            z = b.ln1(h)
            ctx, _ = b.attention(z)
            h = h + b.Wo(ctx)
            h_after_attn = h
            z2 = b.ln2(h)
            fcin_post = relu(b.fc_in(z2))
            h = h + b.fc_out(fcin_post)
            if b is blk:
                top_h_after_attn = h_after_attn
                top_ctx = ctx
                top_z2 = z2
                top_fcin_post = fcin_post
                top_fcin_pre = b.fc_in(z2)
        h_top = h                                       # input to ln_f
        pooled_fwd = model.ln_f(h_top).mean(dim=1)

    # ---- head: invert label target -> pooled target (all CPU/double) ----
    W_head = model.head.weight.detach().double().cpu()
    b_head = model.head.bias.detach().double().cpu()
    gt = gt.cpu()
    pooled_fwd_c = pooled_fwd.double().cpu()
    mu_p, S_p = _empirical_mean_cov(pooled_fwd_c)
    if method == "naive":
        t_pooled = invert_layer(W_head, b_head, gt, pooled_fwd_c, None, "naive", eps=OLS_EPS)
    else:
        t_pooled = invert_layer(W_head, b_head, gt, pooled_fwd_c, None, "kfac_a", mu_a=mu_p, Sigma_a=S_p)

    # ---- mean-pool inverse: broadcast across tokens, then LN_f inverse ----
    d_model = _D_MODEL[task]
    t_lnf_out = t_pooled.unsqueeze(1).expand(n, SEQ_LEN, d_model).reshape(-1, d_model)
    h_top_flat = _flat(h_top.double().cpu())
    t_h_top = _ln_inverse(t_lnf_out, h_top_flat, model.ln_f)        # (n*T, d)

    # ---- top block MLP sublayer: h_top = h_after_attn + fc_out(...) ----
    t_fcout = t_h_top - _flat(top_h_after_attn.double().cpu())      # target for fc_out output
    info[f"b{top}.fc_out"] = {"target_residual":
                              _fit_linear(blk.fc_out, _flat(top_fcin_post.double().cpu()), t_fcout)}
    fitted.append(f"b{top}.fc_out")
    # invert fc_out (NO activation after it) -> target for its input fcin_post
    Wfo = blk.fc_out.weight.detach().double().cpu()
    bfo = blk.fc_out.bias.detach().double().cpu()
    a_in_fo = _flat(top_fcin_post.double().cpu())          # fc_out input (d_ff)
    fcin_pre_flat = _flat(top_fcin_pre.double().cpu())     # fc_in output, pre-ReLU
    mu_fo, S_fo = _empirical_mean_cov(a_in_fo)
    if method == "naive":
        t_fcin_post = invert_layer(Wfo, bfo, t_fcout, t_fcout, None, "naive", eps=OLS_EPS)
    else:
        t_fcin_post = invert_layer(Wfo, bfo, t_fcout, t_fcout, None, "kfac_a", mu_a=mu_fo, Sigma_a=S_fo)
    # invert fc_in's ReLU -> pre-activation target, then fit fc_in (input = ln2_out)
    t_fcin_pre = invert_activation(t_fcin_post, fcin_pre_flat, relu)
    info[f"b{top}.fc_in"] = {"target_residual":
                             _fit_linear(blk.fc_in, _flat(top_z2.double().cpu()), t_fcin_pre)}
    fitted.append(f"b{top}.fc_in")

    # ---- top block attention sublayer: h_after_attn = h_in + Wo(ctx) ----
    # target for h_after_attn comes from the MLP-sublayer residual: the MLP
    # path wants h_after_attn ~= t_h_top - fc_out_new(...); use the simpler
    # identity-residual target t_h_top minus the (now refit) MLP contribution.
    with torch.no_grad():
        new_fcout = blk.fc_out(relu(blk.fc_in(top_z2)))
    t_h_after_attn = t_h_top - _flat(new_fcout.double().cpu())
    t_o = t_h_after_attn - _flat(h_in_blocks[top].double().cpu())   # target for Wo output
    info[f"b{top}.Wo"] = {"target_residual":
                          _fit_linear(blk.Wo, _flat(top_ctx.double().cpu()), t_o)}
    fitted.append(f"b{top}.Wo")

    # Optional: also fit the head directly via OLS on the (now-rebuilt) pooled rep.
    acc_no_headfit = _eval(model, test_loader, device)
    acc_with_headfit = None
    if fit_head:
        with torch.no_grad():
            h2 = model.embed(tok) + model.pos.unsqueeze(0)
            for b in model.blocks:
                h2 = b(h2)
            pooled_new = model.ln_f(h2).mean(dim=1)
        pooled_new_c = pooled_new.double().cpu()
        info["head"] = {"target_residual":
                        _fit_linear(model.head, pooled_new_c, gt.cpu())}
        fitted.append("head")
        acc_with_headfit = _eval(model, test_loader, device)
        print(f"  +fit-head acc = {acc_with_headfit:.4f}  (delta {acc_with_headfit - acc_no_headfit:+.4f})")

    acc = acc_with_headfit if fit_head else acc_no_headfit

    # Wq/Wk/Wv (this block) and the entire lower block are UNREACHABLE:
    # back-prop would have to invert softmax(QK^T)V, which is opaque.
    unreachable = [nm for nm in model.linear_layer_names_deepest_first() if nm not in fitted]

    wall = time.time() - t0
    print(f"  attention-opaque retrain acc = {acc:.4f}  delta={acc - rand_acc:+.4f}  ({wall:.1f}s)")
    print(f"  fitted (reachable): {fitted}")
    print(f"  frozen-random (attention wall): {unreachable}")

    result = {
        "battery": "B-retrain-probe-attn-opaque", "exp_id": "e8", "task": task,
        "seed": SEED, "method": method, "fit_head": fit_head,
        "random_init_acc": rand_acc,
        "retrained_acc_no_headfit": acc_no_headfit,
        "retrained_acc": acc,
        "delta": acc - rand_acc,
        "wall_s": wall,
        "fitted_layers": fitted,
        "unreachable_layers": unreachable,
        "per_layer_residuals": {nm: info.get(nm, {}).get("target_residual") for nm in fitted},
        "interpretation": (
            "Only head + top-block {Wo, fc_in, fc_out} are reachable without "
            "inverting softmax attention. Q/K/V and the entire lower block stay "
            "random => attention is a label-retrain wall, analogous to max-pool in E3."
        ),
    }
    _save_json(results_dir / "b_retrain.json", result)
    return result


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--option", choices=("train", "distill", "retrain", "all"), default="all")
    ap.add_argument("--task", choices=("majority", "pointer"), default="majority")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--method", choices=("naive", "kfac_a"), default="kfac_a")
    ap.add_argument("--fit-head", action="store_true", default=False)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    globals()["SEED"] = args.seed

    # Per-task defaults
    global _CURRENT_TASK
    _CURRENT_TASK = args.task
    epochs = args.epochs
    if epochs is None:
        epochs = 60 if args.task == "pointer" else TRAIN_EPOCHS
    lr = args.lr
    if lr is None:
        lr = 3e-4   # same default for both tasks (pointer task achieves 99.98% with this)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[E8] device={device}  option={args.option}  task={args.task}")

    results_dir = (REPO_ROOT / "benchmark" / "results" / "diagnostic"
                   / "arch_coverage" / "e8" / args.task)
    weights_path = WEIGHTS_DIR / f"tiny_transformer_{args.task}.pt"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_set, test_set, cap_tokens, cap_labels = _datasets(args.task)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    def _load_trained():
        if not weights_path.exists():
            print(f"[E8] no weights at {weights_path}; run --option train first.")
            return None
        m = _make_model(device, args.task)
        m.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        return m

    if args.option in ("train", "all"):
        battery_train(train_set, test_loader, device=device, epochs=epochs, lr=lr,
                      weights_path=weights_path, results_dir=results_dir, task=args.task)

    if args.option in ("distill", "all"):
        teacher = _load_trained()
        if teacher is None:
            return 1
        battery_distill(teacher, cap_tokens, test_loader, device=device,
                        results_dir=results_dir, task=args.task)

    if args.option in ("retrain", "all"):
        battery_retrain(cap_tokens, cap_labels, test_loader, device=device,
                        method=args.method, fit_head=args.fit_head,
                        results_dir=results_dir, task=args.task)

    print("\n[E8] === COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
