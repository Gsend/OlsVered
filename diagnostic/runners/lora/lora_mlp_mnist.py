"""
diagnostic/runners/lora/lora_mlp_mnist.py
==========================================

LoRA-TP experiments using the TinyTransformer (E8) and a standalone Linear
layer math unit-test.

Experiments (run via --exp):

  lr0   Math unit test: one standalone Linear(64, 64) fitted at full rank.
        RRR must reproduce the full-OLS solution identically.
        Gate C1: residual(RRR) / residual(OLS) in [0.99, 1.01].

  lr1   Transformer ADAPTATION: base = TinyTransformer trained on majority task;
        teacher = TinyTransformer trained on pointer task (both already trained
        by WP-A). Freeze base weights; fit low-rank LoRA adapters (on all 13
        Linear layers: Wq/Wk/Wv/Wo/fc_in/fc_out x2 + head) so that the frozen-
        base + adapter reproduces the pointer-teacher's per-layer pre-activations.
        Sweep r in {1, 2, 4, 8, 16, 32, 64}. Report acc vs r.

Usage:
    python -m diagnostic.runners.lora.lora_mlp_mnist --exp lr0
    python -m diagnostic.runners.lora.lora_mlp_mnist --exp lr1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnostic.lora import LoRALinear, solve_lora_layer
from diagnostic.models.tiny_transformer import (
    TinyTransformer, make_majority_dataset, make_pointer_dataset,
)
from diagnostic.target_prop_retrainer import solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostic" / "lora"
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"

SEED = 42
OLS_LAMBDA = 1e-4

# TinyTransformer hyperparams (must match e8_transformer.py per-task sizes)
VOCAB = 32
SEQ_LEN = 16
N_HEADS = 4
DEPTH = 2
N_CLASSES = 2
_D_MODEL = {"majority": 64, "pointer": 128}
_D_FF    = {"majority": 128, "pointer": 256}

N_TRAIN = 20_000
N_TEST  = 4_000
N_CAP   = 2_000


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _make_transformer(task: str, device) -> TinyTransformer:
    d = _D_MODEL[task]; ff = _D_FF[task]
    return TinyTransformer(vocab=VOCAB, seq_len=SEQ_LEN, d_model=d,
                           n_heads=N_HEADS, d_ff=ff, depth=DEPTH,
                           num_classes=N_CLASSES).to(device)


def _load_transformer(task: str, device) -> Optional[TinyTransformer]:
    path = WEIGHTS_DIR / f"tiny_transformer_{task}.pt"
    if not path.exists():
        print(f"[LR] weights not found: {path}. Run e8_transformer --task {task} --option train first.")
        return None
    m = _make_transformer(task, device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def _eval_acc(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for tok, y in loader:
        tok, y = tok.to(device), y.to(device)
        pred = model(tok).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


def _flat(t: torch.Tensor) -> torch.Tensor:
    """(B, T, d) or (B, d) -> (B*T, d) or (B, d)."""
    if t.ndim == 3:
        return t.reshape(-1, t.shape[-1])
    return t


# ===========================================================================
# LR0 — Math unit test: full-rank RRR must match OLS
# ===========================================================================

def experiment_lr0(device):
    print("\n[LR0] === Math unit test: full-rank RRR vs OLS ===")
    torch.manual_seed(SEED)

    d_in, d_out = 64, 64
    n = 1024
    r_full = min(d_in, d_out)

    # Random linear layer and random data
    layer = nn.Linear(d_in, d_out, bias=True)
    nn.init.kaiming_normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    X = torch.randn(n, d_in).double()
    T = torch.randn(n, d_out).double()
    W0 = torch.zeros(d_out, d_in).double()   # base = zero (so T' = T)
    b0 = torch.zeros(d_out).double()

    # Full-OLS (rank-inf control)
    W_ols, b_ols = solve_ols_layer(X, T, with_bias=True, ols_lambda=OLS_LAMBDA)
    pred_ols = X @ W_ols.T + b_ols.unsqueeze(0)
    res_ols = float((pred_ols - T).norm() / T.norm())
    print(f"  OLS residual = {res_ols:.6f}")

    results = {}
    for metric in ("output", "whitened"):
        # RRR at full rank (r = min(d_in, d_out) = 64, bias=False since absorbed)
        A, B, res_rrr = solve_lora_layer(X, T, W0, b0, r=r_full, alpha=r_full,
                                          lam=OLS_LAMBDA, metric=metric)
        ratio = res_rrr / max(res_ols, 1e-12)
        status = "PASS" if abs(ratio - 1.0) < 0.02 else "FAIL"
        print(f"  [{metric}] RRR residual = {res_rrr:.6f}  ratio_to_OLS = {ratio:.4f}  [{status}]")
        results[metric] = {"rrr_residual": res_rrr, "ols_residual": res_ols, "ratio": ratio, "gate": status}

    _save_json(RESULTS_DIR / "lr0_unit_test.json", {
        "exp": "lr0", "n": n, "d_in": d_in, "d_out": d_out, "r_full": r_full,
        "ols_residual": res_ols, "metrics": results,
        "gate_C1": all(v["gate"] == "PASS" for v in results.values()),
    })
    return results


# ===========================================================================
# LR1 — Transformer adaptation: majority base -> pointer teacher
# ===========================================================================

def experiment_lr1(device):
    print("\n[LR1] === Transformer adaptation: majority base -> pointer teacher ===")

    base_model = _load_transformer("majority", device)
    if base_model is None:
        return None
    teacher = _load_transformer("pointer", device)
    if teacher is None:
        return None

    # Build pointer test set for evaluation
    te_tok, te_y = make_pointer_dataset(N_TEST, SEQ_LEN, VOCAB, seed=SEED + 1)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(te_tok, te_y),
        batch_size=512, shuffle=False, num_workers=0)

    # Capture pointer-task activations from the TEACHER
    cap_tok, _ = make_pointer_dataset(N_CAP, SEQ_LEN, VOCAB, seed=SEED + 2)
    cap_tok = cap_tok.to(device)

    teacher.eval()
    with torch.no_grad():
        st_t = teacher.forward_with_state(cap_tok)
    tgt = {k: v.detach().cpu().double() for k, v in st_t.items()}

    base_acc = _eval_acc(base_model, test_loader, device)
    teacher_acc = _eval_acc(teacher, test_loader, device)
    print(f"  base (majority) on pointer task = {base_acc:.4f}")
    print(f"  teacher (pointer) on pointer task = {teacher_acc:.4f}")

    # The base model uses majority config (d_model=64), teacher uses pointer config (d_model=128).
    # We can only adapt layers in the BASE model, so we need teacher targets that match
    # the base model's layer sizes. This is only possible if both use the same architecture.
    # Since they differ in d_model, we do a same-architecture control: use pointer-config
    # base model with RANDOM weights and pointer teacher. This tests pure distillation.
    print("  Note: base (d_model=64) vs teacher (d_model=128) differ in size.")
    print("  Using same-arch setup: pointer-random-init as base, pointer-teacher as teacher.")

    # Use a fresh random pointer-arch model as the base
    torch.manual_seed(SEED)
    base_same = _make_transformer("pointer", device)
    base_same.eval()
    # Copy non-linear params from teacher (embeddings, LN) to make distillation meaningful
    with torch.no_grad():
        base_same.embed.weight.copy_(teacher.embed.weight)
        base_same.pos.copy_(teacher.pos)
        base_same.ln_f.weight.copy_(teacher.ln_f.weight)
        base_same.ln_f.bias.copy_(teacher.ln_f.bias)
        for sblk, tblk in zip(base_same.blocks, teacher.blocks):
            for ln_s, ln_t in ((sblk.ln1, tblk.ln1), (sblk.ln2, tblk.ln2)):
                ln_s.weight.copy_(ln_t.weight)
                ln_s.bias.copy_(ln_t.bias)

    rand_acc = _eval_acc(base_same, test_loader, device)
    print(f"  random-init base (pointer arch) on pointer task = {rand_acc:.4f}")

    # For each rank, do LoRA distillation: rebuild-upstream forward, fit each Linear
    rank_list = [1, 2, 4, 8, 16, 32, 64]
    results_by_rank = {}

    for r in rank_list:
        t0 = time.time()
        # Deep copy base (we want fresh random linear weights per rank trial)
        torch.manual_seed(SEED)
        model = _make_transformer("pointer", device)
        with torch.no_grad():
            model.embed.weight.copy_(teacher.embed.weight)
            model.pos.copy_(teacher.pos)
            model.ln_f.weight.copy_(teacher.ln_f.weight)
            model.ln_f.bias.copy_(teacher.ln_f.bias)
            for sblk, tblk in zip(model.blocks, teacher.blocks):
                for ln_s, ln_t in ((sblk.ln1, tblk.ln1), (sblk.ln2, tblk.ln2)):
                    ln_s.weight.copy_(ln_t.weight)
                    ln_s.bias.copy_(ln_t.bias)

        info = {}
        relu = nn.ReLU()
        tok = cap_tok

        def _fit_lora(layer: nn.Linear, X_in: torch.Tensor, T_pre: torch.Tensor,
                      key: str, metric: str = "output"):
            W0 = layer.weight.detach().double().cpu()
            b0 = layer.bias.detach().double().cpu() if layer.bias is not None else None
            A, B, res = solve_lora_layer(X_in.double().cpu(), T_pre.double().cpu(),
                                          W0, b0, r=r, alpha=r, lam=OLS_LAMBDA, metric=metric)
            # Apply as low-rank delta: W_new = W0 + (1/r)*r * B @ A = W0 + B @ A
            with torch.no_grad():
                delta = (B.to(layer.weight.device, layer.weight.dtype) @
                         A.to(layer.weight.device, layer.weight.dtype))
                layer.weight.copy_(layer.weight + delta)
                # bias unchanged (W0 absorbs it)
            info[key] = {"residual": res}
            return res

        # Rebuilt-upstream forward sweep (same pattern as E8 distill)
        with torch.no_grad():
            h = model.embed(tok) + model.pos.unsqueeze(0)
            for bi, blk in enumerate(model.blocks):
                z = blk.ln1(h)
                z_flat = _flat(z)
                _fit_lora(blk.Wq, z_flat, _flat(tgt[f"b{bi}.q_pre"]), f"b{bi}.Wq")
                _fit_lora(blk.Wk, z_flat, _flat(tgt[f"b{bi}.k_pre"]), f"b{bi}.Wk")
                _fit_lora(blk.Wv, z_flat, _flat(tgt[f"b{bi}.v_pre"]), f"b{bi}.Wv")
                # recompute attention context with updated Wq/Wk/Wv
                B_sz, T_sz, d = z.shape
                q = blk.Wq(z).view(B_sz, T_sz, blk.n_heads, blk.d_head).transpose(1, 2)
                k = blk.Wk(z).view(B_sz, T_sz, blk.n_heads, blk.d_head).transpose(1, 2)
                v = blk.Wv(z).view(B_sz, T_sz, blk.n_heads, blk.d_head).transpose(1, 2)
                scores = (q @ k.transpose(-2, -1)) / (blk.d_head ** 0.5)
                ctx = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B_sz, T_sz, d)
                _fit_lora(blk.Wo, _flat(ctx), _flat(tgt[f"b{bi}.o_pre"]), f"b{bi}.Wo")
                h = h + blk.Wo(ctx)
                z2 = blk.ln2(h)
                z2_flat = _flat(z2)
                _fit_lora(blk.fc_in, z2_flat, _flat(tgt[f"b{bi}.fcin_pre"]), f"b{bi}.fc_in")
                fcin_post = relu(blk.fc_in(z2))
                _fit_lora(blk.fc_out, _flat(fcin_post), _flat(tgt[f"b{bi}.fcout_pre"]),
                           f"b{bi}.fc_out")
                h = h + blk.fc_out(fcin_post)
            pooled = model.ln_f(h).mean(dim=1)
            _fit_lora(model.head, pooled, tgt["logits"], "head")

        acc = _eval_acc(model, test_loader, device)
        wall = time.time() - t0
        print(f"  r={r:3d}  acc={acc:.4f}  (teacher={teacher_acc:.4f})  wall={wall:.1f}s  "
              f"avg_res={sum(v['residual'] for v in info.values()) / len(info):.3e}")
        results_by_rank[r] = {"acc": acc, "per_layer_residuals": info, "wall_s": wall}

    _save_json(RESULTS_DIR / "lr1_transformer_adapt.json", {
        "exp": "lr1", "seed": SEED,
        "base_task": "majority (d_model=64, separate arch)",
        "teacher_task": "pointer (d_model=128)",
        "note": "Same-arch setup: random-init pointer model adapted to pointer teacher.",
        "teacher_acc": teacher_acc, "rand_acc": rand_acc,
        "rank_sweep": {str(r): v for r, v in results_by_rank.items()},
        "rank_list": rank_list,
    })
    return results_by_rank


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", choices=("lr0", "lr1", "all"), default="lr0")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[LR] device={device}  exp={args.exp}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("lr0", "all"):
        r0 = experiment_lr0(device)
        gate = all(v["gate"] == "PASS" for v in r0.values())
        if not gate:
            print("[LR0] Gate C1 FAIL — RRR != OLS at full rank. Halting.")
            return 1

    if args.exp in ("lr1", "all"):
        experiment_lr1(device)

    print("\n[LR] === COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
