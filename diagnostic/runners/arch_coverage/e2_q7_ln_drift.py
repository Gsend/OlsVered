"""
diagnostic/runners/arch_coverage/e2_q7_ln_drift.py
==================================================

E2 open question Q7: does LayerNorm PREVENT iteration collapse, or does it just
START HIGHER (and decline gradually because there is more to lose)?

Background. The B-iter batteries used kfac + moment-match for BOTH models:
  E1 DeepMLP (plain ReLU):  iter1 = 0.29, then collapse to ~0.11 (stagnation)
  E2 LNMLP   (LayerNorm):   iter1 = 0.74, then gradual monotone decline
But moment-match HURTS plain-ReLU far more than LN-ReLU at depth 10
(E1 kfac+mom = 0.29 vs no-mom = 0.89; E2 kfac+mom = 0.74). So the B-iter
comparison is contaminated by the moment-match interaction.

This experiment removes the confound by running BOTH models under BOTH
conditions (no-mom and mom-match), from random init, for N_ITER iterations,
and tracking per-layer activation RMS at every iteration.

Decision rules:
  - Matched no-mom start (~0.88 both): if plain-ReLU collapses while LN declines
    gradually  -> LN PREVENTS collapse (genuine mechanism).
  - If both drift similarly at matched start -> the B-iter gap was the
    moment-match confound, not LN ("LN just started higher" via mom-rescue).
  - Activation RMS: if LN holds activation scale ~constant across iterations
    while plain-ReLU explodes/vanishes -> direct evidence LN bounds the drift.

Usage:
    python -m diagnostic.runners.arch_coverage.e2_q7_ln_drift
    python -m diagnostic.runners.arch_coverage.e2_q7_ln_drift --n-iter 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from diagnostic.capture import collect_activations
from diagnostic.models.deep_mlp import DeepMLP
from diagnostic.models.lnmlp import LNMLP
from diagnostic.runners.phase1_gt_retrain import gt_target_retrain, make_gt_logit_target
from diagnostic.runners.phase1_mlp_mnist import _eval, _load_mnist

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e2"

N_LAYERS = 10
HIDDEN_DIM = 256
N_SAMPLES = 16_384
SEED = 42
GT_MARGIN = 5.0
OLS_LAMBDA = 1e-4
OLS_EPS = 1e-4


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _build_loader_list(dataset, max_samples: int, batch_size: int = 128):
    subset = Subset(dataset, list(range(min(len(dataset), max_samples * 4))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, label_chunks = [], []
    for x, y in loader:
        images.append(x)
        label_chunks.append(y)
    labels = torch.cat(label_chunks, dim=0)[:max_samples]
    return images, labels


def _activation_rms(model, loader_list, layers_df, device, n_samples):
    """Per-layer post-activation RMS (deepest-first index L0..L9)."""
    captured = collect_activations(
        model, loader_list, layers_df,
        max_samples=n_samples, seq_subsample=n_samples, device=device,
    )
    rms = {}
    for i, layer in enumerate(layers_df):
        a_post = captured[layer]["a_post"].double()
        rms[f"L{i}"] = float(a_post.pow(2).mean().sqrt().item())
    return rms


def _make_model(tag, device):
    if tag == "deepmlp":
        return DeepMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
    return LNMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)


def run_drift(tag, mom, train_set, test_loader, *, device, n_iter, n_samples):
    label = f"{tag}_{'mom' if mom else 'nomom'}"
    print(f"\n[Q7] === {label}: {n_iter} iterations from random init ===")
    torch.manual_seed(SEED)
    model = _make_model(tag, device)
    layers_df = model.get_layers_deepest_first()
    rand_acc = _eval(model, test_loader, device)

    loader_list, labels = _build_loader_list(train_set, n_samples)
    gt_targets = make_gt_logit_target(labels, n_classes=10, margin=GT_MARGIN)

    accs = [rand_acc]
    act_rms = [_activation_rms(model, loader_list, layers_df, device, n_samples)]
    diverged_at = None
    t0 = time.time()
    for it in range(n_iter):
        try:
            gt_target_retrain(
                model=model, chain_layers=layers_df, work_layers=layers_df,
                deepest_gt_targets=gt_targets, dataloader_list=loader_list,
                method="kfac_a", correct_target_mean=mom, correct_target_cov=mom,
                eps=OLS_EPS, ols_lambda=OLS_LAMBDA, max_samples=n_samples, device=device,
            )
            acc = _eval(model, test_loader, device)
        except Exception as exc:
            diverged_at = it + 1
            print(f"  iter {it + 1:2d}: DIVERGED ({type(exc).__name__}: {str(exc)[:60]})")
            break
        if acc != acc:  # NaN
            diverged_at = it + 1
            print(f"  iter {it + 1:2d}: DIVERGED (NaN accuracy)")
            break
        accs.append(acc)
        try:
            act_rms.append(_activation_rms(model, loader_list, layers_df, device, n_samples))
        except Exception:
            act_rms.append({f"L{i}": float("nan") for i in range(len(layers_df))})
        print(f"  iter {it + 1:2d}: acc={acc:.4f}")
    wall = time.time() - t0

    # mean activation RMS across layers, per iteration (scale-drift summary)
    mean_rms = [sum(d.values()) / len(d) for d in act_rms]
    print(f"  trajectory: {' -> '.join(f'{a:.4f}' for a in accs)}")
    print(f"  mean act-RMS: {' -> '.join(f'{r:.3g}' for r in mean_rms)}")
    if diverged_at is not None:
        print(f"  -> diverged at iter {diverged_at}")
    return {
        "label": label, "tag": tag, "mom_match": mom,
        "random_init_acc": rand_acc, "iter_accs": accs,
        "iter1_acc": accs[1] if len(accs) > 1 else None,
        "final_acc": accs[-1],
        "diverged_at": diverged_at,
        "act_rms_per_iter": act_rms, "mean_act_rms_per_iter": mean_rms,
        "wall_s": wall,
    }


def _plot(results, n_iter):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    style = {"deepmlp_nomom": "C0-o", "lnmlp_nomom": "C1-s",
             "deepmlp_mom": "C0--^", "lnmlp_mom": "C1--v"}
    for r in results:
        xs = range(len(r["iter_accs"]))
        ax1.plot(xs, r["iter_accs"], style.get(r["label"], "-"), label=r["label"], markersize=4)
        ax2.plot(range(len(r["mean_act_rms_per_iter"])), r["mean_act_rms_per_iter"],
                 style.get(r["label"], "-"), label=r["label"], markersize=4)
    ax1.set_xlabel("iteration"); ax1.set_ylabel("test accuracy")
    ax1.set_title("Q7: accuracy drift (matched start at no-mom)"); ax1.legend(fontsize=8)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("mean post-activation RMS")
    ax2.set_title("Q7: activation-scale drift"); ax2.set_yscale("log"); ax2.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "q7_ln_drift.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Q7] device={device}  n_iter={args.n_iter}  n_samples={args.n_samples}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    results = []
    for tag in ("deepmlp", "lnmlp"):
        for mom in (False, True):
            results.append(run_drift(
                tag, mom, train_set, test_loader,
                device=device, n_iter=args.n_iter, n_samples=args.n_samples))

    _plot(results, args.n_iter)

    # --- analysis summary ---
    by = {r["label"]: r for r in results}
    print("\n[Q7] === ANALYSIS ===")
    print(f"  {'config':<18s} {'iter1':>8s} {'final':>8s} {'drop':>8s} "
          f"{'rms_i0':>8s} {'rms_fin':>8s} {'diverge':>8s}")
    for lbl in ("deepmlp_nomom", "lnmlp_nomom", "deepmlp_mom", "lnmlp_mom"):
        r = by.get(lbl)
        if r is None:
            continue
        i1 = r["iter1_acc"]; fin = r["final_acc"]
        rms0 = r["mean_act_rms_per_iter"][0]; rmsf = r["mean_act_rms_per_iter"][-1]
        dv = r.get("diverged_at")
        i1s = f"{i1:.4f}" if i1 is not None else "  --  "
        drop = f"{i1 - fin:+.4f}" if i1 is not None else "  --  "
        print(f"  {lbl:<18s} {i1s:>8s} {fin:>8.4f} {drop:>8s} "
              f"{rms0:>8.3g} {rmsf:>8.3g} {str(dv):>8s}")

    print("\n  Matched-start (no-mom) verdict:")
    dn, ln = by["deepmlp_nomom"], by["lnmlp_nomom"]
    print(f"    deepmlp no-mom: {dn['iter1_acc']:.4f} -> {dn['final_acc']:.4f} "
          f"(drop {dn['iter1_acc'] - dn['final_acc']:+.4f})")
    print(f"    lnmlp   no-mom: {ln['iter1_acc']:.4f} -> {ln['final_acc']:.4f} "
          f"(drop {ln['iter1_acc'] - ln['final_acc']:+.4f})")
    print("    -> If deepmlp drop >> lnmlp drop at matched start, LN prevents collapse.")
    print("    -> If drops are similar, the B-iter gap was the moment-match confound.")

    payload = {
        "task": "e2_q7_ln_drift", "seed": SEED, "n_iter": args.n_iter,
        "n_samples": args.n_samples, "results": results,
    }
    _save_json(RESULTS_DIR / "q7_ln_drift.json", payload)
    print("\n[Q7] === COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
