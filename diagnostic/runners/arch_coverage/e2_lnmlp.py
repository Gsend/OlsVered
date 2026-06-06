"""
diagnostic/runners/arch_coverage/e2_lnmlp.py
=============================================

E2: LNMLP (MLP + LayerNorm+ReLU activations) on MNIST classification.

Standardized battery — identical to E1 except model uses LNReLU activations:
  B-train          — train via backprop, cache weights
  B-distill        — single-pass TP+OLS from trained weights (own outputs as target)
  B-retrain-rand   — single-pass TP+OLS from random init (GT one-hot targets)
  B-residual-profile — per-layer residual curve, deepest→shallowest
  B-iter           — 10 iterations from random init, track acc + residuals
  B-subset         — layer-subset sweep {fc10 only … all 10}
  B-K-vs-N         — {naive, kfac_a} × {no_mom, mom_match} on random init

Outputs: benchmark/results/diagnostic/arch_coverage/e2/{battery_item}.json
Weights: benchmark/weights/arch_coverage/lnmlp_10l.pt

Usage:
    python -m diagnostic.runners.arch_coverage.e2_lnmlp
    python -m diagnostic.runners.arch_coverage.e2_lnmlp --skip-train
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
from torch.utils.data import DataLoader, Subset

from diagnostic.capture import collect_activations
from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.models.lnmlp import LNMLP
from diagnostic.multi_step import _empirical_mean_cov, _moment_match
from diagnostic.runners.phase1_gt_retrain import gt_target_retrain, make_gt_logit_target
from diagnostic.runners.phase1_mlp_mnist import _eval, _load_mnist
from diagnostic.target_prop_retrainer import retrain_via_target_prop, solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = (
    REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e2"
)
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"
WEIGHTS_PATH = WEIGHTS_DIR / "lnmlp_10l.pt"

N_LAYERS = 10
HIDDEN_DIM = 256
N_SAMPLES = 16_384
SEED = 42
GT_MARGIN = 5.0
OLS_LAMBDA = 1e-4
OLS_EPS = 1e-4
N_ITER = 10
TRAIN_EPOCHS = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _train(model: nn.Module, train_loader, test_loader, *, epochs: int,
           device: torch.device, lr: float = 1e-3) -> float:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()
        acc = _eval(model, test_loader, device)
        print(f"  [train] epoch {epoch + 1:2d}/{epochs}  "
              f"loss={running / seen:.4f}  test_acc={acc:.4f}")
    print(f"  [train] total: {time.time() - t0:.1f}s")
    return _eval(model, test_loader, device)


def _build_loader_list(dataset, max_samples: int, batch_size: int = 128):
    subset_size = min(len(dataset), max_samples * 4)
    subset = Subset(dataset, list(range(subset_size)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, label_chunks = [], []
    for x, y in loader:
        images.append(x)
        label_chunks.append(y)
    labels = torch.cat(label_chunks, dim=0)[:max_samples]
    return images, labels


def _layer_name(chain_idx: int) -> str:
    return f"fc{N_LAYERS - chain_idx}"


# ---------------------------------------------------------------------------
# Battery items
# ---------------------------------------------------------------------------

def b_train(train_set, test_loader, *, device: torch.device) -> dict:
    print("\n[E2] === B-train: training LNMLP-10L ===")
    model = LNMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    t0 = time.time()
    acc = _train(model, train_loader, test_loader, epochs=TRAIN_EPOCHS, device=device)
    wall_s = time.time() - t0
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"  [B-train] test_acc={acc:.4f}  wall={wall_s:.1f}s  weights -> {WEIGHTS_PATH}")
    result = {
        "battery": "B-train", "exp_id": "e2",
        "test_acc": acc, "epochs": TRAIN_EPOCHS,
        "n_layers": N_LAYERS, "hidden_dim": HIDDEN_DIM, "wall_s": wall_s,
    }
    _save_json(RESULTS_DIR / "b_train.json", result)
    return result


def b_distill(trained_model: LNMLP, loader_list: list,
              test_loader, *, device: torch.device) -> dict:
    print("\n[E2] === B-distill: self-distillation from trained model ===")
    layers_df = trained_model.get_layers_deepest_first()
    t0 = time.time()
    res = retrain_via_target_prop(
        trained_model, layers_df, loader_list,
        method="kfac_a",
        correct_target_mean=True, correct_target_cov=True,
        eps=OLS_EPS, ols_lambda=OLS_LAMBDA,
        max_samples=N_SAMPLES, seq_subsample=N_SAMPLES,
        device=device, n_iterations=1,
    )
    acc = _eval(res.model, test_loader, device)
    wall_s = time.time() - t0
    per_layer_res = {
        _layer_name(k): v["target_residual"]
        for k, v in res.per_layer_info.items()
    }
    print(f"  [B-distill] test_acc={acc:.4f}  wall={wall_s:.1f}s")
    for name in [f"fc{i + 1}" for i in range(N_LAYERS)]:
        r = per_layer_res.get(name)
        print(f"    {name}: residual={r:.3g}" if r is not None else f"    {name}: --")
    result = {
        "battery": "B-distill", "exp_id": "e2",
        "test_acc": acc, "wall_s": wall_s,
        "per_layer_residuals": per_layer_res,
    }
    _save_json(RESULTS_DIR / "b_distill.json", result)
    return result


def b_retrain_rand(train_set, test_loader, *, device: torch.device,
                   n_samples: int) -> dict:
    print("\n[E2] === B-retrain-rand: random init + GT labels, single pass ===")
    torch.manual_seed(SEED)
    model = LNMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
    rand_acc = _eval(model, test_loader, device)
    print(f"  random-init test_acc={rand_acc:.4f}")

    loader_list, labels = _build_loader_list(train_set, n_samples)
    gt_targets = make_gt_logit_target(labels, n_classes=10, margin=GT_MARGIN)
    layers_df = model.get_layers_deepest_first()

    t0 = time.time()
    info = gt_target_retrain(
        model=model, chain_layers=layers_df, work_layers=layers_df,
        deepest_gt_targets=gt_targets, dataloader_list=loader_list,
        method="kfac_a", correct_target_mean=True, correct_target_cov=True,
        eps=OLS_EPS, ols_lambda=OLS_LAMBDA, max_samples=n_samples, device=device,
    )
    acc = _eval(model, test_loader, device)
    wall_s = time.time() - t0
    per_layer_res = {_layer_name(k): v["target_residual"] for k, v in info.items()}
    print(f"  [B-retrain-rand] test_acc={acc:.4f}  delta={acc - rand_acc:+.4f}  wall={wall_s:.1f}s")
    for name in [f"fc{i + 1}" for i in range(N_LAYERS)]:
        r = per_layer_res.get(name)
        print(f"    {name}: residual={r:.3g}" if r is not None else f"    {name}: --")
    result = {
        "battery": "B-retrain-rand", "exp_id": "e2", "seed": SEED,
        "random_init_acc": rand_acc, "retrained_acc": acc,
        "delta": acc - rand_acc, "wall_s": wall_s,
        "per_layer_residuals": per_layer_res,
    }
    _save_json(RESULTS_DIR / "b_retrain_rand.json", result)
    return result


def b_residual_profile(distill_result: dict, retrain_rand_result: dict) -> dict:
    print("\n[E2] === B-residual-profile ===")
    layer_names = [f"fc{i + 1}" for i in range(N_LAYERS)]
    dr = distill_result.get("per_layer_residuals", {})
    rr = retrain_rand_result.get("per_layer_residuals", {})
    profile = {
        name: {"distill": dr.get(name), "retrain_rand": rr.get(name)}
        for name in layer_names
    }
    print(f"  {'layer':<8s}  {'distill':>10s}  {'retrain_rand':>12s}")
    for name in layer_names:
        d = profile[name]["distill"]
        r = profile[name]["retrain_rand"]
        print(f"  {name:<8s}  "
              f"{(f'{d:.3g}' if d is not None else '--'):>10s}  "
              f"{(f'{r:.3g}' if r is not None else '--'):>12s}")
    result = {
        "battery": "B-residual-profile", "exp_id": "e2",
        "layer_order": layer_names, "profile": profile,
        "distill_acc": distill_result.get("test_acc"),
        "retrain_rand_acc": retrain_rand_result.get("retrained_acc"),
    }
    _save_json(RESULTS_DIR / "b_residual_profile.json", result)
    _plot_residual_profile(profile, layer_names)
    return result


def _plot_residual_profile(profile: dict, layer_names: List[str]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    distill_vals = [profile[n]["distill"] for n in layer_names]
    retrain_vals = [profile[n]["retrain_rand"] for n in layer_names]
    x = range(len(layer_names))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, distill_vals, "o-", label="B-distill (trained start)")
    ax.plot(x, retrain_vals, "s--", label="B-retrain-rand (random init)")
    ax.set_xticks(list(x)); ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylabel("OLS target residual"); ax.set_title("E2 LNMLP-10L: per-layer residual profile")
    ax.legend(); ax.set_yscale("log"); fig.tight_layout()
    out = RESULTS_DIR / "b_residual_profile.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def b_iter(train_set, test_loader, *, device: torch.device, n_samples: int) -> dict:
    print(f"\n[E2] === B-iter: {N_ITER} iterations from random init ===")
    torch.manual_seed(SEED)
    model = LNMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
    rand_acc = _eval(model, test_loader, device)
    layers_df = model.get_layers_deepest_first()
    print(f"  random-init test_acc={rand_acc:.4f}")

    loader_list, labels = _build_loader_list(train_set, n_samples)
    gt_targets = make_gt_logit_target(labels, n_classes=10, margin=GT_MARGIN)

    iter_accs = [rand_acc]
    t0 = time.time()
    for it in range(N_ITER):
        info = gt_target_retrain(
            model=model, chain_layers=layers_df, work_layers=layers_df,
            deepest_gt_targets=gt_targets, dataloader_list=loader_list,
            method="kfac_a", correct_target_mean=True, correct_target_cov=True,
            eps=OLS_EPS, ols_lambda=OLS_LAMBDA, max_samples=n_samples, device=device,
        )
        acc = _eval(model, test_loader, device)
        iter_accs.append(acc)
        print(f"  iter {it + 1:2d}: acc={acc:.4f}  delta={acc - rand_acc:+.4f}")
    wall_s = time.time() - t0
    traj = " -> ".join(f"{a:.4f}" for a in iter_accs)
    print(f"  [B-iter] done in {wall_s:.1f}s\n  trajectory: {traj}")
    _plot_iter_trajectory(iter_accs)
    result = {
        "battery": "B-iter", "exp_id": "e2", "seed": SEED, "n_iterations": N_ITER,
        "random_init_acc": rand_acc, "iter_accs": iter_accs,
        "final_acc": iter_accs[-1], "wall_s": wall_s,
    }
    _save_json(RESULTS_DIR / "b_iter.json", result)
    return result


def _plot_iter_trajectory(iter_accs: List[float]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(iter_accs)), iter_accs, "o-")
    ax.set_xlabel("iteration"); ax.set_ylabel("test accuracy")
    ax.set_title("E2 LNMLP-10L: B-iter accuracy trajectory"); fig.tight_layout()
    out = RESULTS_DIR / "b_iter.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def _run_subset_retrain(
    trained_model: LNMLP,
    target_layer_names: List[str],
    gt_targets: torch.Tensor,
    loader_list: list,
    test_loader,
    *, device: torch.device, n_samples: int,
) -> float:
    work_model = copy.deepcopy(trained_model)
    df_names = [f"fc{N_LAYERS - i}" for i in range(N_LAYERS)]
    df_layers = [getattr(work_model, name) for name in df_names]

    captured = collect_activations(
        work_model, loader_list, df_layers,
        max_samples=n_samples, seq_subsample=n_samples, device=device,
    )
    n_cap = captured[df_layers[0]]["a_post"].shape[0]
    n_use = min(n_cap, gt_targets.shape[0])
    gt_tgt = gt_targets[:n_use].double()

    forward_priors = {}
    for layer in df_layers:
        a_in = captured[layer]["a_in"].double()[:n_use]
        forward_priors[layer] = _empirical_mean_cov(a_in)

    targets_per_layer: Dict[nn.Linear, torch.Tensor] = {}
    cur_tgt = gt_tgt.clone()
    for step_idx, layer in enumerate(df_layers):
        cap = captured[layer]
        a_in_t = cap["a_in"].double()[:n_use]
        a_pre_t = cap["a_pre"].double()[:n_use]
        activation = cap["activation"]
        mu_a, Sigma_a = forward_priors[layer]
        targets_per_layer[layer] = cur_tgt.clone()

        W_old = layer.weight.detach().to(device=a_in_t.device, dtype=a_in_t.dtype)
        b_old = (layer.bias.detach().to(device=a_in_t.device, dtype=a_in_t.dtype)
                 if layer.bias is not None else None)
        a_hat = invert_layer(
            W_old, b_old, cur_tgt,
            a_pre_forward=a_pre_t, activation=activation,
            method="kfac_a", mu_a=mu_a, Sigma_a=Sigma_a,
        )
        if step_idx + 1 < len(df_layers):
            tgt_mean = a_in_t.mean(dim=0)
            _, tgt_Sigma = _empirical_mean_cov(a_in_t)
            cur_tgt = _moment_match(a_hat, tgt_mean, tgt_Sigma)
        else:
            cur_tgt = a_hat

    target_set = set(target_layer_names)
    for sweep_pos in range(N_LAYERS):
        chain_idx = N_LAYERS - 1 - sweep_pos
        name = df_names[chain_idx]
        if name not in target_set:
            continue
        layer = df_layers[chain_idx]
        re_cap = collect_activations(
            work_model, loader_list, [layer],
            max_samples=n_samples, seq_subsample=n_samples, device=device,
        )
        X = re_cap[layer]["a_in"].double()[:n_use]
        a_pre_fwd = re_cap[layer]["a_pre"].double()[:n_use]
        act_after = captured[layer]["activation"]
        tgt_apost = targets_per_layer[layer]
        t_pre = invert_activation(tgt_apost, a_pre_fwd, act_after)
        W_new, b_new = solve_ols_layer(
            X, t_pre, with_bias=(layer.bias is not None), ols_lambda=OLS_LAMBDA
        )
        with torch.no_grad():
            layer.weight.copy_(W_new.to(device=layer.weight.device, dtype=layer.weight.dtype))
            if layer.bias is not None and b_new is not None:
                layer.bias.copy_(b_new.to(device=layer.bias.device, dtype=layer.bias.dtype))

    return _eval(work_model, test_loader, device)


def b_subset(trained_model: LNMLP, gt_targets: torch.Tensor,
             loader_list: list, test_loader, *, device: torch.device, n_samples: int) -> dict:
    print("\n[E2] === B-subset: layer-subset sweep from trained model ===")
    trained_acc = _eval(trained_model, test_loader, device)
    print(f"  trained baseline acc={trained_acc:.4f}")
    df_names = [f"fc{N_LAYERS - i}" for i in range(N_LAYERS)]
    subset_results = {}
    for k in range(1, N_LAYERS + 1):
        subset = df_names[:k]
        t0 = time.time()
        acc = _run_subset_retrain(
            trained_model, subset, gt_targets, loader_list, test_loader,
            device=device, n_samples=n_samples,
        )
        wall_s = time.time() - t0
        label = f"deepest_{k}"
        subset_results[label] = {
            "n_retrained": k, "target_layers": subset,
            "acc": acc, "delta": acc - trained_acc, "wall_s": wall_s,
        }
        print(f"  {label} ({k:2d} layers): acc={acc:.4f}  delta={acc - trained_acc:+.4f}  ({wall_s:.1f}s)")
    result = {
        "battery": "B-subset", "exp_id": "e2",
        "trained_acc": trained_acc, "subsets": subset_results,
    }
    _save_json(RESULTS_DIR / "b_subset.json", result)
    _plot_subset_curve(subset_results, trained_acc)
    return result


def _plot_subset_curve(subset_results: dict, trained_acc: float) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = sorted(subset_results, key=lambda k: subset_results[k]["n_retrained"])
    ns = [subset_results[k]["n_retrained"] for k in labels]
    accs = [subset_results[k]["acc"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, accs, "o-", label="TP+OLS retrain (trained start)")
    ax.axhline(trained_acc, color="gray", linestyle="--",
               label=f"trained baseline ({trained_acc:.3f})")
    ax.set_xlabel("# layers retrained (deepest-first)"); ax.set_ylabel("test accuracy")
    ax.set_title("E2 LNMLP-10L: B-subset layer-sweep"); ax.legend(); fig.tight_layout()
    out = RESULTS_DIR / "b_subset.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def b_k_vs_n(train_set, test_loader, *, device: torch.device, n_samples: int) -> dict:
    print("\n[E2] === B-K-vs-N: method × momentum comparison ===")
    torch.manual_seed(SEED)
    base_model = LNMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
    rand_acc = _eval(base_model, test_loader, device)
    print(f"  random-init test_acc={rand_acc:.4f}")

    loader_list, labels = _build_loader_list(train_set, n_samples)
    gt_targets = make_gt_logit_target(labels, n_classes=10, margin=GT_MARGIN)

    combos = [
        ("naive",  False, False, "naive_no_mom"),
        ("naive",  True,  True,  "naive_mom_match"),
        ("kfac_a", False, False, "kfac_a_no_mom"),
        ("kfac_a", True,  True,  "kfac_a_mom_match"),
    ]
    combo_results = {}
    for method, mean_corr, cov_corr, label in combos:
        model = copy.deepcopy(base_model)
        layers_df = model.get_layers_deepest_first()
        t0 = time.time()
        info = gt_target_retrain(
            model=model, chain_layers=layers_df, work_layers=layers_df,
            deepest_gt_targets=gt_targets, dataloader_list=loader_list,
            method=method, correct_target_mean=mean_corr, correct_target_cov=cov_corr,
            eps=OLS_EPS, ols_lambda=OLS_LAMBDA, max_samples=n_samples, device=device,
        )
        acc = _eval(model, test_loader, device)
        wall_s = time.time() - t0
        combo_results[label] = {
            "method": method, "correct_target_mean": mean_corr,
            "correct_target_cov": cov_corr, "acc": acc,
            "delta": acc - rand_acc, "wall_s": wall_s,
        }
        print(f"  {label:<22s}: acc={acc:.4f}  delta={acc - rand_acc:+.4f}  ({wall_s:.1f}s)")
    result = {
        "battery": "B-K-vs-N", "exp_id": "e2", "seed": SEED,
        "random_init_acc": rand_acc, "combos": combo_results,
    }
    _save_json(RESULTS_DIR / "b_k_vs_n.json", result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-distill", action="store_true")
    ap.add_argument("--skip-retrain-rand", action="store_true")
    ap.add_argument("--skip-iter", action="store_true")
    ap.add_argument("--skip-subset", action="store_true")
    ap.add_argument("--skip-k-vs-n", action="store_true")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    globals()["SEED"] = args.seed

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[E2] LNMLP-{N_LAYERS}L  device={device}  "
          f"n_samples={args.n_samples}  hidden_dim={HIDDEN_DIM}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[E2] loading MNIST...")
    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    if args.skip_train and WEIGHTS_PATH.exists():
        print(f"[E2] loading cached weights from {WEIGHTS_PATH}")
        trained_model = LNMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
        trained_model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        )
        trained_acc = _eval(trained_model, test_loader, device)
        print(f"[E2] loaded; test_acc={trained_acc:.4f}")
        train_result = {"battery": "B-train", "test_acc": trained_acc, "loaded_from_cache": True}
    else:
        train_result = b_train(train_set, test_loader, device=device)
        trained_model = LNMLP(n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
        trained_model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        )

    loader_list, labels = _build_loader_list(train_set, args.n_samples)
    gt_targets = make_gt_logit_target(labels, n_classes=10, margin=GT_MARGIN)

    distill_result = {}
    if not args.skip_distill:
        distill_result = b_distill(trained_model, loader_list, test_loader, device=device)

    retrain_rand_result = {}
    if not args.skip_retrain_rand:
        retrain_rand_result = b_retrain_rand(
            train_set, test_loader, device=device, n_samples=args.n_samples
        )

    if distill_result and retrain_rand_result:
        b_residual_profile(distill_result, retrain_rand_result)

    if not args.skip_iter:
        b_iter(train_set, test_loader, device=device, n_samples=args.n_samples)

    if not args.skip_subset:
        b_subset(trained_model, gt_targets, loader_list, test_loader,
                 device=device, n_samples=args.n_samples)

    if not args.skip_k_vs_n:
        b_k_vs_n(train_set, test_loader, device=device, n_samples=args.n_samples)

    print(f"\n[E2] === COMPLETE ===")
    print(f"  B-train acc:       {train_result.get('test_acc', '?')}")
    print(f"  B-distill acc:     {distill_result.get('test_acc', '?')}")
    print(f"  B-retrain-rand acc:{retrain_rand_result.get('retrained_acc', '?')}")
    print(f"  Results at: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
