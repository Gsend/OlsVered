"""
diagnostic/runners/arch_coverage/e4_mlp_autoencoder.py
========================================================

E4: MlpAutoencoder (784→256→128→64→128→256→784) on MNIST reconstruction.

Standardized battery (continuous-target variant — P-A):
  B-train          — train via backprop (MSE loss), cache weights
  B-distill        — single-pass TP+OLS from trained weights (own recon as target)
  B-retrain-rand   — single-pass TP+OLS from random init (GT = input images)
  B-residual-profile — per-layer residual curve
  B-iter           — 10 iterations from random init
  B-subset         — layer-subset sweep (deepest-first)
  B-K-vs-N         — {naive, kfac_a} × {no_mom, mom_match}

Eval metric: MSE reconstruction loss on the test set (lower is better).

Outputs: benchmark/results/diagnostic/arch_coverage/e4/{battery_item}.json
Weights: benchmark/weights/arch_coverage/mlp_autoencoder.pt

Usage:
    python -m diagnostic.runners.arch_coverage.e4_mlp_autoencoder
    python -m diagnostic.runners.arch_coverage.e4_mlp_autoencoder --skip-train
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
from diagnostic.models.mlp_autoencoder import MlpAutoencoder
from diagnostic.multi_step import _empirical_mean_cov, _moment_match
from diagnostic.runners.phase1_mlp_mnist import _load_mnist
from diagnostic.target_prop_retrainer import solve_ols_layer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = (
    REPO_ROOT / "benchmark" / "results" / "diagnostic" / "arch_coverage" / "e4"
)
WEIGHTS_DIR = REPO_ROOT / "benchmark" / "weights" / "arch_coverage"
WEIGHTS_PATH = WEIGHTS_DIR / "mlp_autoencoder.pt"

N_LAYERS = MlpAutoencoder.N_LAYERS  # 6
N_SAMPLES = 16_384
SEED = 42
OLS_LAMBDA = 1e-4
OLS_EPS = 1e-4
N_ITER = 10
TRAIN_EPOCHS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: None)
    print(f"  -> saved {path.name}")


def _eval_mse(model: nn.Module, loader, device: torch.device) -> float:
    """Evaluate reconstruction MSE (average per-sample mean squared error)."""
    model.eval()
    total_mse, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            x_flat = x.flatten(1)
            recon = model(x)
            total_mse += F.mse_loss(recon, x_flat, reduction="sum").item()
            n += x_flat.shape[0]
    return total_mse / n


def _train(model: nn.Module, train_loader, test_loader, *,
           epochs: int, device: torch.device, lr: float = 1e-3) -> float:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            x_flat = x.flatten(1)
            opt.zero_grad()
            loss = F.mse_loss(model(x), x_flat)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()
        mse = _eval_mse(model, test_loader, device)
        print(f"  [train] epoch {epoch + 1:2d}/{epochs}  "
              f"loss={running / seen:.4f}  test_mse={mse:.4f}")
    print(f"  [train] total: {time.time() - t0:.1f}s")
    return _eval_mse(model, test_loader, device)


def _build_loader_list(dataset, max_samples: int, batch_size: int = 128):
    """Return (images_list, images_flat_tensor) — GT targets for autoencoder are the inputs."""
    subset_size = min(len(dataset), max_samples * 4)
    subset = Subset(dataset, list(range(subset_size)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    images = []
    for x, _ in loader:
        images.append(x)
    images_cat = torch.cat(images, dim=0)[:max_samples]
    images_flat = images_cat.flatten(1)  # (N, 784)
    return images, images_flat


def _layer_name(chain_idx: int) -> str:
    return f"fc{N_LAYERS - chain_idx}"


# ---------------------------------------------------------------------------
# Core single-pass TP retrain for autoencoder
# ---------------------------------------------------------------------------

def _ae_tp_retrain(
    model: MlpAutoencoder,
    loader_list: list,
    gt_targets: torch.Tensor,          # (N, 784) — target reconstructions
    n_samples: int,
    method: str,
    correct_target_mean: bool,
    correct_target_cov: bool,
    device: torch.device,
) -> Dict[int, dict]:
    """Single-pass TP+OLS for the autoencoder chain."""
    layers_df = model.get_layers_deepest_first()  # fc6 → fc1

    captured = collect_activations(
        model, loader_list, layers_df,
        max_samples=n_samples, seq_subsample=n_samples, device=device,
    )

    n_cap = captured[layers_df[0]]["a_post"].shape[0]
    n_use = min(n_cap, gt_targets.shape[0])
    deepest_target = gt_targets[:n_use].double()

    # Forward priors
    forward_priors = {}
    for layer in layers_df:
        a_in = captured[layer]["a_in"].double()[:n_use]
        forward_priors[layer] = _empirical_mean_cov(a_in)

    # Back-prop chain
    targets_per_layer: Dict[nn.Linear, torch.Tensor] = {}
    cur_target = deepest_target.clone()
    for step_idx, layer in enumerate(layers_df):
        cap = captured[layer]
        a_pre_t = cap["a_pre"].double()[:n_use]
        activation = cap["activation"]
        mu_a, Sigma_a = forward_priors[layer]
        targets_per_layer[layer] = cur_target.clone()

        W = layer.weight.detach().to(device=a_pre_t.device, dtype=a_pre_t.dtype)
        b = (layer.bias.detach().to(device=a_pre_t.device, dtype=a_pre_t.dtype)
             if layer.bias is not None else None)

        if method == "naive":
            a_hat = invert_layer(W, b, cur_target, a_pre_t, activation, "naive", eps=OLS_EPS)
        else:
            a_hat = invert_layer(W, b, cur_target, a_pre_t, activation, "kfac_a",
                                 mu_a=mu_a, Sigma_a=Sigma_a)

        next_target = a_hat
        if correct_target_mean and step_idx + 1 < len(layers_df):
            a_in_t = cap["a_in"].double()[:n_use]
            tgt_mean = a_in_t.mean(dim=0)
            if correct_target_cov:
                _, tgt_Sigma = _empirical_mean_cov(a_in_t)
                next_target = _moment_match(a_hat, tgt_mean, tgt_Sigma)
            else:
                from diagnostic.multi_step import _shift_mean_to
                next_target = _shift_mean_to(a_hat, tgt_mean)
        cur_target = next_target

    # Forward OLS sweep — shallowest first
    info: Dict[int, dict] = {}
    n_chain = len(layers_df)
    prev_a_post_new = None

    for sweep_pos in range(n_chain):
        chain_idx = n_chain - 1 - sweep_pos  # fc1=idx5, ..., fc6=idx0
        layer = layers_df[chain_idx]
        cap = captured[layer]
        a_pre_fwd = cap["a_pre"].double()[:n_use]
        act_after = cap["activation"]
        target_a_post = targets_per_layer[layer]

        X = cap["a_in"].double()[:n_use] if sweep_pos == 0 else prev_a_post_new
        t_pre = invert_activation(target_a_post, a_pre_fwd, act_after)

        W_new, b_new = solve_ols_layer(
            X, t_pre, with_bias=(layer.bias is not None), ols_lambda=OLS_LAMBDA
        )
        pred = X @ W_new.T
        if b_new is not None:
            pred = pred + b_new.unsqueeze(0)
        residual = (pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30)

        with torch.no_grad():
            layer.weight.copy_(W_new.to(layer.weight.device, layer.weight.dtype))
            if layer.bias is not None and b_new is not None:
                layer.bias.copy_(b_new.to(layer.bias.device, layer.bias.dtype))

        if act_after is not None:
            with torch.no_grad():
                prev_a_post_new = act_after(pred.float()).double()
        else:
            prev_a_post_new = pred

        info[chain_idx] = {"target_residual": residual.item()}

    return info


# ---------------------------------------------------------------------------
# Battery items
# ---------------------------------------------------------------------------

def b_train(train_set, test_loader, *, device: torch.device) -> dict:
    print("\n[E4] === B-train: training MlpAutoencoder ===")
    model = MlpAutoencoder().to(device)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    t0 = time.time()
    mse = _train(model, train_loader, test_loader, epochs=TRAIN_EPOCHS, device=device)
    wall_s = time.time() - t0
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"  [B-train] test_mse={mse:.6f}  wall={wall_s:.1f}s  weights -> {WEIGHTS_PATH}")
    result = {
        "battery": "B-train", "exp_id": "e4",
        "test_mse": mse, "epochs": TRAIN_EPOCHS,
        "dims": MlpAutoencoder.DIMS, "wall_s": wall_s,
    }
    _save_json(RESULTS_DIR / "b_train.json", result)
    return result


def b_distill(trained_model: MlpAutoencoder, loader_list: list,
              test_loader, *, device: torch.device) -> dict:
    """Self-distillation: start from trained weights, target = own reconstruction."""
    print("\n[E4] === B-distill: self-distillation from trained model ===")
    # Collect own reconstructions as targets
    trained_model.eval()
    recon_chunks = []
    with torch.no_grad():
        for batch in loader_list:
            x = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            recon_chunks.append(trained_model(x).cpu())
    distill_targets = torch.cat(recon_chunks, dim=0)[:N_SAMPLES].double()

    model = copy.deepcopy(trained_model)
    images_list = loader_list
    t0 = time.time()
    info = _ae_tp_retrain(
        model, images_list, distill_targets, N_SAMPLES,
        method="kfac_a", correct_target_mean=True, correct_target_cov=True,
        device=device,
    )
    mse = _eval_mse(model, test_loader, device)
    wall_s = time.time() - t0
    per_layer_res = {_layer_name(k): v["target_residual"] for k, v in info.items()}
    print(f"  [B-distill] test_mse={mse:.6f}  wall={wall_s:.1f}s")
    for name in [f"fc{i + 1}" for i in range(N_LAYERS)]:
        r = per_layer_res.get(name)
        print(f"    {name}: residual={r:.3g}" if r is not None else f"    {name}: --")
    result = {
        "battery": "B-distill", "exp_id": "e4",
        "test_mse": mse, "wall_s": wall_s,
        "per_layer_residuals": per_layer_res,
    }
    _save_json(RESULTS_DIR / "b_distill.json", result)
    return result


def b_retrain_rand(train_set, test_loader, *, device: torch.device, n_samples: int) -> dict:
    """Random init + GT targets = input images, single pass."""
    print("\n[E4] === B-retrain-rand: random init + GT images, single pass ===")
    torch.manual_seed(SEED)
    model = MlpAutoencoder().to(device)
    rand_mse = _eval_mse(model, test_loader, device)
    print(f"  random-init test_mse={rand_mse:.4f}")

    images_list, gt_targets = _build_loader_list(train_set, n_samples)

    t0 = time.time()
    info = _ae_tp_retrain(
        model, images_list, gt_targets, n_samples,
        method="kfac_a", correct_target_mean=True, correct_target_cov=True,
        device=device,
    )
    mse = _eval_mse(model, test_loader, device)
    wall_s = time.time() - t0
    per_layer_res = {_layer_name(k): v["target_residual"] for k, v in info.items()}
    print(f"  [B-retrain-rand] test_mse={mse:.6f}  wall={wall_s:.1f}s")
    for name in [f"fc{i + 1}" for i in range(N_LAYERS)]:
        r = per_layer_res.get(name)
        print(f"    {name}: residual={r:.3g}" if r is not None else f"    {name}: --")
    result = {
        "battery": "B-retrain-rand", "exp_id": "e4", "seed": SEED,
        "random_init_mse": rand_mse, "retrained_mse": mse,
        "delta": mse - rand_mse, "wall_s": wall_s,
        "per_layer_residuals": per_layer_res,
    }
    _save_json(RESULTS_DIR / "b_retrain_rand.json", result)
    return result


def b_residual_profile(distill_result: dict, retrain_rand_result: dict) -> dict:
    print("\n[E4] === B-residual-profile ===")
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
        "battery": "B-residual-profile", "exp_id": "e4",
        "layer_order": layer_names, "profile": profile,
        "distill_mse": distill_result.get("test_mse"),
        "retrain_rand_mse": retrain_rand_result.get("retrained_mse"),
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
    ax.set_ylabel("OLS target residual"); ax.set_title("E4 MlpAutoencoder: per-layer residual profile")
    ax.legend(); ax.set_yscale("log"); fig.tight_layout()
    out = RESULTS_DIR / "b_residual_profile.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def b_iter(train_set, test_loader, *, device: torch.device, n_samples: int) -> dict:
    print(f"\n[E4] === B-iter: {N_ITER} iterations from random init ===")
    torch.manual_seed(SEED)
    model = MlpAutoencoder().to(device)
    rand_mse = _eval_mse(model, test_loader, device)
    print(f"  random-init test_mse={rand_mse:.4f}")

    images_list, gt_targets = _build_loader_list(train_set, n_samples)

    iter_mses = [rand_mse]
    t0 = time.time()
    for it in range(N_ITER):
        _ae_tp_retrain(
            model, images_list, gt_targets, n_samples,
            method="kfac_a", correct_target_mean=True, correct_target_cov=True,
            device=device,
        )
        mse = _eval_mse(model, test_loader, device)
        iter_mses.append(mse)
        print(f"  iter {it + 1:2d}: mse={mse:.6f}  delta={mse - rand_mse:+.6f}")
    wall_s = time.time() - t0
    traj = " -> ".join(f"{m:.4f}" for m in iter_mses)
    print(f"  [B-iter] done in {wall_s:.1f}s\n  trajectory: {traj}")
    _plot_iter_trajectory(iter_mses)
    result = {
        "battery": "B-iter", "exp_id": "e4", "seed": SEED, "n_iterations": N_ITER,
        "random_init_mse": rand_mse, "iter_mses": iter_mses,
        "final_mse": iter_mses[-1], "wall_s": wall_s,
    }
    _save_json(RESULTS_DIR / "b_iter.json", result)
    return result


def _plot_iter_trajectory(iter_mses: List[float]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(iter_mses)), iter_mses, "o-")
    ax.set_xlabel("iteration"); ax.set_ylabel("test MSE (reconstruction)")
    ax.set_title("E4 MlpAutoencoder: B-iter MSE trajectory"); fig.tight_layout()
    out = RESULTS_DIR / "b_iter.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def _run_subset_retrain(
    trained_model: MlpAutoencoder,
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
        a_pre_t = cap["a_pre"].double()[:n_use]
        activation = cap["activation"]
        mu_a, Sigma_a = forward_priors[layer]
        targets_per_layer[layer] = cur_tgt.clone()
        W = layer.weight.detach().to(device=a_pre_t.device, dtype=a_pre_t.dtype)
        b = (layer.bias.detach().to(device=a_pre_t.device, dtype=a_pre_t.dtype)
             if layer.bias is not None else None)
        a_hat = invert_layer(W, b, cur_tgt, a_pre_t, activation,
                             "kfac_a", mu_a=mu_a, Sigma_a=Sigma_a)
        if step_idx + 1 < len(df_layers):
            tgt_mean = cap["a_in"].double()[:n_use].mean(dim=0)
            _, tgt_Sigma = _empirical_mean_cov(cap["a_in"].double()[:n_use])
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
        t_pre = invert_activation(targets_per_layer[layer], a_pre_fwd, act_after)
        W_new, b_new = solve_ols_layer(X, t_pre, with_bias=(layer.bias is not None),
                                       ols_lambda=OLS_LAMBDA)
        with torch.no_grad():
            layer.weight.copy_(W_new.to(layer.weight.device, layer.weight.dtype))
            if layer.bias is not None and b_new is not None:
                layer.bias.copy_(b_new.to(layer.bias.device, layer.bias.dtype))

    return _eval_mse(work_model, test_loader, device)


def b_subset(trained_model: MlpAutoencoder, gt_targets: torch.Tensor,
             loader_list: list, test_loader, *, device: torch.device, n_samples: int) -> dict:
    print("\n[E4] === B-subset: layer-subset sweep from trained model ===")
    trained_mse = _eval_mse(trained_model, test_loader, device)
    print(f"  trained baseline mse={trained_mse:.6f}")
    df_names = [f"fc{N_LAYERS - i}" for i in range(N_LAYERS)]
    subset_results = {}
    for k in range(1, N_LAYERS + 1):
        subset = df_names[:k]
        t0 = time.time()
        mse = _run_subset_retrain(
            trained_model, subset, gt_targets, loader_list, test_loader,
            device=device, n_samples=n_samples,
        )
        wall_s = time.time() - t0
        label = f"deepest_{k}"
        subset_results[label] = {
            "n_retrained": k, "target_layers": subset,
            "mse": mse, "delta": mse - trained_mse, "wall_s": wall_s,
        }
        print(f"  {label} ({k:2d} layers): mse={mse:.6f}  "
              f"delta={mse - trained_mse:+.6f}  ({wall_s:.1f}s)")
    result = {
        "battery": "B-subset", "exp_id": "e4",
        "trained_mse": trained_mse, "subsets": subset_results,
    }
    _save_json(RESULTS_DIR / "b_subset.json", result)
    _plot_subset_curve(subset_results, trained_mse)
    return result


def _plot_subset_curve(subset_results: dict, trained_mse: float) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = sorted(subset_results, key=lambda k: subset_results[k]["n_retrained"])
    ns = [subset_results[k]["n_retrained"] for k in labels]
    mses = [subset_results[k]["mse"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, mses, "o-", label="TP+OLS retrain (trained start)")
    ax.axhline(trained_mse, color="gray", linestyle="--",
               label=f"trained baseline ({trained_mse:.4f})")
    ax.set_xlabel("# layers retrained (deepest-first)"); ax.set_ylabel("test MSE")
    ax.set_title("E4 MlpAutoencoder: B-subset layer-sweep"); ax.legend(); fig.tight_layout()
    out = RESULTS_DIR / "b_subset.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> plot saved to {out.name}")


def b_k_vs_n(train_set, test_loader, *, device: torch.device, n_samples: int) -> dict:
    print("\n[E4] === B-K-vs-N: method × momentum comparison ===")
    torch.manual_seed(SEED)
    base_model = MlpAutoencoder().to(device)
    rand_mse = _eval_mse(base_model, test_loader, device)
    print(f"  random-init test_mse={rand_mse:.4f}")

    images_list, gt_targets = _build_loader_list(train_set, n_samples)

    combos = [
        ("naive",  False, False, "naive_no_mom"),
        ("naive",  True,  True,  "naive_mom_match"),
        ("kfac_a", False, False, "kfac_a_no_mom"),
        ("kfac_a", True,  True,  "kfac_a_mom_match"),
    ]
    combo_results = {}
    for method, mean_corr, cov_corr, label in combos:
        model = copy.deepcopy(base_model)
        t0 = time.time()
        _ae_tp_retrain(
            model, images_list, gt_targets, n_samples,
            method=method, correct_target_mean=mean_corr,
            correct_target_cov=cov_corr, device=device,
        )
        mse = _eval_mse(model, test_loader, device)
        wall_s = time.time() - t0
        combo_results[label] = {
            "method": method, "correct_target_mean": mean_corr,
            "correct_target_cov": cov_corr, "mse": mse,
            "delta": mse - rand_mse, "wall_s": wall_s,
        }
        print(f"  {label:<22s}: mse={mse:.6f}  delta={mse - rand_mse:+.6f}  ({wall_s:.1f}s)")
    result = {
        "battery": "B-K-vs-N", "exp_id": "e4", "seed": SEED,
        "random_init_mse": rand_mse, "combos": combo_results,
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
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[E4] MlpAutoencoder  device={device}  "
          f"n_samples={args.n_samples}  dims={MlpAutoencoder.DIMS}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[E4] loading MNIST...")
    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    if args.skip_train and WEIGHTS_PATH.exists():
        print(f"[E4] loading cached weights from {WEIGHTS_PATH}")
        trained_model = MlpAutoencoder().to(device)
        trained_model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        )
        trained_mse = _eval_mse(trained_model, test_loader, device)
        print(f"[E4] loaded; test_mse={trained_mse:.6f}")
        train_result = {"battery": "B-train", "test_mse": trained_mse, "loaded_from_cache": True}
    else:
        train_result = b_train(train_set, test_loader, device=device)
        trained_model = MlpAutoencoder().to(device)
        trained_model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        )

    images_list, gt_targets = _build_loader_list(train_set, args.n_samples)

    distill_result = {}
    if not args.skip_distill:
        distill_result = b_distill(trained_model, images_list, test_loader, device=device)

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
        b_subset(trained_model, gt_targets, images_list, test_loader,
                 device=device, n_samples=args.n_samples)

    if not args.skip_k_vs_n:
        b_k_vs_n(train_set, test_loader, device=device, n_samples=args.n_samples)

    print(f"\n[E4] === COMPLETE ===")
    print(f"  B-train mse:       {train_result.get('test_mse', '?')}")
    print(f"  B-distill mse:     {distill_result.get('test_mse', '?')}")
    print(f"  B-retrain-rand mse:{retrain_rand_result.get('retrained_mse', '?')}")
    print(f"  Results at: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
