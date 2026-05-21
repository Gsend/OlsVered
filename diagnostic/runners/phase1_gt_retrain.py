"""
diagnostic/runners/phase1_gt_retrain.py
=======================================

Tests the framework's RETRAINING capability (vs. its distillation capability).

The distillation test (phase1_final_layer_refit.py) showed that target-prop +
forward-sweep OLS perfectly recovers a reference model's behavior when the
reference's intermediate outputs are available as per-layer targets.

This experiment tests the harder question: can the framework retrain a
network using only ground-truth LABELS as the deepest target, with no
intermediate references? This is the "real retraining" scenario.

Setup:
  - Use the cached trained MnistMLP as a starting point.
  - Inject random weight perturbation to simulate a "damaged" model that
    needs retraining (or use --random-init to start from scratch).
  - Set the deepest target = one-hot logits derived from GT labels.
  - Run TP retrain (kfac + moment match).
  - Evaluate.

Compares retraining accuracy under three setups:
  1. damaged model + TP retrain alone
  2. damaged model + TP retrain + fc4 refit against the same GT one-hot
  3. damaged model with no retraining (the "floor")

Usage:
    python -m diagnostic.runners.phase1_gt_retrain
    python -m diagnostic.runners.phase1_gt_retrain --noise-std 0.1
    python -m diagnostic.runners.phase1_gt_retrain --random-init
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from diagnostic.capture import collect_activations
from diagnostic.inversion import invert_activation, invert_layer
from diagnostic.multi_step import (
    _empirical_mean_cov,
    _moment_match,
    _shift_mean_to,
)
from diagnostic.runners.phase1_mlp_mnist import (
    MnistMLP,
    REPO_ROOT,
    RESULTS_DIR,
    WEIGHTS_PATH,
    _eval,
    _load_mnist,
)
from diagnostic.target_prop_retrainer import solve_ols_layer


OUT_JSON = RESULTS_DIR / "phase1_gt_retrain.json"


def make_gt_logit_target(labels_tensor, n_classes, margin=5.0):
    """Convert label indices to one-hot logit targets:
        target[j, GT_class] = +margin
        target[j, other]    = -margin
    """
    n = labels_tensor.shape[0]
    t = -margin * torch.ones(n, n_classes, dtype=torch.float64)
    t[torch.arange(n), labels_tensor.long()] = +margin
    return t


def gt_target_retrain(
    model,
    chain_layers,  # deepest-first: [fc4, fc3, fc2, fc1]
    work_layers,
    deepest_gt_targets,  # (n, 10) one-hot logit targets
    dataloader_list,
    *,
    method="kfac_a",
    correct_target_mean=True,
    correct_target_cov=True,
    eps=1e-4,
    sigma2=None,
    ols_lambda=1e-4,
    max_samples=4096,
    device=None,
):
    """Run a single TP retrain pass using GT one-hot logits as the deepest
    layer's a_post target (instead of the model's own forward outputs)."""
    # 1. Capture forward activations from current model
    captured = collect_activations(
        model, dataloader_list, chain_layers,
        max_samples=max_samples, seq_subsample=max_samples,
        device=device,
    )

    # 2. Forward priors per layer
    forward_priors = {}
    for layer in chain_layers:
        a_in = captured[layer]["a_in"].double()
        forward_priors[layer] = _empirical_mean_cov(a_in)

    # 3. Truncate GT targets to match captured sample count
    n_captured = captured[chain_layers[0]]["a_post"].shape[0]
    n_gt = deepest_gt_targets.shape[0]
    n_use = min(n_captured, n_gt)
    deepest_target = deepest_gt_targets[:n_use].double()

    # 4. Chain back-prop using GT targets at the deepest layer
    targets_per_layer = {}
    cur_target_a_post = deepest_target.clone()
    for step_idx, layer in enumerate(chain_layers):
        cap = captured[layer]
        a_in_truth = cap["a_in"].double()[:n_use]
        a_pre_truth = cap["a_pre"].double()[:n_use]
        a_post_truth = cap["a_post"].double()[:n_use]
        activation = cap["activation"]
        mu_a, Sigma_a = forward_priors[layer]

        targets_per_layer[layer] = cur_target_a_post.clone()

        W_old = layer.weight.detach().to(
            device=a_in_truth.device, dtype=a_in_truth.dtype)
        b_old = (layer.bias.detach().to(
            device=a_in_truth.device, dtype=a_in_truth.dtype)
            if layer.bias is not None else None)

        if method == "naive":
            a_hat = invert_layer(
                W_old, b_old, cur_target_a_post,
                a_pre_forward=a_pre_truth,
                activation=activation, method="naive", eps=eps,
            )
        else:
            a_hat = invert_layer(
                W_old, b_old, cur_target_a_post,
                a_pre_forward=a_pre_truth,
                activation=activation, method="kfac_a",
                mu_a=mu_a, Sigma_a=Sigma_a, sigma2=sigma2,
            )

        next_target = a_hat
        if correct_target_mean and step_idx + 1 < len(chain_layers):
            tgt_mean = a_in_truth.mean(dim=0)
            if correct_target_cov:
                _, tgt_Sigma = _empirical_mean_cov(a_in_truth)
                next_target = _moment_match(a_hat, tgt_mean, tgt_Sigma)
            else:
                next_target = _shift_mean_to(a_hat, tgt_mean)
        cur_target_a_post = next_target

    # 5. Per-layer OLS solve — FORWARD SWEEP (shallowest -> deepest).
    # Entry retrained layer uses captured a_in; every deeper retrained layer
    # uses the post-activation output of the just-rebuilt previous layer.
    info = {}
    n_chain = len(chain_layers)
    prev_a_post_new = None

    for sweep_pos in range(n_chain):
        chain_idx = n_chain - 1 - sweep_pos  # deepest-first key preserved
        chain_layer = chain_layers[chain_idx]
        work_layer = work_layers[chain_idx]

        cap = captured[chain_layer]
        a_pre_forward = cap["a_pre"].double()[:n_use]
        activation_after = cap["activation"]
        target_a_post = targets_per_layer[chain_layer]

        if sweep_pos == 0:
            X = cap["a_in"].double()[:n_use]
        else:
            X = prev_a_post_new

        t_pre = invert_activation(target_a_post, a_pre_forward, activation_after)
        W_new, b_new = solve_ols_layer(
            X, t_pre,
            with_bias=(work_layer.bias is not None),
            ols_lambda=ols_lambda,
        )

        with torch.no_grad():
            target_device = work_layer.weight.device
            target_dtype = work_layer.weight.dtype
            work_layer.weight.copy_(W_new.to(device=target_device, dtype=target_dtype))
            if work_layer.bias is not None and b_new is not None:
                work_layer.bias.copy_(b_new.to(device=target_device, dtype=target_dtype))

            pred = X @ W_new.T
            if work_layer.bias is not None and b_new is not None:
                pred = pred + b_new.unsqueeze(0)
            residual = (pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30)

            a_pre_new = pred
            if activation_after is None or isinstance(activation_after, nn.Identity):
                prev_a_post_new = a_pre_new
            else:
                try:
                    act_param = next(activation_after.parameters())
                    prev_a_post_new = activation_after(
                        a_pre_new.to(device=act_param.device, dtype=act_param.dtype)
                    ).to(device=a_pre_new.device, dtype=a_pre_new.dtype)
                except StopIteration:
                    prev_a_post_new = activation_after(a_pre_new)

        info[chain_idx] = {
            "d_in": int(X.shape[1]),
            "d_out": int(t_pre.shape[1]),
            "target_residual": float(residual.item()),
            "n_samples": int(X.shape[0]),
        }
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-train", action="store_true", default=True)
    ap.add_argument("--max-samples", type=int, default=16384)
    ap.add_argument("--retrain-data", choices=("test", "train"), default="train")
    ap.add_argument("--noise-std", type=float, default=0.05,
                    help="Std of Gaussian noise injected into the trained weights.")
    ap.add_argument("--random-init", action="store_true",
                    help="Start from a randomly-initialized model (not the trained one).")
    ap.add_argument("--gt-margin", type=float, default=5.0,
                    help="One-hot logit magnitude for GT targets.")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--sigma2", type=float, default=None)
    ap.add_argument("--ols-lambda", type=float, default=1e-4)
    ap.add_argument("--method", choices=("naive", "kfac_a"), default="kfac_a")
    ap.add_argument("--no-mean-correct", action="store_true",
                    help="Disable target mean correction between chain steps.")
    ap.add_argument("--no-cov-correct", action="store_true",
                    help="Disable target covariance moment match between chain steps.")
    ap.add_argument("--n-iterations", type=int, default=1,
                    help="Number of TP-retrain passes. Each pass re-captures "
                         "activations from the updated work model; the GT "
                         "one-hot targets stay fixed across iterations.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[gt-retrain] device = {device}  retrain_data={args.retrain_data}  "
          f"max_samples={args.max_samples}  gt_margin={args.gt_margin}")

    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    # Build the starting model (damaged from baseline, or random init)
    starting_model = MnistMLP().to(device)
    if args.random_init:
        print(f"[gt-retrain] starting from random init")
    else:
        if not WEIGHTS_PATH.exists():
            print(f"[gt-retrain] no cached weights at {WEIGHTS_PATH}.")
            return 1
        starting_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        # Add noise
        if args.noise_std > 0:
            print(f"[gt-retrain] adding noise with std={args.noise_std} to trained weights")
            with torch.no_grad():
                for p in starting_model.parameters():
                    p.add_(args.noise_std * torch.randn_like(p))

    starting_acc = _eval(starting_model, test_loader, device)
    print(f"[gt-retrain] starting model test_acc = {starting_acc:.4f}")

    # Retrain dataloader: yields (image, label) so we can grab labels
    if args.retrain_data == "train":
        subset_size = min(len(train_set), max(args.max_samples * 4, 4096))
        retrain_subset = Subset(train_set, list(range(subset_size)))
    else:
        subset_size = min(len(test_set), max(args.max_samples * 4, 4096))
        retrain_subset = Subset(test_set, list(range(subset_size)))
    retrain_loader = DataLoader(retrain_subset, batch_size=128,
                                shuffle=False, num_workers=0)

    images = []
    labels_list = []
    for x, y in retrain_loader:
        images.append(x)
        labels_list.append(y)
    labels = torch.cat(labels_list, dim=0)[:args.max_samples]
    print(f"[gt-retrain] retrain set: {len(labels)} samples")

    def batch_iter():
        for x, _ in retrain_loader:
            yield x
    loader_list = list(batch_iter())

    # Build GT one-hot targets
    gt_targets = make_gt_logit_target(labels, n_classes=10, margin=args.gt_margin)
    print(f"[gt-retrain] GT targets shape={tuple(gt_targets.shape)}, "
          f"margin={args.gt_margin}")

    # === Variant 1: TP retrain with GT targets, shrinking-chain iteration ===
    # Iter k retrains the deepest (N - k + 1) layers. The shallowest layer
    # retrained at iter k is frozen for all subsequent iters. This refines
    # the deeper layers against fresh X without redoing the noisiest shallow
    # back-prop targets.
    print(f"\n[gt-retrain] Variant 1: TP retrain ({args.method}+mean={not args.no_mean_correct}+cov={not args.no_cov_correct}) "
          f"with GT one-hot targets, shrinking chain, n_iter={args.n_iterations}...")
    tp_model = copy.deepcopy(starting_model)
    all_chain = [tp_model.fc4, tp_model.fc3, tp_model.fc2, tp_model.fc1]
    n_total = len(all_chain)
    n_iters_effective = min(args.n_iterations, n_total)
    if n_iters_effective < args.n_iterations:
        print(f"[gt-retrain]   note: requested {args.n_iterations} iterations but "
              f"only {n_total} layers available; capping at {n_iters_effective}.")
    iter_accs = []
    iter_infos = []
    t0 = time.time()
    for iter_idx in range(n_iters_effective):
        current_chain = all_chain[: n_total - iter_idx]
        n_layers = len(current_chain)
        layer_names = [f"fc{4 - all_chain.index(l)}" for l in current_chain]
        tp_info = gt_target_retrain(
            model=tp_model,
            chain_layers=current_chain,
            work_layers=current_chain,
            deepest_gt_targets=gt_targets,
            dataloader_list=loader_list,
            method=args.method,
            correct_target_mean=(not args.no_mean_correct),
            correct_target_cov=(not args.no_cov_correct),
            eps=args.eps,
            sigma2=args.sigma2,
            ols_lambda=args.ols_lambda,
            max_samples=args.max_samples,
            device=device,
        )
        iter_acc = _eval(tp_model, test_loader, device)
        iter_accs.append(iter_acc)
        iter_infos.append(tp_info)
        print(f"[gt-retrain]   iter {iter_idx + 1} (retraining {n_layers} layers: "
              f"{layer_names}): test_acc = {iter_acc:.4f}  "
              f"(delta vs starting = {iter_acc - starting_acc:+.4f})")
        for li in sorted(tp_info.keys()):
            info = tp_info[li]
            print(f"[gt-retrain]     layer {li}: d_in={info['d_in']}, d_out={info['d_out']}  "
                  f"target_residual={info['target_residual']:.3g}")
    t1 = time.time()
    tp_acc = iter_accs[-1]
    print(f"[gt-retrain]   {n_iters_effective} iterations done in {t1 - t0:.1f}s")
    if n_iters_effective > 1:
        traj = " -> ".join(f"{a:.4f}" for a in iter_accs)
        print(f"[gt-retrain]   accuracy trajectory: {traj}")

    # === Variant 2: TP + fc4 OLS against GT targets ===
    print(f"\n[gt-retrain] Variant 2: TP + fc4 OLS refit (target = GT one-hot)...")
    cap_work = collect_activations(
        tp_model, loader_list, [tp_model.fc4],
        max_samples=args.max_samples, seq_subsample=args.max_samples,
        device=device,
    )
    X_to_fc4 = cap_work[tp_model.fc4]["a_in"].double()
    n_use = min(X_to_fc4.shape[0], gt_targets.shape[0])
    X_to_fc4 = X_to_fc4[:n_use]
    gt_for_fc4 = gt_targets[:n_use]
    t2 = time.time()
    W_new, b_new = solve_ols_layer(
        X_to_fc4, gt_for_fc4, with_bias=True, ols_lambda=args.ols_lambda
    )
    with torch.no_grad():
        tp_model.fc4.weight.copy_(W_new.to(
            device=tp_model.fc4.weight.device, dtype=tp_model.fc4.weight.dtype))
        tp_model.fc4.bias.copy_(b_new.to(
            device=tp_model.fc4.bias.device, dtype=tp_model.fc4.bias.dtype))
    t3 = time.time()
    tp_plus_fc4_acc = _eval(tp_model, test_loader, device)
    print(f"[gt-retrain]   fc4 refit done in {t3 - t2:.1f}s; "
          f"TP+fc4-GT test_acc = {tp_plus_fc4_acc:.4f}  "
          f"(delta vs starting = {tp_plus_fc4_acc - starting_acc:+.4f})")

    # === Variant 3: fc4-only OLS on the starting (un-TP'd) model ===
    print(f"\n[gt-retrain] Variant 3: fc4-only OLS on STARTING model (no TP)...")
    fc4_only_model = copy.deepcopy(starting_model)
    cap_fc4_only = collect_activations(
        fc4_only_model, loader_list, [fc4_only_model.fc4],
        max_samples=args.max_samples, seq_subsample=args.max_samples,
        device=device,
    )
    X_only = cap_fc4_only[fc4_only_model.fc4]["a_in"].double()[:n_use]
    t4 = time.time()
    W_o, b_o = solve_ols_layer(
        X_only, gt_for_fc4, with_bias=True, ols_lambda=args.ols_lambda
    )
    with torch.no_grad():
        fc4_only_model.fc4.weight.copy_(W_o.to(
            device=fc4_only_model.fc4.weight.device, dtype=fc4_only_model.fc4.weight.dtype))
        fc4_only_model.fc4.bias.copy_(b_o.to(
            device=fc4_only_model.fc4.bias.device, dtype=fc4_only_model.fc4.bias.dtype))
    t5 = time.time()
    fc4_only_acc = _eval(fc4_only_model, test_loader, device)
    print(f"[gt-retrain]   fc4-only done in {t5 - t4:.1f}s; "
          f"fc4-only test_acc = {fc4_only_acc:.4f}  "
          f"(delta vs starting = {fc4_only_acc - starting_acc:+.4f})")

    print(f"\n[gt-retrain] === SUMMARY ===")
    print(f"  starting model acc:           {starting_acc:.4f}")
    print(f"  TP retrain (kfac+moment, GT): {tp_acc:.4f}  "
          f"(lift {tp_acc - starting_acc:+.4f})")
    print(f"  + fc4 refit against GT:       {tp_plus_fc4_acc:.4f}  "
          f"(lift {tp_plus_fc4_acc - starting_acc:+.4f})")
    print(f"  fc4-only on starting (no TP): {fc4_only_acc:.4f}  "
          f"(lift {fc4_only_acc - starting_acc:+.4f})")

    payload = {
        "task": "phase1_gt_retrain",
        "starting_acc": starting_acc,
        "noise_std": args.noise_std,
        "random_init": args.random_init,
        "gt_margin": args.gt_margin,
        "max_samples": args.max_samples,
        "n_iterations": args.n_iterations,
        "tp_gt_acc": tp_acc,
        "tp_gt_iter_accs": iter_accs,
        "tp_plus_fc4_gt_acc": tp_plus_fc4_acc,
        "fc4_only_acc": fc4_only_acc,
        "tp_per_layer_final": tp_info,
        "tp_per_layer_per_iter": iter_infos,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: None)
    print(f"\n[gt-retrain] saved results to {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
