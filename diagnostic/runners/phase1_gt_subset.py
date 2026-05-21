"""
diagnostic/runners/phase1_gt_subset.py
======================================

Subset OLS-retrain experiments on MnistMLP with GT one-hot logit targets.

Group A (trained start):
  A1: fc3 only
  A2: fc2 only
  A3: fc1 only
  A4: fc4 + fc2
  A5: fc4 + fc1
  (fc4 only was established at 0.9802 in phase1_gt_retrain.)

Group B (random-init start, seeded):
  B1: all 4 layers, iterated n_iterations times

Chain back-prop runs through ALL 4 layers using the starting model's weights
once (stable targets). OLS is applied only to the requested layers, in
shallowest-first order; between OLS updates the layer-X is re-captured by
running the current (partially-updated) model forward, so non-contiguous
retrained sets see correct X.

Run:
    python -m diagnostic.runners.phase1_gt_subset
    python -m diagnostic.runners.phase1_gt_subset --seed 0 --n-iterations 5
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
from diagnostic.runners.phase1_gt_retrain import (
    gt_target_retrain,
    make_gt_logit_target,
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


OUT_JSON = RESULTS_DIR / "phase1_gt_subset.json"

ALL_DNAMES = ["fc4", "fc3", "fc2", "fc1"]  # deepest-first


def run_subset_experiment(
    starting_model,
    target_layer_names,
    deepest_gt_targets,
    dataloader_list,
    *,
    method="kfac_a",
    correct_target_mean=True,
    correct_target_cov=True,
    eps=1e-4,
    sigma2=None,
    ols_lambda=1e-4,
    max_samples=16384,
    device=None,
):
    """Single-pass OLS retrain on a subset of layers, with chain back-prop
    through the full layer stack using the starting model's weights."""
    work_model = copy.deepcopy(starting_model)
    all_layers = [getattr(work_model, n) for n in ALL_DNAMES]

    # Initial full capture from the starting model
    captured = collect_activations(
        work_model, dataloader_list, all_layers,
        max_samples=max_samples, seq_subsample=max_samples, device=device,
    )

    n_captured = captured[all_layers[0]]["a_post"].shape[0]
    n_use = min(n_captured, deepest_gt_targets.shape[0])
    gt_target = deepest_gt_targets[:n_use].double()

    forward_priors = {}
    for layer in all_layers:
        a_in = captured[layer]["a_in"].double()[:n_use]
        forward_priors[layer] = _empirical_mean_cov(a_in)

    # Chain back-prop through ALL 4 layers using starting-model weights.
    # Builds a stable per-layer target dict; later OLS draws from it.
    targets_per_layer = {}
    cur_target = gt_target.clone()
    for step_idx, layer in enumerate(all_layers):
        cap = captured[layer]
        a_in_truth = cap["a_in"].double()[:n_use]
        a_pre_truth = cap["a_pre"].double()[:n_use]
        activation = cap["activation"]
        mu_a, Sigma_a = forward_priors[layer]

        targets_per_layer[layer] = cur_target.clone()

        W_old = layer.weight.detach().to(
            device=a_in_truth.device, dtype=a_in_truth.dtype)
        b_old = (layer.bias.detach().to(
            device=a_in_truth.device, dtype=a_in_truth.dtype)
            if layer.bias is not None else None)

        if method == "naive":
            a_hat = invert_layer(
                W_old, b_old, cur_target,
                a_pre_forward=a_pre_truth, activation=activation,
                method="naive", eps=eps)
        else:
            a_hat = invert_layer(
                W_old, b_old, cur_target,
                a_pre_forward=a_pre_truth, activation=activation,
                method="kfac_a", mu_a=mu_a, Sigma_a=Sigma_a, sigma2=sigma2)

        next_target = a_hat
        if correct_target_mean and step_idx + 1 < len(all_layers):
            tgt_mean = a_in_truth.mean(dim=0)
            if correct_target_cov:
                _, tgt_Sigma = _empirical_mean_cov(a_in_truth)
                next_target = _moment_match(a_hat, tgt_mean, tgt_Sigma)
            else:
                next_target = _shift_mean_to(a_hat, tgt_mean)
        cur_target = next_target

    # OLS only on the requested target layers, shallowest-first, with a fresh
    # capture of each layer's X from the (partially updated) work model.
    target_set = set(target_layer_names)
    info = {}
    for chain_idx in reversed(range(len(all_layers))):  # shallowest -> deepest
        layer_name = ALL_DNAMES[chain_idx]
        if layer_name not in target_set:
            continue
        layer = all_layers[chain_idx]

        re_cap = collect_activations(
            work_model, dataloader_list, [layer],
            max_samples=max_samples, seq_subsample=max_samples, device=device,
        )
        X = re_cap[layer]["a_in"].double()[:n_use]
        a_pre_forward = re_cap[layer]["a_pre"].double()[:n_use]
        activation_after = captured[layer]["activation"]
        target_a_post = targets_per_layer[layer]

        t_pre = invert_activation(target_a_post, a_pre_forward, activation_after)
        has_bias = layer.bias is not None
        W_new, b_new = solve_ols_layer(
            X, t_pre, with_bias=has_bias, ols_lambda=ols_lambda)

        with torch.no_grad():
            layer.weight.copy_(W_new.to(
                device=layer.weight.device, dtype=layer.weight.dtype))
            if has_bias and b_new is not None:
                layer.bias.copy_(b_new.to(
                    device=layer.bias.device, dtype=layer.bias.dtype))

            pred = X @ W_new.T
            if has_bias and b_new is not None:
                pred = pred + b_new.unsqueeze(0)
            residual = (pred - t_pre).norm() / t_pre.norm().clamp(min=1e-30)

        info[layer_name] = {
            "d_in": int(X.shape[1]),
            "d_out": int(t_pre.shape[1]),
            "target_residual": float(residual.item()),
            "n_samples": int(X.shape[0]),
        }

    return work_model, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-samples", type=int, default=16384)
    ap.add_argument("--retrain-data", choices=("test", "train"), default="train")
    ap.add_argument("--gt-margin", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--sigma2", type=float, default=None)
    ap.add_argument("--ols-lambda", type=float, default=1e-4)
    ap.add_argument("--method", choices=("naive", "kfac_a"), default="kfac_a")
    ap.add_argument("--no-mean-correct", action="store_true")
    ap.add_argument("--no-cov-correct", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed used for Group B's random init.")
    ap.add_argument("--n-iterations", type=int, default=5,
                    help="Iterations for Group B (random-init all-4 retrain).")
    ap.add_argument("--skip-group-a", action="store_true",
                    help="Skip the trained-start subset experiments (A1-A5).")
    ap.add_argument("--skip-group-b", action="store_true",
                    help="Skip the random-init iteration experiment (B1).")
    ap.add_argument("--skip-group-c", action="store_true",
                    help="Skip the random-init layer-subset experiments (C1-C4).")
    ap.add_argument("--resample-each-iter", action="store_true", default=True,
                    help="(Default ON) Draw a fresh random subset of the train "
                         "set for each Group-B iteration. Deterministic given "
                         "--seed.")
    ap.add_argument("--no-resample", action="store_true",
                    help="Disable per-iteration resampling; use a fixed subset "
                         "for all Group-B iterations.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[gt-subset] device = {device}  seed = {args.seed}  "
          f"max_samples = {args.max_samples}")

    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    if not WEIGHTS_PATH.exists():
        print(f"[gt-subset] no cached weights at {WEIGHTS_PATH}.")
        return 1
    trained_model = MnistMLP().to(device)
    trained_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    trained_acc = _eval(trained_model, test_loader, device)
    print(f"[gt-subset] trained baseline acc = {trained_acc:.4f}")

    # Build retrain dataloader
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

    def batch_iter():
        for x, _ in retrain_loader:
            yield x
    loader_list = list(batch_iter())

    gt_targets = make_gt_logit_target(labels, n_classes=10, margin=args.gt_margin)

    common_kwargs = dict(
        method=args.method,
        correct_target_mean=(not args.no_mean_correct),
        correct_target_cov=(not args.no_cov_correct),
        eps=args.eps,
        sigma2=args.sigma2,
        ols_lambda=args.ols_lambda,
        max_samples=args.max_samples,
        device=device,
    )

    results = {}

    # ----- Group A -----
    group_a_exps = [
        ("A1_fc3_only",    ["fc3"]),
        ("A2_fc2_only",    ["fc2"]),
        ("A3_fc1_only",    ["fc1"]),
        ("A4_fc4_and_fc2", ["fc4", "fc2"]),
        ("A5_fc4_and_fc1", ["fc4", "fc1"]),
    ]

    if args.skip_group_a:
        print(f"\n[gt-subset] === Group A: SKIPPED (--skip-group-a) ===")
        group_a_exps = []
    else:
        print(f"\n[gt-subset] === Group A: subset retrain on trained model ===")
    for name, target_layers in group_a_exps:
        t0 = time.time()
        retrained_model, info = run_subset_experiment(
            trained_model, target_layers, gt_targets, loader_list,
            **common_kwargs,
        )
        t1 = time.time()
        acc = _eval(retrained_model, test_loader, device)
        results[name] = {
            "target_layers": target_layers,
            "starting_acc": trained_acc,
            "retrained_acc": acc,
            "delta": acc - trained_acc,
            "wall_s": t1 - t0,
            "layer_info": info,
        }
        print(f"  {name:<20s} target={target_layers}  "
              f"acc={acc:.4f}  delta={acc - trained_acc:+.4f}  ({t1-t0:.1f}s)")
        for lname, li in info.items():
            print(f"    {lname}: d_in={li['d_in']}, d_out={li['d_out']}  "
                  f"residual={li['target_residual']:.3g}")

    # ----- Random model creation (shared by Groups B and C) -----
    need_random = not (args.skip_group_b and args.skip_group_c)
    random_model = None
    random_acc = None
    if need_random:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        random_model = MnistMLP().to(device)
        random_acc = _eval(random_model, test_loader, device)
        print(f"\n[gt-subset] random-init starting acc = {random_acc:.4f}  "
              f"(seed={args.seed})")
        print(f"[gt-subset]   test_loader uses {len(test_set)} HELD-OUT MNIST "
              f"test samples (never seen during retraining).")

    # ----- Group B -----
    if args.skip_group_b:
        print(f"\n[gt-subset] === Group B: SKIPPED (--skip-group-b) ===")
    else:
        resample = not args.no_resample
        print(f"\n[gt-subset] === Group B: random init, all 4 layers, "
              f"n_iter={args.n_iterations}, resample_per_iter={resample} ===")

        work_model = copy.deepcopy(random_model)
        chain_layers = [work_model.fc4, work_model.fc3, work_model.fc2, work_model.fc1]

        # Independent RNG for the iteration resampling — does NOT touch the
        # model-init seed sequence, so the random-init model is unchanged.
        sample_rng = torch.Generator()
        sample_rng.manual_seed(args.seed + 1)
        train_n = len(train_set)

        iter_accs = []
        iter_infos = []
        iter_sample_indices = []
        t0 = time.time()
        for iter_idx in range(args.n_iterations):
            if resample:
                perm = torch.randperm(train_n, generator=sample_rng)
                indices = perm[: args.max_samples].tolist()
                iter_subset = Subset(train_set, indices)
                iter_loader = DataLoader(iter_subset, batch_size=128,
                                         shuffle=False, num_workers=0)
                it_images = []
                it_labels = []
                for x, y in iter_loader:
                    it_images.append(x)
                    it_labels.append(y)
                it_labels = torch.cat(it_labels, dim=0)[: args.max_samples]
                it_gt_targets = make_gt_logit_target(
                    it_labels, n_classes=10, margin=args.gt_margin)
                iter_loader_list = it_images
                iter_sample_indices.append(indices[:32])
            else:
                iter_loader_list = loader_list
                it_gt_targets = gt_targets

            tp_info = gt_target_retrain(
                model=work_model,
                chain_layers=chain_layers,
                work_layers=chain_layers,
                deepest_gt_targets=it_gt_targets,
                dataloader_list=iter_loader_list,
                **common_kwargs,
            )
            iter_acc = _eval(work_model, test_loader, device)
            iter_accs.append(iter_acc)
            iter_infos.append(tp_info)
            print(f"  B1 iter {iter_idx + 1}: acc={iter_acc:.4f}  "
                  f"delta_vs_random={iter_acc - random_acc:+.4f}")
            for li in sorted(tp_info.keys()):
                info = tp_info[li]
                print(f"    layer {li}: d_in={info['d_in']}, d_out={info['d_out']}  "
                      f"residual={info['target_residual']:.3g}")
        t1 = time.time()
        results["B1_random_init_all_4_iter"] = {
            "starting_acc": random_acc,
            "iter_accs": iter_accs,
            "final_acc": iter_accs[-1],
            "delta_vs_random": iter_accs[-1] - random_acc,
            "wall_s": t1 - t0,
            "seed": args.seed,
            "n_iterations": args.n_iterations,
            "resample_per_iter": resample,
            "iter_sample_indices_head": iter_sample_indices,
            "per_iter_layer_info": iter_infos,
        }
        print(f"  B1 done in {t1 - t0:.1f}s; trajectory: "
              f"{' -> '.join(f'{a:.4f}' for a in iter_accs)}")

    # ----- Group C: random init, progressively more layers retrained -----
    group_c_exps = [
        ("C1_fc4_only",        ["fc4"]),
        ("C2_fc4_and_fc3",     ["fc4", "fc3"]),
        ("C3_fc4_fc3_fc2",     ["fc4", "fc3", "fc2"]),
        ("C4_all_four",        ["fc4", "fc3", "fc2", "fc1"]),
    ]
    if args.skip_group_c:
        print(f"\n[gt-subset] === Group C: SKIPPED (--skip-group-c) ===")
        group_c_exps = []
    else:
        print(f"\n[gt-subset] === Group C: random init, layer subsets, single pass ===")
    for name, target_layers in group_c_exps:
        t0c = time.time()
        retrained_model, info = run_subset_experiment(
            random_model, target_layers, gt_targets, loader_list,
            **common_kwargs,
        )
        t1c = time.time()
        acc = _eval(retrained_model, test_loader, device)
        results[name] = {
            "target_layers": target_layers,
            "starting_acc": random_acc,
            "retrained_acc": acc,
            "delta_vs_random": acc - random_acc,
            "wall_s": t1c - t0c,
            "layer_info": info,
        }
        print(f"  {name:<22s} target={target_layers}  "
              f"acc={acc:.4f}  delta_vs_random={acc - random_acc:+.4f}  ({t1c-t0c:.1f}s)")
        for lname, li in info.items():
            print(f"    {lname}: d_in={li['d_in']}, d_out={li['d_out']}  "
                  f"residual={li['target_residual']:.3g}")

    # ----- Summary -----
    print(f"\n[gt-subset] === SUMMARY ===")
    print(f"  trained baseline:     {trained_acc:.4f}")
    if random_acc is not None:
        print(f"  random-init starting: {random_acc:.4f}  (seed={args.seed})")
    if group_a_exps:
        print()
        print(f"  Group A (trained start, single pass):")
        for name, _ in group_a_exps:
            r = results[name]
            print(f"    {name:<22s} target={r['target_layers']!s:<22s} "
                  f"acc={r['retrained_acc']:.4f}  delta={r['delta']:+.4f}")
    if "B1_random_init_all_4_iter" in results:
        print()
        print(f"  Group B (random init, all 4, {args.n_iterations} iter):")
        r = results["B1_random_init_all_4_iter"]
        print(f"    trajectory: {' -> '.join(f'{a:.4f}' for a in r['iter_accs'])}")
        print(f"    final acc:  {r['final_acc']:.4f}  "
              f"delta_vs_random={r['delta_vs_random']:+.4f}")
    if group_c_exps:
        print()
        print(f"  Group C (random init, layer subsets, single pass):")
        for name, _ in group_c_exps:
            r = results[name]
            print(f"    {name:<22s} target={r['target_layers']!s:<28s} "
                  f"acc={r['retrained_acc']:.4f}  "
                  f"delta_vs_random={r['delta_vs_random']:+.4f}")

    payload = {
        "task": "phase1_gt_subset",
        "trained_acc": trained_acc,
        "random_init_starting_acc": random_acc,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "gt_margin": args.gt_margin,
        "method": args.method,
        "correct_target_mean": not args.no_mean_correct,
        "correct_target_cov": not args.no_cov_correct,
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: None)
    print(f"\n[gt-subset] saved results to {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
