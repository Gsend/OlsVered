"""
diagnostic/runners/phase1_retrain_via_targets.py
=================================================

Phase 1 retraining experiment using target-prop + per-layer OLS.

Compares accuracy of the retrained MLP under 6 configurations:

  1. naive (no corrections) -> OLS
  2. naive + mean only       -> OLS
  3. naive + moment match    -> OLS
  4. kfac_a (no corrections) -> OLS
  5. kfac_a + mean only      -> OLS
  6. kfac_a + moment match   -> OLS

Options:

  --retrain-data {test|train}   data source for retraining capture
  --max-samples N               number of samples per layer (up to 10k for
                                test, 60k for train)
  --n-iterations N              multi-pass iteration count (1 = single pass,
                                k>1 = BCD-style; only affects K when prior
                                stats change between iters, but applied
                                uniformly)

Usage:
    python -m diagnostic.runners.phase1_retrain_via_targets --no-train
    python -m diagnostic.runners.phase1_retrain_via_targets --no-train --max-samples 8192
    python -m diagnostic.runners.phase1_retrain_via_targets --no-train --retrain-data train --max-samples 32768
    python -m diagnostic.runners.phase1_retrain_via_targets --no-train --n-iterations 5
    python -m diagnostic.runners.phase1_retrain_via_targets --no-train --retrain-data train --max-samples 16384 --n-iterations 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from diagnostic.runners.phase1_mlp_mnist import (
    MnistMLP,
    REPO_ROOT,
    RESULTS_DIR,
    WEIGHTS_PATH,
    _eval,
    _load_mnist,
    _train,
)
from diagnostic.target_prop_retrainer import retrain_via_target_prop


OUT_JSON = RESULTS_DIR / "phase1_retrain_via_targets.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-train", action="store_true",
                    help="Skip training; require cached weights.")
    ap.add_argument("--max-samples", type=int, default=4096,
                    help="Activations captured per layer for OLS solve. "
                         "Up to 10k for test, 60k for train.")
    ap.add_argument("--retrain-data", choices=("test", "train"), default="test",
                    help="Data source for retraining capture. 'train' gives "
                         "access to 60k samples; 'test' caps at 10k.")
    ap.add_argument("--n-iterations", type=int, default=1,
                    help="Multi-pass iteration count for the retrainer. "
                         "Iter k>1 re-captures from updated model.")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--sigma2", type=float, default=None)
    ap.add_argument("--ols-lambda", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[retrain] device = {device}")
    print(f"[retrain] retrain_data={args.retrain_data}  "
          f"max_samples={args.max_samples}  n_iterations={args.n_iterations}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[retrain] loading MNIST...")
    train_set, test_set = _load_mnist(Path(args.data_root))
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512,
                             shuffle=False, num_workers=0)

    model = MnistMLP().to(device)

    if args.no_train:
        if not WEIGHTS_PATH.exists():
            print(f"[retrain] no cached weights at {WEIGHTS_PATH}. "
                  "Run phase1_mlp_mnist.py first or omit --no-train.")
            return 1
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        baseline_acc = _eval(model, test_loader, device)
        print(f"[retrain] loaded weights; baseline test_acc = {baseline_acc:.4f}")
    else:
        print(f"[retrain] training {args.epochs} epochs...")
        baseline_acc = _train(model, train_loader, test_loader,
                              epochs=args.epochs, device=device, lr=args.lr)
        torch.save(model.state_dict(), WEIGHTS_PATH)

    # Build retraining dataloader from chosen source
    if args.retrain_data == "train":
        # Cap subset size at max_samples * 4 (to allow capture's seq_subsample
        # to subsample within); MNIST train has 60k samples
        subset_size = min(len(train_set), max(args.max_samples * 4, 4096))
        retrain_subset = Subset(train_set, list(range(subset_size)))
        print(f"[retrain] using {subset_size} TRAIN samples for retraining")
    else:
        subset_size = min(len(test_set), max(args.max_samples * 4, 4096))
        retrain_subset = Subset(test_set, list(range(subset_size)))
        print(f"[retrain] using {subset_size} TEST samples for retraining")
    retrain_loader = DataLoader(retrain_subset, batch_size=128,
                                shuffle=False, num_workers=0)

    def batch_iter():
        for x, _ in retrain_loader:
            yield x

    layers_descending_names = ["fc4", "fc3", "fc2", "fc1"]

    # Eval callback used per iteration for accuracy tracking
    def make_eval_fn(use_test_loader):
        def _ev(m):
            return _eval(m, use_test_loader, device)
        return _ev
    eval_fn = make_eval_fn(test_loader)

    configs = [
        ("naive_no_correction",   "naive",  False, False),
        ("naive_mean_only",       "naive",  True,  False),
        ("naive_moment_match",    "naive",  True,  True),
        ("kfac_no_correction",    "kfac_a", False, False),
        ("kfac_mean_only",        "kfac_a", True,  False),
        ("kfac_moment_match",     "kfac_a", True,  True),
    ]

    results = []
    for cfg_name, method, ct_mean, ct_cov in configs:
        # Materialize dataloader so we can re-iterate across iterations
        loader_iter = list(batch_iter())
        layers = [getattr(model, n) for n in layers_descending_names]

        print(f"\n[retrain] === config: {cfg_name} ===")
        t0 = time.time()
        result = retrain_via_target_prop(
            model=model,
            layers_to_retrain_deepest_first=layers,
            dataloader=loader_iter,
            method=method,
            correct_target_mean=ct_mean,
            correct_target_cov=ct_cov,
            eps=args.eps,
            sigma2=args.sigma2,
            ols_lambda=args.ols_lambda,
            max_samples=args.max_samples,
            seq_subsample=args.max_samples,
            device=device,
            return_copy=True,
            n_iterations=args.n_iterations,
            eval_fn=eval_fn,
        )
        t1 = time.time()
        retrained_acc = _eval(result.model, test_loader, device)

        # Trajectory of accuracy across iterations
        iter_accs = [h.get("accuracy") for h in result.iteration_history]
        traj_str = " -> ".join(f"{a:.4f}" if a is not None else "n/a"
                               for a in iter_accs)
        print(f"[retrain]   done in {t1 - t0:.1f}s; "
              f"final retrained_acc = {retrained_acc:.4f}  "
              f"(delta = {retrained_acc - baseline_acc:+.4f})")
        print(f"[retrain]   accuracy trajectory: {traj_str}")
        for layer_idx, info in result.per_layer_info.items():
            print(f"[retrain]   layer {layer_idx} (final): "
                  f"d_in={info['d_in']}, d_out={info['d_out']}  "
                  f"weight_delta={info['weight_change_frob']:.3g}  "
                  f"target_residual={info['target_residual']:.3g}")

        results.append({
            "config": cfg_name,
            "method": method,
            "correct_target_mean": ct_mean,
            "correct_target_cov": ct_cov,
            "baseline_acc": baseline_acc,
            "retrained_acc": retrained_acc,
            "acc_delta": retrained_acc - baseline_acc,
            "wall_s": t1 - t0,
            "iter_accuracies": iter_accs,
            "final_per_layer": result.per_layer_info,
        })

    payload = {
        "task": "phase1_retrain_via_targets",
        "baseline_acc": baseline_acc,
        "max_samples": args.max_samples,
        "retrain_data": args.retrain_data,
        "n_iterations": args.n_iterations,
        "ols_lambda": args.ols_lambda,
        "configs": results,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: None)
    print(f"\n[retrain] saved results to {OUT_JSON}")

    print(f"\n[retrain] ===== ACCURACY SUMMARY =====")
    print(f"  baseline test_acc = {baseline_acc:.4f}")
    print(f"  retrain_data = {args.retrain_data}  "
          f"max_samples = {args.max_samples}  "
          f"n_iterations = {args.n_iterations}")
    print(f"  {'config':<24s} {'final_acc':>10s} {'delta':>10s}  {'wall_s':>8s}")
    for r in results:
        print(f"  {r['config']:<24s} "
              f"{r['retrained_acc']:>10.4f} "
              f"{r['acc_delta']:>+10.4f}  "
              f"{r['wall_s']:>8.1f}")
    print()

    if args.n_iterations > 1:
        print(f"[retrain] ===== ACCURACY TRAJECTORY (per iteration) =====")
        for r in results:
            accs = r['iter_accuracies']
            traj = " -> ".join(f"{a:.4f}" if a is not None else "n/a" for a in accs)
            print(f"  {r['config']:<24s} {traj}")
        print()

    best = max(results, key=lambda r: r['retrained_acc'])
    print(f"[retrain] best variant: {best['config']} "
          f"(acc = {best['retrained_acc']:.4f}, "
          f"delta vs baseline = {best['acc_delta']:+.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
