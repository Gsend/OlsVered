"""
diagnostic/runners/phase1_final_layer_refit.py
==============================================

Stack a final-layer (fc4) OLS refit on top of the best target-prop retraining
result, to fix the chain-noise distortion at the classifier head.

The chain back-prop retrains fc1-fc4. Earlier layers (fc1, fc2, fc3) get
noisy targets and end up slightly different from the original; this means
the retrained model's intermediate representation feeding into fc4 has
shifted relative to what the original fc4 was trained on. A single OLS
solve on fc4 — using the retrained model's actual a_in_to_fc4 and the
ORIGINAL model's fc4 output as the target — adjusts fc4 to map the new
intermediate representation back to the desired output.

This is "frozen earlier layers + final-layer OLS refit", which is exactly
the linear probing pattern that's known to work well for adapting deep
representations.

Run:
    python -m diagnostic.runners.phase1_final_layer_refit
    python -m diagnostic.runners.phase1_final_layer_refit --retrain-data train --max-samples 16384
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from diagnostic.capture import collect_activations
from diagnostic.runners.phase1_mlp_mnist import (
    MnistMLP,
    REPO_ROOT,
    RESULTS_DIR,
    WEIGHTS_PATH,
    _eval,
    _load_mnist,
)
from diagnostic.target_prop_retrainer import (
    retrain_via_target_prop,
    solve_ols_layer,
)


OUT_JSON = RESULTS_DIR / "phase1_final_layer_refit.json"


def final_layer_refit(
    work_model,
    original_model,
    final_layer_name,
    dataloader_list,
    *,
    device,
    ols_lambda=1e-4,
    max_samples=4096,
):
    """OLS-refit a single layer of work_model so it maps work_model's
    current input activations to the ORIGINAL model's output at that layer.

    Steps:
      1. Capture a_in_to_final from work_model's forward pass.
      2. Capture a_in_to_final from original model + compute its target output.
      3. Solve OLS: W, b s.t. work_model.X @ W^T + b ≈ original.output
      4. Copy new W, b into work_model's layer.

    Same data is used for both captures, so sample j aligns between work
    and original captures.
    """
    final_layer_work = getattr(work_model, final_layer_name)
    final_layer_orig = getattr(original_model, final_layer_name)

    # Capture work_model's input to fc4 (changed because fc1-fc3 are retrained)
    cap_work = collect_activations(
        work_model, dataloader_list, [final_layer_work],
        max_samples=max_samples, seq_subsample=max_samples,
        device=device,
    )
    # Capture original model's pre-activation output of fc4 (the target)
    cap_orig = collect_activations(
        original_model, dataloader_list, [final_layer_orig],
        max_samples=max_samples, seq_subsample=max_samples,
        device=device,
    )

    X_work = cap_work[final_layer_work]["a_in"].double()
    # a_pre is the pre-activation output (= logits since fc4 has no following activation)
    target = cap_orig[final_layer_orig]["a_pre"].double()

    # Truncate to a common length in case captures landed at different sample counts
    n = min(X_work.shape[0], target.shape[0])
    X_work = X_work[:n]
    target = target[:n]

    print(f"  [refit] X_work shape={tuple(X_work.shape)}  "
          f"target shape={tuple(target.shape)}")

    W_new, b_new = solve_ols_layer(
        X_work, target, with_bias=(final_layer_work.bias is not None),
        ols_lambda=ols_lambda,
    )

    target_device = final_layer_work.weight.device
    target_dtype = final_layer_work.weight.dtype
    with torch.no_grad():
        old_W = final_layer_work.weight.detach().clone()
        new_W = W_new.to(device=target_device, dtype=target_dtype)
        final_layer_work.weight.copy_(new_W)
        w_change = (new_W - old_W).norm().item()

        b_change = 0.0
        if final_layer_work.bias is not None and b_new is not None:
            old_b = final_layer_work.bias.detach().clone()
            new_b = b_new.to(device=target_device, dtype=target_dtype)
            final_layer_work.bias.copy_(new_b)
            b_change = (new_b - old_b).norm().item()

        # Residual
        pred = X_work @ W_new.T
        if final_layer_work.bias is not None and b_new is not None:
            pred = pred + b_new.unsqueeze(0)
        residual = (pred - target).norm() / target.norm().clamp(min=1e-30)

    return {
        "weight_delta": w_change,
        "bias_delta": b_change,
        "target_residual": float(residual.item()),
        "n_samples_used": int(n),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-train", action="store_true", default=True,
                    help="Use cached weights.")
    ap.add_argument("--max-samples", type=int, default=4096,
                    help="Activations captured per layer.")
    ap.add_argument("--retrain-data", choices=("test", "train"), default="test")
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
    print(f"[refit] device = {device}")
    print(f"[refit] retrain_data={args.retrain_data}  max_samples={args.max_samples}")

    train_set, test_set = _load_mnist(Path(args.data_root))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    # Load the original model
    original_model = MnistMLP().to(device)
    if not WEIGHTS_PATH.exists():
        print(f"[refit] no cached weights at {WEIGHTS_PATH}. Run phase1_mlp_mnist.py first.")
        return 1
    original_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    baseline_acc = _eval(original_model, test_loader, device)
    print(f"[refit] baseline (original model) test_acc = {baseline_acc:.4f}")

    # Build retrain dataloader
    if args.retrain_data == "train":
        subset_size = min(len(train_set), max(args.max_samples * 4, 4096))
        retrain_subset = Subset(train_set, list(range(subset_size)))
    else:
        subset_size = min(len(test_set), max(args.max_samples * 4, 4096))
        retrain_subset = Subset(test_set, list(range(subset_size)))
    retrain_loader = DataLoader(retrain_subset, batch_size=128,
                                shuffle=False, num_workers=0)

    def batch_iter():
        for x, _ in retrain_loader:
            yield x

    loader_list = list(batch_iter())

    # Run the best TP-retrainer config: kfac_a + moment_match
    print(f"\n[refit] running TP retrain (kfac + moment match, single-pass)...")
    chain_layers = [original_model.fc4, original_model.fc3,
                    original_model.fc2, original_model.fc1]
    t0 = time.time()
    tp_result = retrain_via_target_prop(
        model=original_model,
        layers_to_retrain_deepest_first=chain_layers,
        dataloader=loader_list,
        method="kfac_a",
        correct_target_mean=True,
        correct_target_cov=True,
        eps=args.eps,
        sigma2=args.sigma2,
        ols_lambda=args.ols_lambda,
        max_samples=args.max_samples,
        seq_subsample=args.max_samples,
        device=device,
        return_copy=True,
        n_iterations=1,  # single-pass (multi-pass diverges)
    )
    tp_time = time.time() - t0
    tp_model = tp_result.model
    tp_acc = _eval(tp_model, test_loader, device)
    print(f"[refit] TP-retrain done in {tp_time:.1f}s; tp_acc = {tp_acc:.4f}  "
          f"(delta vs baseline = {tp_acc - baseline_acc:+.4f})")
    for layer_idx, info in tp_result.per_layer_info.items():
        print(f"[refit]   TP layer {layer_idx}: "
              f"d_in={info['d_in']}, d_out={info['d_out']}  "
              f"weight_delta={info['weight_change_frob']:.3g}  "
              f"target_residual={info['target_residual']:.3g}")

    # === Final-layer refit on fc4 ===
    print(f"\n[refit] applying final-layer OLS refit on fc4...")
    t1 = time.time()
    refit_info = final_layer_refit(
        work_model=tp_model,
        original_model=original_model,
        final_layer_name="fc4",
        dataloader_list=loader_list,
        device=device,
        ols_lambda=args.ols_lambda,
        max_samples=args.max_samples,
    )
    refit_time = time.time() - t1
    refit_acc = _eval(tp_model, test_loader, device)
    print(f"[refit] fc4 refit done in {refit_time:.1f}s; "
          f"refit_acc = {refit_acc:.4f}  "
          f"(delta vs baseline = {refit_acc - baseline_acc:+.4f})")

    # === Forward-sweep refit: fc1 -> fc2 -> fc3 -> fc4 ===
    # Each layer is refit to match the ORIGINAL model's output at that layer,
    # using work_model's CURRENT input (which reflects earlier-layer refits).
    print(f"\n[refit] applying forward-sweep OLS refit (fc1 -> fc2 -> fc3 -> fc4)...")
    t2 = time.time()
    layer_names_forward = ["fc1", "fc2", "fc3", "fc4"]
    sweep_info = {}
    for lname in layer_names_forward:
        info = final_layer_refit(
            work_model=tp_model,
            original_model=original_model,
            final_layer_name=lname,
            dataloader_list=loader_list,
            device=device,
            ols_lambda=args.ols_lambda,
            max_samples=args.max_samples,
        )
        intermediate_acc = _eval(tp_model, test_loader, device)
        print(f"  [refit] {lname}: target_residual={info['target_residual']:.3g}  "
              f"weight_delta={info['weight_delta']:.3g}  "
              f"acc_after={intermediate_acc:.4f}")
        sweep_info[lname] = {**info, "acc_after": intermediate_acc}
    sweep_time = time.time() - t2
    sweep_acc = _eval(tp_model, test_loader, device)
    print(f"[refit] forward-sweep done in {sweep_time:.1f}s; "
          f"sweep_acc = {sweep_acc:.4f}  "
          f"(delta vs baseline = {sweep_acc - baseline_acc:+.4f})")
    print(f"[refit]   fc4 target_residual = {refit_info['target_residual']:.3g}")

    print(f"\n[refit] === SUMMARY ===")
    print(f"  baseline (original):     {baseline_acc:.4f}")
    print(f"  TP retrain (kfac+mom):   {tp_acc:.4f}  (delta {tp_acc - baseline_acc:+.4f})")
    print(f"  + fc4 refit only:        {refit_acc:.4f}  (delta {refit_acc - baseline_acc:+.4f})")
    print(f"  + forward sweep all:     {sweep_acc:.4f}  (delta {sweep_acc - baseline_acc:+.4f})")
    print(f"  fc4-refit lift over TP:  {refit_acc - tp_acc:+.4f}")
    print(f"  sweep lift over fc4:     {sweep_acc - refit_acc:+.4f}")

    payload = {
        "task": "phase1_final_layer_refit",
        "baseline_acc": baseline_acc,
        "tp_retrain_acc": tp_acc,
        "tp_plus_fc4refit_acc": refit_acc,
        "tp_plus_sweep_acc": sweep_acc,
        "max_samples": args.max_samples,
        "retrain_data": args.retrain_data,
        "tp_per_layer": tp_result.per_layer_info,
        "fc4_refit_info": refit_info,
        "sweep_info": sweep_info,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: None)
    print(f"\n[refit] saved results to {OUT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
