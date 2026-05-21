"""
diagnostic/runners/phase1_mlp_mnist.py
======================================

Phase 1 of the inversion-drift diagnostic - MLP on MNIST.

Trains a 4-layer MLP (784 -> 512 -> 256 -> 128 -> 10) on MNIST, runs the
single-layer drift experiment AND the multi-step back-target chain (the
real H3 test), saves JSON + plots.

Outputs:
  benchmark/results/diagnostic/phase1_mlp_mnist.json
  benchmark/results/diagnostic/phase1_mlp_mnist_chain.json
  benchmark/results/diagnostic/phase1_plots/

Usage:
    python -m diagnostic.runners.phase1_mlp_mnist
    python -m diagnostic.runners.phase1_mlp_mnist --no-train
    python -m diagnostic.runners.phase1_mlp_mnist --chain-depth 4
    python -m diagnostic.runners.phase1_mlp_mnist --no-train --sigma2 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diagnostic.experiment import (
    LayerDriftReport,
    run_drift_experiment,
    save_reports,
)
from diagnostic.multi_step import ChainStepReport, run_multi_step_chain
from diagnostic.plotting import generate_default_plots


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "diagnostic"
WEIGHTS_PATH = RESULTS_DIR / "mnist_mlp.pt"
JSON_OUT = RESULTS_DIR / "phase1_mlp_mnist.json"
CHAIN_JSON_OUT = RESULTS_DIR / "phase1_mlp_mnist_chain.json"
PLOTS_DIR = RESULTS_DIR / "phase1_plots"


class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.fc4(x)


def _load_mnist(data_root: Path):
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        sys.stderr.write(
            "torchvision is required for Phase 1. Install via:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install torchvision\n"
        )
        sys.exit(1)
    tform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(
        root=str(data_root), train=True, download=True, transform=tform)
    test = torchvision.datasets.MNIST(
        root=str(data_root), train=False, download=True, transform=tform)
    return train, test


def _train(model, train_loader, test_loader, *, epochs, device, lr=1e-3):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
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
        acc = _eval(model, test_loader, device)
        print(f"  epoch {epoch + 1:2d}/{epochs}: "
              f"train_loss={running/seen:.4f}  test_acc={acc:.4f}")
    print(f"  total train time: {time.time() - t0:.1f}s")
    return _eval(model, test_loader, device)


def _eval(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _build_remaining_forward_fns(model: MnistMLP):
    """Build a dict mapping each fc layer to a function that runs the model
    forward starting at that layer's input."""

    def remaining_from(layer_input_into: nn.Linear):
        layers_in_order = [
            (model.fc1, model.act1),
            (model.fc2, model.act2),
            (model.fc3, model.act3),
            (model.fc4, None),
        ]
        start = next(i for i, (lin, _) in enumerate(layers_in_order)
                     if lin is layer_input_into)
        target_device = layer_input_into.weight.device
        target_dtype = layer_input_into.weight.dtype

        def fn(a):
            x = a.to(device=target_device, dtype=target_dtype)
            for i in range(start, len(layers_in_order)):
                lin, act = layers_in_order[i]
                x = lin(x)
                if act is not None:
                    x = act(x)
            return x
        return fn

    return {
        model.fc1: remaining_from(model.fc1),
        model.fc2: remaining_from(model.fc2),
        model.fc3: remaining_from(model.fc3),
        model.fc4: remaining_from(model.fc4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--max-samples", type=int, default=4096)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--sigma2", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--no-functional", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--chain-depth", type=int, default=4,
                    help="Depth for multi-step back-target chain (0 to skip).")
    ap.add_argument("--correct-target-mean", action="store_true",
                    help="Shift each chained target's mean to match the "
                         "forward-pass mean before the next inversion.")
    ap.add_argument("--correct-target-cov", action="store_true",
                    help="Apply full moment matching (mean + covariance "
                         "whitening/recoloring) to each chained target. "
                         "Requires --correct-target-mean.")
    ap.add_argument("--chain-passes", type=int, default=1,
                    help="Number of chain passes for self-consistent prior. "
                         "1 = forward-pass prior throughout. 2+ = use each "
                         "pass's recovered output statistics as the next "
                         "pass's prior. Only affects Method K.")
    args = ap.parse_args(argv)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[phase1] device = {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[phase1] loading MNIST...")
    train_set, test_set = _load_mnist(Path(args.data_root))
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512,
                             shuffle=False, num_workers=0)

    model = MnistMLP().to(device)

    if args.no_train:
        if not WEIGHTS_PATH.exists():
            print(f"[phase1] no cached weights at {WEIGHTS_PATH}")
            return 1
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        acc = _eval(model, test_loader, device)
        print(f"[phase1] loaded weights; test_acc = {acc:.4f}")
    else:
        print(f"[phase1] training {args.epochs} epochs...")
        acc = _train(model, train_loader, test_loader,
                     epochs=args.epochs, device=device, lr=args.lr)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"[phase1] cached weights; test_acc = {acc:.4f}")

    diag_subset = Subset(test_set, list(range(min(len(test_set),
                                                  args.max_samples * 4))))
    diag_loader = DataLoader(diag_subset, batch_size=128,
                             shuffle=False, num_workers=0)

    def batch_iter():
        for x, _ in diag_loader:
            yield x

    layers = [model.fc1, model.fc2, model.fc3, model.fc4]
    layer_labels = [
        "fc1 (784->512)", "fc2 (512->256)",
        "fc3 (256->128)", "fc4 (128->10)",
    ]
    rem_fns = None if args.no_functional else _build_remaining_forward_fns(model)

    print(f"[phase1] running single-layer diagnostic on {len(layers)} layers...")
    t0 = time.time()
    reports = run_drift_experiment(
        model=model, layers=layers, dataloader=batch_iter(),
        methods=("naive", "kfac_a"),
        eps=args.eps, sigma2=args.sigma2,
        max_samples=args.max_samples, seq_subsample=args.max_samples,
        device=device, layer_labels=layer_labels,
        remaining_forward_fns=rem_fns, use_predicted_cov_check=True,
    )
    print(f"[phase1] single-layer done in {time.time() - t0:.1f}s")

    save_reports(reports, JSON_OUT,
                 model_name="MnistMLP", task="mnist",
                 extra={"phase": 1, "test_accuracy": acc,
                        "device": str(device),
                        "args": {k: (str(v) if isinstance(v, Path) else v)
                                 for k, v in vars(args).items()}})
    print(f"[phase1] saved single-layer reports to {JSON_OUT}")

    if not args.no_plots:
        produced = generate_default_plots(reports, PLOTS_DIR)
        print(f"[phase1] {len(produced)} plots in {PLOTS_DIR}")

    # ---- Single-layer summary ----
    print("\n[phase1] ===== SINGLE-LAYER SUMMARY =====")
    print(f"{'layer':<22s} {'method':<8s} {'cov_frob':>10s} "
          f"{'mean_drift':>12s} {'gauss_kl':>10s} {'logit_mse':>11s}")
    for r in reports:
        lm = (f"{r.logit_mse:>11.4g}" if r.logit_mse is not None
              else f"{'--':>11s}")
        print(f"{r.layer_label:<22s} {r.method:<8s} "
              f"{r.cov_frob:>10.4g} {r.mean_drift:>12.4g} "
              f"{r.gauss_kl_sym:>10.4g} {lm}")
    print()

    by_layer = {}
    for r in reports:
        by_layer.setdefault(r.layer_label, {})[r.method] = r
    k_wins_cov = sum(1 for m in by_layer.values()
                     if "naive" in m and "kfac_a" in m
                     and m["kfac_a"].cov_frob < m["naive"].cov_frob)
    n_cmp = sum(1 for m in by_layer.values()
                if "naive" in m and "kfac_a" in m)
    print(f"[phase1] H2 distributional: K beats N on cov_frob in {k_wins_cov}/{n_cmp}")

    k_wins_log = 0
    n_log = 0
    for m in by_layer.values():
        if ("naive" in m and "kfac_a" in m
                and m["naive"].logit_mse is not None
                and m["kfac_a"].logit_mse is not None):
            n_log += 1
            if m["kfac_a"].logit_mse < m["naive"].logit_mse:
                k_wins_log += 1
    if n_log:
        print(f"[phase1] H3 functional:     K beats N on logit_mse in {k_wins_log}/{n_log}")
    else:
        print("[phase1] H3 functional:     n/a")

    # ---- Multi-step chain ----
    if args.chain_depth > 0:
        deepest_first = [model.fc4, model.fc3, model.fc2, model.fc1]
        chain_labels_all = [
            "step0: fc4 (GT target)",
            "step1: fc3 (chained)",
            "step2: fc2 (chained)",
            "step3: fc1 (chained)",
        ]
        depth = min(args.chain_depth, len(deepest_first))
        chain_layers = deepest_first[:depth]
        chain_labels = chain_labels_all[:depth]

        print(f"\n[phase1] running multi-step chain (depth={depth})...")
        t1 = time.time()

        def chain_batch_iter():
            for x, _ in diag_loader:
                yield x

        chain_reports = run_multi_step_chain(
            model=model, layers_descending=chain_layers,
            dataloader=chain_batch_iter(),
            methods=("naive", "kfac_a"),
            eps=args.eps, sigma2=args.sigma2,
            max_samples=args.max_samples, seq_subsample=args.max_samples,
            device=device, layer_labels=chain_labels,
            correct_target_mean=args.correct_target_mean,
            correct_target_cov=args.correct_target_cov,
            n_passes=args.chain_passes,
        )
        print(f"[phase1] chain done in {time.time() - t1:.1f}s")

        payload = {
            "model": "MnistMLP", "task": "mnist_chain",
            "chain_depth": depth,
            "reports": [r.to_dict() for r in chain_reports],
        }
        with open(CHAIN_JSON_OUT, "w") as fh:
            json.dump(payload, fh, indent=2, default=lambda o: None)
        print(f"[phase1] saved chain reports to {CHAIN_JSON_OUT}")

        print("\n[phase1] ===== MULTI-STEP CHAIN SUMMARY =====")
        print(f"{'step':<6s} {'layer':<28s} {'method':<8s} "
              f"{'cov_frob':>10s} {'mean_drift':>12s} {'cov_growth':>11s}")
        for r in chain_reports:
            cg = (f"{r.cov_frob_growth:>11.3g}"
                  if r.cov_frob_growth is not None else f"{'--':>11s}")
            print(f"step{r.chain_idx:<2d} {r.layer_label[:28]:<28s} "
                  f"{r.method:<8s} {r.cov_frob:>10.4g} "
                  f"{r.mean_drift:>12.4g} {cg}")
        print()

        deepest_step = depth - 1
        try:
            last_n = next(r for r in chain_reports
                          if r.chain_idx == deepest_step and r.method == "naive")
            last_k = next(r for r in chain_reports
                          if r.chain_idx == deepest_step and r.method == "kfac_a")
            ratio = (max(last_n.cov_frob, last_k.cov_frob)
                     / max(min(last_n.cov_frob, last_k.cov_frob), 1e-30))
            winner = "wins" if last_k.cov_frob < last_n.cov_frob else "loses"
            print(f"[phase1] H3-real (chain): after {depth} steps, final cov_frob "
                  f"naive={last_n.cov_frob:.4g}, kfac={last_k.cov_frob:.4g}  "
                  f"(K {winner} by {ratio:.1f}x)")
        except StopIteration:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
