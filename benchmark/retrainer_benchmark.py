"""
OlsSMLayerRetrainer Benchmark  —  all modes vs LoRA on a real HuggingFace model.

Compares every retraining mode on the same pre-trained model and data budget:

  Baselines:
    adam          — Full AdamW fine-tuning (all parameters), 1 epoch
    adam_head     — AdamW on classifier head only (frozen backbone), 1 epoch
    lora_r4       — LoRA rank-4 (PEFT or manual), 1 epoch AdamW on adapters
    lora_r8       — LoRA rank-8, same

  OLS retrainer (one data pass, exact closed-form solution):
    ols_n1        — Last 1 Linear layer via OLS
    ols_n2        — Last 2 Linear layers via BCD-OLS
    ols_n4        — Last 4 Linear layers via BCD-OLS
    ols_n8        — Last 8 Linear layers via BCD-OLS
    ols_all       — ALL Linear layers via BCD-OLS

  OLS + LoRA (OLS bulk correction, then LoRA on residual):
    ols_lora_r2   — OLS(n=2) + LoRA residual rank-2
    ols_lora_r4   — OLS(n=2) + LoRA residual rank-4
    ols_lora_r8   — OLS(n=2) + LoRA residual rank-8

Default model:  bert-base-uncased  (110M params)
Default task:   SST-2 sentiment classification (GLUE)

Usage:
  python benchmark/retrainer_benchmark.py
  python benchmark/retrainer_benchmark.py --model distilbert-base-uncased --modes ols_n1,lora_r4,ols_lora_r4
  python benchmark/retrainer_benchmark.py --task sst2 --batch-size 32 --modes all
  python benchmark/retrainer_benchmark.py --no-plots
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from optimizer.layer_retrainer import OlsSMLayerRetrainer

# ---------------------------------------------------------------------------
# All available modes
# ---------------------------------------------------------------------------

ALL_MODES = [
    "adam",
    "adam_head",
    "lora_r4",
    "lora_r8",
    "ols_n1",
    "ols_n2",
    "ols_n4",
    "ols_n8",
    "ols_all",
    "ols_lora_r2",
    "ols_lora_r4",
    "ols_lora_r8",
    "als_lora_n1_r4",
    "als_lora_n2_r4",
    "als_lora_n4_r4",
]

MODE_LABELS = {
    "adam":           "AdamW (full)",
    "adam_head":      "AdamW (head only)",
    "lora_r4":        "LoRA r=4",
    "lora_r8":        "LoRA r=8",
    "ols_n1":         "OLS  N=1",
    "ols_n2":         "OLS  N=2 (BCD)",
    "ols_n4":         "OLS  N=4 (BCD)",
    "ols_n8":         "OLS  N=8 (BCD)",
    "ols_all":        "OLS  all layers",
    "ols_lora_r2":    "OLS N=2 + LoRA r=2",
    "ols_lora_r4":    "OLS N=2 + LoRA r=4",
    "ols_lora_r8":    "OLS N=2 + LoRA r=8",
    "als_lora_n1_r4": "ALS-LoRA N=1 r=4",
    "als_lora_n2_r4": "ALS-LoRA N=2 r=4",
    "als_lora_n4_r4": "ALS-LoRA N=4 r=4",
}

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ModeResult:
    mode: str
    label: str
    accuracy: float = 0.0          # eval accuracy (0-1)
    loss: float = float("inf")     # eval cross-entropy loss
    wall_s: float = 0.0            # wall-clock time in seconds
    peak_mem_gb: float = 0.0       # peak GPU memory
    n_trainable: int = 0           # trainable parameter count
    n_total: int = 0               # total parameter count
    n_data_passes: float = 0.0     # effective passes over training data
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None    # set if mode failed

    def to_dict(self) -> dict:
        d = {
            "mode": self.mode, "label": self.label,
            "accuracy": round(self.accuracy, 4),
            "loss": round(self.loss, 4),
            "wall_s": round(self.wall_s, 2),
            "peak_mem_gb": round(self.peak_mem_gb, 3),
            "n_trainable": self.n_trainable,
            "n_total": self.n_total,
            "n_data_passes": round(self.n_data_passes, 2),
        }
        d.update(self.extra)
        if self.error:
            d["error"] = self.error
        return d

# ---------------------------------------------------------------------------
# Manual LoRA implementation (fallback when peft is not installed)
# ---------------------------------------------------------------------------

class ManualLoraLinear(nn.Module):
    """Drop-in replacement for nn.Linear with a rank-r LoRA adapter.

    W_eff = W_frozen + scale * B @ A
    where A:(r, d_in), B:(d_out, r), scale = alpha/r.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 16.0):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.scale = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

        # Trainable adapters (standard init: A~N(0,0.02), B=0)
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        adapter = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + adapter


def inject_manual_lora(model: nn.Module, rank: int, target_modules: List[str]) -> nn.Module:
    """Replace named Linear modules with ManualLoraLinear wrappers."""
    for name, module in list(model.named_modules()):
        # Check if this module's simple name matches any target
        simple = name.split(".")[-1]
        if simple in target_modules and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            parent = model if not parent_name else _get_module(model, parent_name)
            setattr(parent, simple, ManualLoraLinear(module, rank=rank))
    return model


def _get_module(model: nn.Module, path: str) -> nn.Module:
    parts = path.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    tokenizer,
    task: str,
    batch_size: int,
    max_train_samples: Optional[int],
    max_eval_samples: int,
    device: torch.device,
):
    """Load and tokenise an HF dataset, return (train_loader, eval_loader)."""
    print(f"  Loading dataset '{task}' ...")

    if task == "sst2":
        raw = load_dataset("glue", "sst2")
        text_col, label_col = "sentence", "label"
        num_labels = 2
    elif task == "mrpc":
        raw = load_dataset("glue", "mrpc")
        text_col, label_col = ("sentence1", "sentence2"), "label"
        num_labels = 2
    elif task == "imdb":
        raw = load_dataset("imdb")
        text_col, label_col = "text", "label"
        num_labels = 2
    else:
        raise ValueError(f"Unknown task: {task}. Supported: sst2, mrpc, imdb")

    def tokenise(batch):
        if isinstance(text_col, tuple):
            return tokenizer(
                batch[text_col[0]], batch[text_col[1]],
                truncation=True, max_length=128, padding="max_length",
            )
        return tokenizer(
            batch[text_col], truncation=True,
            max_length=128, padding="max_length",
        )

    raw = raw.map(tokenise, batched=True, desc="Tokenising")
    raw = raw.rename_column(label_col, "labels")
    raw.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = raw["train"]
    eval_ds  = raw["validation"] if "validation" in raw else raw["test"]

    if max_train_samples and max_train_samples < len(train_ds):
        train_ds = train_ds.select(range(max_train_samples))
    if max_eval_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(max_eval_samples))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    eval_loader  = torch.utils.data.DataLoader(
        eval_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"  Train: {len(train_ds)} samples  |  Eval: {len(eval_ds)} samples")
    return train_loader, eval_loader, num_labels

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    """Returns (accuracy, avg_cross_entropy_loss)."""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total, total_loss / total

# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _peak_mem(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    return 0.0


def _reset_peak(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def run_adam(
    model_init: nn.Module,
    train_loader,
    eval_loader,
    device: torch.device,
    head_only: bool = False,
    lr: float = 2e-5,
    epochs: int = 1,
) -> ModeResult:
    """AdamW fine-tuning — full model or head-only."""
    mode = "adam_head" if head_only else "adam"
    print(f"\n{'─'*60}\n[{MODE_LABELS[mode]}]")

    model = copy.deepcopy(model_init).to(device)

    if head_only:
        for name, p in model.named_parameters():
            p.requires_grad = "classifier" in name or "pooler" in name
    else:
        for p in model.parameters():
            p.requires_grad = True

    n_train, n_total = count_trainable(model)
    print(f"  Trainable params: {n_train:,} / {n_total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    _reset_peak(device)
    t0 = time.time()
    model.train()

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"  step {step}/{len(train_loader)}  loss={loss.item():.4f}")

    wall = time.time() - t0
    acc, loss_val = evaluate(model, eval_loader, device)
    mem = _peak_mem(device)

    print(f"  → accuracy={acc:.4f}  loss={loss_val:.4f}  time={wall:.0f}s  mem={mem:.2f}GB")
    return ModeResult(
        mode=mode, label=MODE_LABELS[mode],
        accuracy=acc, loss=loss_val, wall_s=wall, peak_mem_gb=mem,
        n_trainable=n_train, n_total=n_total, n_data_passes=epochs,
    )


def run_lora(
    model_init: nn.Module,
    train_loader,
    eval_loader,
    device: torch.device,
    rank: int,
    lr: float = 3e-4,
    epochs: int = 1,
    target_modules: Optional[List[str]] = None,
) -> ModeResult:
    """LoRA fine-tuning — uses PEFT if available, manual LoRA otherwise."""
    mode = f"lora_r{rank}"
    print(f"\n{'─'*60}\n[{MODE_LABELS[mode]}]")

    model = copy.deepcopy(model_init)

    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]

    if HAS_PEFT:
        print("  Using HuggingFace PEFT LoRA")
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_cfg)
    else:
        print("  peft not installed — using manual LoRA")
        for name, p in model.named_parameters():
            p.requires_grad = False
        inject_manual_lora(model, rank=rank, target_modules=target_modules)

    model = model.to(device)
    n_train, n_total = count_trainable(model)
    print(f"  Trainable params: {n_train:,} / {n_total:,}")

    optimizer  = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    _reset_peak(device)
    t0 = time.time()
    model.train()

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None \
                   else F.cross_entropy(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"  step {step}/{len(train_loader)}  loss={loss.item():.4f}")

    wall = time.time() - t0
    acc, loss_val = evaluate(model, eval_loader, device)
    mem = _peak_mem(device)

    print(f"  → accuracy={acc:.4f}  loss={loss_val:.4f}  time={wall:.0f}s  mem={mem:.2f}GB")
    return ModeResult(
        mode=mode, label=MODE_LABELS[mode],
        accuracy=acc, loss=loss_val, wall_s=wall, peak_mem_gb=mem,
        n_trainable=n_train, n_total=n_total, n_data_passes=epochs,
        extra={"peft_used": HAS_PEFT},
    )


def run_lora_als(
    model_init: nn.Module,
    train_loader,
    eval_loader,
    device: torch.device,
    n_layers: int,
    rank: int = 4,
    als_sweeps: int = 5,
    lambda_reg: float = 1e-4,
    num_labels: int = 2,
) -> ModeResult:
    """ALS-LoRA: train LoRA matrices via Alternating Least Squares, no gradients.

    Replaces AdamW for the LoRA sub-problem with exact OLS alternating between
    fixing B (solve A) and fixing A (solve B).  Base weights W are never modified.

    Each ALS sweep is one dataset pass — 5 sweeps ≈ 5× the cost of OLS N=1,
    far fewer than the hundreds of gradient steps AdamW needs.

    This answers: "can OLS replace Adam in LoRA matrices?"
    """
    mode = f"als_lora_n{n_layers}_r{rank}"
    print(f"\n{'─'*60}\n[{MODE_LABELS.get(mode, mode)}]")

    model = copy.deepcopy(model_init).to(device)
    for p in model.parameters():
        p.requires_grad = False

    def target_fn(y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y.long(), num_classes=num_labels).float()

    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    # residual_mode mirrors run_ols: True for N=1 (proven stable), False for N>1
    # (full-replace BCD; residual_mode=True with BCD causes exponential blowup).
    residual_mode = (n_layers == 1)

    retrainer = OlsSMLayerRetrainer(
        model,
        n_layers=n_layers,
        lambda_reg=lambda_reg,
        lora_rank=rank,
        lora_sweeps=als_sweeps,
        lora_lambda=lambda_reg,
        bcd_mode="gauss_seidel",
        residual_mode=residual_mode,
        bcd_step_size=1.0,
        verbose=True,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    # LoRA params are not nn.Parameters here (they're plain tensors inside the
    # retrainer), so count them manually: 2 × rank × (d_in + d_out) per layer
    lora_params = sum(
        rank * (layer.in_features + layer.out_features)
        for layer in retrainer.get_retrained_layers()
    )
    print(f"  ALS-LoRA on last {n_layers} of {n_linear} Linear layers  "
          f"(rank={rank}, {lora_params:,} adapter params, {als_sweeps} ALS sweeps)")

    class HFLoaderAdapter:
        def __init__(self, loader):
            self._loader = loader
        def __iter__(self):
            for batch in self._loader:
                x = {k: batch[k] for k in ("input_ids", "attention_mask")}
                y = batch["labels"]
                yield x, y
        def __len__(self):
            return len(self._loader)

    _reset_peak(device)
    t0 = time.time()

    retrainer.retrain_lora_als(HFLoaderAdapter(train_loader), target_fn)
    retrainer.remove_hooks()

    wall = time.time() - t0
    acc, loss_val = evaluate(model, eval_loader, device)
    mem = _peak_mem(device)

    print(f"  → accuracy={acc:.4f}  loss={loss_val:.4f}  "
          f"time={wall:.0f}s  mem={mem:.2f}GB  als_sweeps={als_sweeps}")
    return ModeResult(
        mode=mode, label=MODE_LABELS.get(mode, mode),
        accuracy=acc, loss=loss_val, wall_s=wall, peak_mem_gb=mem,
        n_trainable=lora_params, n_total=n_total,
        n_data_passes=als_sweeps,
        extra={"n_layers_als": n_layers, "lora_rank": rank, "als_sweeps": als_sweeps},
    )


def run_ols(
    model_init: nn.Module,
    train_loader,
    eval_loader,
    device: torch.device,
    n_layers: int,
    lora_rank: int = 0,
    max_sweeps: int = 5,
    lambda_reg: float = 1e-4,
    num_labels: int = 2,
    bcd_mode: str = "gauss_seidel",
) -> ModeResult:
    """OlsSMLayerRetrainer — BCD with optional LoRA residual stage."""
    if lora_rank > 0:
        mode = f"ols_lora_r{lora_rank}"
    elif n_layers == -1:
        mode = "ols_all"
    else:
        mode = f"ols_n{n_layers}"

    print(f"\n{'─'*60}\n[{MODE_LABELS.get(mode, mode)}]")

    model = copy.deepcopy(model_init).to(device)
    # Freeze everything (OlsSM retrainer does direct weight replacement)
    for p in model.parameters():
        p.requires_grad = False

    # target_fn: one-hot targets for the classifier head.
    # With residual_mode=True, OLS solves for ΔW such that (W_old+ΔW)@x → y_onehot.
    # For a pretrained model with large logits (e.g. [-5, 8]), the residual
    # (y_onehot - W_old@x) points in the right direction and the solution
    # fine-tunes around the good initialisation rather than replacing weights.
    def target_fn(y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y.long(), num_classes=num_labels).float()

    # Resolve n_layers=-1 to all linear layers
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    actual_n = n_linear if n_layers == -1 else n_layers

    # residual_mode=True  (N=1):  solve for ΔW — stays close to pretrained init,
    #                             proven effective at 84.98% on SST-2.
    # residual_mode=False (N>1):  solve for W directly — proper Gauss-Seidel BCD.
    #   Residual mode + multi-layer BCD is unstable: partial updates leave layers
    #   under-corrected, each subsequent solve must compensate, causing exponential
    #   blowup in max|ΔW|.  Full-replace BCD has exact sub-solves at each step,
    #   which guarantees monotone convergence for Gauss-Seidel.
    residual_mode = (actual_n == 1)

    retrainer = OlsSMLayerRetrainer(
        model,
        n_layers=actual_n,
        lambda_reg=lambda_reg,
        max_sweeps=max_sweeps,
        tol=1e-5,
        lora_rank=lora_rank,
        lora_sweeps=3,
        bcd_mode=bcd_mode,
        residual_mode=residual_mode,
        bcd_step_size=1.0,
        verbose=True,
    )
    n_train, n_total = count_trainable(model)
    print(f"  Retraining {actual_n} of {n_linear} Linear layers  "
          f"({'+ LoRA r=' + str(lora_rank) if lora_rank else 'pure OLS'})")

    # Wrap the HF loader so the retrainer receives:
    #   batch_x = {'input_ids': LongTensor, 'attention_mask': LongTensor}
    #   batch_y = LongTensor (class labels)
    # The retrainer's _model_forward helper unpacks the dict via model(**batch_x),
    # preserving token dtypes (Long) that BERT's embedding layer requires.
    class HFLoaderAdapter:
        """Wraps an HF dataloader to yield (dict_inputs, labels) tuples."""
        def __init__(self, loader):
            self._loader = loader
        def __iter__(self):
            for batch in self._loader:
                x = {
                    "input_ids":      batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                yield x, batch["labels"].to(device)
        def __len__(self):
            return len(self._loader)

    _reset_peak(device)
    t0 = time.time()

    history = retrainer.retrain(
        HFLoaderAdapter(train_loader),
        target_fn=target_fn,
    )
    retrainer.remove_hooks()

    wall = time.time() - t0
    acc, loss_val = evaluate(model, eval_loader, device)
    mem = _peak_mem(device)

    print(f"  → accuracy={acc:.4f}  loss={loss_val:.4f}  "
          f"time={wall:.0f}s  mem={mem:.2f}GB  sweeps={history['n_sweeps']}")
    return ModeResult(
        mode=mode, label=MODE_LABELS.get(mode, mode),
        accuracy=acc, loss=loss_val, wall_s=wall, peak_mem_gb=mem,
        n_trainable=n_train, n_total=n_total,
        n_data_passes=history["n_sweeps"],
        extra={
            "n_layers_retrained": actual_n,
            "n_linear_total": n_linear,
            "bcd_converged": history["converged"],
            "bcd_sweeps": history["n_sweeps"],
            "lora_fitted": history["lora_fitted"],
        },
    )

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(results: List[ModeResult], out_dir: Path, model_name: str, task: str):
    if not HAS_MPL:
        print("[WARN] matplotlib not available — skipping plots.")
        return

    valid = [r for r in results if r.error is None]
    if not valid:
        return

    modes  = [r.label for r in valid]
    accs   = [r.accuracy * 100 for r in valid]
    losses = [r.loss for r in valid]
    times  = [r.wall_s for r in valid]
    mems   = [r.peak_mem_gb for r in valid]

    x = np.arange(len(modes))
    colours = []
    for r in valid:
        if r.mode.startswith("adam"):      colours.append("#4C72B0")
        elif r.mode.startswith("lora"):    colours.append("#DD8452")
        elif "lora" in r.mode:             colours.append("#55A868")
        else:                              colours.append("#C44E52")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"OlsSM Retrainer Benchmark\n"
        f"Model: {model_name}  |  Task: {task.upper()}",
        fontsize=13, fontweight="bold",
    )

    def bar(ax, values, title, ylabel, fmt="{:.2f}", colour_override=None):
        bars = ax.bar(x, values, color=colour_override or colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar_, val in zip(bars, values):
            ax.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() + 0.002 * max(values),
                    fmt.format(val), ha="center", va="bottom", fontsize=7)

    bar(axes[0, 0], accs,   "Accuracy (%)",           "Accuracy (%)",      fmt="{:.1f}%")
    bar(axes[0, 1], losses, "Eval Cross-Entropy Loss", "Loss",              fmt="{:.3f}")
    bar(axes[1, 0], times,  "Wall-Clock Time (s)",     "Seconds",           fmt="{:.0f}s")
    bar(axes[1, 1], mems,   "Peak GPU Memory (GB)",    "GB",                fmt="{:.2f}")

    # Trainable parameter annotation on accuracy chart
    for i, r in enumerate(valid):
        if r.n_trainable > 0:
            axes[0, 0].text(i, 0.5, f"{r.n_trainable/1e6:.1f}M",
                            ha="center", va="bottom", fontsize=6, color="white",
                            fontweight="bold", rotation=90)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="AdamW baselines"),
        Patch(facecolor="#DD8452", label="Pure LoRA"),
        Patch(facecolor="#55A868", label="OLS + LoRA"),
        Patch(facecolor="#C44E52", label="Pure OLS"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    tag = model_name.replace("/", "_").replace("-", "_")
    plot_path = out_dir / f"retrainer_{tag}_{task}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {plot_path}")
    plt.close()

    # ── Accuracy vs Time scatter ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for r, c in zip(valid, colours):
        ax2.scatter(r.wall_s, r.accuracy * 100, color=c, s=120, zorder=5)
        ax2.annotate(r.label, (r.wall_s, r.accuracy * 100),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title(
        f"Accuracy vs Time — {model_name} on {task.upper()}\n"
        "(upper-left = better)",
        fontsize=11,
    )
    ax2.grid(alpha=0.3)
    scatter_path = out_dir / f"retrainer_{tag}_{task}_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"  Scatter plot saved: {scatter_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(results: List[ModeResult]):
    print("\n" + "═" * 80)
    print(f"  {'MODE':<22}  {'ACC':>7}  {'LOSS':>7}  {'TIME(s)':>8}  "
          f"{'MEM(GB)':>8}  {'PARAMS(M)':>10}  {'PASSES':>7}")
    print("─" * 80)
    for r in results:
        if r.error:
            print(f"  {r.label:<22}  {'ERROR':>7}  {r.error[:40]}")
            continue
        params_m = r.n_trainable / 1e6
        print(
            f"  {r.label:<22}  {r.accuracy*100:>6.2f}%  {r.loss:>7.4f}  "
            f"{r.wall_s:>8.0f}  {r.peak_mem_gb:>8.3f}  "
            f"{params_m:>9.1f}M  {r.n_data_passes:>7.1f}"
        )
    print("═" * 80)

    # Best accuracy
    valid = [r for r in results if r.error is None]
    if valid:
        best = max(valid, key=lambda r: r.accuracy)
        fastest_good = min(
            (r for r in valid if r.accuracy >= best.accuracy - 0.01),
            key=lambda r: r.wall_s,
        )
        print(f"\n  Best accuracy : {best.label}  ({best.accuracy*100:.2f}%)")
        print(f"  Fastest within 1pp of best: {fastest_good.label}  "
              f"({fastest_good.wall_s:.0f}s, {fastest_good.accuracy*100:.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark OlsSMLayerRetrainer vs LoRA on a HuggingFace model"
    )
    p.add_argument("--model",    default="bert-base-uncased",
                   help="HuggingFace model name or path (default: bert-base-uncased)")
    p.add_argument("--task",     default="sst2",
                   choices=["sst2", "mrpc", "imdb"],
                   help="Classification task / dataset (default: sst2)")
    p.add_argument("--modes",    default="all",
                   help=f"Comma-separated list of modes, or 'all'. "
                        f"Available: {', '.join(ALL_MODES)}")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--max-train",   type=int, default=None,
                   help="Cap training set size (default: use full dataset)")
    p.add_argument("--max-eval",    type=int, default=872,
                   help="Max eval samples (default: 872 = full SST-2 dev)")
    p.add_argument("--epochs",      type=int, default=1,
                   help="Epochs for Adam/LoRA modes (default: 1)")
    p.add_argument("--max-sweeps",  type=int, default=5,
                   help="Max BCD sweeps for OLS modes (default: 5)")
    p.add_argument("--lambda-reg",  type=float, default=1e-4,
                   help="OLS regularisation lambda (default: 1e-4)")
    p.add_argument("--adam-lr",     type=float, default=2e-5)
    p.add_argument("--lora-lr",     type=float, default=3e-4)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir",  default=str(OUT))
    p.add_argument("--bcd-mode",    default="gauss_seidel",
                   choices=["jacobi", "gauss_seidel"],
                   help="BCD variant for OLS modes: jacobi (default) or gauss_seidel. "
                        "gauss_seidel updates each layer immediately so subsequent "
                        "layers see the correction; costs N forward passes per sweep "
                        "instead of 1, but converges monotonically.")
    p.add_argument("--no-plots",    action="store_true")
    p.add_argument("--tag",         default="",
                   help="Optional tag appended to output filenames")
    return p.parse_args()


def main():
    args = parse_args()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    if not HAS_DATASETS:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve modes
    if args.modes.strip().lower() == "all":
        modes = ALL_MODES
    else:
        modes = [m.strip() for m in args.modes.split(",")]
        invalid = [m for m in modes if m not in ALL_MODES]
        if invalid:
            print(f"ERROR: unknown modes: {invalid}. Valid: {ALL_MODES}")
            sys.exit(1)

    print("=" * 70)
    print(f"  OlsSM Retrainer Benchmark")
    print(f"  Model  : {args.model}")
    print(f"  Task   : {args.task}")
    print(f"  Modes  : {', '.join(modes)}")
    print(f"  Device : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 70)

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print("\nLoading model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # We need to know num_labels before building the loader
    num_labels = 2  # all default tasks are binary

    model_init = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    n_total = sum(p.numel() for p in model_init.parameters())
    print(f"  {args.model}  ({n_total/1e6:.1f}M parameters)")

    # ── Build data loaders ────────────────────────────────────────────────────
    print("\nPreparing data ...")
    train_loader, eval_loader, num_labels = build_dataloaders(
        tokenizer, args.task, args.batch_size,
        args.max_train, args.max_eval, device,
    )

    # Baseline: eval pretrained model (no fine-tuning)
    print("\nEvaluating pretrained model (zero-shot baseline) ...")
    model_init.to(device)
    pt_acc, pt_loss = evaluate(model_init, eval_loader, device)
    print(f"  Pretrained accuracy: {pt_acc*100:.2f}%  loss: {pt_loss:.4f}")
    model_init.cpu()

    # ── Run each mode ─────────────────────────────────────────────────────────
    results: List[ModeResult] = [
        ModeResult(mode="pretrained", label="Pretrained (no FT)",
                   accuracy=pt_acc, loss=pt_loss, wall_s=0,
                   n_trainable=0, n_total=n_total, n_data_passes=0)
    ]

    for mode in modes:
        try:
            if mode == "adam":
                r = run_adam(model_init, train_loader, eval_loader, device,
                             head_only=False, lr=args.adam_lr, epochs=args.epochs)
            elif mode == "adam_head":
                r = run_adam(model_init, train_loader, eval_loader, device,
                             head_only=True,  lr=args.adam_lr, epochs=args.epochs)
            elif mode == "lora_r4":
                r = run_lora(model_init, train_loader, eval_loader, device,
                             rank=4, lr=args.lora_lr, epochs=args.epochs)
            elif mode == "lora_r8":
                r = run_lora(model_init, train_loader, eval_loader, device,
                             rank=8, lr=args.lora_lr, epochs=args.epochs)
            elif mode == "ols_n1":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=1, max_sweeps=1, lambda_reg=args.lambda_reg,
                            num_labels=num_labels, bcd_mode=args.bcd_mode)
            elif mode == "ols_n2":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=2, max_sweeps=args.max_sweeps,
                            lambda_reg=args.lambda_reg, num_labels=num_labels,
                            bcd_mode=args.bcd_mode)
            elif mode == "ols_n4":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=4, max_sweeps=args.max_sweeps,
                            lambda_reg=args.lambda_reg, num_labels=num_labels,
                            bcd_mode=args.bcd_mode)
            elif mode == "ols_n8":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=8, max_sweeps=args.max_sweeps,
                            lambda_reg=args.lambda_reg, num_labels=num_labels,
                            bcd_mode=args.bcd_mode)
            elif mode == "ols_all":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=-1, max_sweeps=args.max_sweeps,
                            lambda_reg=args.lambda_reg, num_labels=num_labels,
                            bcd_mode=args.bcd_mode)
            elif mode == "ols_lora_r2":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=2, lora_rank=2, max_sweeps=args.max_sweeps,
                            lambda_reg=args.lambda_reg, num_labels=num_labels,
                            bcd_mode=args.bcd_mode)
            elif mode == "ols_lora_r4":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=2, lora_rank=4, max_sweeps=args.max_sweeps,
                            lambda_reg=args.lambda_reg, num_labels=num_labels,
                            bcd_mode=args.bcd_mode)
            elif mode == "ols_lora_r8":
                r = run_ols(model_init, train_loader, eval_loader, device,
                            n_layers=2, lora_rank=8, max_sweeps=args.max_sweeps,
                            lambda_reg=args.lambda_reg, num_labels=num_labels,
                            bcd_mode=args.bcd_mode)
            elif mode == "als_lora_n1_r4":
                r = run_lora_als(model_init, train_loader, eval_loader, device,
                                 n_layers=1, rank=4, als_sweeps=args.max_sweeps,
                                 lambda_reg=args.lambda_reg, num_labels=num_labels)
            elif mode == "als_lora_n2_r4":
                r = run_lora_als(model_init, train_loader, eval_loader, device,
                                 n_layers=2, rank=4, als_sweeps=args.max_sweeps,
                                 lambda_reg=args.lambda_reg, num_labels=num_labels)
            elif mode == "als_lora_n4_r4":
                r = run_lora_als(model_init, train_loader, eval_loader, device,
                                 n_layers=4, rank=4, als_sweeps=args.max_sweeps,
                                 lambda_reg=args.lambda_reg, num_labels=num_labels)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as exc:
            import traceback
            print(f"\n  [ERROR in {mode}]: {exc}")
            traceback.print_exc()
            r = ModeResult(
                mode=mode, label=MODE_LABELS.get(mode, mode),
                error=str(exc),
            )

        results.append(r)

        # Save incrementally after every mode
        tag = f"_{args.tag}" if args.tag else ""
        model_tag = args.model.replace("/", "_").replace("-", "_")
        json_path = out_dir / f"retrainer_{model_tag}_{args.task}{tag}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "task": args.task,
                    "modes_run": [r.mode for r in results],
                    "pretrained_accuracy": pt_acc,
                    "results": [r.to_dict() for r in results],
                },
                f, indent=2,
            )
        print(f"  (results saved → {json_path})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        make_plots(
            [r for r in results if r.mode != "pretrained"],
            out_dir, args.model, args.task,
        )

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()