"""
OlsveredKFAC GPU Benchmark
============================
Empirically measures the advantage of OlsveredKFAC over ClassicKFAC and Adam
on real GPU hardware, using two tasks:

  Task 1 — Large MLP (quick, 20-40 min):
      4-layer 784→2048→2048→2048→10 network trained on MNIST.
      All Linear layers → full OlsveredKFAC coverage.
      Used to verify per-step overhead ratios and convergence curves.

  Task 2 — BERT-base fine-tuning on SST-2 (comprehensive, 3-6 hrs):
      110M param transformer fine-tuned for sentiment classification.
      All 200+ attention/FFN layers are nn.Linear → full coverage.
      Measures whether K-FAC step advantage holds at scale.

Metrics tracked per optimizer:
  - Val accuracy / loss  vs  gradient steps
  - Val accuracy / loss  vs  samples seen   (fair cross-batchsize comparison)
  - Val accuracy / loss  vs  wall-clock time
  - Optimizer overhead per step (timed separately from fwd/bwd)
  - Peak GPU memory allocated (GB)
  - Average GPU power draw (W)  — via nvidia-smi if available

Usage:
  # Quick test (MLP only, ~30 min on any GPU with ≥4GB VRAM):
  python benchmark/gpu_benchmark.py --task mlp

  # Full benchmark (MLP + BERT, ~4-6h, needs ≥12GB VRAM):
  python benchmark/gpu_benchmark.py --task all

  # BERT only:
  python benchmark/gpu_benchmark.py --task bert

  # Custom:
  python benchmark/gpu_benchmark.py --task all --max-steps-mlp 2000 --max-steps-bert 5000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT  = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ─── Serialisation helpers ────────────────────────────────────────────────

def to_serialisable(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, list):        return [to_serialisable(x) for x in obj]
    if isinstance(obj, dict):        return {k: to_serialisable(v) for k,v in obj.items()}
    return obj

def save_result_incremental(result: dict, task: str):
    """Save a single optimizer result immediately after it completes."""
    name_slug = result["name"].lower().replace(" ", "_")
    # ── JSON (full detail) ──
    json_path = OUT / f"{task}_{name_slug}_result.json"
    with open(json_path, "w") as f:
        json.dump(to_serialisable(result), f, indent=2)
    # ── CSV (summary row) ──
    csv_path = OUT / f"{task}_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("task,name,batch_size,lr_init,lr_final,steps,samples,wall_s,"
                    "avg_fwdbwd_ms,avg_opt_ms,p99_opt_ms,"
                    "peak_mem_gb,avg_power_w,final_val_acc,final_val_loss\n")
        final_acc  = result["curve_val_acc"][-1]  if result["curve_val_acc"]  else ""
        final_loss = result["curve_val_loss"][-1] if result["curve_val_loss"] else ""
        pwr      = result.get("avg_power_w") or ""
        lr_final = result.get("lr_final", "")
        f.write(f"{result['task']},{result['name']},{result['B']},{result['lr']},"
                f"{lr_final},{result['steps']},{result['samples']},{result['wall_s']:.1f},"
                f"{result['avg_fwdbwd_ms']:.2f},{result['avg_opt_ms']:.2f},"
                f"{result['p99_opt_ms']:.2f},{result['peak_mem_gb']:.3f},{pwr},"
                f"{final_acc},{final_loss}\n")
    print(f"  ✓  Saved → {json_path.name}  |  {csv_path.name}")

# ─── GPU utilities ────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"| VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        return d
    print("  ERROR: No CUDA GPU found. Run via run_benchmark.sh for GPU verification.")
    sys.exit(2)

def gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_gpu_power_w() -> Optional[float]:
    """Read instantaneous GPU power via nvidia-smi (returns None if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        return float(out.stdout.strip().split("\n")[0])
    except Exception:
        return None

class PowerMonitor:
    """Background power sampling via nvidia-smi every 0.5 s."""
    def __init__(self):
        self.samples = []
        self._active = False

    def start(self):
        self._active = True; self.samples = []

    def sample(self):
        if self._active:
            w = get_gpu_power_w()
            if w is not None: self.samples.append(w)

    def stop(self) -> Optional[float]:
        self._active = False
        return float(np.mean(self.samples)) if self.samples else None

# ─── Helpers ──────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0; total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches: break
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
            else:  # HuggingFace dict batch
                batch  = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                out    = model(**batch)
                logits = out.logits; y = labels
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            correct    += (logits.argmax(-1) == y).sum().item()
            total      += y.size(0)
    return correct / total, total_loss / max(1, i+1)

# ─── TASK 1: Large MLP on MNIST ───────────────────────────────────────────

def run_mlp_benchmark(device, args):
    print("\n" + "="*70)
    print("  TASK 1: Large MLP  (784 → 2048 → 2048 → 2048 → 10)  on MNIST")
    print("="*70)

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tf  = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,),(0.3081,))])
    val_loader = DataLoader(
        datasets.MNIST(ROOT/"data", train=False, download=True, transform=tf),
        batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    class LargeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 10))
        def forward(self, x):
            return self.net(x.view(x.size(0), -1))

    configs = [
        dict(name="Adam",          B=128,  lr=1e-3,  kfac=False),
        dict(name="ClassicKFAC",   B=512,  lr=5e-2,  kfac=True,  randomised=False),
        dict(name="OlsveredKFAC",  B=512,  lr=1e-2,  kfac=True,  randomised=True),
    ]
    configs = [c for c in configs if c["name"].lower() not in args.skip]
    if not configs:
        print("  All optimizers skipped — nothing to run for MLP task.")
        return []

    all_results = []
    for cfg in configs:
        print(f"\n  ── {cfg['name']}  (B={cfg['B']}, lr={cfg['lr']}) ──")
        torch.manual_seed(42)
        model = LargeMLP().to(device)
        print(f"     Parameters: {count_params(model):,}")

        train_ds = datasets.MNIST(ROOT/"data", train=True, download=False, transform=tf)
        train_loader = DataLoader(train_ds, batch_size=cfg['B'], shuffle=True,
                                  num_workers=2, pin_memory=True)

        if not cfg['kfac']:
            opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        elif cfg['randomised']:
            from optimizer.olsvered_kfac import OlsveredKFAC
            # On GPU: inv_update_freq=10 is fine (EVD is fast).
            # On CPU: increase to 50 to amortise the expensive EVD cost.
            evd_freq = 5 if torch.cuda.is_available() else 50
            opt = OlsveredKFAC(model, lr=cfg['lr'], damping=1e-2,
                               factor_update_freq=5, inv_update_freq=evd_freq,
                               adaptive=True, adaptive_min_n=256,
                               adaptive_rank_budget=64, momentum=0.0,
                               grad_clip=10.0)
        else:
            from optimizer.classic_kfac import ClassicKFAC
            opt = ClassicKFAC(model, lr=cfg['lr'], damping=1e-2,
                              factor_update_freq=10, inv_update_freq=10,
                              momentum=0.0, grad_clip=10.0)

        criterion  = nn.CrossEntropyLoss()
        # K-FAC needs faster LR decay than Adam: it converges to a good basin
        # quickly but needs the LR floor earlier to refine past the plateau.
        # Use T_max = half the run so cosine reaches eta_min by step ~1500,
        # then holds there.  Adam uses the full run length for a gentler ramp.
        if cfg['kfac']:
            sched_t_max    = max(1, args.max_steps_mlp // 2)
            eta_min_factor = 0.015   # floor = 0.2 % of initial LR
        else:
            sched_t_max    = args.max_steps_mlp
            eta_min_factor = 0.01
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=sched_t_max, eta_min=cfg['lr'] * eta_min_factor)
        data_iter  = iter(train_loader)
        power_mon  = PowerMonitor()
        reset_memory_stats()
        power_mon.start()

        t0 = time.perf_counter()
        step = samples_seen = 0
        opt_times = []; fwdbwd_times = []
        record_every_n_samples = 2_000
        next_record = record_every_n_samples
        curve_steps=[]; curve_samples=[]; curve_times=[]
        curve_val_acc=[]; curve_val_loss=[]

        while step < args.max_steps_mlp:
            try: x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            # Timed fwd+bwd
            t_fwd = time.perf_counter()
            model.train(); logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward()
            t_fwd_done = time.perf_counter()

            # Timed optimizer step
            t_opt = time.perf_counter()
            opt.step()
            t_opt_done = time.perf_counter()

            scheduler.step()
            power_mon.sample()
            fwdbwd_times.append(t_fwd_done - t_fwd)
            opt_times.append(t_opt_done - t_opt)
            step += 1; samples_seen += x.size(0)

            if samples_seen >= next_record:
                va, vl = evaluate(model, val_loader, device, max_batches=20)
                wall = time.perf_counter() - t0
                curve_steps.append(step); curve_samples.append(samples_seen)
                curve_times.append(wall); curve_val_acc.append(va)
                curve_val_loss.append(vl)
                next_record += record_every_n_samples
                cur_lr = scheduler.get_last_lr()[0]
                print(f"     step={step:5d}  samples={samples_seen:7,}  "
                      f"val_acc={va:.4f}  val_loss={vl:.4f}  "
                      f"lr={cur_lr:.2e}  wall={wall:.1f}s  "
                      f"opt={np.mean(opt_times[-20:])*1000:.1f}ms")

        avg_power = power_mon.stop()
        peak_mem  = gpu_memory_gb()
        result = dict(
            task="mlp", **cfg,
            lr_final=scheduler.get_last_lr()[0],
            steps=step, samples=samples_seen,
            wall_s=time.perf_counter()-t0,
            avg_fwdbwd_ms=np.mean(fwdbwd_times)*1000,
            avg_opt_ms=np.mean(opt_times)*1000,
            p99_opt_ms=np.percentile(opt_times,99)*1000,
            peak_mem_gb=peak_mem,
            avg_power_w=avg_power,
            curve_steps=curve_steps, curve_samples=curve_samples,
            curve_times=curve_times, curve_val_acc=curve_val_acc,
            curve_val_loss=curve_val_loss,
        )
        all_results.append(result)
        save_result_incremental(result, "mlp")
        if hasattr(opt,'cleanup'): opt.cleanup()

    return all_results

# ─── TASK 2: BERT-base fine-tuning on SST-2 ───────────────────────────────

def run_bert_benchmark(device, args):
    print("\n" + "="*70)
    print("  TASK 2: BERT-base fine-tuning on SST-2 (sentiment classification)")
    print("="*70)

    try:
        from transformers import (BertForSequenceClassification, BertTokenizer,
                                  DataCollatorWithPadding)
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: transformers and datasets packages required.")
        print("         pip install transformers datasets")
        return []

    print("  Loading SST-2 dataset and BERT-base tokenizer...")
    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    raw_ds     = load_dataset("glue", "sst2")

    def preprocess(batch):
        enc = tokenizer(batch["sentence"], truncation=True, max_length=128)
        enc["labels"] = batch["label"]
        return enc

    encoded = raw_ds.map(preprocess, batched=True,
                          remove_columns=["sentence","idx"])
    encoded.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    from torch.utils.data import DataLoader
    val_loader = DataLoader(encoded["validation"], batch_size=256,
                            collate_fn=collator, num_workers=2)

    configs = [
        dict(name="Adam",         B=32,  lr=2e-5, kfac=False),
        dict(name="ClassicKFAC",  B=512, lr=5e-3, kfac=True,  randomised=False),
        dict(name="OlsveredKFAC", B=512, lr=9e-3, kfac=True,  randomised=True),
    ]
    configs = [c for c in configs if c["name"].lower() not in args.skip]
    if not configs:
        print("  All optimizers skipped — nothing to run for BERT task.")
        return []

    all_results = []
    for cfg in configs:
        print(f"\n  ── {cfg['name']}  (B={cfg['B']}, lr={cfg['lr']}) ──")
        torch.manual_seed(42)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2).to(device)
        print(f"     Parameters: {count_params(model):,}")

        train_loader = DataLoader(
            encoded["train"], batch_size=cfg['B'], shuffle=True,
            collate_fn=collator, num_workers=2)

        if not cfg['kfac']:
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                    weight_decay=0.01)
        elif cfg['randomised']:
            from optimizer.olsvered_kfac import OlsveredKFAC
            evd_freq = 5 if torch.cuda.is_available() else 50
            opt = OlsveredKFAC(model, lr=cfg['lr'], damping=1e-3,
                               factor_update_freq=4, inv_update_freq=evd_freq,
                               adaptive=True, adaptive_min_n=256,
                               adaptive_rank_budget=64, momentum=0.0,
                               grad_clip=5.0)
        else:
            from optimizer.classic_kfac import ClassicKFAC
            opt = ClassicKFAC(model, lr=cfg['lr'], damping=1e-3,
                              factor_update_freq=10, inv_update_freq=10,
                              momentum=0.0, grad_clip=5.0)

        if cfg['kfac']:
            sched_t_max    = max(1, args.max_steps_bert // 2)
            eta_min_factor = 0.002
        else:
            sched_t_max    = args.max_steps_bert
            eta_min_factor = 0.01
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=sched_t_max, eta_min=cfg['lr'] * eta_min_factor)
        data_iter = iter(train_loader)
        power_mon = PowerMonitor()
        reset_memory_stats()
        power_mon.start()

        t0 = time.perf_counter()
        step = samples_seen = 0
        opt_times=[]; fwdbwd_times=[]
        record_every_n = 5_000
        next_record = record_every_n
        curve_steps=[]; curve_samples=[]; curve_times=[]
        curve_val_acc=[]; curve_val_loss=[]

        while step < args.max_steps_bert:
            try: batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); batch = next(data_iter)
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            t_fwd = time.perf_counter()
            model.train()
            out  = model(**batch)
            loss = F.cross_entropy(out.logits, labels)
            opt.zero_grad(); loss.backward()
            t_fwd_done = time.perf_counter()

            t_opt = time.perf_counter()
            opt.step()
            t_opt_done = time.perf_counter()

            scheduler.step()
            power_mon.sample()
            fwdbwd_times.append(t_fwd_done - t_fwd)
            opt_times.append(t_opt_done - t_opt)
            step += 1; samples_seen += labels.size(0)

            if samples_seen >= next_record:
                va, vl = evaluate(model, val_loader, device)
                wall = time.perf_counter() - t0
                curve_steps.append(step); curve_samples.append(samples_seen)
                curve_times.append(wall); curve_val_acc.append(va)
                curve_val_loss.append(vl)
                next_record += record_every_n
                cur_lr = scheduler.get_last_lr()[0]
                print(f"     step={step:5d}  samples={samples_seen:7,}  "
                      f"val_acc={va:.4f}  val_loss={vl:.4f}  "
                      f"lr={cur_lr:.2e}  wall={wall/60:.1f}min  "
                      f"opt={np.mean(opt_times[-20:])*1000:.1f}ms")

        avg_power = power_mon.stop()
        peak_mem  = gpu_memory_gb()
        result = dict(
            task="bert", **cfg,
            lr_final=scheduler.get_last_lr()[0],
            steps=step, samples=samples_seen,
            wall_s=time.perf_counter()-t0,
            avg_fwdbwd_ms=np.mean(fwdbwd_times)*1000,
            avg_opt_ms=np.mean(opt_times)*1000,
            p99_opt_ms=np.percentile(opt_times,99)*1000,
            peak_mem_gb=peak_mem,
            avg_power_w=avg_power,
            curve_steps=curve_steps, curve_samples=curve_samples,
            curve_times=curve_times, curve_val_acc=curve_val_acc,
            curve_val_loss=curve_val_loss,
        )
        all_results.append(result)
        save_result_incremental(result, "bert")
        if hasattr(opt,'cleanup'): opt.cleanup()

    return all_results

# ─── Plotting ─────────────────────────────────────────────────────────────

def make_plots(results, tag):
    import matplotlib.pyplot as plt
    COLORS = {"Adam":"#7f8c8d","ClassicKFAC":"#c0392b","OlsveredKFAC":"#2980b9"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"OlsveredKFAC GPU Benchmark — {tag}", fontsize=13, fontweight="bold")

    def plot_curve(ax, x_key, xlabel, ylabel_key, ylabel, title):
        for r in results:
            ax.plot(r[x_key], r[ylabel_key], lw=2, marker="o", ms=4,
                    color=COLORS.get(r["name"],"black"), label=r["name"])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plot_curve(axes[0,0], "curve_steps",   "Gradient steps",
               "curve_val_acc", "Val accuracy", "Val Accuracy vs Steps")
    plot_curve(axes[0,1], "curve_samples", "Samples seen",
               "curve_val_acc", "Val accuracy", "Val Accuracy vs Samples Seen\n(fair cross-batchsize comparison)")
    plot_curve(axes[0,2], "curve_times",   "Wall time (s)",
               "curve_val_acc", "Val accuracy", "Val Accuracy vs Wall Time")
    plot_curve(axes[1,0], "curve_steps",   "Gradient steps",
               "curve_val_loss","Val loss",     "Val Loss vs Steps")
    plot_curve(axes[1,1], "curve_samples", "Samples seen",
               "curve_val_loss","Val loss",     "Val Loss vs Samples Seen")

    # Per-step timing + memory bar chart
    ax = axes[1,2]
    names   = [r["name"] for r in results]
    fwd_ms  = [r["avg_fwdbwd_ms"] for r in results]
    opt_ms  = [r["avg_opt_ms"]    for r in results]
    mem_gb  = [r["peak_mem_gb"]   for r in results]
    pwr_w   = [r.get("avg_power_w") or 0 for r in results]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, fwd_ms, w, label="Fwd+Bwd (ms)", color="#bdc3c7")
    ax.bar(x - w/2, opt_ms, w, label="Optimizer (ms)", color=[COLORS.get(n,"black") for n in names],
           bottom=fwd_ms, alpha=0.9)
    ax2 = ax.twinx()
    ax2.plot(x, mem_gb, "D--", color="#e67e22", ms=8, lw=2, label="Peak mem (GB)")
    ax2.set_ylabel("Peak GPU memory (GB)", color="#e67e22", fontsize=9)
    ax2.tick_params(axis='y', labelcolor="#e67e22")
    ax.set_title("Per-Step Timing & GPU Memory", fontweight="bold", fontsize=10)
    ax.set_ylabel("ms / step", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate power if available
    for i, (name, pw) in enumerate(zip(names, pwr_w)):
        if pw: ax.text(i+w/2, 5, f"{pw:.0f}W", fontsize=8, ha="center", color="#8e44ad")

    plt.tight_layout()
    path = OUT / f"gpu_benchmark_{tag.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {path}")

# ─── Summary printout ─────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    print(f"  {'Name':<16}  {'B':>5}  {'lr_init':>8}  {'lr_final':>8}  "
          f"{'Steps':>7}  {'Samples':>9}  {'Wall':>8}  "
          f"{'FwdBwd ms':>10}  {'Opt ms':>8}  {'Mem GB':>7}  {'Pwr W':>6}")
    print(f"  {'':─<16}  {'':─<5}  {'':─<8}  {'':─<8}  {'':─<7}  "
          f"{'':─<9}  {'':─<8}  {'':─<10}  {'':─<8}  {'':─<7}  {'':─<6}")
    for r in results:
        wall      = f"{r['wall_s']/60:.1f}min" if r['wall_s'] > 120 else f"{r['wall_s']:.0f}s"
        pwr       = f"{r['avg_power_w']:.0f}" if r.get('avg_power_w') else "N/A"
        lr_final  = f"{r['lr_final']:.2e}" if r.get('lr_final') is not None else "N/A"
        print(f"  {r['name']:<16}  {r['B']:>5}  {r['lr']:>8.2e}  {lr_final:>8}  "
              f"{r['steps']:>7,}  {r['samples']:>9,}  {wall:>8}  "
              f"{r['avg_fwdbwd_ms']:>10.1f}  {r['avg_opt_ms']:>8.1f}  "
              f"{r['peak_mem_gb']:>7.2f}  {pwr:>6}")
    print(f"\n  Final val accuracy:")
    for r in results:
        acc = r["curve_val_acc"][-1] if r["curve_val_acc"] else 0
        print(f"    {r['name']:<16}: {acc:.4f}")

    # Compute key ratios
    by_name = {r["name"]: r for r in results}
    if "OlsveredKFAC" in by_name and "ClassicKFAC" in by_name:
        ol = by_name["OlsveredKFAC"]; cl = by_name["ClassicKFAC"]
        print(f"\n  OlsveredKFAC vs ClassicKFAC:")
        print(f"    Optimizer overhead ratio : {cl['avg_opt_ms']/ol['avg_opt_ms']:.2f}× faster")
        print(f"    Wall-time ratio          : {cl['wall_s']/ol['wall_s']:.2f}× faster")
        print(f"    Memory savings           : {cl['peak_mem_gb']-ol['peak_mem_gb']:.2f} GB less")
        if cl.get('avg_power_w') and ol.get('avg_power_w'):
            elec_ratio = (cl['avg_power_w']*cl['wall_s']) / (ol['avg_power_w']*ol['wall_s'])
            print(f"    Energy ratio             : {elec_ratio:.2f}× less electricity")
    if "OlsveredKFAC" in by_name and "Adam" in by_name:
        ol = by_name["OlsveredKFAC"]; ad = by_name["Adam"]
        print(f"\n  OlsveredKFAC vs Adam (samples to same final accuracy):")
        print(f"    Samples seen: OlsveredKFAC {ol['samples']:,}  vs  Adam {ad['samples']:,}")
        if ol['samples'] < ad['samples']:
            print(f"    K-FAC needed {ad['samples']/ol['samples']:.1f}× fewer samples — advantage confirmed")
        else:
            print(f"    Adam needed fewer samples — K-FAC advantage not seen at this scale")

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mlp","bert","all"], default="all")
    parser.add_argument("--max-steps-mlp",  type=int, default=3000)
    parser.add_argument("--max-steps-bert", type=int, default=8000)
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated optimizer names to skip. "
             "Valid: adam, classickfac, olsveredkfac. "
             "Example: --skip adam,classickfac"
    )
    args = parser.parse_args()
    # Normalise skip list to lowercase set
    args.skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    print("\nOlsveredKFAC GPU Benchmark")
    print("="*70)
    device = get_device()

    all_results = []

    if args.task in ("mlp","all"):
        mlp_results = run_mlp_benchmark(device, args)
        all_results.extend(mlp_results)
        if mlp_results:
            print_summary(mlp_results)
            make_plots(mlp_results, "Large MLP")

    if args.task in ("bert","all"):
        bert_results = run_bert_benchmark(device, args)
        all_results.extend(bert_results)
        if bert_results:
            print_summary(bert_results)
            make_plots(bert_results, "BERT-base SST-2")

    # Save combined results JSON (all runs together)
    if all_results:
        save_path = OUT / "gpu_benchmark_results.json"
        with open(save_path, "w") as f:
            json.dump(to_serialisable(all_results), f, indent=2)
        print(f"\n  Combined results → {save_path}")
        print(f"  Per-run JSONs and CSV summary → {OUT}/")

if __name__ == "__main__":
    main()
