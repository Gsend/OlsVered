"""
Theoretical FLOP / Memory / Wall-Time comparison
  Classical K-FAC  vs  OlsveredKFAC (randomised EVD rank-k)

For a single linear layer  W : d_in → d_out
and a mini-batch of size B, update frequency f, rank k.

Notation
--------
  B        batch size
  d_in     input  features of the layer
  d_out    output features of the layer
  f        factor+inverse update frequency (every f steps)
  k        rank used by OlsveredKFAC
  p        oversampling in the randomised range-finder  (p=10)
  n_iter   power iterations in randomised EVD          (n_iter=2)

FLOPs counted as multiply-adds (each = 2 actual FLOPs for hardware).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ─── per-layer FLOP helpers ────────────────────────────────────────────────

def flops_gram_accum(B, d_in, d_out):
    """X^T X and δ^T δ  – paid every step, identical for both methods."""
    return B * d_in**2 + B * d_out**2          # outer-product sums

def flops_classic_inv(d_in, d_out):
    """Dense Cholesky + triangular solve ≈ d³/3 + d³/3  ≈  d³."""
    # numpy/LAPACK dgehrd-based: ~(2/3)d³ for Cholesky, + (1/3)d³ for solve
    return d_in**3 + d_out**3                  # amortised over f steps

def flops_randomised_evd(d_in, d_out, k, p=10, n_iter=2):
    """
    Halko-Martinsson-Tropp randomised EVD, applied to both A and G.
      Phase 1 (range finder + power iters): (2+4*n_iter)*k * d²
      Phase 2 (small EVD):  O((k+p)³) – negligible
    """
    k_eff = k + p
    phase1_A = (2 + 4 * n_iter) * k_eff * d_in**2
    phase1_G = (2 + 4 * n_iter) * k_eff * d_out**2
    return phase1_A + phase1_G                 # amortised over f steps

def flops_classic_apply(d_in, d_out):
    """G_inv @ grad_W @ A_inv  (two dense matmuls)."""
    return d_out**2 * d_in + d_out * d_in**2   # = d_in*d_out*(d_in+d_out)

def flops_olsvered_apply(d_in, d_out, k):
    """
    Q_G diag Q_G^T  grad_W  Q_A diag Q_A^T
    Four matmuls each O(k * d_in * d_out):
      Q_G^T @ grad_W : k*d_out*d_in
      Q_G  @ result  : d_out*k*d_in
      result @ Q_A   : d_out*d_in*k
      result @ Q_A^T : d_out*k*d_in    (each counted once)
    """
    return 4 * k * d_in * d_out

# ─── per-layer MEMORY helpers (floats = 4 bytes FP32) ─────────────────────

def mem_gram_accum(d_in, d_out):
    """A_sum + G_sum – identical for both methods (always present)."""
    return (d_in**2 + d_out**2)   # in floats

def mem_classic_extra(d_in, d_out):
    """A_inv + G_inv stored between updates."""
    return d_in**2 + d_out**2

def mem_olsvered_extra(d_in, d_out, k):
    """Q_A, lam_A, Q_G, lam_G stored between updates."""
    return k * (d_in + d_out) + 2 * k

# ─── full-model analysis ──────────────────────────────────────────────────

def analyse_model(name, layers, B, f=10, rank_budget=64,
                  adaptive_min_n=128, p=10, n_iter=2):
    """
    layers : list of (d_in, d_out) tuples  (linear layers only)
    Returns dict with per-method totals.
    """
    # Adaptive rank: for small layers use full EVD; otherwise use rank_budget
    def eff_rank(d_in, d_out):
        n = min(d_in, d_out)
        if n < adaptive_min_n:
            return None          # full EVD (same cost as Classic)
        return min(rank_budget, n)

    gram_total   = 0
    cl_inv_total = 0; cl_app_total = 0
    ol_evd_total = 0; ol_app_total = 0
    cl_mem_total = 0; ol_mem_total = 0
    common_mem   = 0

    for d_in, d_out in layers:
        k = eff_rank(d_in, d_out)
        if k is None:                       # full EVD for small layers
            k = min(d_in, d_out)

        gram  = flops_gram_accum(B, d_in, d_out)
        c_inv = flops_classic_inv(d_in, d_out)
        o_evd = flops_randomised_evd(d_in, d_out, k, p, n_iter)
        c_app = flops_classic_apply(d_in, d_out)
        o_app = flops_olsvered_apply(d_in, d_out, k)

        gram_total   += gram
        cl_inv_total += c_inv
        cl_app_total += c_app
        ol_evd_total += o_evd
        ol_app_total += o_app
        common_mem   += mem_gram_accum(d_in, d_out)
        cl_mem_total += mem_classic_extra(d_in, d_out)
        ol_mem_total += mem_olsvered_extra(d_in, d_out, k)

    # Amortised cost per step  =  gram  +  (inv/evd)/f  +  apply
    cl_step = gram_total + cl_inv_total / f + cl_app_total
    ol_step = gram_total + ol_evd_total / f + ol_app_total

    # Memory in MB  (4 bytes per float32)
    common_mb = common_mem * 4 / 1e6
    cl_mb     = (common_mem + cl_mem_total) * 4 / 1e6
    ol_mb     = (common_mem + ol_mem_total) * 4 / 1e6

    return dict(
        model=name, B=B, f=f, rank=rank_budget,
        gram_GF   = gram_total   / 1e9,
        cl_inv_GF = cl_inv_total / 1e9,
        ol_evd_GF = ol_evd_total / 1e9,
        cl_app_GF = cl_app_total / 1e9,
        ol_app_GF = ol_app_total / 1e9,
        cl_step_GF= cl_step / 1e9,
        ol_step_GF= ol_step / 1e9,
        speedup   = cl_step / ol_step,
        cl_mem_MB = cl_mb,
        ol_mem_MB = ol_mb,
        mem_ratio = cl_mb / ol_mb,
    )

# ─── model architectures ─────────────────────────────────────────────────

def bert_base_layers():
    """BERT-base: 12 transformer blocks.
    Each block:
      3 attention projections W_Q, W_K, W_V : 768→768
      1 output projection W_O              : 768→768
      2 FFN projections                    : 768→3072, 3072→768
    """
    block = [(768,768),(768,768),(768,768),(768,768),  # Q,K,V,O
             (768,3072),(3072,768)]                     # FFN
    return block * 12

def gpt2_medium_layers():
    """GPT-2 medium: 24 transformer blocks, d=1024, FFN=4096."""
    block = [(1024,1024)]*4 + [(1024,4096),(4096,1024)]
    return block * 24

def large_mlp_layers():
    """Large MLP: 784→4096→4096→2048→1024→10."""
    return [(784,4096),(4096,4096),(4096,2048),(2048,1024),(1024,10)]

def resnet50_fc_layers():
    """ResNet-50 conv layers approximated as (in_ch*kk, out_ch) linear layers.
    Only the dominant ones: 64 conv layers of various sizes.
    We focus on the big blocks: 256/512/1024/2048-channel bottlenecks.
    Conv k=3×3: equivalent d_in = in_ch*9, d_out = out_ch (spatial dims folded into B).
    """
    layers = []
    # layer 1: 64→64→256 bottleneck ×3
    layers += [(64*1,64),(64*9,64),(64*1,256)] * 3
    # layer 2: 128→128→512 bottleneck ×4
    layers += [(256*1,128),(128*9,128),(128*1,512)] * 4
    # layer 3: 256→256→1024 bottleneck ×6
    layers += [(512*1,256),(256*9,256),(256*1,1024)] * 6
    # layer 4: 512→512→2048 bottleneck ×3
    layers += [(1024*1,512),(512*9,512),(512*1,2048)] * 3
    # final FC
    layers += [(2048,1000)]
    return layers


# ─── run analyses ────────────────────────────────────────────────────────

models = {
    "Large MLP\n(4×4096)":   large_mlp_layers(),
    "BERT-base\n(12L d=768)": bert_base_layers(),
    "GPT-2 Medium\n(24L d=1024)": gpt2_medium_layers(),
    "ResNet-50\n(bottleneck)": resnet50_fc_layers(),
}

batch_sizes = [512, 1024, 2048, 4096]
rank_budget = 64
f_update    = 10

print("=" * 90)
print(f"  THEORETICAL COMPARISON: Classical K-FAC  vs  OlsveredKFAC  (rank={rank_budget}, f={f_update})")
print("=" * 90)

all_results = {}
for model_name, layers in models.items():
    short = model_name.replace("\n", " ")
    print(f"\n{'─'*90}")
    print(f"  Model: {short}   ({len(layers)} linear layers)")
    print(f"{'─'*90}")
    print(f"  {'Batch':>6}  {'Classic GF/step':>16}  {'Olsvered GF/step':>16}  "
          f"{'Speedup':>8}  {'Cl mem MB':>10}  {'Ol mem MB':>10}  {'Mem ratio':>9}")
    print(f"  {'':─<6}  {'':─<16}  {'':─<16}  {'':─<8}  {'':─<10}  {'':─<10}  {'':─<9}")
    rows = []
    for B in batch_sizes:
        r = analyse_model(short, layers, B, f=f_update, rank_budget=rank_budget)
        rows.append(r)
        print(f"  {B:>6}  {r['cl_step_GF']:>16.2f}  {r['ol_step_GF']:>16.2f}  "
              f"  {r['speedup']:>6.1f}×  {r['cl_mem_MB']:>10.1f}  {r['ol_mem_MB']:>10.1f}  "
              f"  {r['mem_ratio']:>7.1f}×")
    all_results[model_name] = rows

# ─── breakdown at B=1024 ─────────────────────────────────────────────────
print(f"\n\n{'=' * 90}")
print(f"  FLOP BREAKDOWN  (B=1024, rank={rank_budget}, f={f_update})")
print("=" * 90)
B_show = 1024
for model_name, layers in models.items():
    short = model_name.replace("\n", " ")
    r = analyse_model(short, layers, B_show, f=f_update, rank_budget=rank_budget)
    total_cl = r['cl_step_GF']
    total_ol = r['ol_step_GF']
    print(f"\n  {short}")
    print(f"    {'Component':<30} {'Classic (GF)':>14}  {'%':>5}   {'Olsvered (GF)':>14}  {'%':>5}")
    print(f"    {'':─<30} {'':─<14}  {'':─<5}   {'':─<14}  {'':─<5}")
    def row(name, cl_v, ol_v):
        print(f"    {name:<30} {cl_v:>14.3f}  {100*cl_v/total_cl:>4.1f}%   "
              f"{ol_v:>14.3f}  {100*ol_v/total_ol:>4.1f}%")
    row("Gram accumulation (shared)", r['gram_GF'], r['gram_GF'])
    row("EVD / Inverse (amortised)", r['cl_inv_GF']/f_update, r['ol_evd_GF']/f_update)
    row("Natural gradient apply",   r['cl_app_GF'], r['ol_app_GF'])
    print(f"    {'TOTAL':─<30} {total_cl:>14.3f}           {total_ol:>14.3f}   "
          f"→  {r['speedup']:.1f}× speedup")
    print(f"    Memory (optimizer state):    Classic={r['cl_mem_MB']:.0f} MB   "
          f"Olsvered={r['ol_mem_MB']:.0f} MB   ({r['mem_ratio']:.1f}× less)")

# ─── wall-time estimate (V100 GPU effective throughput) ──────────────────
print(f"\n\n{'=' * 90}")
print("  ESTIMATED WALL TIME PER STEP  (V100 GPU, 30% peak = ~4.5 TF/s effective)")
print("=" * 90)
TFLOPS_EFF = 4.5e12  # 30% of V100 FP32 peak

# Forward/backward cost (independent of optimizer):
# ~2× parameters × B for a single fwd+bwd pass (rough rule of thumb)
def fwd_bwd_gflops(layers, B):
    total = 0
    for d_in, d_out in layers:
        # fwd: B*d_in*d_out  matmul; bwd: ~2×
        total += 3 * B * d_in * d_out
    return total / 1e9

print(f"\n  {'Model':<28}  {'B':>5}  {'Fwd/Bwd ms':>11}  {'Cl opt ms':>10}  "
      f"{'Ol opt ms':>10}  {'Cl tot ms':>10}  {'Ol tot ms':>10}  {'Walltime ratio':>14}")
print(f"  {'':─<28}  {'':─<5}  {'':─<11}  {'':─<10}  {'':─<10}  {'':─<10}  {'':─<10}  {'':─<14}")

for model_name, layers in models.items():
    short = model_name.replace("\n", " ")
    for B in [512, 1024, 2048]:
        fb  = fwd_bwd_gflops(layers, B) / (TFLOPS_EFF/1e9) * 1e3   # ms
        r   = analyse_model(short, layers, B, f=f_update, rank_budget=rank_budget)
        cl_opt = r['cl_step_GF'] / (TFLOPS_EFF/1e9) * 1e3
        ol_opt = r['ol_step_GF'] / (TFLOPS_EFF/1e9) * 1e3
        cl_tot = fb + cl_opt
        ol_tot = fb + ol_opt
        # ratio: how much slower vs just fwd/bwd (Adam ≈ 0% optimizer overhead for big models)
        ratio = cl_tot / ol_tot
        print(f"  {short:<28}  {B:>5}  {fb:>11.2f}  {cl_opt:>10.2f}  "
              f"{ol_opt:>10.2f}  {cl_tot:>10.2f}  {ol_tot:>10.2f}  {ratio:>13.2f}×")

# ─── plots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Classical K-FAC vs OlsveredKFAC — Theoretical Analysis\n"
             f"(rank={rank_budget}, update freq f={f_update}, adaptive EVD)",
             fontsize=13, fontweight="bold")

colors = {"Classic": "#e05252", "Olsvered": "#3d85c8"}

for ax, (model_name, rows) in zip(axes.flat, all_results.items()):
    Bs = [r["B"] for r in rows]
    cl = [r["cl_step_GF"] for r in rows]
    ol = [r["ol_step_GF"] for r in rows]
    sp = [r["speedup"] for r in rows]

    ax2 = ax.twinx()
    ax.plot(Bs, cl, "o-", color=colors["Classic"],  lw=2, ms=7, label="ClassicKFAC")
    ax.plot(Bs, ol, "s-", color=colors["Olsvered"], lw=2, ms=7, label="OlsveredKFAC")
    ax2.bar([b*1.08 for b in Bs], sp, width=[b*0.15 for b in Bs],
            color="gold", alpha=0.6, label="Speedup ×")
    ax2.set_ylabel("Speedup (GFLOPs ratio ×)", color="goldenrod", fontsize=9)
    ax2.tick_params(axis='y', labelcolor='goldenrod')
    ax2.set_ylim(0, max(sp)*1.4)

    ax.set_title(model_name.replace("\n", "  "), fontsize=11, fontweight="bold")
    ax.set_xlabel("Batch size", fontsize=9)
    ax.set_ylabel("GFLOPs per step (optimizer only)", fontsize=9)
    ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # annotate memory
    cl_mb = rows[-1]["cl_mem_MB"]
    ol_mb = rows[-1]["ol_mem_MB"]
    ax.text(0.5, 0.05,
            f"Memory (B=4096): Classic={cl_mb:.0f}MB  Olsvered={ol_mb:.0f}MB  "
            f"({rows[-1]['mem_ratio']:.1f}× less)",
            transform=ax.transAxes, fontsize=7.5, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig.savefig(OUT / "theoretical_comparison.png", dpi=150, bbox_inches="tight")
print(f"\n  Plot saved → benchmark/results/theoretical_comparison.png")

# ─── memory breakdown table ─────────────────────────────────────────────
print(f"\n\n{'=' * 90}")
print(f"  OPTIMIZER STATE MEMORY (MB)  —  how much GPU RAM the optimizer consumes")
print(f"  (Gram accumulators included; rank={rank_budget})")
print("=" * 90)
print(f"  {'Model':<30}  {'Classic MB':>10}  {'Olsvered MB':>11}  {'Ratio':>7}  {'Savings MB':>10}")
print(f"  {'':─<30}  {'':─<10}  {'':─<11}  {'':─<7}  {'':─<10}")
for model_name, layers in models.items():
    short = model_name.replace("\n", " ")
    r = analyse_model(short, layers, 1024)
    print(f"  {short:<30}  {r['cl_mem_MB']:>10.1f}  {r['ol_mem_MB']:>11.1f}  "
          f"{r['mem_ratio']:>6.1f}×  {r['cl_mem_MB']-r['ol_mem_MB']:>10.1f}")

print("\nDone.")
