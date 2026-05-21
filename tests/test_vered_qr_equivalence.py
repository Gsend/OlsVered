"""
tests/test_vered_qr_equivalence.py

Verification harness for the Vered QR GPU optimization.

Runs a baseline VeredKFAC implementation and a candidate (optimized) one on
identical inputs, then compares observable outputs to verify they produce
numerically equivalent results.

Use cases:

(1) BEFORE any optimization is implemented (no-op check):
    Both factories build the same VeredKFAC class.  The harness should report
    all checks pass with epsilon-level differences (FP non-determinism only).
    This validates the harness logic itself.

(2) AFTER implementing batched QR (or another optimization):
    `make_candidate_optimizer()` is updated to construct the optimized version.
    The harness then catches any numerical regression at fp32 tolerance.

What gets compared, in increasing strictness:

    1. Loss trajectory:    per-step training loss values, max-relative-error
                           over the run.  Tolerance: 0.1% (1e-3 relative).

    2. dW direction:       cosine-similarity of the per-step parameter update
                           (param.data after - before) per layer.
                           Tolerance: > 0.9999 in fp32.

    3. dW magnitude:       L2 norm ratio of the per-step update per layer.
                           Tolerance: within 1% (0.99 < ratio < 1.01).

    4. Final-state diff:   element-wise relative diff of all model parameters
                           after N steps.  Tolerance: < 1e-4 in fp32.

The R-factor comparison requires inspecting VeredKFAC internals; it is
included as a "soft" check (warns if internals aren't accessible) and can
be enabled by adjusting `_extract_r_factors()` to match the actual class.

Usage:
    python tests/test_vered_qr_equivalence.py                      # full run, default 100 steps
    python tests/test_vered_qr_equivalence.py --n-steps 50         # quicker, less coverage
    python tests/test_vered_qr_equivalence.py --strict             # fail on any soft warning
    python tests/test_vered_qr_equivalence.py --device cuda        # default; set to 'cpu' for pure determinism
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------- Tiny model used for verification ------------------------------
# Deliberately small so each step is ~50-100 ms and the harness completes
# in under 1 minute.  Mirrors SmallGPT's structure but at smaller scale.

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=False)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (b, n_heads, s, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, s, d)
        x = x + self.attn_out(ctx)
        x = x + self.ffn2(F.gelu(self.ffn1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 512, seq_len: int = 32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


def make_synthetic_data(vocab: int, n_batches: int, batch_size: int,
                        seq_len: int, seed: int, device: str):
    """Generate a fixed, reproducible sequence of input batches."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randint(0, vocab, (batch_size, seq_len + 1), generator=g)
        # Last token is the target for the previous; teacher-forcing
        batches.append((x[:, :-1].to(device), x[:, 1:].to(device)))
    return batches


# ---------- Optimizer factories -------------------------------------------
# Edit these when implementing a new optimization to point at the new class.

def make_baseline_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Build the baseline VeredKFAC optimizer.  This is the reference; never
    change unless the baseline implementation itself changes."""
    from optimizer.vered_kfac import VeredKFAC
    return VeredKFAC(
        model,
        lr=8e-3,
        damping=1e-4,
        momentum=0.3,
        gamma=0.9,
        grad_clip=300.0,
    )


def make_candidate_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Build the candidate (potentially optimized) VeredKFAC.

    BEFORE any optimization: returns the same as make_baseline_optimizer.
    This makes the harness a no-op check that validates its own logic.

    AFTER implementing batched QR: change this to construct the optimized
    version (e.g., VeredKFAC(..., use_batched_qr=True) or a separate class).
    """
    from optimizer.vered_kfac import VeredKFAC
    return VeredKFAC(
        model,
        lr=8e-3,
        damping=1e-4,
        momentum=0.3,
        gamma=0.9,
        grad_clip=300.0,
    )


def _extract_r_factors(opt) -> Optional[Dict[str, torch.Tensor]]:
    """Pull out per-layer R-factors from a VeredKFAC optimizer for comparison.

    This is a SOFT check — if VeredKFAC's internals don't expose R-factors
    in the form expected here, returns None and the harness skips this check.

    Edit this function once you know exactly how VeredKFAC stores R-factors
    (e.g., opt.state[layer]['R_X'] and opt.state[layer]['R_G']).
    """
    factors = {}
    try:
        # Try common attribute paths.  Adjust to actual VeredKFAC internals.
        for layer_name, layer_state in (opt.state.items() if hasattr(opt, 'state') else []):
            if isinstance(layer_state, dict):
                if 'R_X' in layer_state:
                    factors[f"{layer_name}_R_X"] = layer_state['R_X'].detach().clone()
                if 'R_G' in layer_state:
                    factors[f"{layer_name}_R_G"] = layer_state['R_G'].detach().clone()
    except Exception:
        return None
    return factors if factors else None


# ---------- Comparison primitives -----------------------------------------

@dataclass
class StepCapture:
    """Per-step state captured for comparison."""
    loss: float
    dW: Dict[str, torch.Tensor] = field(default_factory=dict)
    R: Dict[str, torch.Tensor] = field(default_factory=dict)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    a_norm = a_flat.norm()
    b_norm = b_flat.norm()
    if a_norm < 1e-12 or b_norm < 1e-12:
        return 1.0  # both ~zero
    return (a_flat @ b_flat / (a_norm * b_norm)).item()


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """max-relative-error: max |a - b| / max(|a|, |b|, eps)."""
    diff = (a - b).abs()
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
    return (diff / denom).max().item()


def canonical_R(R: torch.Tensor) -> torch.Tensor:
    """Canonicalize R-factor sign by ensuring positive diagonal.

    QR is unique only up to sign per row of R.  Multiplying row i by -1
    while flipping column i of Q gives an equivalent factorization.  Forcing
    diag(R) > 0 makes R unique and comparable across implementations.
    """
    if R.dim() < 2:
        return R
    diag_signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    diag_signs = torch.where(diag_signs == 0, torch.ones_like(diag_signs), diag_signs)
    # Multiply each row of R by its diagonal sign
    return R * diag_signs.unsqueeze(-1)


# ---------- Run a model with captured state ------------------------------

def run_with_capture(model, optimizer, data, n_steps: int,
                     pad_id: int = 0) -> Tuple[List[StepCapture], List[Dict[str, torch.Tensor]]]:
    captures: List[StepCapture] = []
    final_state: List[Dict[str, torch.Tensor]] = []

    model.train()
    for step in range(n_steps):
        x, y = data[step % len(data)]
        # Snapshot params before step
        pre = {n: p.detach().clone() for n, p in model.named_parameters()}

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_id,
        )
        loss.backward()
        optimizer.step()

        # Compute dW per layer
        dW = {n: (p.detach() - pre[n]).clone()
              for n, p in model.named_parameters()}

        # Try to grab R-factors (soft check)
        R_factors = _extract_r_factors(optimizer) or {}

        captures.append(StepCapture(
            loss=loss.item(),
            dW=dW,
            R=R_factors,
        ))

    final_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    return captures, final_state


# ---------- The harness itself --------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    detail: str = ""


def verify_qr_equivalence(
    n_steps: int = 100,
    seed: int = 42,
    device: str = "cuda",
    vocab: int = 256,
    batch_size: int = 8,
    seq_len: int = 32,
    d_model: int = 128,
    n_layers: int = 2,
    tolerance_loss_pct: float = 1e-3,
    tolerance_dW_cos: float = 0.9999,
    tolerance_dW_norm_ratio: float = 0.01,
    tolerance_param_rel: float = 1e-4,
    tolerance_R_rel: float = 1e-5,
    strict: bool = False,
) -> List[CheckResult]:
    """Run the full equivalence verification.  Returns a list of CheckResult.

    Use strict=True to require all soft checks to be wired up; otherwise
    missing soft checks are warnings, not failures.
    """

    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU")
        device = "cpu"

    # CRITICAL: deterministic operations.  GPU FP non-determinism may still
    # introduce small differences even at the same seed; we expect them to
    # be near zero for the no-op check.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate fixed data once; both runs see identical batches.
    data = make_synthetic_data(vocab, max(20, n_steps // 5),
                                batch_size, seq_len, seed, device)

    print(f"\n[run] baseline:  building model + optimizer (seed={seed})")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_b = TinyGPT(vocab=vocab, d_model=d_model, n_layers=n_layers,
                      seq_len=seq_len).to(device)
    opt_b = make_baseline_optimizer(model_b)

    t0 = time.perf_counter()
    captures_b, final_b = run_with_capture(model_b, opt_b, data, n_steps)
    t_b = time.perf_counter() - t0
    print(f"[run] baseline:  {n_steps} steps in {t_b:.1f}s "
          f"({t_b * 1000 / n_steps:.0f} ms/step)")

    print(f"\n[run] candidate: building model + optimizer (seed={seed})")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_c = TinyGPT(vocab=vocab, d_model=d_model, n_layers=n_layers,
                      seq_len=seq_len).to(device)
    opt_c = make_candidate_optimizer(model_c)

    t0 = time.perf_counter()
    captures_c, final_c = run_with_capture(model_c, opt_c, data, n_steps)
    t_c = time.perf_counter() - t0
    print(f"[run] candidate: {n_steps} steps in {t_c:.1f}s "
          f"({t_c * 1000 / n_steps:.0f} ms/step)")

    if t_b > 0:
        speedup = t_b / max(t_c, 1e-6)
        print(f"[perf] candidate vs baseline speedup: {speedup:.2f}x")

    # ---- Comparisons ----
    results: List[CheckResult] = []

    # Check 1: loss trajectory
    losses_b = torch.tensor([c.loss for c in captures_b])
    losses_c = torch.tensor([c.loss for c in captures_c])
    rel_loss_err = ((losses_b - losses_c).abs() /
                    losses_b.abs().clamp_min(1e-12)).max().item()
    results.append(CheckResult(
        name="Loss trajectory (max relative error)",
        passed=rel_loss_err < tolerance_loss_pct,
        metric=rel_loss_err,
        threshold=tolerance_loss_pct,
        detail=f"baseline final={losses_b[-1]:.4f}, candidate final={losses_c[-1]:.4f}",
    ))

    # Check 2: dW direction (per layer, per step) - take worst-case cosine sim
    worst_cos = 1.0
    worst_step = -1
    worst_layer = ""
    for step in range(n_steps):
        for layer_name in captures_b[step].dW:
            if layer_name not in captures_c[step].dW:
                continue
            dW_b = captures_b[step].dW[layer_name]
            dW_c = captures_c[step].dW[layer_name]
            if dW_b.norm() < 1e-10 and dW_c.norm() < 1e-10:
                continue
            cs = cosine_sim(dW_b, dW_c)
            if cs < worst_cos:
                worst_cos = cs
                worst_step = step
                worst_layer = layer_name
    results.append(CheckResult(
        name="dW direction (worst per-layer per-step cosine sim)",
        passed=worst_cos > tolerance_dW_cos,
        metric=worst_cos,
        threshold=tolerance_dW_cos,
        detail=f"worst at step {worst_step}, layer {worst_layer}",
    ))

    # Check 3: dW magnitude ratio (per layer, averaged)
    norm_ratios = []
    for step in range(n_steps):
        for layer_name in captures_b[step].dW:
            if layer_name not in captures_c[step].dW:
                continue
            n_b = captures_b[step].dW[layer_name].norm().item()
            n_c = captures_c[step].dW[layer_name].norm().item()
            if n_b < 1e-10 and n_c < 1e-10:
                continue
            if n_b < 1e-10:
                continue
            norm_ratios.append(n_c / n_b)
    if norm_ratios:
        ratios = torch.tensor(norm_ratios)
        worst_ratio_dev = max((ratios - 1).abs().max().item(), 0.0)
    else:
        worst_ratio_dev = 0.0
    results.append(CheckResult(
        name="dW magnitude (worst |ratio - 1| across all layers/steps)",
        passed=worst_ratio_dev < tolerance_dW_norm_ratio,
        metric=worst_ratio_dev,
        threshold=tolerance_dW_norm_ratio,
        detail=f"sampled {len(norm_ratios)} (layer, step) pairs",
    ))

    # Check 4: final-state element-wise relative diff
    max_param_rel = 0.0
    worst_param = ""
    for name in final_b:
        if name not in final_c:
            continue
        rel = relative_error(final_b[name], final_c[name])
        if rel > max_param_rel:
            max_param_rel = rel
            worst_param = name
    results.append(CheckResult(
        name="Final params (worst per-element relative diff)",
        passed=max_param_rel < tolerance_param_rel,
        metric=max_param_rel,
        threshold=tolerance_param_rel,
        detail=f"worst layer: {worst_param}",
    ))

    # Check 5 (soft): R-factor diff per layer per step
    have_R = any(c.R for c in captures_b) and any(c.R for c in captures_c)
    if have_R:
        worst_R_rel = 0.0
        worst_R_step = -1
        worst_R_layer = ""
        for step in range(n_steps):
            for layer_name in captures_b[step].R:
                if layer_name not in captures_c[step].R:
                    continue
                R_b = canonical_R(captures_b[step].R[layer_name])
                R_c = canonical_R(captures_c[step].R[layer_name])
                if R_b.shape != R_c.shape:
                    continue
                rel = relative_error(R_b, R_c)
                if rel > worst_R_rel:
                    worst_R_rel = rel
                    worst_R_step = step
                    worst_R_layer = layer_name
        results.append(CheckResult(
            name="R-factor (worst per-layer per-step relative diff, sign-canonical)",
            passed=worst_R_rel < tolerance_R_rel,
            metric=worst_R_rel,
            threshold=tolerance_R_rel,
            detail=f"worst at step {worst_R_step}, layer {worst_R_layer}",
        ))
    else:
        msg = "R-factor extraction not wired up (_extract_r_factors returned empty)"
        if strict:
            results.append(CheckResult(
                name="R-factor (NOT WIRED UP)",
                passed=False,
                metric=0.0,
                threshold=tolerance_R_rel,
                detail=msg,
            ))
        else:
            print(f"\n[soft warn] {msg}")
            print("            Edit _extract_r_factors() to enable R-factor comparison.")

    return results


# ---------- Reporting ------------------------------------------------------

def print_report(results: List[CheckResult]):
    print()
    print("=" * 76)
    print("  Verification report")
    print("=" * 76)
    width_name = max(len(r.name) for r in results) + 2
    print(f"  {'Check':<{width_name}}  {'metric':>14}  {'threshold':>14}  {'pass'}")
    print("  " + "-" * (width_name + 36))
    for r in results:
        ok = "PASS" if r.passed else "FAIL"
        m = f"{r.metric:.6g}"
        t = f"{r.threshold:.6g}"
        print(f"  {r.name:<{width_name}}  {m:>14}  {t:>14}  {ok}")
        if r.detail:
            print(f"    -> {r.detail}")
    print()
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    if n_pass == n_total:
        print(f"  ALL {n_total} CHECKS PASSED")
    else:
        print(f"  {n_pass}/{n_total} checks passed; {n_total - n_pass} FAILED")
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=100,
                    help="Number of training steps to compare (default: 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--strict", action="store_true",
                    help="Treat missing soft checks (R-factors) as failures")
    ap.add_argument("--save-json", type=str, default=None,
                    help="Optional path to save report as JSON")
    args = ap.parse_args()

    print("=" * 76)
    print("  Vered QR equivalence verification")
    print(f"  steps={args.n_steps}  seed={args.seed}  device={args.device}")
    print("  (no-op check: both factories build the same VeredKFAC by default)")
    print("=" * 76)

    results = verify_qr_equivalence(
        n_steps=args.n_steps,
        seed=args.seed,
        device=args.device,
        strict=args.strict,
    )
    print_report(results)

    if args.save_json:
        out = [
            {"name": r.name, "passed": r.passed,
             "metric": r.metric, "threshold": r.threshold, "detail": r.detail}
            for r in results
        ]
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"  Saved report to {args.save_json}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
