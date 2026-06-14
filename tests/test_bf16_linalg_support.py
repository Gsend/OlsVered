"""
tests/test_bf16_linalg_support.py

Probe PyTorch + cuSOLVER for native bf16 support on the linear-algebra
operations we need for true-bf16 Vered K-FAC:
  - torch.linalg.qr
  - torch.linalg.solve_triangular
  - torch.linalg.cholesky
  - torch.linalg.inv (for Classic K-FAC control)

For each op, we test on CPU and CUDA (if available):
  1. Does the call succeed when given bf16 inputs?
  2. Is the output actually bf16 (or did it silently upcast)?
  3. Does the result match the fp32 result within bf16 precision?

Run:    python tests\test_bf16_linalg_support.py
Output: a support matrix and a "GO / NO-GO" verdict per device.
"""
from __future__ import annotations
import sys, traceback
import torch


def _check(name, fn, dev_name):
    """Run fn(), classify the result.  Returns ('ok'|'silent_upcast'|'fail', detail)."""
    try:
        result = fn()
        # Normalize: if fn returned a single tensor or tuple
        outs = result if isinstance(result, tuple) else (result,)
        dtypes = [t.dtype for t in outs if isinstance(t, torch.Tensor)]
        if not dtypes:
            return ("unknown", "no tensor output")
        if all(d == torch.bfloat16 for d in dtypes):
            return ("ok", f"native bf16, dtypes={dtypes}")
        if any(d == torch.float32 for d in dtypes):
            return ("silent_upcast", f"upcast to fp32, dtypes={dtypes}")
        return ("unknown", f"unexpected dtypes={dtypes}")
    except (NotImplementedError, RuntimeError) as e:
        return ("fail", f"{type(e).__name__}: {str(e)[:150]}")
    except Exception as e:
        return ("fail", f"{type(e).__name__}: {str(e)[:150]}")


def probe_device(dev):
    dev_name = str(dev)
    print(f"\n=== {dev_name} ===")
    torch.manual_seed(42)
    p, n = 256, 32

    # Build well-conditioned bf16 inputs to avoid spurious errors from
    # ill-conditioning at bf16 precision.
    A_full = torch.randn(p, n, device=dev, dtype=torch.float32)
    A_bf16 = A_full.to(torch.bfloat16)
    # Symmetric positive-definite Gram (for Cholesky and inv tests)
    SPD_full = A_full.T @ A_full + torch.eye(n, device=dev)
    SPD_bf16 = SPD_full.to(torch.bfloat16)
    # Upper-triangular for triangular-solve test
    R_full = torch.triu(torch.randn(n, n, device=dev, dtype=torch.float32))
    R_full = R_full + torch.eye(n, device=dev) * 2.0   # ensure unit-diagonal-ish
    R_bf16 = R_full.to(torch.bfloat16)
    b_bf16 = torch.randn(n, 4, device=dev, dtype=torch.bfloat16)

    results = []

    # 1. torch.linalg.qr (bf16 input)
    status, detail = _check("qr", lambda: torch.linalg.qr(A_bf16, mode="reduced"),
                              dev_name)
    print(f"  qr(bf16 input)               -> {status:>14}   {detail}")
    results.append(("qr", status))

    # 2. torch.linalg.solve_triangular (bf16 R, bf16 b)
    status, detail = _check("solve_triangular",
        lambda: torch.linalg.solve_triangular(R_bf16, b_bf16, upper=True),
        dev_name)
    print(f"  solve_triangular(bf16)       -> {status:>14}   {detail}")
    results.append(("solve_triangular", status))

    # 3. torch.linalg.cholesky (bf16 SPD)
    status, detail = _check("cholesky", lambda: torch.linalg.cholesky(SPD_bf16),
                              dev_name)
    print(f"  cholesky(bf16 SPD)           -> {status:>14}   {detail}")
    results.append(("cholesky", status))

    # 4. torch.linalg.inv (bf16 SPD)  — for Classic K-FAC control
    status, detail = _check("inv", lambda: torch.linalg.inv(SPD_bf16), dev_name)
    print(f"  inv(bf16 SPD)                -> {status:>14}   {detail}")
    results.append(("inv", status))

    # 5. plain bf16 matmul (sanity — should always work)
    status, detail = _check("matmul",
        lambda: A_bf16.T @ A_bf16, dev_name)
    print(f"  matmul(bf16 @ bf16)          -> {status:>14}   {detail}")
    results.append(("matmul", status))

    # 6. fp16 (for comparison — cuSOLVER often supports fp16 separately)
    A_fp16 = A_full.to(torch.float16) if dev.type == "cuda" else None
    if A_fp16 is not None:
        status, detail = _check("qr_fp16",
            lambda: torch.linalg.qr(A_fp16, mode="reduced"), dev_name)
        print(f"  qr(fp16 input) [for compare] -> {status:>14}   {detail}")
        results.append(("qr_fp16", status))

    # Verdict for this device
    needed = {"qr", "solve_triangular", "cholesky", "inv"}
    by_op = dict(results)
    ok_count = sum(1 for op in needed if by_op.get(op) == "ok")
    print(f"\n  Verdict: {ok_count}/{len(needed)} required ops have native bf16 on {dev_name}")
    if ok_count == len(needed):
        print(f"  → GO: native bf16 K-FAC is one-line implementation away")
    elif ok_count >= 2:
        print(f"  → PARTIAL: need to hand-roll {len(needed) - ok_count} op(s); others come free")
    else:
        print(f"  → NO-GO on native path: need hand-rolled bf16 implementations for all critical ops")
    return results


def main():
    print(f"PyTorch version:         {torch.__version__}")
    print(f"CUDA available:          {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch):  {torch.version.cuda}")
        print(f"GPU:                     {torch.cuda.get_device_name(0)}")
        cc = torch.cuda.get_device_capability(0)
        print(f"Compute capability:      {cc[0]}.{cc[1]}  "
              f"(tensor cores require ≥7.0; bf16 tensor cores require ≥8.0)")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    all_results = {}
    for dev in devices:
        all_results[str(dev)] = probe_device(dev)

    # Cross-device summary
    print("\n" + "=" * 72)
    print("Cross-device support matrix")
    print("=" * 72)
    ops = ["qr", "solve_triangular", "cholesky", "inv", "matmul"]
    print(f"  {'op':>20}  " + "  ".join(f"{str(d):>10}" for d in devices))
    by_dev = {str(d): dict(all_results[str(d)]) for d in devices}
    for op in ops:
        cells = [by_dev[str(d)].get(op, "—") for d in devices]
        print(f"  {op:>20}  " + "  ".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
