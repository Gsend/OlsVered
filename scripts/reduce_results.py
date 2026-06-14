"""
scripts/reduce_results.py

Trim the size of benchmark/results/ so the repo stays under ~25 MB.

Three-phase strategy:
  Phase 1: delete files from experiments that were dropped (PINN bench).
  Phase 2: downsample `per_step` in the multi-seed result JSONs by 10x
           (keep every 10th step).  Loss curves still resolve cleanly at
           plot-axis resolution but file size drops by ~10x.
  Phase 3: delete intermediate screen JSONs whose winners are already
           captured in the surviving multi-seed runs.

Each phase reports before/after sizes.  Use --dry-run to preview.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmark" / "results"


def _human_size(bytes_: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def _total_size(paths) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


# ---- Phase 1: delete dropped-experiment files ------------------------------

DROPPED_PATTERNS = [
    "pinn_bench_*.json",        # PINN bench dropped (capture-mode test only)
    "pinn_burgers_*.json",      # earlier PINN intermediates
    "pinn_screen_*.json",       # PINN screen results
    "pinn_winners.json",        # PINN tuner output
]


def phase1_delete_dropped(dry_run=False):
    print("\n=== Phase 1: delete dropped-experiment files ===")
    total = 0
    n = 0
    for pat in DROPPED_PATTERNS:
        for p in RES.glob(pat):
            sz = p.stat().st_size
            total += sz
            n += 1
            if not dry_run:
                p.unlink()
            print(f"  {'(dry) ' if dry_run else ''}rm  {p.name:<50} {_human_size(sz):>10}")
    print(f"  Phase 1 total: {n} files, {_human_size(total)}")
    return total


# ---- Phase 2: downsample per_step in multi-seed JSONs ----------------------

DOWNSAMPLE_PATTERNS = [
    "ae_mnist_fp32_*_seed*.json",
    "ae_mnist_bf16_*_seed*.json",
    "per_step_4way_*.json",
    "adamw_tune_*.json",
    "kfac_wd_*.json",
]
DOWNSAMPLE_RATE = 10   # keep every Nth step


def phase2_downsample(dry_run=False):
    print(f"\n=== Phase 2: downsample per_step by {DOWNSAMPLE_RATE}x ===")
    total_before = 0
    total_after = 0
    n = 0
    for pat in DOWNSAMPLE_PATTERNS:
        for p in RES.glob(pat):
            try:
                # Tolerant load (some legacy files have trailing junk)
                text = p.read_text()
                try:
                    d = json.loads(text)
                except json.JSONDecodeError:
                    d, _ = json.JSONDecoder().raw_decode(text)
            except Exception as e:
                print(f"  ? skip {p.name}: {e}")
                continue
            recs = d.get("per_step", [])
            if not recs:
                continue
            sz_before = p.stat().st_size
            # Keep every Nth record (preserve first and last)
            kept = recs[::DOWNSAMPLE_RATE]
            if kept[-1] != recs[-1]:
                kept.append(recs[-1])
            d["per_step"] = kept
            d["per_step_downsample_rate"] = DOWNSAMPLE_RATE
            new_text = json.dumps(d, indent=2, default=str)
            sz_after = len(new_text.encode("utf-8"))
            total_before += sz_before
            total_after += sz_after
            n += 1
            if not dry_run:
                p.write_text(new_text)
            print(f"  {'(dry) ' if dry_run else ''}{p.name:<50} "
                  f"{_human_size(sz_before)} -> {_human_size(sz_after)}  "
                  f"({len(recs)} -> {len(kept)} steps)")
    saved = total_before - total_after
    print(f"  Phase 2 total: {n} files, {_human_size(total_before)} -> "
          f"{_human_size(total_after)}  (saved {_human_size(saved)})")
    return saved


# ---- Phase 3: delete obsolete intermediate screens -------------------------

OBSOLETE_PATTERNS = [
    "ae_mnist_screen_*.json",        # AE screen 1 - winners captured in main sweep
    "ae_mnist_screen2_*.json",       # AE screen 2 - same
    "ae_mnist_screen3_*.json",       # AE screen 3 - same
    "ae_mnist_adamw_screen_*.json",  # AdamW screen - same
    "ae_mnist_adamw_screen2_*.json", # AdamW screen 2 - same
    "ae_mnist_adamw_bf16_screen_*.json",  # AdamW bf16 screen
    "vered_2dscreen_*.json",         # Old transformer 2D screens
    "classic_2dscreen_*.json",
    "vered_grid_*.json",
    "classic_grid_*.json",
    "vered_clip_*.json",
    "vered_damp_*.json",
    "vered_mom_*.json",
    "vered_gamma_*.json",
    "vered_wgso_*.json",
]


def phase3_delete_obsolete(dry_run=False):
    print("\n=== Phase 3: delete obsolete intermediate screens ===")
    total = 0
    n = 0
    for pat in OBSOLETE_PATTERNS:
        for p in RES.glob(pat):
            sz = p.stat().st_size
            total += sz
            n += 1
            if not dry_run:
                p.unlink()
    print(f"  {'(dry) ' if dry_run else ''}removed {n} obsolete screen files, "
          f"{_human_size(total)}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="preview without modifying")
    parser.add_argument("--skip-phase3", action="store_true",
                         help="keep intermediate screen files (safer)")
    args = parser.parse_args()

    print(f"Results folder: {RES}")
    before = _total_size(RES.glob("*.json")) + _total_size(RES.rglob("*.png"))
    print(f"Initial size: {_human_size(before)}")

    s1 = phase1_delete_dropped(args.dry_run)
    s2 = phase2_downsample(args.dry_run)
    s3 = 0 if args.skip_phase3 else phase3_delete_obsolete(args.dry_run)

    after = _total_size(RES.glob("*.json")) + _total_size(RES.rglob("*.png"))
    print(f"\nFinal size: {_human_size(after)} "
          f"(saved {_human_size(before - after)})")
    if args.dry_run:
        print("\n[DRY RUN] — no files were modified.  Re-run without --dry-run "
              "to apply.")


if __name__ == "__main__":
    main()
