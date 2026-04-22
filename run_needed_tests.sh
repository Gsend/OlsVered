#!/usr/bin/env bash
# =============================================================================
#  run_needed_tests.sh  —  Re-run all benchmark tests that need fresh results
#
#  What this script addresses:
#    1. bert_olsveredkfac  — MISSING (crashed before JSON save, damping bug fixed)
#    2. bert_classickfac   — STALE   (pre-hooks-fix run at 221ms/step)
#    3. All other tasks    — NO_HW_RECORD (run before hardware provenance was added)
#
#  Fixes included in this run vs prior results:
#    - Hardware provenance stamped into every result JSON ("hw" field)
#    - BERT OlsSmKFAC damping bug fixed: gradual cosine decay post-threshold
#      instead of instantaneous 15x drop that caused model collapse to ~49% acc
#    - Madar LU-solve path active for small layers (lu_max_dim=512):
#      more stable conditioning, no explicit matrix inverse formed
#    - Transformer uses best LRs from prior sweep (0.008 for both K-FAC optimizers)
#
#  Usage:
#    bash run_needed_tests.sh              # run all 4 tasks (recommended)
#    bash run_needed_tests.sh --bert-only  # run only missing/stale BERT tests
#    bash run_needed_tests.sh --no-setup   # skip env setup (already installed)
#    bash run_needed_tests.sh --dry-run    # print what would run, don't execute
#
#  Expected wall time (RTX PRO 6000 / A100 80GB):
#    --all:       ~90 min  (MLP ~6m + CIFAR ~10m + Transformer ~20m + BERT ~55m)
#    --bert-only: ~40 min  (OlsSmKFAC ~15m + ClassicKFAC ~15m + Adam skipped)
#
#  Requirements:
#    - NVIDIA GPU with >= 32 GB VRAM  (ClassicKFAC BERT peaks at 25.7 GB)
#    - 32 GB RAM, 4+ CPU cores
#    - Internet access for HuggingFace downloads (BERT, SST-2, WikiText-2)
#    - If HF is blocked: pre-cache model before running
#        python3 -c "from transformers import BertTokenizer, BertForSequenceClassification; \
#                    BertTokenizer.from_pretrained('bert-base-uncased'); \
#                    BertForSequenceClassification.from_pretrained('bert-base-uncased')"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmark/results"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${CYAN}══ $* ══${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
BERT_ONLY=false
RUN_SETUP=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bert-only)  BERT_ONLY=true;  shift ;;
        --no-setup)   RUN_SETUP=false; shift ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        --help|-h)
            sed -n '3,36p' "$0" | sed 's/^#  \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1  (use --help)" >&2; exit 1 ;;
    esac
done

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}OlsVeredKFAC — Needed Tests Runner${NC}"
echo "============================================="
echo "  Mode:        $([ "$BERT_ONLY" = true ] && echo 'BERT only (missing + stale)' || echo 'Full benchmark (all 4 tasks)')"
echo "  Setup:       $([ "$RUN_SETUP" = true ] && echo 'enabled' || echo 'skipped (--no-setup)')"
echo "  Dry run:     $([ "$DRY_RUN"   = true ] && echo 'YES — will not execute' || echo 'no')"
echo ""

# ── Pre-flight: report what is missing / stale ────────────────────────────────
section "Result audit"

python3 - <<'PYEOF'
import json
from pathlib import Path

base = Path("benchmark/results")
rows = [
    ("mlp",         "adam",         "OK — will re-run for hw record"),
    ("mlp",         "olsveredkfac", "OK — will re-run for hw record"),
    ("mlp",         "classickfac",  "OK — will re-run for hw record"),
    ("cifar",       "adam",         "OK — will re-run for hw record"),
    ("cifar",       "olsveredkfac", "OK — will re-run for hw record"),
    ("cifar",       "classickfac",  "OK — will re-run for hw record"),
    ("transformer", "adam",         "OK — will re-run for hw record"),
    ("transformer", "olsveredkfac", "OK — will re-run for hw record"),
    ("transformer", "classickfac",  "OK — will re-run for hw record"),
    ("bert",        "adam",         "OK — will re-run for hw record"),
    ("bert",        "olsveredkfac", "MISSING — first successful run"),
    ("bert",        "classickfac",  "STALE   — pre-hooks-fix (221ms/step)"),
]
for task, opt, note in rows:
    fpath = base / f"{task}_{opt}_result.json"
    exists = "✓" if fpath.exists() else "✗"
    print(f"  {exists}  {task:<12} {opt:<14}  {note}")
PYEOF

echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    warn "Dry run — exiting without making changes."
    exit 0
fi

# ── Step 1: Clear stale checkpoints ───────────────────────────────────────────
section "Clearing stale checkpoints"

if [[ "$BERT_ONLY" == "true" ]]; then
    # Only clear BERT checkpoints (keep MLP/CIFAR/Transformer if they exist)
    info "Removing stale BERT checkpoints (OlsSmKFAC + ClassicKFAC) ..."
    rm -fv "${RESULTS_DIR}"/bert_ckpt_olssmkfac_*.pt   2>/dev/null || true
    rm -fv "${RESULTS_DIR}"/bert_ckpt_classickfac_*.pt 2>/dev/null || true
    info "Keeping bert_adam_result.json (result is valid, has hw record from this run)"
else
    # Full re-run: clear ALL BERT checkpoints so every optimizer starts fresh
    info "Removing all BERT checkpoints for a clean full run ..."
    rm -fv "${RESULTS_DIR}"/bert_ckpt_*.pt 2>/dev/null || true
    info "Prior result JSONs are preserved; they will be overwritten after each run."
fi

# ── Step 2: Build run commands ────────────────────────────────────────────────
section "Preparing run"

# Best transformer LRs from prior sweep (lr=0.008 beat all others for both K-FAC optimizers)
LR_OLS_TRANSFORMER="0.008"
LR_CLS_TRANSFORMER="0.008"

SETUP_FLAG=""
[[ "$RUN_SETUP" == "false" ]] && SETUP_FLAG="--no-setup"

if [[ "$BERT_ONLY" == "true" ]]; then
    # Adam BERT result is fine — skip it, run only the two K-FAC optimizers
    CMD="bash run_benchmark.sh --task bert --skip adam ${SETUP_FLAG}"
    info "Will run: ${CMD}"
    info "Expected wall time: ~40 min"
else
    CMD="bash run_benchmark.sh --task all \
  --lr-ols-transformer ${LR_OLS_TRANSFORMER} \
  --lr-cls-transformer ${LR_CLS_TRANSFORMER} \
  ${SETUP_FLAG}"
    info "Will run: ${CMD}"
    info "Expected wall time: ~90 min"
    info "  MLP:         ~6 min"
    info "  CIFAR-10:    ~10 min"
    info "  Transformer: ~20 min  (lr=${LR_OLS_TRANSFORMER} for both K-FAC)"
    info "  BERT:        ~55 min  (Adam ~15m + OlsSmKFAC ~15m + ClassicKFAC ~25m)"
fi

echo ""
info "Results will be saved incrementally to benchmark/results/"
info "Each JSON will include a 'hw' block with full pod hardware info."
info "Each result is git-committed and pushed immediately after saving."
echo ""
info "If your SSH connection drops, reconnect and run:"
info "  tmux attach -t benchmark"
echo ""

# ── Step 3: Launch ────────────────────────────────────────────────────────────
section "Launching"

cd "${SCRIPT_DIR}"
eval "$CMD"
