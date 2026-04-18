#!/usr/bin/env bash
# =============================================================================
#  run_all_retrainer_tests.sh  —  Full retrainer benchmark suite
# =============================================================================
#
#  Sequentially runs every retraining mode with a fixed seed for reproducible,
#  comparable results.  Each mode writes to its own tagged JSON so nothing
#  gets overwritten.  Pretrained baseline is measured in every run and should
#  be identical across runs once --seed pins the classifier-head init.
#
#  On a single 16 GB GPU, expect roughly:
#      ols_n1            ~1.5 min
#      ols_n2 (GS)       ~15 min
#      ols_n2 (Jacobi)   ~15 min   (only if --include-jacobi)
#      ols_n4            ~27 min
#      als_lora_n1_r4    ~2 min
#      als_lora_n2_r4    ~15 min
#      als_lora_n4_r4    ~28 min
#      ols_lora_r4       ~17 min
#      ─────────────────────────
#      Full suite (no baselines)   ~1h 45m
#      + --include-baselines       +~15 min
#      + --include-jacobi          +~15 min
#
#  Usage:
#    bash run_all_retrainer_tests.sh [options]
#
#  Options:
#    --seed N             Master seed (default: 42)
#    --model NAME         HuggingFace model (default: bert-base-uncased)
#    --task NAME          Dataset (default: sst2)
#    --include-baselines  Also run Adam/LoRA baselines (adam, adam_head, lora_r4)
#    --include-jacobi     Also run OLS N=2 with Jacobi BCD (default: GS only)
#    --skip-existing      Skip modes whose tagged JSON already exists
#    --dry-run            Print the commands without executing anything
#    --setup              Let the wrapper re-install deps (default: --no-setup)
#    --help               Show this help
#
#  Output:
#    Results : benchmark/results/retrainer_<model>_<task>_<tag>.json
#    Log     : benchmark/results/run_all_retrainer_tests_<timestamp>.log
# =============================================================================

set -uo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SEED=42
MODEL="bert-base-uncased"
TASK="sst2"
INCLUDE_BASELINES=false
INCLUDE_JACOBI=false
SKIP_EXISTING=false
DRY_RUN=false
NO_SETUP=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="${SCRIPT_DIR}/run_retrainer_benchmark.sh"
RESULTS_DIR="${SCRIPT_DIR}/benchmark/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/run_all_retrainer_tests_${TIMESTAMP}.log"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${CYAN}══ $* ══${NC}"; }

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)              SEED="$2";                shift 2 ;;
        --model)             MODEL="$2";               shift 2 ;;
        --task)              TASK="$2";                shift 2 ;;
        --include-baselines) INCLUDE_BASELINES=true;   shift   ;;
        --include-jacobi)    INCLUDE_JACOBI=true;      shift   ;;
        --skip-existing)     SKIP_EXISTING=true;       shift   ;;
        --dry-run)           DRY_RUN=true;             shift   ;;
        --setup)             NO_SETUP=false;           shift   ;;
        --no-setup)          NO_SETUP=true;            shift   ;;
        --help|-h)
            sed -n '3,50p' "$0" | sed 's/^#  \?//'
            exit 0 ;;
        *)
            error "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -x "$WRAPPER" && ! -f "$WRAPPER" ]]; then
    error "Wrapper not found: $WRAPPER"
    exit 2
fi
mkdir -p "$RESULTS_DIR"

# ── Test matrix:  mode  bcd_mode  tag  extra_args ────────────────────────────
# tag becomes the suffix in the output filename.
declare -a TEST_LIST=(
  "ols_n1         gauss_seidel  n1"
  "ols_n2         gauss_seidel  n2_gs"
  "ols_n4         gauss_seidel  n4"
  "als_lora_n1_r4 gauss_seidel  alslora_n1_r4"
  "als_lora_n2_r4 gauss_seidel  alslora_n2_r4"
  "als_lora_n4_r4 gauss_seidel  alslora_n4_r4"
  "ols_lora_r4    gauss_seidel  olslora_r4"
)

if $INCLUDE_JACOBI; then
    TEST_LIST+=("ols_n2 jacobi n2_jacobi")
fi

if $INCLUDE_BASELINES; then
    TEST_LIST+=(
      "adam      gauss_seidel  adam"
      "adam_head gauss_seidel  adam_head"
      "lora_r4   gauss_seidel  lora_r4"
    )
fi

MODEL_SANITIZED="${MODEL//[-\/]/_}"

# ── Common wrapper args ──────────────────────────────────────────────────────
COMMON_ARGS=(
    --model "$MODEL"
    --task "$TASK"
    --seed "$SEED"
)
$NO_SETUP && COMMON_ARGS+=(--no-setup)

# ── Plan summary ─────────────────────────────────────────────────────────────
section "Plan"
info "Model        : $MODEL"
info "Task         : $TASK"
info "Seed         : $SEED"
info "Wrapper      : $WRAPPER"
info "Results dir  : $RESULTS_DIR"
info "Log file     : $LOG_FILE"
$DRY_RUN    && info "Mode         : DRY RUN (no commands executed)"
$SKIP_EXISTING && info "Skip existing: yes"
echo ""
info "Modes to run (${#TEST_LIST[@]}):"
for entry in "${TEST_LIST[@]}"; do
    read -r mode bcd tag <<< "$entry"
    echo "    • $mode  (bcd=$bcd, tag=$tag)"
done

# ── Run loop ─────────────────────────────────────────────────────────────────
section "Running"
declare -a FAILED=()
declare -a SKIPPED=()
declare -a COMPLETED=()
SUITE_START=$(date +%s)

# Everything below is tee'd to the log file so you can re-read it later.
{
for entry in "${TEST_LIST[@]}"; do
    read -r mode bcd tag <<< "$entry"
    OUT_JSON="${RESULTS_DIR}/retrainer_${MODEL_SANITIZED}_${TASK}_${tag}.json"

    if $SKIP_EXISTING && [[ -f "$OUT_JSON" ]]; then
        warn "Skipping $mode — result already exists: $(basename "$OUT_JSON")"
        SKIPPED+=("$mode")
        continue
    fi

    section "mode=$mode  bcd=$bcd  tag=$tag"
    CMD=(bash "$WRAPPER" "${COMMON_ARGS[@]}"
         --modes "$mode"
         --bcd-mode "$bcd"
         --tag "$tag")

    info "$(date '+%H:%M:%S')  Command: ${CMD[*]}"

    if $DRY_RUN; then
        SKIPPED+=("$mode (dry-run)")
        continue
    fi

    MODE_START=$(date +%s)
    if "${CMD[@]}"; then
        MODE_END=$(date +%s)
        info "$mode completed in $((MODE_END - MODE_START))s  →  $(basename "$OUT_JSON")"
        COMPLETED+=("$mode")
    else
        MODE_END=$(date +%s)
        error "$mode FAILED after $((MODE_END - MODE_START))s"
        FAILED+=("$mode")
    fi
done
} 2>&1 | tee "$LOG_FILE"

SUITE_END=$(date +%s)

# ── Post-suite summary ───────────────────────────────────────────────────────
section "Summary"
info "Total wall time : $(( (SUITE_END - SUITE_START) / 60 ))m $(( (SUITE_END - SUITE_START) % 60 ))s"
info "Completed       : ${#COMPLETED[@]}  ${COMPLETED[*]:-}"
(( ${#SKIPPED[@]}  > 0 )) && warn  "Skipped         : ${#SKIPPED[@]}  ${SKIPPED[*]}"
(( ${#FAILED[@]}   > 0 )) && error "Failed          : ${#FAILED[@]}  ${FAILED[*]}"

# ── Optional results table (requires jq) ─────────────────────────────────────
if command -v jq &>/dev/null && ! $DRY_RUN; then
    section "Results table"
    printf "%-28s  %9s  %8s  %8s  %8s\n" "Mode (tag)" "Accuracy" "Loss" "WallSec" "PeakMem"
    printf "%-28s  %9s  %8s  %8s  %8s\n" "──────────" "────────" "────" "───────" "───────"
    # Pretrained baseline from the first completed run — should be identical
    # across all runs once --seed is fixed.
    BASELINE_PRINTED=false
    for entry in "${TEST_LIST[@]}"; do
        read -r mode bcd tag <<< "$entry"
        OUT_JSON="${RESULTS_DIR}/retrainer_${MODEL_SANITIZED}_${TASK}_${tag}.json"
        [[ -f "$OUT_JSON" ]] || continue

        if ! $BASELINE_PRINTED; then
            jq -r '.results[] | select(.mode == "pretrained") |
                [.label, (.accuracy*100|(.*100|round)/100|tostring+"%"),
                 (.loss|tostring), (.wall_s|tostring), (.peak_mem_gb|tostring+" GB")]
                | @tsv' "$OUT_JSON" 2>/dev/null | \
                awk -F'\t' '{printf "%-28s  %9s  %8s  %8s  %8s\n", $1, $2, $3, $4, $5}'
            BASELINE_PRINTED=true
        fi

        jq -r --arg mode "$mode" --arg tag "$tag" \
            '.results[] | select(.mode == $mode) |
             [.label + " (" + $tag + ")",
              (.accuracy*100|(.*100|round)/100|tostring+"%"),
              (.loss|tostring), (.wall_s|tostring),
              (.peak_mem_gb|tostring+" GB")]
             | @tsv' "$OUT_JSON" 2>/dev/null | \
            awk -F'\t' '{printf "%-28s  %9s  %8s  %8s  %8s\n", $1, $2, $3, $4, $5}'
    done
else
    info "Install 'jq' to get a results table at the end. Skipping."
fi

# Non-zero exit if anything failed
(( ${#FAILED[@]} > 0 )) && exit 1
exit 0
