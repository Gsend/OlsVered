#!/usr/bin/env bash
# =============================================================================
#  run_benchmark.sh  —  OlsveredKFAC GPU Benchmark launcher
# =============================================================================
#
#  Usage:
#    bash run_benchmark.sh [options]
#
#  Options:
#    --task      mlp|bert|all          (default: mlp)
#    --skip      adam,classickfac,...  Comma-separated optimizers to skip
#                Valid names: adam, classickfac, olsveredkfac
#    --steps-mlp N                     Max steps for MLP task  (default: 3000)
#    --steps-bert N                    Max steps for BERT task (default: 8000)
#    --no-tmux                         Run directly, without tmux session
#    --session   NAME                  tmux session name (default: benchmark)
#    --help                            Show this help
#
#  Examples:
#    bash run_benchmark.sh --task mlp
#    bash run_benchmark.sh --task mlp --skip adam
#    bash run_benchmark.sh --task all --skip adam,classickfac
#    bash run_benchmark.sh --task mlp --steps-mlp 1000 --no-tmux
#
#  Results are saved incrementally to benchmark/results/ after each optimizer
#  completes — so a dropped connection never loses a finished run.
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TASK="mlp"
SKIP=""
STEPS_MLP=3000
STEPS_BERT=8000
USE_TMUX=true
SESSION="benchmark"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)       TASK="$2";       shift 2 ;;
        --skip)       SKIP="$2";       shift 2 ;;
        --steps-mlp)  STEPS_MLP="$2";  shift 2 ;;
        --steps-bert) STEPS_BERT="$2"; shift 2 ;;
        --no-tmux)    USE_TMUX=false;  shift   ;;
        --session)    SESSION="$2";    shift 2 ;;
        --help|-h)
            sed -n '3,30p' "$0" | sed 's/^#  \?//'
            exit 0 ;;
        *)
            error "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# Validate --task
if [[ ! "$TASK" =~ ^(mlp|bert|all)$ ]]; then
    error "--task must be mlp, bert, or all (got: $TASK)"
    exit 1
fi

# ── GPU verification ──────────────────────────────────────────────────────────
info "Verifying GPU availability..."

GPU_CHECK=$(python3 - <<'PYEOF'
import sys, torch

if not torch.cuda.is_available():
    # Try to give a helpful reason
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            print("NO_DRIVER")
        else:
            print("NO_CUDA")
    except FileNotFoundError:
        print("NO_NVIDIA_SMI")
    sys.exit(1)

name  = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"OK|{name}|{total:.1f}")
sys.exit(0)
PYEOF
) || true

if [[ "$GPU_CHECK" == NO_DRIVER ]]; then
    error "NVIDIA driver not found or too old."
    error "Update your driver: https://www.nvidia.com/Download/index.aspx"
    error "Or pick a newer pod template in Runpod that includes the latest driver."
    exit 2
elif [[ "$GPU_CHECK" == NO_CUDA ]]; then
    error "CUDA is not available. Your PyTorch may not match your CUDA driver."
    error "Try: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 2
elif [[ "$GPU_CHECK" == NO_NVIDIA_SMI ]]; then
    error "nvidia-smi not found — this machine has no NVIDIA GPU."
    error "Rent a GPU pod on Runpod / Lambda Labs before running benchmarks."
    exit 2
elif [[ "$GPU_CHECK" != OK* ]]; then
    error "GPU check failed with unexpected output: $GPU_CHECK"
    exit 2
fi

IFS='|' read -r _ GPU_NAME GPU_MEM <<< "$GPU_CHECK"
info "GPU OK: ${GPU_NAME}  (${GPU_MEM} GB VRAM)"

# Warn if VRAM < 12 GB and BERT is requested
if [[ "$TASK" != "mlp" ]]; then
    VRAM_INT=${GPU_MEM%.*}
    if (( VRAM_INT < 12 )); then
        warn "BERT task needs ≥12 GB VRAM. Detected ${GPU_MEM} GB — it may OOM."
        warn "Consider running --task mlp only on this GPU."
    fi
fi

# ── Build Python command ──────────────────────────────────────────────────────
PY_CMD="cd '${SCRIPT_DIR}' && python3 benchmark/gpu_benchmark.py"
PY_CMD+=" --task ${TASK}"
PY_CMD+=" --max-steps-mlp ${STEPS_MLP}"
PY_CMD+=" --max-steps-bert ${STEPS_BERT}"
[[ -n "$SKIP" ]] && PY_CMD+=" --skip ${SKIP}"

info "Command: ${PY_CMD}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ "$USE_TMUX" == false ]]; then
    info "Running directly (no tmux)..."
    eval "$PY_CMD"
    exit $?
fi

# Check if tmux is available
if ! command -v tmux &>/dev/null; then
    warn "tmux not found — running directly."
    warn "Install with: apt-get install -y tmux"
    eval "$PY_CMD"
    exit $?
fi

# If session already exists, offer to attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
    warn "tmux session '${SESSION}' already exists."
    echo -n "  Attach to existing session? [Y/n]: "
    read -r ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        error "Aborting. Kill the old session first: tmux kill-session -t ${SESSION}"
        exit 1
    fi
    tmux attach -t "$SESSION"
    exit 0
fi

# Launch new tmux session with the benchmark
info "Starting tmux session '${SESSION}'..."
info "If your SSH connection drops, reconnect and run:"
info "  tmux attach -t ${SESSION}"
echo ""

tmux new-session -d -s "$SESSION" \
    "bash -c \"${PY_CMD}; echo ''; echo '=== Benchmark finished. Press Enter to close. ==='; read\""

tmux attach -t "$SESSION"
