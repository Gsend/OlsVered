#!/usr/bin/env bash
# =============================================================================
#  run_benchmark.sh  —  OlsSMKFAC GPU Benchmark launcher
# =============================================================================
#
#  Usage:
#    bash run_benchmark.sh [options]
#
#  Options:
#    --task          mlp|bert|cifar|scaling|transformer|all  (default: mlp)
#                      mlp         — Large MLP on MNIST (~30 min)
#                      cifar       — Deep MLP on CIFAR-10 (~20 min)
#                      scaling     — Width sweep: step-cost + convergence (~5-10 min)
#                      bert        — BERT fine-tuning on SST-2 (~4-6 hrs, needs >=12GB VRAM)
#                      transformer — Small GPT from scratch on WikiText-2 (~30-60 min)
#                      all         — Run all tasks in sequence
#    --skip          adam,classickfac,...    Comma-separated optimizers to skip
#                    Valid names: adam, classickfac, olssmkfac
#    --steps-mlp     N                      Max steps for MLP task              (default: 3000)
#    --steps-bert    N                      Max steps for BERT task             (default: 8000)
#    --steps-cifar   N                      Max steps for CIFAR task            (default: 5000)
#    --steps-scaling N                      Convergence steps per width/optimizer in
#                                           scaling task                        (default: 300)
#    --lr-ols-transformer LR               Override OlsSMKFAC lr in transformer task
#    --lr-cls-transformer LR               Override ClassicKFAC lr in transformer task
#    --lr-sweep-transformer                Run LR sweep [1e-3, 3e-3, 5e-3, 8e-3] for both
#                                           K-FAC optimizers and report best LR
#    --no-setup                        Skip dependency/build checks (if already set up)
#    --no-tmux                         Run directly, without tmux session
#    --session   NAME                  tmux session name (default: benchmark)
#    --help                            Show this help
#
#  First-run setup (done automatically):
#    1. Creates .venv if no virtual environment is active
#    2. Detects CUDA version and installs matching PyTorch
#    3. Installs requirements.txt (numpy, scipy, matplotlib, transformers, etc.)
#    4. Installs Rust toolchain if cargo is not found
#    5. Compiles the olssm Rust extension via maturin
#
#  Examples:
#    bash run_benchmark.sh --task mlp
#    bash run_benchmark.sh --task transformer --lr-sweep-transformer
#    bash run_benchmark.sh --task all --skip adam,classickfac
#    bash run_benchmark.sh --task mlp --no-setup   # skip setup on repeat runs
#
#  Results are saved to benchmark/results/ after each optimizer completes.
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TASK="mlp"
SKIP=""
STEPS_MLP=3000
STEPS_BERT=8000
STEPS_CIFAR=5000
STEPS_SCALING=300
STEPS_TRANSFORMER=5000
LR_OLS_TRANSFORMER=""
LR_CLS_TRANSFORMER=""
LR_SWEEP_TRANSFORMER=false
RUN_SETUP=true
USE_TMUX=true
SESSION="benchmark"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${CYAN}══ $* ══${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)                  TASK="$2";                  shift 2 ;;
        --skip)                  SKIP="$2";                  shift 2 ;;
        --steps-mlp)             STEPS_MLP="$2";             shift 2 ;;
        --steps-bert)            STEPS_BERT="$2";            shift 2 ;;
        --steps-cifar)           STEPS_CIFAR="$2";           shift 2 ;;
        --steps-scaling)         STEPS_SCALING="$2";         shift 2 ;;
        --steps-transformer)     STEPS_TRANSFORMER="$2";     shift 2 ;;
        --lr-ols-transformer)    LR_OLS_TRANSFORMER="$2";    shift 2 ;;
        --lr-cls-transformer)    LR_CLS_TRANSFORMER="$2";    shift 2 ;;
        --lr-sweep-transformer)  LR_SWEEP_TRANSFORMER=true;  shift   ;;
        --no-setup)              RUN_SETUP=false;            shift   ;;
        --no-tmux)               USE_TMUX=false;             shift   ;;
        --session)               SESSION="$2";               shift 2 ;;
        --help|-h)
            sed -n '3,50p' "$0" | sed 's/^#  \?//'
            exit 0 ;;
        *)
            error "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# Validate --task
if [[ ! "$TASK" =~ ^(mlp|bert|cifar|scaling|transformer|all)$ ]]; then
    error "--task must be mlp, bert, cifar, scaling, transformer, or all (got: $TASK)"
    exit 1
fi

# ── Step 1: Virtual environment ───────────────────────────────────────────────
if [[ "$RUN_SETUP" == true ]]; then
    section "Environment"

    VENV_DIR="${SCRIPT_DIR}/.venv"

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        # Not inside any venv
        if [[ ! -d "$VENV_DIR" ]]; then
            info "Creating virtual environment at .venv ..."
            python3 -m venv "$VENV_DIR"
        fi
        info "Activating .venv ..."
        # shellcheck source=/dev/null
        source "${VENV_DIR}/bin/activate"
    else
        info "Using active virtual environment: ${VIRTUAL_ENV}"
    fi

    # ── Step 2: Detect CUDA version and install PyTorch ───────────────────────
    section "PyTorch"

    TORCH_OK=$(python3 -c "import torch; print('OK')" 2>/dev/null || echo "MISSING")

    if [[ "$TORCH_OK" == "MISSING" ]]; then
        info "PyTorch not found — detecting CUDA version..."

        # Read CUDA version from nvidia-smi (e.g. "12.8" → "cu128")
        CUDA_VER=$(nvidia-smi 2>/dev/null \
            | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" \
            | head -1 || echo "")

        if [[ -z "$CUDA_VER" ]]; then
            warn "Could not detect CUDA version — defaulting to cu121."
            CUDA_VER="12.1"
        fi

        # Map "12.8" → "cu128", "12.1" → "cu121", etc.
        CUDA_TAG="cu$(echo "$CUDA_VER" | tr -d '.')"
        TORCH_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

        info "Detected CUDA ${CUDA_VER} — installing PyTorch from ${TORCH_URL} ..."
        pip install torch torchvision --index-url "$TORCH_URL" --quiet
    else
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
        info "PyTorch already installed: ${TORCH_VER}"
    fi

    # ── Step 3: Install requirements.txt ──────────────────────────────────────
    section "Python dependencies"

    if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
        info "Installing requirements.txt ..."
        pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
    else
        warn "requirements.txt not found — skipping."
    fi

    # ── Step 4: Rust toolchain ────────────────────────────────────────────────
    section "Rust toolchain"

    if ! command -v cargo &>/dev/null; then
        info "cargo not found — installing Rust toolchain (this takes ~1 min)..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --default-toolchain stable --no-modify-path
        # shellcheck source=/dev/null
        source "${HOME}/.cargo/env"
        info "Rust installed: $(rustc --version)"
    else
        info "Rust already installed: $(rustc --version)"
    fi

    # ── Step 5: Build olssm Rust extension ────────────────────────────────────
    section "olssm Rust extension"

    OLSSM_OK=$(python3 -c "import olssm; print('OK')" 2>/dev/null || echo "MISSING")

    if [[ "$OLSSM_OK" == "MISSING" ]]; then
        info "Building olssm extension (maturin develop --release) ..."
        (cd "${SCRIPT_DIR}" && maturin develop --release)

        OLSSM_VERIFY=$(python3 -c "import olssm; print('OK')" 2>/dev/null || echo "FAILED")
        if [[ "$OLSSM_VERIFY" != "OK" ]]; then
            error "Build finished but 'import olssm' still fails."
            error "Try manually: cd ${SCRIPT_DIR} && maturin develop --release"
            exit 3
        fi
        info "olssm built and installed (Rust backend active)."
    else
        info "olssm already installed (Rust backend active)."
    fi
fi  # end RUN_SETUP

# ── GPU verification ──────────────────────────────────────────────────────────
section "GPU check"

GPU_CHECK=$(python3 - <<'PYEOF'
import sys, torch

if not torch.cuda.is_available():
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        print("NO_DRIVER" if out.returncode != 0 else "NO_CUDA")
    except FileNotFoundError:
        print("NO_NVIDIA_SMI")
    sys.exit(1)

name  = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"OK|{name}|{total:.1f}")
PYEOF
) || true

case "$GPU_CHECK" in
    NO_DRIVER)
        error "NVIDIA driver not found or too old."
        error "Pick a pod template that includes the NVIDIA driver (e.g. PyTorch on Runpod)."
        exit 2 ;;
    NO_CUDA)
        error "CUDA not available — PyTorch may not match your CUDA driver."
        error "Re-run without --no-setup to reinstall the correct PyTorch version."
        exit 2 ;;
    NO_NVIDIA_SMI)
        error "nvidia-smi not found — this machine has no NVIDIA GPU."
        exit 2 ;;
    OK*)
        IFS='|' read -r _ GPU_NAME GPU_MEM <<< "$GPU_CHECK"
        info "GPU: ${GPU_NAME}  (${GPU_MEM} GB VRAM)" ;;
    *)
        error "GPU check returned unexpected output: ${GPU_CHECK}"
        exit 2 ;;
esac

# Warn if VRAM < 12 GB and BERT is requested
if [[ "$TASK" =~ ^(bert|all)$ ]]; then
    VRAM_INT=${GPU_MEM%.*}
    if (( VRAM_INT < 12 )); then
        warn "BERT task needs >=12 GB VRAM. Detected ${GPU_MEM} GB — it may OOM."
    fi
fi

# ── Build Python command ──────────────────────────────────────────────────────
PY_CMD="cd '${SCRIPT_DIR}' && python3 benchmark/gpu_benchmark.py"
PY_CMD+=" --task ${TASK}"
PY_CMD+=" --max-steps-mlp ${STEPS_MLP}"
PY_CMD+=" --max-steps-bert ${STEPS_BERT}"
PY_CMD+=" --max-steps-cifar ${STEPS_CIFAR}"
PY_CMD+=" --max-steps-scaling ${STEPS_SCALING}"
PY_CMD+=" --max-steps-transformer ${STEPS_TRANSFORMER}"
[[ -n "$SKIP" ]]                       && PY_CMD+=" --skip ${SKIP}"
[[ -n "$LR_OLS_TRANSFORMER" ]]         && PY_CMD+=" --lr-ols-transformer ${LR_OLS_TRANSFORMER}"
[[ -n "$LR_CLS_TRANSFORMER" ]]         && PY_CMD+=" --lr-cls-transformer ${LR_CLS_TRANSFORMER}"
[[ "$LR_SWEEP_TRANSFORMER" == true ]]  && PY_CMD+=" --lr-sweep-transformer"

echo ""
info "Launching: ${PY_CMD}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ "$USE_TMUX" == false ]]; then
    info "Running directly (no tmux)..."
    eval "$PY_CMD"
    exit $?
fi

if ! command -v tmux &>/dev/null; then
    warn "tmux not found — running directly. Install with: apt-get install -y tmux"
    eval "$PY_CMD"
    exit $?
fi

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

info "Starting tmux session '${SESSION}'..."
info "If your SSH connection drops, reconnect and run:"
info "  tmux attach -t ${SESSION}"
echo ""

tmux new-session -d -s "$SESSION" \
    "bash -c \"source '${VENV_DIR:-${VIRTUAL_ENV}}/bin/activate' 2>/dev/null || true; ${PY_CMD}; echo ''; echo '=== Benchmark finished. Press Enter to close. ==='; read\""

tmux attach -t "$SESSION"
