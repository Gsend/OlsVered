#!/usr/bin/env bash
# =============================================================================
#  run_retrainer_benchmark.sh  —  OlsSMLayerRetrainer benchmark launcher
# =============================================================================
#
#  Benchmarks every retraining mode of OlsSMLayerRetrainer against LoRA and
#  AdamW baselines on a real HuggingFace model.
#
#  Usage:
#    bash run_retrainer_benchmark.sh [options]
#
#  Options:
#    --model   NAME     HuggingFace model name (default: bert-base-uncased)
#                         Other good choices:
#                           distilbert-base-uncased  (faster, 66M params)
#                           roberta-base             (stronger baseline, 125M)
#                           bert-large-uncased       (large, 340M, needs >=16GB)
#    --task    NAME     Dataset / task (default: sst2)
#                         sst2   — Stanford Sentiment Treebank (binary, 67K train)
#                         mrpc   — Paraphrase detection (binary, 3.7K train)
#                         imdb   — Movie review sentiment (binary, 25K train)
#    --modes   LIST     Comma-separated modes or 'all' (default: all)
#                         adam, adam_head,
#                         lora_r4, lora_r8,
#                         ols_n1, ols_n2, ols_n4, ols_n8, ols_all,
#                         ols_lora_r2, ols_lora_r4, ols_lora_r8
#    --batch-size  N    Batch size (default: 32)
#    --max-train   N    Cap training samples — useful for quick tests
#                         e.g. --max-train 2000 runs in ~5 min
#    --epochs      N    Epochs for Adam/LoRA modes (default: 1)
#    --max-sweeps  N    Max BCD sweeps for OLS modes (default: 5)
#    --lambda-reg  F    OLS regularisation lambda (default: 1e-4)
#    --bcd-mode    STR  BCD variant for OLS modes (default: gauss_seidel)
#                         jacobi         — simultaneous updates, 1 pass/sweep
#                         gauss_seidel   — sequential updates, N passes/sweep
#    --seed        N    Master RNG seed (default: 42). Pins the random
#                         classifier-head init so the pretrained baseline
#                         is reproducible across runs.
#    --tag         STR  Optional tag appended to result filenames
#    --no-plots         Skip matplotlib plots
#    --no-setup         Skip venv / pip / maturin setup (repeat runs)
#    --no-tmux          Run directly instead of inside a tmux session
#    --session   NAME   tmux session name (default: retrainer_bench)
#    --help             Show this help
#
#  Quick-test examples:
#    bash run_retrainer_benchmark.sh --max-train 2000 --modes ols_n1,lora_r4,ols_lora_r4
#    bash run_retrainer_benchmark.sh --model distilbert-base-uncased --task mrpc
#
#  Full benchmark examples:
#    bash run_retrainer_benchmark.sh
#    bash run_retrainer_benchmark.sh --model bert-base-uncased --task sst2 --modes all
#    bash run_retrainer_benchmark.sh --model roberta-base --epochs 3 --max-sweeps 5
#
#  Results saved to:  benchmark/results/retrainer_<model>_<task>[_tag].json
#  Plots saved to:    benchmark/results/retrainer_<model>_<task>[_tag].png
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="bert-base-uncased"
TASK="sst2"
MODES="all"
BATCH_SIZE=32
MAX_TRAIN=""
EPOCHS=1
MAX_SWEEPS=5
LAMBDA_REG="1e-4"
BCD_MODE="gauss_seidel"
SEED=42
TAG=""
NO_PLOTS=false
RUN_SETUP=true
USE_TMUX=false
SESSION="retrainer_bench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Windows (Git Bash) compatibility ─────────────────────────────────────────
if [[ -n "${WINDIR:-}" ]] || [[ "$OSTYPE" == msys* ]] || [[ "$OSTYPE" == cygwin* ]]; then
    _VENV_SCRIPTS="Scripts"
    # Cargo and MinGW64 are not on PATH by default in Git Bash; add them
    export PATH="${HOME}/.cargo/bin:/c/msys64/mingw64/bin:${PATH}"
    # Pre-generated pyo3 config avoids running the blocked build script on Windows
    export PYO3_CONFIG_FILE="${SCRIPT_DIR}/pyo3-build-config.txt"
    # Resolve a working Python for pre-activation use (venv creation)
    _SYS_PYTHON="${HOME}/AppData/Local/Programs/Python/Python312/python.exe"
    [[ -x "$_SYS_PYTHON" ]] || _SYS_PYTHON="python"
else
    _VENV_SCRIPTS="bin"
    _SYS_PYTHON="python3"
fi

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${CYAN}══ $* ══${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --task)        TASK="$2";        shift 2 ;;
        --modes)       MODES="$2";       shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --max-train)   MAX_TRAIN="$2";   shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --max-sweeps)  MAX_SWEEPS="$2";  shift 2 ;;
        --lambda-reg)  LAMBDA_REG="$2";  shift 2 ;;
        --bcd-mode)    BCD_MODE="$2";    shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        --tag)         TAG="$2";         shift 2 ;;
        --no-plots)    NO_PLOTS=true;    shift   ;;
        --no-setup)    RUN_SETUP=false;  shift   ;;
        --no-tmux)     USE_TMUX=false;   shift   ;;
        --session)     SESSION="$2";     shift 2 ;;
        --help|-h)
            sed -n '3,60p' "$0" | sed 's/^#  \?//'
            exit 0 ;;
        *)
            error "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# ── Step 1: Virtual environment ───────────────────────────────────────────────
if [[ "$RUN_SETUP" == true ]]; then
    section "Environment"

    VENV_DIR="${SCRIPT_DIR}/.venv"

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        if [[ ! -d "$VENV_DIR" ]]; then
            info "Creating virtual environment at .venv ..."
            "$_SYS_PYTHON" -m venv "$VENV_DIR"
        fi
        info "Activating .venv ..."
        # shellcheck source=/dev/null
        source "${VENV_DIR}/${_VENV_SCRIPTS}/activate"
    else
        info "Using active virtual environment: ${VIRTUAL_ENV}"
        VENV_DIR="${VIRTUAL_ENV}"
    fi

    # ── Step 2: PyTorch ───────────────────────────────────────────────────────
    section "PyTorch"

    TORCH_OK=$(python3 -c "import torch; print('OK')" 2>/dev/null || echo "MISSING")

    if [[ "$TORCH_OK" == "MISSING" ]]; then
        info "PyTorch not found — detecting CUDA version..."
        CUDA_VER=$(nvidia-smi 2>/dev/null \
            | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || echo "12.1")
        CUDA_TAG="cu$(echo "$CUDA_VER" | tr -d '.')"
        info "Detected CUDA ${CUDA_VER} — installing PyTorch (${CUDA_TAG})..."
        pip install torch torchvision \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --quiet
    else
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
        info "PyTorch: ${TORCH_VER}"
    fi

    # ── Step 3: Python dependencies ───────────────────────────────────────────
    section "Python dependencies"

    pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet

    # peft is optional but strongly recommended for the LoRA baseline
    PEFT_OK=$(python3 -c "import peft; print('OK')" 2>/dev/null || echo "MISSING")
    if [[ "$PEFT_OK" == "MISSING" ]]; then
        info "Installing peft (HuggingFace PEFT for LoRA baseline)..."
        pip install peft --quiet
    else
        info "peft already installed (LoRA baseline will use HF PEFT)."
    fi

    # ── Step 4: Rust toolchain ────────────────────────────────────────────────
    section "Rust toolchain"

    if ! command -v cargo &>/dev/null; then
        info "Installing Rust toolchain (~1 min)..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --default-toolchain stable --no-modify-path
        # shellcheck source=/dev/null
        source "${HOME}/.cargo/env" 2>/dev/null || export PATH="${HOME}/.cargo/bin:${PATH}"
    else
        info "Rust: $(rustc --version)"
    fi

    # ── Step 5: olssm Rust extension ─────────────────────────────────────────
    section "olssm Rust extension"

    OLSSM_OK=$(python3 -c "import olssm; print('OK')" 2>/dev/null || echo "MISSING")
    if [[ "$OLSSM_OK" == "MISSING" ]]; then
        info "Building olssm (maturin develop --release)..."
        (cd "${SCRIPT_DIR}" && maturin develop --release)
        python3 -c "import olssm" || { error "Build failed."; exit 3; }
        info "olssm built successfully."
    else
        info "olssm already installed (Rust backend active)."
    fi
fi  # end RUN_SETUP

# ── Ensure venv is active (needed when --no-setup skips activation) ───────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    _DEFAULT_VENV="${SCRIPT_DIR}/.venv"
    if [[ -d "${_DEFAULT_VENV}/${_VENV_SCRIPTS}" ]]; then
        # shellcheck source=/dev/null
        source "${_DEFAULT_VENV}/${_VENV_SCRIPTS}/activate" 2>/dev/null || \
            export PATH="${_DEFAULT_VENV}/${_VENV_SCRIPTS}:${PATH}"
    fi
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
section "GPU check"

GPU_CHECK=$(python3 - <<'PYEOF'
import sys, torch
if not torch.cuda.is_available():
    print("NO_CUDA"); sys.exit(1)
name  = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"OK|{name}|{total:.1f}")
PYEOF
) || true

case "$GPU_CHECK" in
    OK*)
        IFS='|' read -r _ GPU_NAME GPU_MEM <<< "$GPU_CHECK"
        info "GPU: ${GPU_NAME}  (${GPU_MEM} GB VRAM)" ;;
    *)
        warn "CUDA not available — benchmark will run on CPU (very slow)."
        GPU_MEM="0" ;;
esac

# Warn for large models
VRAM_INT=${GPU_MEM%.*}
if [[ "$MODEL" == *"large"* ]] && (( ${VRAM_INT:-0} < 16 )); then
    warn "Large model requested with <16 GB VRAM — may OOM."
    warn "Consider --model bert-base-uncased or --max-train 2000 to reduce memory."
fi

# ── Build Python command ──────────────────────────────────────────────────────
# PYTHONUTF8=1 ensures Unicode output works correctly on Windows terminals
PY_CMD="cd '${SCRIPT_DIR}' && PYTHONUTF8=1 python3 benchmark/retrainer_benchmark.py"
PY_CMD+=" --model ${MODEL}"
PY_CMD+=" --task ${TASK}"
PY_CMD+=" --modes ${MODES}"
PY_CMD+=" --batch-size ${BATCH_SIZE}"
PY_CMD+=" --epochs ${EPOCHS}"
PY_CMD+=" --max-sweeps ${MAX_SWEEPS}"
PY_CMD+=" --lambda-reg ${LAMBDA_REG}"
PY_CMD+=" --bcd-mode ${BCD_MODE}"
PY_CMD+=" --seed ${SEED}"
[[ -n "$MAX_TRAIN" ]]  && PY_CMD+=" --max-train ${MAX_TRAIN}"
[[ -n "$TAG" ]]        && PY_CMD+=" --tag ${TAG}"
[[ "$NO_PLOTS" == true ]] && PY_CMD+=" --no-plots"

echo ""
info "Model  : ${MODEL}"
info "Task   : ${TASK}"
info "Modes  : ${MODES}"
[[ -n "$MAX_TRAIN" ]] && info "Train samples capped at: ${MAX_TRAIN}"
info "Command: ${PY_CMD}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ "$USE_TMUX" == false ]]; then
    eval "$PY_CMD"
    exit $?
fi

if ! command -v tmux &>/dev/null; then
    warn "tmux not found — running directly."
    eval "$PY_CMD"
    exit $?
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    warn "tmux session '${SESSION}' already exists."
    echo -n "  Attach to existing session? [Y/n]: "
    read -r ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        error "Aborting. Kill with: tmux kill-session -t ${SESSION}"
        exit 1
    fi
    tmux attach -t "$SESSION"
    exit 0
fi

info "Starting tmux session '${SESSION}'..."
info "Reconnect after SSH drop with: tmux attach -t ${SESSION}"
echo ""

tmux new-session -d -s "$SESSION" \
    "bash -c \"source '${VENV_DIR:-${VIRTUAL_ENV:-/dev/null}}'/bin/activate 2>/dev/null || true; \
               ${PY_CMD}; \
               echo ''; echo '=== Benchmark finished. Press Enter to close. ==='; read\""

tmux attach -t "$SESSION"
