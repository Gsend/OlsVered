#!/usr/bin/env bash
# =============================================================================
#  commit_and_push.sh  —  OlsSMKFAC Git Commit & Push (Linux / Runpod)
# =============================================================================
#  Usage:
#    bash scripts/commit_and_push.sh
#    bash scripts/commit_and_push.sh "optional custom commit message"
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
MSG="${1:-}"

echo ""
echo -e "${CYAN}=== OlsSMKFAC Git Commit & Push ===${NC}"
echo ""

# ── Step 1: Clear stale index.lock ────────────────────────────────────────────
if [ -f ".git/index.lock" ]; then
    echo -e "${YELLOW}[1/4] Removing stale index.lock...${NC}"
    rm -f .git/index.lock
    echo -e "${GREEN}      Removed.${NC}"
else
    echo -e "${GREEN}[1/4] No index.lock — OK.${NC}"
fi

# ── Step 2: Stage relevant files ──────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/4] Staging files...${NC}"

FILES=(
    "benchmark/gpu_benchmark.py"
    "benchmark/training_benchmark.py"
    "benchmark/economic_analysis.py"
    "benchmark/theoretical_analysis.py"
    "benchmark/results/"
    "optimizer/hooks.py"
    "optimizer/olssm_kfac.py"
    "optimizer/classic_kfac.py"
    "run_benchmark.sh"
    "scripts/"
    ".gitignore"
)

for f in "${FILES[@]}"; do
    if [ -e "$f" ]; then
        git add "$f"
        echo "      + $f"
    fi
done

echo ""
echo -e "${CYAN}      Staged:${NC}"
git diff --cached --name-only | sed 's/^/        /'

STAGED=$(git diff --cached --name-only)
if [ -z "$STAGED" ]; then
    echo ""
    echo -e "${YELLOW}  Nothing to commit — working tree clean.${NC}"
    exit 0
fi

# ── Step 3: Commit ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/4] Committing...${NC}"

if [ -z "$MSG" ]; then
    AREAS=""
    echo "$STAGED" | grep -q "benchmark/" && AREAS="${AREAS}benchmark, "
    echo "$STAGED" | grep -q "optimizer/" && AREAS="${AREAS}optimizer, "
    echo "$STAGED" | grep -q "scripts/"   && AREAS="${AREAS}scripts, "
    echo "$STAGED" | grep -q "run_benchmark" && AREAS="${AREAS}run_benchmark, "
    AREAS="${AREAS%, }"
    [ -z "$AREAS" ] && AREAS="misc"
    MSG="Update ${AREAS} — $(date '+%Y-%m-%d')"
fi

git commit -m "$MSG"
echo -e "${GREEN}      Committed: ${MSG}${NC}"

# ── Step 4: Push ──────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/4] Pushing to origin/main...${NC}"
git push origin main

echo ""
echo -e "${GREEN}=== Done! All changes pushed. ===${NC}"
echo ""
