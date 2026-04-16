#!/usr/bin/env bash
# =============================================================================
#  run_ols_modes.sh  —  Run the OLS modes that are missing or had a bug
# =============================================================================
#
#  Runs: ols_n1  ols_n2  ols_n4  ols_lora_r4
#
#  Why a separate script?
#  The benchmark overwrites its JSON file after each mode. The existing
#  retrainer_bert_base_uncased_sst2.json already contains correct results
#  for adam, adam_head and lora_r4, so we must NOT overwrite it.
#  This script therefore uses --tag ols, which saves to:
#      benchmark/results/retrainer_bert_base_uncased_sst2_ols.json
#
#  After both runs finish, this script merges the two JSON files into:
#      benchmark/results/retrainer_bert_base_uncased_sst2_combined.json
#
#  Usage:
#    bash run_ols_modes.sh [--no-setup] [--no-plots]
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmark/results"
MODEL="bert-base-uncased"
TASK="sst2"
EXTRA_ARGS=()

# ── Parse optional flags forwarded to run_retrainer_benchmark.sh ─────────────
for arg in "$@"; do
    case "$arg" in
        --no-setup|--no-plots) EXTRA_ARGS+=("$arg") ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Step 1: Run the four OLS modes ────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Running OLS modes: ols_n1  ols_n2  ols_n4  ols_lora_r4"
echo " Output tag: ols  →  retrainer_bert_base_uncased_sst2_ols.json"
echo "════════════════════════════════════════════════════════════════"
echo ""

bash "${SCRIPT_DIR}/run_retrainer_benchmark.sh" \
    --model   "${MODEL}"  \
    --task    "${TASK}"   \
    --modes   "ols_n1,ols_n2,ols_n4,ols_lora_r4" \
    --tag     ols         \
    "${EXTRA_ARGS[@]}"

# ── Step 2: Merge results ─────────────────────────────────────────────────────
BASE_JSON="${RESULTS_DIR}/retrainer_bert_base_uncased_sst2.json"
OLS_JSON="${RESULTS_DIR}/retrainer_bert_base_uncased_sst2_ols.json"
COMBINED="${RESULTS_DIR}/retrainer_bert_base_uncased_sst2_combined.json"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Merging results into retrainer_bert_base_uncased_sst2_combined.json"
echo "════════════════════════════════════════════════════════════════"

python3 - <<PYEOF
import json, sys
from pathlib import Path

base_path = Path("${BASE_JSON}")
ols_path  = Path("${OLS_JSON}")
out_path  = Path("${COMBINED}")

if not base_path.exists():
    sys.exit(f"ERROR: base results not found: {base_path}")
if not ols_path.exists():
    sys.exit(f"ERROR: OLS results not found: {ols_path}")

with open(base_path) as f:
    base = json.load(f)
with open(ols_path) as f:
    ols = json.load(f)

# Build a dict keyed by mode so we can deduplicate and override stale entries.
# Priority: OLS results take precedence (they have the bug fix applied).
by_mode = {}
for r in base["results"]:
    by_mode[r["mode"]] = r
for r in ols["results"]:
    by_mode[r["mode"]] = r   # override with fresh OLS results

# Canonical mode order
ORDER = [
    "pretrained", "adam", "adam_head",
    "lora_r4", "lora_r8",
    "ols_n1", "ols_n2", "ols_n4", "ols_n8", "ols_all",
    "ols_lora_r2", "ols_lora_r4", "ols_lora_r8",
]
merged_results = [by_mode[m] for m in ORDER if m in by_mode]

combined = {
    "model":               base["model"],
    "task":                base["task"],
    "modes_run":           [r["mode"] for r in merged_results],
    "pretrained_accuracy": base["pretrained_accuracy"],
    "results":             merged_results,
}

with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)

print(f"Merged {len(merged_results)} modes → {out_path}")
for r in merged_results:
    flag = "  [OLS-fixed]" if r["mode"] in [
        "ols_n1", "ols_n2", "ols_n4", "ols_lora_r4"
    ] else ""
    err  = f"  ERROR: {r.get('error')}" if r.get("error") else ""
    acc  = f"  acc={r['accuracy']*100:.2f}%" if r.get("accuracy") else ""
    print(f"  {r['mode']:<20}{acc}{flag}{err}")
PYEOF

echo ""
echo "All done."
echo "  Combined results: ${COMBINED}"
