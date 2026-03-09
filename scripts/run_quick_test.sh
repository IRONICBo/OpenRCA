#!/bin/bash
# Quick smoke test to verify the setup works end-to-end
# Runs only 2 tasks on a single dataset
# Usage: bash scripts/run_quick_test.sh [--no_wandb]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-fintech-copilot}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

NO_WANDB=""
DATASET="${DATASET:-Bank}"
if [[ "$1" == "--no_wandb" ]]; then
    NO_WANDB="--no_wandb"
fi
if [[ "$2" != "" ]]; then
    DATASET="$2"
fi

echo "============================================"
echo "  OpenRCA Quick Smoke Test"
echo "============================================"
echo "  Dataset: $DATASET"
echo "  Tasks:   [0, 1] (2 tasks only)"
echo "============================================"

echo ""
echo "--- Test 1: RCA-Agent (2 tasks) ---"
python -m rca.run_agent_standard \
    --dataset "$DATASET" \
    --start_idx 0 \
    --end_idx 1 \
    --controller_max_step 5 \
    --controller_max_turn 2 \
    --timeout 120 \
    --tag "smoke_test" \
    $NO_WANDB

echo ""
echo "--- Test 2: Balanced Direct (2 tasks) ---"
python -m rca.run_sampling_balanced \
    --dataset "$DATASET" \
    --start_idx 0 \
    --end_idx 1 \
    --mode direct \
    --tag "smoke_test" \
    $NO_WANDB

echo ""
echo "============================================"
echo "  Smoke test passed!"
echo "============================================"
