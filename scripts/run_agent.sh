#!/bin/bash
# Run RCA-Agent evaluation on a single dataset
# Usage: bash scripts/run_agent.sh [options]
#
# Examples:
#   bash scripts/run_agent.sh --dataset Bank
#   bash scripts/run_agent.sh --dataset "Market/cloudbed-1" --start_idx 0 --end_idx 50
#   bash scripts/run_agent.sh --dataset Telecom --no_wandb
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Default parameters
DATASET="${DATASET:-Market/cloudbed-1}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-150}"
MAX_STEP="${MAX_STEP:-25}"
MAX_TURN="${MAX_TURN:-5}"
TIMEOUT="${TIMEOUT:-600}"
TAG="${TAG:-rca}"
SAMPLE_NUM="${SAMPLE_NUM:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-OpenRCA}"
NO_WANDB="${NO_WANDB:-}"

# Parse command-line arguments (override defaults)
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --start_idx) START_IDX="$2"; shift 2 ;;
        --end_idx) END_IDX="$2"; shift 2 ;;
        --max_step) MAX_STEP="$2"; shift 2 ;;
        --max_turn) MAX_TURN="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --sample_num) SAMPLE_NUM="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --no_wandb) NO_WANDB="--no_wandb"; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================"
echo "  OpenRCA Agent Evaluation"
echo "============================================"
echo "  Dataset:    $DATASET"
echo "  Range:      [$START_IDX, $END_IDX]"
echo "  Max Steps:  $MAX_STEP"
echo "  Max Turns:  $MAX_TURN"
echo "  Timeout:    ${TIMEOUT}s"
echo "  Tag:        $TAG"
echo "  Wandb:      ${NO_WANDB:-enabled}"
echo "============================================"

python -m rca.run_agent_standard \
    --dataset "$DATASET" \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --controller_max_step $MAX_STEP \
    --controller_max_turn $MAX_TURN \
    --timeout $TIMEOUT \
    --tag "$TAG" \
    --sample_num $SAMPLE_NUM \
    --wandb_project "$WANDB_PROJECT" \
    $NO_WANDB \
    $EXTRA_ARGS

echo "Done!"
