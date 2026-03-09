#!/bin/bash
# Run sampling-based baseline (balanced or oracle) on a single dataset
# Usage: bash scripts/run_sampling.sh [options]
#
# Examples:
#   bash scripts/run_sampling.sh --method balanced --mode direct --dataset Bank
#   bash scripts/run_sampling.sh --method oracle --mode cot --dataset Telecom
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Default parameters
METHOD="${METHOD:-balanced}"
MODE="${MODE:-direct}"
DATASET="${DATASET:-Market/cloudbed-1}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-150}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-60}"
TAG="${TAG:-lm}"
SAMPLE_NUM="${SAMPLE_NUM:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-OpenRCA}"
NO_WANDB="${NO_WANDB:-}"

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --start_idx) START_IDX="$2"; shift 2 ;;
        --end_idx) END_IDX="$2"; shift 2 ;;
        --sample_interval) SAMPLE_INTERVAL="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --sample_num) SAMPLE_NUM="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --no_wandb) NO_WANDB="--no_wandb"; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================"
echo "  OpenRCA Sampling Baseline"
echo "============================================"
echo "  Method:     $METHOD"
echo "  Mode:       $MODE"
echo "  Dataset:    $DATASET"
echo "  Range:      [$START_IDX, $END_IDX]"
echo "  Interval:   ${SAMPLE_INTERVAL}s"
echo "  Tag:        $TAG"
echo "  Wandb:      ${NO_WANDB:-enabled}"
echo "============================================"

if [ "$METHOD" == "balanced" ]; then
    python -m rca.run_sampling_balanced \
        --dataset "$DATASET" \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --sample_interval $SAMPLE_INTERVAL \
        --mode "$MODE" \
        --tag "$TAG" \
        --sample_num $SAMPLE_NUM \
        --wandb_project "$WANDB_PROJECT" \
        $NO_WANDB \
        $EXTRA_ARGS
elif [ "$METHOD" == "oracle" ]; then
    python -m rca.run_sampling_oracle \
        --dataset "$DATASET" \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --sample_interval $SAMPLE_INTERVAL \
        --mode "$MODE" \
        --tag "$TAG" \
        --sample_num $SAMPLE_NUM \
        --wandb_project "$WANDB_PROJECT" \
        $NO_WANDB \
        $EXTRA_ARGS
else
    echo "ERROR: Unknown method: $METHOD. Use 'balanced' or 'oracle'."
    exit 1
fi

echo "Done!"
