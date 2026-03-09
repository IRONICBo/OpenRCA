#!/bin/bash
# Run full evaluation across all datasets and methods
# Usage: bash scripts/run_all.sh [--no_wandb]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NO_WANDB=""
if [[ "$1" == "--no_wandb" ]]; then
    NO_WANDB="--no_wandb"
fi

DATASETS=("Market/cloudbed-1" "Market/cloudbed-2" "Bank" "Telecom")

echo "============================================"
echo "  OpenRCA Full Evaluation Suite"
echo "============================================"
echo "  Datasets: ${DATASETS[*]}"
echo "  Wandb:    ${NO_WANDB:-enabled}"
echo "============================================"
echo ""

# 1. RCA-Agent on all datasets
echo ">>> Phase 1: RCA-Agent Evaluation"
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Running RCA-Agent on $ds ---"
    bash scripts/run_agent.sh --dataset "$ds" $NO_WANDB || {
        echo "WARNING: RCA-Agent on $ds failed, continuing..."
    }
done

# 2. Balanced Direct LM on all datasets
echo ""
echo ">>> Phase 2: Balanced Direct LM Evaluation"
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Running Balanced Direct on $ds ---"
    bash scripts/run_sampling.sh --method balanced --mode direct --dataset "$ds" $NO_WANDB || {
        echo "WARNING: Balanced Direct on $ds failed, continuing..."
    }
done

# 3. Balanced CoT LM on all datasets
echo ""
echo ">>> Phase 3: Balanced CoT LM Evaluation"
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Running Balanced CoT on $ds ---"
    bash scripts/run_sampling.sh --method balanced --mode cot --dataset "$ds" $NO_WANDB || {
        echo "WARNING: Balanced CoT on $ds failed, continuing..."
    }
done

# 4. Oracle Direct LM on all datasets
echo ""
echo ">>> Phase 4: Oracle Direct LM Evaluation"
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Running Oracle Direct on $ds ---"
    bash scripts/run_sampling.sh --method oracle --mode direct --dataset "$ds" $NO_WANDB || {
        echo "WARNING: Oracle Direct on $ds failed, continuing..."
    }
done

# 5. Oracle CoT LM on all datasets
echo ""
echo ">>> Phase 5: Oracle CoT LM Evaluation"
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "--- Running Oracle CoT on $ds ---"
    bash scripts/run_sampling.sh --method oracle --mode cot --dataset "$ds" $NO_WANDB || {
        echo "WARNING: Oracle CoT on $ds failed, continuing..."
    }
done

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Check wandb dashboard for results."
echo "  Local results in: test/result/"
echo "============================================"
