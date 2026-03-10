#!/bin/bash
# Run full evaluation using WanQing API (Claude Opus 4.5)
#
# Usage:
#   bash scripts/run_wanqing_eval.sh                      # all datasets
#   bash scripts/run_wanqing_eval.sh --dataset Bank       # single dataset
#   bash scripts/run_wanqing_eval.sh --no_wandb           # disable wandb
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-fintech-copilot}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Config
API_CONFIG="${API_CONFIG:-rca/api_config_wanqing.yaml}"
TAG="${TAG:-claude}"
DATASET="${DATASET:-}"
NO_WANDB=""
WANDB_PROJECT="${WANDB_PROJECT:-OpenRCA}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --no_wandb) NO_WANDB="--no_wandb"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --config) API_CONFIG="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ ! -f "$API_CONFIG" ]; then
    echo "ERROR: Config file not found: $API_CONFIG"
    echo "  Create it with your WanQing API key."
    exit 1
fi

# Read model name from config
MODEL_NAME=$(grep "MODEL:" "$API_CONFIG" | awk '{print $2}' | tr -d '"')

DATASETS_ALL=("Market/cloudbed-1" "Market/cloudbed-2" "Bank" "Telecom")
if [ -n "$DATASET" ]; then
    DATASETS=("$DATASET")
else
    DATASETS=("${DATASETS_ALL[@]}")
fi

echo "============================================"
echo "  OpenRCA WanQing API Evaluation"
echo "============================================"
echo "  Model:    $MODEL_NAME"
echo "  Config:   $API_CONFIG"
echo "  Datasets: ${DATASETS[*]}"
echo "  Tag:      $TAG"
echo "  Wandb:    ${NO_WANDB:-enabled}"
echo "============================================"
echo ""

mkdir -p logs

for DS in "${DATASETS[@]}"; do
    TOTAL=$(python -c "import pandas as pd; print(len(pd.read_csv('dataset/${DS}/query.csv')))" 2>/dev/null || echo "?")
    END_IDX=$((TOTAL - 1))
    LOGFILE="logs/wanqing_${DS//\//_}.log"

    echo ">>> Dataset: $DS ($TOTAL tasks, idx 0-$END_IDX)"

    RCA_API_CONFIG="$API_CONFIG" python -m rca.run_agent_standard \
        --dataset "$DS" \
        --start_idx 0 \
        --end_idx $END_IDX \
        --tag "$TAG" \
        --wandb_project "$WANDB_PROJECT" \
        $NO_WANDB \
        2>&1 | tee "$LOGFILE" || {
        echo "WARNING: $DS failed, continuing..."
    }

    echo ""
done

echo "============================================"
echo "  All evaluations complete!"
echo "  Results: test/result/"
echo "  Logs:    logs/wanqing_*.log"
echo "============================================"
