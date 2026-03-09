#!/bin/bash
# End-to-end evaluation with Qwen via vLLM
#
# Single worker (1 vLLM instance):
#   bash scripts/run_qwen_eval.sh --dataset Bank
#
# 8-GPU parallel (8 vLLM instances, 8 workers):
#   bash scripts/run_qwen_eval.sh --parallel 8 --dataset Bank
#   bash scripts/run_qwen_eval.sh --parallel 8                   # all datasets
#
# Remote vLLM:
#   VLLM_HOST=10.0.0.1 bash scripts/run_qwen_eval.sh --dataset Bank
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-fintech-copilot}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# vLLM connection
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Defaults
METHOD="${METHOD:-agent}"
MODE="${MODE:-direct}"
DATASET="${DATASET:-}"
TAG="${TAG:-qwen}"
NO_WANDB=""
WANDB_PROJECT="${WANDB_PROJECT:-OpenRCA}"
PARALLEL="${PARALLEL:-1}"

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --no_wandb) NO_WANDB="--no_wandb"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --vllm_host) VLLM_HOST="$2"; shift 2 ;;
        --vllm_port) VLLM_PORT="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Check first vLLM instance
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo "Checking vLLM server at ${VLLM_URL} ..."
if ! curl -s --max-time 5 "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach vLLM at ${VLLM_URL}"
    echo "  Start with: bash scripts/start_vllm.sh --multi"
    exit 1
fi

SERVED_MODEL=$(curl -s "${VLLM_URL}/models" | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

# Count available instances
AVAILABLE_INSTANCES=0
for i in $(seq 0 $((PARALLEL - 1))); do
    P=$((VLLM_PORT + i))
    if curl -s --max-time 2 "http://${VLLM_HOST}:${P}/v1/models" > /dev/null 2>&1; then
        AVAILABLE_INSTANCES=$((AVAILABLE_INSTANCES + 1))
    fi
done

if [ "$PARALLEL" -gt 1 ] && [ "$AVAILABLE_INSTANCES" -lt "$PARALLEL" ]; then
    echo "WARNING: Requested ${PARALLEL} workers but only ${AVAILABLE_INSTANCES} vLLM instances available."
    echo "  Start more with: NUM_INSTANCES=${PARALLEL} bash scripts/start_vllm.sh --multi"
    PARALLEL=$AVAILABLE_INSTANCES
fi

echo ""
echo "============================================"
echo "  OpenRCA Qwen/vLLM Evaluation"
echo "============================================"
echo "  Model:      ${SERVED_MODEL}"
echo "  Server:     ${VLLM_HOST}:${VLLM_PORT}"
echo "  Instances:  ${AVAILABLE_INSTANCES}"
echo "  Workers:    ${PARALLEL}"
echo "  Method:     ${METHOD}"
echo "  Dataset:    ${DATASET:-all}"
echo "  Wandb:      ${NO_WANDB:-enabled}"
echo "============================================"
echo ""

DATASETS_ALL=("Market/cloudbed-1" "Market/cloudbed-2" "Bank" "Telecom")
if [ -n "$DATASET" ]; then
    DATASETS=("$DATASET")
else
    DATASETS=("${DATASETS_ALL[@]}")
fi

# Create log dir
mkdir -p logs

run_single_worker() {
    # Args: WORKER_ID DATASET METHOD MODE PORT START END
    local WID=$1 DS=$2 MTH=$3 MD=$4 PT=$5 START=$6 END=$7
    local WORKER_URL="http://${VLLM_HOST}:${PT}/v1"
    local WORKER_TAG="${TAG}_w${WID}"

    # Write per-worker config
    local CFG="rca/api_config_worker_${WID}.yaml"
    cat > "$CFG" <<EOFCFG
SOURCE:     "vLLM"
MODEL:      "${SERVED_MODEL}"
API_KEY:    "EMPTY"
API_BASE:   "${WORKER_URL}"
MAX_TOKENS: 8192
MAX_RETRIES: 5
EOFCFG

    if [ "$MTH" == "agent" ]; then
        RCA_API_CONFIG="$CFG" python -m rca.run_agent_standard \
            --dataset "$DS" \
            --start_idx $START --end_idx $END \
            --tag "$WORKER_TAG" \
            --wandb_project "$WANDB_PROJECT" \
            $NO_WANDB $EXTRA_ARGS \
            > "logs/worker_${WID}_${DS//\//_}.log" 2>&1
    else
        RCA_API_CONFIG="$CFG" python -m rca.run_sampling_${MTH} \
            --dataset "$DS" \
            --start_idx $START --end_idx $END \
            --mode "$MD" \
            --tag "$WORKER_TAG" \
            --wandb_project "$WANDB_PROJECT" \
            $NO_WANDB $EXTRA_ARGS \
            > "logs/worker_${WID}_${DS//\//_}.log" 2>&1
    fi
}

for DS in "${DATASETS[@]}"; do
    echo ">>> Dataset: $DS"

    # Get total task count
    TOTAL=$(python -c "import pandas as pd; print(len(pd.read_csv('dataset/${DS}/query.csv')))" 2>/dev/null || echo "150")
    echo "  Total tasks: $TOTAL"

    if [ "$PARALLEL" -le 1 ]; then
        # Single worker — use default config
        cat > rca/api_config.yaml <<EOF
SOURCE:     "vLLM"
MODEL:      "${SERVED_MODEL}"
API_KEY:    "EMPTY"
API_BASE:   "${VLLM_URL}"
MAX_TOKENS: 8192
MAX_RETRIES: 5
EOF
        if [ "$METHOD" == "agent" ]; then
            bash scripts/run_agent.sh --dataset "$DS" --tag "$TAG" \
                --wandb_project "$WANDB_PROJECT" $NO_WANDB $EXTRA_ARGS || {
                echo "WARNING: $DS failed, continuing..."
            }
        else
            bash scripts/run_sampling.sh --method "$METHOD" --mode "$MODE" \
                --dataset "$DS" --tag "$TAG" \
                --wandb_project "$WANDB_PROJECT" $NO_WANDB $EXTRA_ARGS || {
                echo "WARNING: $DS failed, continuing..."
            }
        fi
    else
        # Multi-worker parallel
        CHUNK=$(( (TOTAL + PARALLEL - 1) / PARALLEL ))
        PIDS=()
        echo "  Launching $PARALLEL workers (chunk size: $CHUNK tasks each)..."

        for W in $(seq 0 $((PARALLEL - 1))); do
            START=$((W * CHUNK))
            END=$(( (W + 1) * CHUNK - 1 ))
            if [ $END -ge $TOTAL ]; then END=$((TOTAL - 1)); fi
            if [ $START -ge $TOTAL ]; then break; fi

            PORT=$((VLLM_PORT + W))
            echo "  [Worker $W] tasks [$START, $END] -> port $PORT"

            run_single_worker $W "$DS" "$METHOD" "$MODE" $PORT $START $END &
            PIDS+=($!)
        done

        echo "  Waiting for ${#PIDS[@]} workers..."
        FAILED=0
        for pid in "${PIDS[@]}"; do
            wait $pid || FAILED=$((FAILED + 1))
        done

        if [ $FAILED -gt 0 ]; then
            echo "  WARNING: $FAILED worker(s) failed. Check logs/worker_*.log"
        else
            echo "  All workers done for $DS"
        fi

        # Cleanup per-worker configs
        rm -f rca/api_config_worker_*.yaml
    fi

    echo ""
done

echo "============================================"
echo "  Evaluation complete!"
echo "  Results:  test/result/"
echo "  Logs:     logs/"
echo "  Wandb:    https://wandb.ai"
echo "============================================"
