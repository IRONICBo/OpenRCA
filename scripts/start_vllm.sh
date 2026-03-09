#!/bin/bash
# Start vLLM server(s) for Qwen model on B200 GPUs
#
# Single instance:
#   bash scripts/start_vllm.sh
#
# Multi-instance (8-GPU, one per GPU):
#   bash scripts/start_vllm.sh --multi
#   NUM_INSTANCES=4 bash scripts/start_vllm.sh --multi
set -e

CONDA_ENV="${CONDA_ENV:-fintech-copilot}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
BASE_PORT="${BASE_PORT:-8000}"
GPUS="${GPUS:-1}"
NUM_INSTANCES="${NUM_INSTANCES:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache/huggingface/hub}"
# Max seconds to wait for each instance to become ready
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"
MULTI_MODE=false

for arg in "$@"; do
    case $arg in
        --multi) MULTI_MODE=true ;;
        --model=*) MODEL="${arg#*=}" ;;
    esac
done

# Check vLLM
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM not installed. Install with: pip install vllm"
    exit 1
fi

# Detect GPUs
TOTAL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Detected GPUs: ${TOTAL_GPUS}"
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi

mkdir -p logs

# VL model extra args
VL_ARGS=""
if [[ "$MODEL" == *"VL"* ]] || [[ "$MODEL" == *"vl"* ]]; then
    echo "Detected VL (Vision-Language) model."
fi

if [ "$MULTI_MODE" = true ]; then
    # ========================================
    # Multi-instance: one vLLM per GPU
    # ========================================
    if [ "$NUM_INSTANCES" -gt "$TOTAL_GPUS" ]; then
        NUM_INSTANCES=$TOTAL_GPUS
    fi

    echo "============================================"
    echo "  Starting vLLM Multi-Instance"
    echo "============================================"
    echo "  Model:      $MODEL"
    echo "  Instances:  $NUM_INSTANCES"
    echo "  Ports:      ${BASE_PORT}-$((BASE_PORT + NUM_INSTANCES - 1))"
    echo "  Max Len:    $MAX_MODEL_LEN"
    echo "  Cache:      $MODEL_CACHE"
    echo "  Timeout:    ${WAIT_TIMEOUT}s per instance"
    echo "============================================"
    echo ""

    # Step 1: Pre-download model once (avoid 8 concurrent downloads)
    echo "[Step 1] Pre-downloading model (if needed)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', cache_dir='${MODEL_CACHE}')
print('Model cached.')
" 2>&1 | tail -3
    echo ""

    # Step 2: Start instances one by one
    echo "[Step 2] Starting instances..."
    PIDS=()
    READY_COUNT=0

    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((BASE_PORT + i))

        if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
           lsof -i :$PORT -sTCP:LISTEN >/dev/null 2>&1; then
            echo "  [SKIP] Port $PORT already in use"
            continue
        fi

        echo "  [START] GPU $i -> port $PORT ..."
        CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --port $PORT \
            --tensor-parallel-size 1 \
            --max-model-len $MAX_MODEL_LEN \
            --trust-remote-code \
            --dtype auto \
            --gpu-memory-utilization 0.90 \
            --download-dir "$MODEL_CACHE" \
            $VL_ARGS \
            > "logs/vllm_gpu${i}.log" 2>&1 &
        PIDS+=($!)

        # Wait for this instance to be ready before starting next
        ELAPSED=0
        while [ $ELAPSED -lt $WAIT_TIMEOUT ]; do
            if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
                echo "  [READY] GPU $i -> port $PORT (${ELAPSED}s)"
                READY_COUNT=$((READY_COUNT + 1))
                break
            fi
            # Check if process died
            if ! kill -0 ${PIDS[-1]} 2>/dev/null; then
                echo "  [FAILED] GPU $i -> process died. Check logs/vllm_gpu${i}.log"
                tail -5 "logs/vllm_gpu${i}.log" 2>/dev/null
                break
            fi
            sleep 3
            ELAPSED=$((ELAPSED + 3))
        done

        if [ $ELAPSED -ge $WAIT_TIMEOUT ]; then
            echo "  [TIMEOUT] GPU $i -> port $PORT did not become ready in ${WAIT_TIMEOUT}s"
            echo "  Last log lines:"
            tail -5 "logs/vllm_gpu${i}.log" 2>/dev/null
        fi
    done

    echo ""
    echo "============================================"
    echo "  ${READY_COUNT}/${NUM_INSTANCES} instances ready"
    echo "  PIDs: ${PIDS[*]}"
    echo "  Logs: logs/vllm_gpu*.log"
    echo "============================================"
    echo ""
    echo "To stop: bash scripts/stop_vllm.sh"
    echo ""

    if [ $READY_COUNT -eq 0 ]; then
        echo "ERROR: No instances started successfully."
        echo "Check logs: tail logs/vllm_gpu0.log"
        exit 1
    fi

    # Keep script alive so Ctrl+C kills all
    echo "Press Ctrl+C to stop all instances."
    wait

else
    # ========================================
    # Single instance (tensor parallel)
    # ========================================
    if [ "$GPUS" -gt "$TOTAL_GPUS" ]; then
        GPUS=$TOTAL_GPUS
    fi

    echo "============================================"
    echo "  Starting vLLM (Single Instance)"
    echo "============================================"
    echo "  Model:      $MODEL"
    echo "  Port:       $BASE_PORT"
    echo "  GPUs (TP):  $GPUS"
    echo "  Max Len:    $MAX_MODEL_LEN"
    echo "============================================"

    if ss -tlnp 2>/dev/null | grep -q ":${BASE_PORT} " || \
       lsof -i :$BASE_PORT -sTCP:LISTEN >/dev/null 2>&1; then
        echo "WARNING: Port $BASE_PORT already in use."
        exit 1
    fi

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $BASE_PORT \
        --tensor-parallel-size $GPUS \
        --max-model-len $MAX_MODEL_LEN \
        --trust-remote-code \
        --dtype auto \
        --gpu-memory-utilization 0.90 \
        --download-dir "$MODEL_CACHE" \
        $VL_ARGS
fi
