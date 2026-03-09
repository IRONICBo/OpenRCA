#!/bin/bash
# Start vLLM server(s) for Qwen model on B200 GPUs
#
# Two modes:
#   Single instance (TP across GPUs):
#     bash scripts/start_vllm.sh
#     GPUS=4 bash scripts/start_vllm.sh
#
#   Multi-instance (one per GPU, for max throughput with small models):
#     bash scripts/start_vllm.sh --multi
#     NUM_INSTANCES=8 bash scripts/start_vllm.sh --multi
#
# 8B model on 8x B200: use --multi to get 8 independent instances
set -e

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
BASE_PORT="${BASE_PORT:-8000}"
GPUS="${GPUS:-1}"
NUM_INSTANCES="${NUM_INSTANCES:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MULTI_MODE=false

# Parse args
for arg in "$@"; do
    case $arg in
        --multi) MULTI_MODE=true ;;
        --model=*) MODEL="${arg#*=}" ;;
        *) MODEL="${arg}" ;;
    esac
done

# Check vLLM
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM not installed. Install with:"
    echo "  pip install vllm"
    exit 1
fi

# Detect available GPUs
TOTAL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Detected GPUs: ${TOTAL_GPUS}"

if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi

# Build extra args for VL models
VL_ARGS=""
if [[ "$MODEL" == *"VL"* ]] || [[ "$MODEL" == *"vl"* ]]; then
    echo "Detected VL (Vision-Language) model."
fi

if [ "$MULTI_MODE" = true ]; then
    # ========================================
    # Multi-instance mode: one vLLM per GPU
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
    echo "  GPUs:       0-$((NUM_INSTANCES - 1)) (1 GPU each)"
    echo "  Max Len:    $MAX_MODEL_LEN"
    echo "============================================"
    echo ""

    PIDS=()
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((BASE_PORT + i))

        # Check port
        if lsof -i :$PORT -sTCP:LISTEN >/dev/null 2>&1; then
            echo "  [SKIP] GPU $i - Port $PORT already in use"
            continue
        fi

        echo "  [START] GPU $i -> port $PORT"
        CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --port $PORT \
            --tensor-parallel-size 1 \
            --max-model-len $MAX_MODEL_LEN \
            --trust-remote-code \
            --dtype auto \
            --gpu-memory-utilization 0.90 \
            $VL_ARGS \
            > "logs/vllm_gpu${i}.log" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "All instances starting. PIDs: ${PIDS[*]}"
    echo "Logs: logs/vllm_gpu*.log"
    echo ""
    echo "Waiting for servers to be ready..."
    sleep 5

    READY=0
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((BASE_PORT + i))
        for attempt in $(seq 1 60); do
            if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
                echo "  [READY] GPU $i -> port $PORT"
                READY=$((READY + 1))
                break
            fi
            sleep 2
        done
    done

    echo ""
    echo "${READY}/${NUM_INSTANCES} instances ready."
    echo ""
    echo "To stop all: kill ${PIDS[*]}"
    echo "Or: bash scripts/stop_vllm.sh"

    # Wait for all
    wait

else
    # ========================================
    # Single instance mode (tensor parallel)
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

    # Check port
    if lsof -i :$BASE_PORT -sTCP:LISTEN >/dev/null 2>&1; then
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
        $VL_ARGS
fi
