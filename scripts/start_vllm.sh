#!/bin/bash
# Start vLLM server for Qwen model on B200 GPU
# Usage:
#   bash scripts/start_vllm.sh                          # default Qwen2.5-72B-Instruct
#   bash scripts/start_vllm.sh Qwen/Qwen2.5-7B-Instruct   # custom model
#   GPUS=4 bash scripts/start_vllm.sh                   # specify GPU count
set -e

MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Number of frames for video input (VL models), 0 to disable
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"

echo "============================================"
echo "  Starting vLLM Server"
echo "============================================"
echo "  Model:          $MODEL"
echo "  Port:           $PORT"
echo "  GPUs (TP):      $GPUS"
echo "  Max Model Len:  $MAX_MODEL_LEN"
echo "============================================"

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM not installed. Install with:"
    echo "  pip install vllm"
    exit 1
fi

# Check if port is already in use
if lsof -i :$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "WARNING: Port $PORT is already in use."
    echo "  Existing vLLM server might be running."
    echo "  Use 'kill \$(lsof -t -i:$PORT)' to stop it, or set PORT=xxxx."
    exit 1
fi

# Build extra args for VL models
EXTRA_ARGS=""
if [[ "$MODEL" == *"VL"* ]] || [[ "$MODEL" == *"vl"* ]]; then
    echo "  Detected VL (Vision-Language) model."
    # Limit multimodal inputs per prompt to avoid OOM
    if [ -n "$LIMIT_MM_PER_PROMPT" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --limit-mm-per-prompt image=$LIMIT_MM_PER_PROMPT"
    fi
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $PORT \
    --tensor-parallel-size $GPUS \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    $EXTRA_ARGS
