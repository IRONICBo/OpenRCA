#!/bin/bash
# Run evaluation with Qwen3-30B-A3B via 8-GPU vLLM
#
# Usage:
#   # Step 1: Start vLLM (in another terminal)
#   MAX_MODEL_LEN=262144 MODEL=/home/bo/models/Qwen3-30B-A3B-Instruct-2507 bash scripts/start_vllm.sh --multi
#
#   # Step 2: Run evaluation
#   bash scripts/run_30b_eval.sh                      # all datasets
#   bash scripts/run_30b_eval.sh --dataset Bank       # single dataset
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Override model name to match vLLM served model
export MODEL_NAME="/home/bo/models/Qwen3-30B-A3B-Instruct-2507"

bash scripts/run_qwen_eval.sh \
    --parallel 8 \
    --tag qwen30b \
    "$@"
