#!/bin/bash
# Stop all vLLM server instances
echo "Stopping all vLLM instances..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && \
    echo "All vLLM instances stopped." || \
    echo "No running vLLM instances found."
