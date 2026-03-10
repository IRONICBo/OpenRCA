#!/bin/bash
# Stop all vLLM server instances (covers both `vllm serve` and ray workers)
echo "Stopping all vLLM instances..."

# Kill vllm serve processes
pkill -f "vllm serve" 2>/dev/null && \
    echo "  Killed vllm serve processes." || \
    echo "  No vllm serve processes found."

# Kill any legacy vllm.entrypoints processes
pkill -f "vllm.entrypoints" 2>/dev/null && \
    echo "  Killed vllm entrypoint processes." || \
    echo "  No vllm entrypoint processes found."

# Kill ray workers spawned by vLLM
pkill -f "ray::" 2>/dev/null && \
    echo "  Killed ray worker processes." || \
    echo "  No ray processes found."

echo "Done."
