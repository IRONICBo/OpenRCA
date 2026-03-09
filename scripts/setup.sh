#!/bin/bash
# Setup script for OpenRCA on B200 environment (conda)
# Usage: bash scripts/setup.sh
set -e

echo "============================================"
echo "  OpenRCA Setup for B200 Environment"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-fintech-copilot}"

# 1. Activate conda environment
echo "[1/5] Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "  Python: $(which python)"

# 2. Install dependencies
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install vLLM
echo "[3/5] Installing vLLM..."
if python -c "import vllm" 2>/dev/null; then
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)")
    echo "  vLLM already installed: v${VLLM_VER}"
else
    echo "  Installing vLLM (this may take a few minutes)..."
    pip install vllm
fi

# 4. wandb
echo "[4/5] Configuring wandb..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "  WANDB_API_KEY not set. Set it with:"
    echo "    export WANDB_API_KEY=your_api_key"
    echo "  Or run: wandb login"
else
    echo "  WANDB_API_KEY detected."
fi

# 5. Dataset
echo "[5/5] Checking dataset..."
MISSING=false
for ds in "Bank" "Telecom" "Market/cloudbed-1" "Market/cloudbed-2"; do
    if [ -f "dataset/$ds/query.csv" ] && [ -f "dataset/$ds/record.csv" ]; then
        echo "  [OK] $ds"
    else
        echo "  [MISSING] $ds"
        MISSING=true
    fi
done

if [ "$MISSING" = true ]; then
    echo ""
    read -p "  Download datasets from Google Drive? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash scripts/download_dataset.sh
    else
        echo "  Skipped. Run later: bash scripts/download_dataset.sh"
    fi
fi

# Create log directory
mkdir -p logs

# GPU check
echo ""
echo "============================================"
echo "  Environment Check"
echo "============================================"
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NGPUS" -gt 0 ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    echo "  GPUs:    $NGPUS x $GPU_NAME"
else
    echo "  GPUs:    None detected (CPU only)"
fi
echo "  Conda:   $CONDA_ENV"
echo "  Python:  $(python --version 2>&1)"
echo "  vLLM:    $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"
echo "  wandb:   $(python -c 'import wandb; print(wandb.__version__)' 2>/dev/null || echo 'not installed')"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Quick start (8-GPU B200):"
echo ""
echo "  # Terminal 1: Start 8 vLLM instances (one per GPU)"
echo "  conda activate $CONDA_ENV"
echo "  bash scripts/start_vllm.sh --multi"
echo ""
echo "  # Terminal 2: Run evaluation with 8 parallel workers"
echo "  conda activate $CONDA_ENV"
echo "  bash scripts/run_qwen_eval.sh --parallel 8 --dataset Bank"
echo ""
echo "  # Stop vLLM when done"
echo "  bash scripts/stop_vllm.sh"
