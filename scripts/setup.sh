#!/bin/bash
# Setup script for OpenRCA on B200 environment
set -e

echo "============================================"
echo "  OpenRCA Setup for B200 Environment"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Login to wandb (optional)
echo "[3/4] Configuring wandb..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "  WANDB_API_KEY not set. You can set it with:"
    echo "    export WANDB_API_KEY=your_api_key"
    echo "  Or run: wandb login"
else
    echo "  WANDB_API_KEY detected."
fi

# Check dataset
echo "[4/4] Checking dataset..."
if [ -d "dataset" ]; then
    echo "  Dataset directory found."
    for ds in "Bank" "Telecom" "Market/cloudbed-1" "Market/cloudbed-2"; do
        if [ -f "dataset/$ds/query.csv" ] && [ -f "dataset/$ds/record.csv" ]; then
            echo "    [OK] $ds"
        else
            echo "    [MISSING] $ds - Please download from Google Drive"
        fi
    done
else
    echo "  Dataset directory NOT found. Please download datasets from Google Drive."
    echo "  See dataset/README.md for instructions."
fi

# Check API config
echo ""
echo "============================================"
echo "  Configuration Check"
echo "============================================"
if [ -f "rca/api_config.yaml" ]; then
    echo "  api_config.yaml found."
    echo "  Please ensure your API_KEY is configured:"
    echo "    vim rca/api_config.yaml"
else
    echo "  WARNING: rca/api_config.yaml not found!"
fi

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "Run experiments with:"
echo "  bash scripts/run_agent.sh --dataset Bank"
echo "  bash scripts/run_all.sh"
