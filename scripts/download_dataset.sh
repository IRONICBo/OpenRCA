#!/bin/bash
# Download OpenRCA datasets from Google Drive
# Usage: bash scripts/download_dataset.sh
#
# Requires: gdown (pip install gdown)
# Google Drive folder: https://drive.google.com/drive/folders/1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GDRIVE_FOLDER_ID="1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM"

echo "============================================"
echo "  OpenRCA Dataset Download"
echo "============================================"

# Check gdown
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Download entire folder
echo "Downloading dataset from Google Drive..."
echo "  Folder ID: ${GDRIVE_FOLDER_ID}"
echo "  Target:    dataset/"
echo ""

# gdown supports folder download with --folder flag
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" \
    -O dataset/ --remaining-ok

echo ""
echo "Verifying downloaded files..."
DATASETS=("Bank" "Telecom" "Market/cloudbed-1" "Market/cloudbed-2")
ALL_OK=true
for ds in "${DATASETS[@]}"; do
    if [ -f "dataset/$ds/query.csv" ] && [ -f "dataset/$ds/record.csv" ]; then
        TELEMETRY_COUNT=$(find "dataset/$ds/telemetry" -name "*.csv" 2>/dev/null | wc -l)
        echo "  [OK] $ds (${TELEMETRY_COUNT} telemetry files)"
    else
        echo "  [MISSING] $ds"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo "All datasets downloaded successfully!"
else
    echo "Some datasets are missing. You may need to download them manually."
    echo "  URL: https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"
    echo ""
    echo "If gdown fails with access errors, try:"
    echo "  1. Open the URL in browser and verify access"
    echo "  2. Use 'gdown --fuzzy' or download manually"
    echo "  3. For large files: gdown --id <FILE_ID> -O <output>"
fi

echo ""
echo "Dataset directory structure:"
ls -la dataset/ 2>/dev/null || echo "  dataset/ not found"
