#!/bin/bash
# Step 1: Run zero-shot inference to generate predictions CSV
#
# Prerequisites:
#   1. Install BuildingsBench: pip install buildings-bench
#      Or clone from: https://github.com/NREL/BuildingsBench
#   2. Download BDG-2 data and model checkpoint (see BuildingsBench README)
#   3. Set BUILDINGS_BENCH environment variable to data directory
#
# Output: results/predictions_TransformerWithGaussian-L_bdg2_raw.csv
#   (~1.4 GB, 9.2M rows: per-hour predictions for 611 BDG-2 buildings)
#
# Note: This step requires GPU and takes ~2-4 hours.
#       If you have the predictions CSV, skip to Step 2.

set -e

# Check environment
if [ -z "$BUILDINGS_BENCH" ]; then
    echo "ERROR: Set BUILDINGS_BENCH=/path/to/data"
    exit 1
fi

python scripts/zero_shot.py \
    --model TransformerWithGaussian-L \
    --checkpoint checkpoints/TransformerWithGaussian-L_best.pt \
    --benchmark bdg-2 \
    --save_predictions \
    --results_path results/

echo "Predictions saved to results/predictions_TransformerWithGaussian-L_bdg2_raw.csv"
echo "Proceed to Step 2: python scripts/02_reproduce_paper.py"
