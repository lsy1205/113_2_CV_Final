#!/bin/bash

GOLDEN_DIR="../TA_Golden/test"
PREDICT_DIR="../test_dust_transformer"

for PLY_PATH in "$GOLDEN_DIR"/*; do
    echo "Processing $PLY_PATH"
    PLY_FILE=$(basename "$PLY_PATH")
    # echo "$PLY_FILE"    
    python utils.py --gt_ply "$PLY_PATH" --rec_ply "$PREDICT_DIR/$PLY_FILE"
done