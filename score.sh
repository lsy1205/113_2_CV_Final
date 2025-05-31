#!/bin/bash

GOLDEN_DIR="./Golden"
PREDICT_DIR="./predict"

for SEQ_PATH in "$GOLDEN_DIR"/*/t*/*; do
    REL_DIR=$(dirname "${SEQ_PATH#$GOLDEN_DIR/}")   # stairs/train
    SEQ_NAME=$(basename "$SEQ_PATH")                # seq-XX

    
    GT_PATH="$SEQ_PATH"
    REC_PATH="$PREDICT_DIR/$REL_DIR/$SEQ_NAME"
    
    echo "Score for Ground Truth: $GT_PATH -> Reconstruction: $REC_PATH"
    
    python utils.py --gt_ply "$SEQ_PATH" --rec_ply "$REC_PATH"
done