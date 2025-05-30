#!/bin/bash

SCENES_DIR="./7SCENES"
POSE_DIR="./predict_pose"
PLY_DIR="./predict"

for SEQ_PATH in "$SCENES_DIR"/*/t*/*; do
    REL_DIR=$(dirname "${SEQ_PATH#$SCENES_DIR/}")   # stairs/train
    SEQ_NAME=$(basename "$SEQ_PATH")                # seq-XX

    POSE_PATH="$POSE_DIR/$REL_DIR/$SEQ_NAME"
    echo "Using pose path: $POSE_PATH"

    OUT_DIR="$PLY_DIR/$REL_DIR"
    mkdir -p "$OUT_DIR"

    PLY_PATH="$OUT_DIR/${SEQ_NAME}.ply"
    echo "Processing $SEQ_PATH -> $PLY_PATH"
    python seq2ply_pred.py --seq_path "$SEQ_PATH" --predict_pose_path "$POSE_PATH" --ply_path "$PLY_PATH"
done