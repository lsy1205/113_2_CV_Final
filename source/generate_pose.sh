#!/bin/bash

SCENES_DIR="../7SCENES"
POSE_DIR="./pose_fast"

for SEQ_PATH in "$SCENES_DIR"/*/t*/*; do
    REL_DIR=$(dirname "${SEQ_PATH#$SCENES_DIR/}")   # stairs/train
    SEQ_DIR=$(basename "$SEQ_PATH")                # seq-XX
    OUT_DIR="$POSE_DIR/$REL_DIR/$SEQ_DIR"
    echo "Processing $SEQ_PATH -> $OUT_DIR"
    mkdir -p "$OUT_DIR"
    python fast3r/usage.py --seq_path "$SEQ_PATH" --pose_path "$OUT_DIR"
done