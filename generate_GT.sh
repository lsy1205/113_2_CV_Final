#!/bin/bash

SCENES_DIR="./7SCENES"
GOLDEN_DIR="./Golden"

for SEQ_PATH in "$SCENES_DIR"/*/t*/seq-*; do
    REL_DIR=$(dirname "${SEQ_PATH#$SCENES_DIR/}")   # stairs/train
    SEQ_NAME=$(basename "$SEQ_PATH")                # seq-XX
    OUT_DIR="$GOLDEN_DIR/$REL_DIR"
    mkdir -p "$OUT_DIR"
    PLY_PATH="$OUT_DIR/${SEQ_NAME}.ply"
    echo "Processing $SEQ_PATH -> $PLY_PATH"
    python seq2ply.py --seq_path "$SEQ_PATH" --ply_path "$PLY_PATH"
done