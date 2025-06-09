#!/bin/bash

SCENES_DIR="../7SCENES"
PRED_DIR="../predict"

for SEQ_PATH in "$SCENES_DIR"/*/te*/seq*; do
    REL_DIR=$(dirname "${SEQ_PATH#$SCENES_DIR/}")   # stairs/train
    SEQ_NAME=$(basename "$SEQ_PATH")                # seq-XX
    OUT_DIR="$PRED_DIR/$REL_DIR"
    mkdir -p "$OUT_DIR"
    PLY_PATH="$OUT_DIR/${SEQ_NAME}.ply"
    echo "Processing $SEQ_PATH -> $PLY_PATH"
    python seq2ply.py --seq_path "$SEQ_PATH" --ply_path "$PLY_PATH"
done

TEST_DIR="../test"

mkdir -p "$TEST_DIR"
for SEQ_PATH in "$PRED_DIR"/*/test/seq-*; do
    SECOND_DIR=$(echo "$SEQ_PATH" | cut -d'/' -f3)
    SEQ_NAME=$(basename "$SEQ_PATH")
    echo "copy $SEQ_PATH -> $TEST_DIR/$SECOND_DIR-$SEQ_NAME"
    cp $SEQ_PATH $TEST_DIR/$SECOND_DIR-$SEQ_NAME
done

echo "All .ply files have been moved to the 'test' folder."

rm -rf "$PRED_DIR"