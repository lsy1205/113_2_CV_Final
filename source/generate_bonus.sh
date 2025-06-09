#!/bin/bash

SCENES_DIR="../7SCENES"
POSE_DIR="./pose_dust"
PRED_DIR="../predict_bonus"
PYTHON_PATH=./seq2ply_pred.py

for SEQ_PATH in "$SCENES_DIR"/*/te*/sp*; do
    REL_DIR=$(dirname "${SEQ_PATH#$SCENES_DIR/}")   # stairs/train
    SEQ_NAME=$(basename "$SEQ_PATH")                # seq-XX

    POSE_PATH="$POSE_DIR/$REL_DIR/$SEQ_NAME"
    echo "Using pose path: $POSE_PATH"

    OUT_DIR="$PRED_DIR/$REL_DIR"
    mkdir -p "$OUT_DIR"

    PLY_PATH="$OUT_DIR/${SEQ_NAME}.ply"
    echo "Processing $SEQ_PATH -> $PLY_PATH"
    python $PYTHON_PATH --seq_path "$SEQ_PATH" --predict_pose_path "$POSE_PATH" --ply_path "$PLY_PATH" --kf_every 1
done

GOAL_PATH="../bonus"

for PLY_PATH in "$PRED_DIR"/*/test/sp*; do
    # echo "Processing $PLY_PATH"
    PLY_FILE=$(basename "$PLY_PATH")
    SCENE=$(echo "$PLY_PATH" | cut -d'/' -f3)
    # echo "Copying $PLY_FILE to $GOAL_PATH"
    cp "$PLY_PATH" "$GOAL_PATH/$SCENE-$PLY_FILE"
    echo "Copied $PLY_PATH to $GOAL_PATH/$SCENE-$PLY_FILE"
done

rm -rf "$PRED_DIR"