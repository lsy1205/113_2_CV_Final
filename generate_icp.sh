#!/bin/bash

SCENES_DIR="./7SCENES"
POSE_DIR="./Transformer/refined_pose"
ICP_DIR="./icp_pose"

for SEQ_PATH in "$SCENES_DIR"/*/test/seq-*; do
    REL_DIR=$(dirname "${SEQ_PATH#$SCENES_DIR/}")   # stairs/train
    SEQ_NAME=$(basename "$SEQ_PATH")                # seq-XX

    POSE_PATH="$POSE_DIR/$REL_DIR/$SEQ_NAME"

    OUT_DIR="$ICP_DIR/$REL_DIR/$SEQ_NAME"
    mkdir -p "$OUT_DIR"

    echo "Processing $SEQ_PATH with pose path: $POSE_PATH -> $OUT_DIR"

    python ICP_correction.py --color_dir $SEQ_PATH \
                             --depth_dir $SEQ_PATH \
                             --pred_pose_dir $POSE_PATH \
                             --out_pose_dir $OUT_DIR 
done