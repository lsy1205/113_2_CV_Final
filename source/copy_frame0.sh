#!/bin/bash

DATASET_DIR="../7SCENES"
DEST_DIR="./predict_pose"

for SEQ_PATH in "$DATASET_DIR"/*/te*/*; do
    POSE_FILE="$SEQ_PATH/frame-000000.pose.txt"
    
    REL_DIR="${SEQ_PATH#$DATASET_DIR/}"   # stairs/train
    DEST_PATH="$DEST_DIR/$REL_DIR"
    mkdir -p "$DEST_PATH"

    echo "Copying pose file: $POSE_FILE to $DEST_PATH/frame-000000.pose.txt"
    cp "$POSE_FILE" "$DEST_PATH/frame-000000.pose.txt"
done