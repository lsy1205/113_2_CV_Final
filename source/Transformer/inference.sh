#!/bin/bash

DATASET_DIR="../../7SCENES"
POSE_DIR="../pose_fast"
DEST_DIR="./refined_pose"
CHECKPOINT_PATH="./checkpoints/best.pth"

for SEQ_PATH in "$DATASET_DIR"/*/t*/*; do
    POSE_FILE="$SEQ_PATH/frame-000000.pose.txt"
    
    REL_DIR="${SEQ_PATH#$DATASET_DIR/}"   # stairs/train
    DEST_PATH="$DEST_DIR/$REL_DIR"
    mkdir -p "$DEST_PATH"

    echo "Copying pose file: $POSE_FILE to $DEST_PATH/frame-000000.pose.txt"
    cp "$POSE_FILE" "$DEST_PATH/frame-000000.pose.txt"
done

echo "Finish copying frame 0 pose files."

echo "Start to inference poses of all frames."

python inference.py --data_root $DATASET_DIR \
                    --predict_root $POSE_DIR \
                    --ckpt $CHECKPOINT_PATH \
                    --output_root $DEST_DIR