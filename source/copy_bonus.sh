#!/bin/bash
GOAL_PATH="../bonus"
TEST_DIR="./predict"

for PLY_PATH in "$TEST_DIR"/*/test/sp*; do
    # echo "Processing $PLY_PATH"
    PLY_FILE=$(basename "$PLY_PATH")
    SCENE=$(echo "$PLY_PATH" | cut -d'/' -f3)
    # echo "Copying $PLY_FILE to $GOAL_PATH"
    cp "$PLY_PATH" "$GOAL_PATH/$SCENE-$PLY_FILE"
    echo "Copied $PLY_PATH to $GOAL_PATH/$SCENE-$PLY_FILE"
done
