#!/bin/bash

if [ ! -d "Golden" ]; then
  echo "Goldne Path Does not exist."
  exit 1
fi

GOLDEN_DIR="./Golden"
if [ ! -d "test" ]; then
  mkdir test
  echo "create 'test' folder."
fi

TEST_DIR="./test"

for SEQ_PATH in "$GOLDEN_DIR"/*/test/seq-*; do
    SECOND_DIR=$(echo "$SEQ_PATH" | cut -d'/' -f3)
    SEQ_NAME=$(basename "$SEQ_PATH")
    echo "copy $SEQ_PATH -> $TEST_DIR/$SECOND_DIR-$SEQ_NAME"
    cp $SEQ_PATH $TEST_DIR/$SECOND_DIR-$SEQ_NAME
done


echo "All .ply files have been moved to the 'test' folder."