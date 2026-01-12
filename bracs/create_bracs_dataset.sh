#!/bin/bash

SOURCE_DATASET="/app/bracs_datasets/BRACS_RoI/latest_version"
OUTPUT_BASE="/app/BRACS_multimag"

python bracs_create_multimag_dataset.py \
    --source "$SOURCE_DATASET" \
    --output "$OUTPUT_BASE" \
    --method progressive \
    --patch-size 224 \
    --mpps 0.25 0.375 0.5 0.75 1.0 1.5 2.0