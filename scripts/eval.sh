#!/usr/bin/env bash

# Evaluation script for protein masked language model

set -e  # Exit on any error

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> [additional_args]"
    echo "Example: $0 checkpoints/toy_mlm/best_model.pt"
    echo "         $0 checkpoints/toy_mlm/epoch_10.pt --limit 1000"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift  # Remove first argument

echo "Evaluating model..."
echo "Checkpoint: $CHECKPOINT_PATH"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Run evaluation
python -m protein_mlm.evaluate --checkpoint "$CHECKPOINT_PATH" "$@"

echo "Evaluation completed!"