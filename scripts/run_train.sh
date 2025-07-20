#!/usr/bin/env bash

# Training script for protein masked language model

set -e  # Exit on any error

echo "Starting protein MLM training..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check if CUDA is available
if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" | grep -q "True"; then
    echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
fi

# Run training
python -m protein_mlm.train --config configs/default.yaml "$@"

echo "Training completed!"