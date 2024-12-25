#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
python src/train.py 2>&1 | tee logs/training_${timestamp}.log