#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Fine-tuning UNet on Oxford-IIIT Pet..."
.venv/bin/python run_experiment.py train
echo "Training complete. Checkpoint saved to results/checkpoints/best_model.pt"
