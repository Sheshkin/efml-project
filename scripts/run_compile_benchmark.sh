#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Running torch.compile benchmark..."
python run_experiment.py compile
