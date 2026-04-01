#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Running baseline FP16 benchmark..."
.venv/bin/python run_experiment.py baseline
