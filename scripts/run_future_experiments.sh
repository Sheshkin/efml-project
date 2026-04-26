#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Running six core experiments..."
python run_future_experiments.py
