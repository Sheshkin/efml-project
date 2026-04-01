#!/usr/bin/env bash
# Download Oxford-IIIT Pet segmentation dataset via torchvision
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "Downloading Oxford-IIIT Pet dataset..."
cd "$ROOT"
.venv/bin/python - <<'EOF'
from torchvision.datasets import OxfordIIITPet
print("Downloading trainval split...")
OxfordIIITPet(root="./data", split="trainval", target_types="segmentation", download=True)
print("Downloading test split...")
OxfordIIITPet(root="./data", split="test", target_types="segmentation", download=True)
print("Dataset downloaded successfully.")
EOF
echo "Done."
