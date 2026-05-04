#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venv_skin_disease}"

echo "=== Creating Virtual Environment ==="
# Using standard venv
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Upgrading pip ==="
python -m pip install --upgrade pip

echo "=== Installing PyTorch for ROCm 6.2 ==="
# CRITICAL: Force pip to use the ROCm wheel index
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio

echo "=== Installing other dependencies ==="
pip install -r requirements.txt

echo "=== Verifying ROCm GPU access ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count  : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU Name      : {torch.cuda.get_device_name(0)}')
"

echo "Done! Activate with: source $VENV_DIR/bin/activate"
