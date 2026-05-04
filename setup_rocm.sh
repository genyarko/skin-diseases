#!/usr/bin/env bash
# Exact copy of the original hackathon ROCm setup script.
# Uses the droplet's pre-built PyTorch ROCm wheel via --system-site-packages.

set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venv_skin_disease}"

echo "[setup] creating venv at $VENV_DIR (--system-site-packages)"
# CRITICAL: This flag is what makes the AMD Droplet work
python3 -m venv --system-site-packages "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub

# ---------------------------------------------------------------
# Guard: if a torch dep pulled in a CUDA build, remove it so the
# --system-site-packages ROCm torch wins again.
# ---------------------------------------------------------------
python - <<'PY'
import sys, torch, subprocess
hip = getattr(torch.version, "hip", None)
if not hip:
    print(f"[setup] venv torch={torch.__version__} has no HIP — uninstalling so system ROCm torch is used")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
else:
    print(f"[setup] venv torch has ROCm ({hip}) — ok")
PY

# ---------------------------------------------------------------
# GPU visibility sanity
# ---------------------------------------------------------------
python - <<'PY'
import torch
print(f"cuda avail   : {torch.cuda.is_available()}")
print(f"device count : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  GPU Name   : {torch.cuda.get_device_name(0)}")
PY

echo "[setup] Done. Run: source $VENV_DIR/bin/activate"
