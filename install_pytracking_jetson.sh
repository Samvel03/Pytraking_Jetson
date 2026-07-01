#!/usr/bin/env bash
set -euo pipefail

# Jetson PyTracking + DiMP + torch2trt pip/venv installer
#
# Expected layout, based on your machine:
#
#   ~/Desktop/pytracking/
#       env/                 <- created by this script
#       pytracking/          <- actual visionml/pytracking repo, contains ltr/
#       torch2trt/           <- NVIDIA-AI-IOT/torch2trt repo
#       conversion/          <- optional conversion scripts
#
# Usage:
#   bash install.sh
#
# Optional:
#   bash install.sh /home/jetson/Desktop/pytracking
#
# Notes:
# - Uses venv, not conda.
# - Uses --system-site-packages so JetPack / apt TensorRT is visible.
# - Uses PYTHONNOUSERSITE=1 while installing to avoid accidentally satisfying
#   packages from ~/.local instead of installing into env.
# - spatial-correlation-sampler is skipped by default because it is only needed
#   for KYS tracker and commonly tries to download/compile another torch.
#   Set INSTALL_SPATIAL_CORR=1 if you really need it.

PROJECT_ROOT="${1:-$PWD}"
ENV_DIR="$PROJECT_ROOT/env"
PYTRACKING_REPO="$PROJECT_ROOT/pytracking"
TORCH2TRT_REPO="$PROJECT_ROOT/torch2trt"
NETWORK_DIR="$PYTRACKING_REPO/pytracking/networks"

TORCH_WHL="${TORCH_WHL:-/home/jetson/Downloads/torch-2.7.0-cp310-cp310-linux_aarch64.whl}"
TORCHVISION_WHL="${TORCHVISION_WHL:-/home/jetson/Downloads/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl}"

INSTALL_SPATIAL_CORR="${INSTALL_SPATIAL_CORR:-0}"
DOWNLOAD_DIMP50="${DOWNLOAD_DIMP50:-1}"

echo "============================================================"
echo "Jetson PyTracking pip/venv installer"
echo "PROJECT_ROOT:       $PROJECT_ROOT"
echo "PYTRACKING_REPO:    $PYTRACKING_REPO"
echo "TORCH2TRT_REPO:     $TORCH2TRT_REPO"
echo "TORCH_WHL:          $TORCH_WHL"
echo "TORCHVISION_WHL:    $TORCHVISION_WHL"
echo "============================================================"

if [ ! -d "$PYTRACKING_REPO" ]; then
    echo "ERROR: pytracking repo not found at: $PYTRACKING_REPO"
    echo "Expected: $PROJECT_ROOT/pytracking"
    exit 1
fi

if [ ! -d "$TORCH2TRT_REPO" ]; then
    echo "ERROR: torch2trt repo not found at: $TORCH2TRT_REPO"
    echo "Expected: $PROJECT_ROOT/torch2trt"
    exit 1
fi

if [ ! -f "$TORCH_WHL" ]; then
    echo "ERROR: torch wheel not found:"
    echo "  $TORCH_WHL"
    echo "Set it manually, for example:"
    echo "  TORCH_WHL=/path/to/torch.whl bash install.sh"
    exit 1
fi

if [ ! -f "$TORCHVISION_WHL" ]; then
    echo "ERROR: torchvision wheel not found:"
    echo "  $TORCHVISION_WHL"
    echo "Set it manually, for example:"
    echo "  TORCHVISION_WHL=/path/to/torchvision.whl bash install.sh"
    exit 1
fi

echo ""
echo "****************** Installing apt packages ******************"
sudo apt-get update
sudo apt-get install -y \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    ninja-build \
    libturbojpeg \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev

echo ""
echo "****************** Checking system TensorRT ******************"
python3 - <<'PY'
import tensorrt as trt
print("system tensorrt:", trt.__version__)
PY

echo ""
echo "****************** Recreating venv with system TensorRT access ******************"
if [ -d "$ENV_DIR" ]; then
    rm -rf "$ENV_DIR"
fi

python3 -m venv --system-site-packages "$ENV_DIR"

# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

echo ""
echo "****************** Confirming active Python/pip ******************"
which python
which pip
python --version

# Avoid ~/.local packages being used to satisfy installs.
export PYTHONNOUSERSITE=1

echo ""
echo "****************** Upgrading build tools ******************"
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "****************** Installing PyTorch / TorchVision wheels ******************"
python -m pip install --force-reinstall "$TORCH_WHL"
python -m pip install --force-reinstall --no-deps "$TORCHVISION_WHL"

echo ""
echo "****************** Installing PyTracking Python dependencies ******************"
# Keep numpy below 2 because older scientific/vision packages and TensorFlow on
# Jetson often expect NumPy 1.x. opencv-python has newer builds that request
# numpy>=2, so install a compatible OpenCV first, then pin numpy back.
python -m pip install \
    "numpy==1.26.4" \
    "opencv-python==4.9.0.80" \
    "matplotlib" \
    "pandas" \
    "tqdm" \
    "visdom" \
    "scikit-image" \
    "tikzplotlib" \
    "gdown" \
    "cython" \
    "pycocotools" \
    "lvis" \
    "jpeg4py" \
    "yacs" \
    "easydict" \
    "tensorboard"

echo ""
echo "****************** Initializing PyTracking submodules ******************"
cd "$PYTRACKING_REPO"
git submodule update --init --recursive || true

echo ""
echo "****************** Building PreciseRoIPooling if present ******************"
if [ -d "$PYTRACKING_REPO/ltr/external/PreciseRoIPooling/pytorch/prroi_pool" ]; then
    cd "$PYTRACKING_REPO/ltr/external/PreciseRoIPooling/pytorch/prroi_pool"
    python setup.py build_ext --inplace || {
        echo "WARNING: PreciseRoIPooling build failed. DiMP may still run depending on tracker path."
    }
else
    echo "PreciseRoIPooling source folder not found; skipping."
fi

echo ""
echo "****************** Installing torch2trt into venv ******************"
cd "$TORCH2TRT_REPO"
# Prefer modern editable install. Do not use: pip install setup.py
python -m pip install --no-build-isolation --no-deps -e .

echo ""
echo "****************** Optional spatial-correlation-sampler ******************"
if [ "$INSTALL_SPATIAL_CORR" = "1" ]; then
    # Disable build isolation so pip does not try to download a different torch.
    python -m pip install --no-build-isolation spatial-correlation-sampler || {
        echo "WARNING: spatial-correlation-sampler failed. This is required only for KYS tracker."
    }
else
    echo "Skipping spatial-correlation-sampler. Set INSTALL_SPATIAL_CORR=1 to install it."
fi

echo ""
echo "****************** Setting PYTHONPATH for ltr and pytracking ******************"
# conversion/conversion.py is outside the real repo in your layout, so it needs
# this path to find 'ltr.admin.loading'.
ENV_EXPORT_FILE="$ENV_DIR/bin/pytracking_env.sh"
cat > "$ENV_EXPORT_FILE" <<EOF
export PYTHONPATH="$PYTRACKING_REPO:\${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
EOF

# Also patch activate so future 'source env/bin/activate' works automatically.
if ! grep -q "pytracking_env.sh" "$ENV_DIR/bin/activate"; then
    cat >> "$ENV_DIR/bin/activate" <<EOF

# PyTracking local path
if [ -f "\$VIRTUAL_ENV/bin/pytracking_env.sh" ]; then
    source "\$VIRTUAL_ENV/bin/pytracking_env.sh"
fi
EOF
fi

# Load it for this install session.
source "$ENV_EXPORT_FILE"

echo ""
echo "****************** Creating local environment files ******************"
cd "$PYTRACKING_REPO"
python - <<'PY'
try:
    from pytracking.evaluation.environment import create_default_local_file
    create_default_local_file()
    print("created pytracking local.py")
except Exception as e:
    print("WARNING: failed to create pytracking local.py:", repr(e))

try:
    from ltr.admin.environment import create_default_local_file
    create_default_local_file()
    print("created ltr local.py")
except Exception as e:
    print("WARNING: failed to create ltr local.py:", repr(e))
PY

echo ""
echo "****************** Downloading DiMP50 network ******************"
mkdir -p "$NETWORK_DIR"

if [ "$DOWNLOAD_DIMP50" = "1" ]; then
    if [ -f "$NETWORK_DIR/dimp50.pth" ]; then
        echo "dimp50.pth already exists; skipping download."
    else
        # This is a large file. If the network fails, rerun the script or run this command manually.
        gdown --fuzzy "https://drive.google.com/uc?id=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN" \
            -O "$NETWORK_DIR/dimp50.pth" || {
            echo "WARNING: DiMP50 download failed or was interrupted."
            echo "Run manually later:"
            echo "  source $ENV_DIR/bin/activate"
            echo "  gdown --fuzzy 'https://drive.google.com/uc?id=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN' -O '$NETWORK_DIR/dimp50.pth'"
        }
    fi
else
    echo "Skipping DiMP50 download. Set DOWNLOAD_DIMP50=1 to download."
fi

echo ""
echo "****************** Final verification ******************"
cd "$PROJECT_ROOT"
python - <<'PY'
import os
import sys

print("python:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

import torch
import torchvision
import tensorrt as trt
from torch2trt import torch2trt
import cv2
import numpy as np

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("tensorrt:", trt.__version__)
print("cuda available:", torch.cuda.is_available())
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)

import ltr.admin.loading as ltr_loading
print("ltr import: OK")

import pytracking
print("pytracking import: OK")

print("torch2trt import: OK")
PY

echo ""
echo "============================================================"
echo "Installation complete."
echo ""
echo "Use the environment like this:"
echo "  cd $PROJECT_ROOT"
echo "  source env/bin/activate"
echo "============================================================"
