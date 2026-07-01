#!/usr/bin/env bash
set -euo pipefail

# Creates a runnable Jetson workspace from this GitHub repo.
#
# Default:
#   REPO_ROOT = current repo
#   WORKSPACE = /home/jetson/Desktop/pytracking
#
# Usage:
#   bash scripts/bootstrap_workspace.sh
#
# Optional:
#   bash scripts/bootstrap_workspace.sh /home/jetson/Desktop/pytracking

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${1:-/home/jetson/Desktop/pytracking}"

PYTRACKING_URL="${PYTRACKING_URL:-https://github.com/visionml/pytracking.git}"
TORCH2TRT_URL="${TORCH2TRT_URL:-https://github.com/NVIDIA-AI-IOT/torch2trt.git}"

echo "============================================================"
echo "Bootstrap PyTracking Jetson TRT workspace"
echo "REPO_ROOT:  $REPO_ROOT"
echo "WORKSPACE:  $WORKSPACE"
echo "============================================================"

mkdir -p "$WORKSPACE"

echo ""
echo "****************** Cloning PyTracking ******************"
if [ -d "$WORKSPACE/pytracking/.git" ]; then
    echo "PyTracking already exists: $WORKSPACE/pytracking"
else
    git clone "$PYTRACKING_URL" "$WORKSPACE/pytracking"
fi

echo ""
echo "****************** Cloning torch2trt ******************"
if [ -d "$WORKSPACE/torch2trt/.git" ]; then
    echo "torch2trt already exists: $WORKSPACE/torch2trt"
else
    git clone "$TORCH2TRT_URL" "$WORKSPACE/torch2trt"
fi

echo ""
echo "****************** Copying conversion scripts ******************"
mkdir -p "$WORKSPACE/conversion"
cp "$REPO_ROOT/conversion/conversion_dimp50.py" "$WORKSPACE/conversion/"
cp "$REPO_ROOT/conversion/create_runtime_ckpt.py" "$WORKSPACE/conversion/"

if [ -f "$REPO_ROOT/conversion/test_trt_backbone.py" ]; then
    cp "$REPO_ROOT/conversion/test_trt_backbone.py" "$WORKSPACE/conversion/"
fi

echo ""
echo "****************** Copying installer ******************"
cp "$REPO_ROOT/install_pytracking_jetson.sh" "$WORKSPACE/install_pytracking_jetson.sh"
chmod +x "$WORKSPACE/install_pytracking_jetson.sh"

echo ""
echo "============================================================"
echo "Workspace ready:"
echo "  $WORKSPACE"
echo ""
echo "Next:"
echo "  cd $WORKSPACE"
echo "  bash install_pytracking_jetson.sh $WORKSPACE"
echo "============================================================"
