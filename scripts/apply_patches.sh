#!/usr/bin/env bash
set -euo pipefail

# Apply DiMP50 TensorRT patches into an existing PyTracking checkout.
#
# Usage:
#   bash scripts/apply_patches.sh /home/jetson/Desktop/pytracking
#
# Expected workspace:
#   /home/jetson/Desktop/pytracking/
#       pytracking/
#           ltr/
#           pytracking/
#       conversion/
#       torch2trt/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${1:-/home/jetson/Desktop/pytracking}"

PYTRACKING_REPO="$WORKSPACE/pytracking"

echo "============================================================"
echo "Apply DiMP50 TensorRT patches"
echo "REPO_ROOT:       $REPO_ROOT"
echo "WORKSPACE:       $WORKSPACE"
echo "PYTRACKING_REPO: $PYTRACKING_REPO"
echo "============================================================"

if [ ! -d "$PYTRACKING_REPO/ltr" ]; then
    echo "ERROR: ltr/ not found at:"
    echo "  $PYTRACKING_REPO/ltr"
    echo "Run bootstrap first:"
    echo "  bash scripts/bootstrap_workspace.sh $WORKSPACE"
    exit 1
fi

if [ ! -d "$PYTRACKING_REPO/pytracking" ]; then
    echo "ERROR: pytracking/ package not found at:"
    echo "  $PYTRACKING_REPO/pytracking"
    exit 1
fi

echo ""
echo "****************** Backing up original files ******************"
cp "$PYTRACKING_REPO/ltr/admin/loading.py" \
   "$PYTRACKING_REPO/ltr/admin/loading.py.bak" || true

cp "$PYTRACKING_REPO/pytracking/utils/loading.py" \
   "$PYTRACKING_REPO/pytracking/utils/loading.py.bak" || true

cp "$PYTRACKING_REPO/pytracking/parameter/dimp/dimp50.py" \
   "$PYTRACKING_REPO/pytracking/parameter/dimp/dimp50.py.bak" || true

echo ""
echo "****************** Applying patches ******************"
cp "$REPO_ROOT/patches/ltr_admin_loading_trt.py" \
   "$PYTRACKING_REPO/ltr/admin/loading.py"

cp "$REPO_ROOT/patches/pytracking_utils_loading_trt.py" \
   "$PYTRACKING_REPO/pytracking/utils/loading.py"

cp "$REPO_ROOT/patches/dimp50_trt.py" \
   "$PYTRACKING_REPO/pytracking/parameter/dimp/dimp50.py"

echo ""
echo "****************** Patching old torchvision model_urls import if present ******************"
RESNET_FILE="$PYTRACKING_REPO/ltr/models/backbone/resnet.py"

if grep -q "from torchvision.models.resnet import model_urls" "$RESNET_FILE"; then
    python3 - "$RESNET_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

old = "from torchvision.models.resnet import model_urls"
new = """try:
    from torchvision.models.resnet import model_urls
except ImportError:
    model_urls = {}"""

text = text.replace(old, new)
path.write_text(text)

print(f"Patched: {path}")
PY
else
    echo "model_urls import patch not needed."
fi

echo ""
echo "============================================================"
echo "Patches applied."
echo ""
echo "Next:"
echo "  cd $WORKSPACE"
echo "  source env/bin/activate"
echo "  python conversion/create_runtime_ckpt.py --in pytracking/pytracking/networks/dimp50.pth --out conversion/dimp50_runtime.pth --drop_prefix feature_extractor. --strip_optimizer"
echo "  python conversion/conversion_dimp50.py --pytracking-root $PYTRACKING_REPO --checkpoint pytracking/pytracking/networks/dimp50.pth --output-dir conversion --precision fp16 --batches 1 13"
echo "============================================================"
