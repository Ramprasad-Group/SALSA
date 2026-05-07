#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root (this script lives at src/setup-scripts/setup.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Repository root: $REPO_ROOT"

# ------------------------------------------------------------------
# 1. Ensure git submodules are initialised
# ------------------------------------------------------------------
echo "==> Initialising git submodules..."
git -C "$REPO_ROOT" submodule update --init --recursive

# ------------------------------------------------------------------
# 2. Ensure Git LFS model artifacts are available
# ------------------------------------------------------------------
if ! git lfs version >/dev/null 2>&1; then
    echo "ERROR: Git LFS is required for model checkpoints. Install Git LFS, then rerun setup."
    exit 1
fi

echo "==> Pulling Git LFS model artifacts..."
git -C "$REPO_ROOT" lfs install --local
git -C "$REPO_ROOT" lfs pull

# ------------------------------------------------------------------
# 3. Install MMDetection 2.x stack (mmcv-full + mmdet)
#    Must come before ChartDete, whose setup.py imports torch + mmdet.
# ------------------------------------------------------------------
CUDA_TAG="${CUDA_TAG:-cu117}"
MMCV_PIN="${MMCV_PIN:-mmcv-full==1.7.2}"
MMDET_PIN="${MMDET_PIN:-mmdet==2.28.2}"
MMENGINE_PIN="${MMENGINE_PIN:-}"

# Detect Torch version (major.minor)
# Try importing torch first; fall back to pip metadata if the import fails
# (e.g. missing Intel runtime libs on first install).
TORCH_VER="$(python - <<'PY'
import re, sys
try:
    import torch
    v = torch.__version__
except Exception:
    from importlib.metadata import version
    v = version("torch")
v = v.split("+")[0]
m = re.match(r"^\d+\.\d+", v)
if not m:
    print("ERROR: cannot determine torch version", file=sys.stderr)
    sys.exit(1)
print(m.group(0))
PY
)"

echo "Torch version: $TORCH_VER  |  CUDA tag: $CUDA_TAG"

BASE_URL="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_VER}/index.html"
echo "==> Using wheel index: $BASE_URL"

if [[ -n "$MMENGINE_PIN" ]]; then
  python -m pip install -U ${MMENGINE_PIN}
fi

python -m pip install -U -f "$BASE_URL" ${MMCV_PIN}
python -m pip install -U ${MMDET_PIN}

# ------------------------------------------------------------------
# 4. Install ChartDete as editable so its custom mmdet heads are available
#    --no-build-isolation ensures setup.py sees the env's torch/mmdet.
# ------------------------------------------------------------------
if [ -d "$REPO_ROOT/extern/ChartDete" ] && [ "$(ls -A "$REPO_ROOT/extern/ChartDete")" ]; then
    echo "==> Installing ChartDete as editable package..."
    python -m pip install "$REPO_ROOT/extern/ChartDete" --no-build-isolation --no-deps
else
    echo "WARNING: extern/ChartDete is empty. Submodule init may have failed."
    exit 1
fi

# ------------------------------------------------------------------
# 5. Patch configuration files
# ------------------------------------------------------------------
CHARTDETE_CONFIG="$REPO_ROOT/extern/ChartDete/configs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py"
if [ -f "$CHARTDETE_CONFIG" ]; then
    echo "==> Patching ChartDete config..."
    python "$SCRIPT_DIR/patch_config.py" "$CHARTDETE_CONFIG"
fi

if [ -f "$REPO_ROOT/extern/LineFormer/infer.py" ]; then
    echo "==> Patching LineFormer infer.py..."
    python "$SCRIPT_DIR/patch_lineformer.py"
fi

# ------------------------------------------------------------------
# 6. Install the plextract package in editable mode
# ------------------------------------------------------------------
echo "==> Installing plextract (editable)..."
python -m pip install -e "$REPO_ROOT"

echo ""
echo "===== Setup complete ====="
echo "Launch the UI:  streamlit run src/ui/streamlit_app.py"
