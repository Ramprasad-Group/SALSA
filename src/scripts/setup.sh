#!/usr/bin/env bash
set -euo pipefail

# Install subdirectories as editable so they can be locally modified
python -m pip install -e extern/ChartDete
python -m pip install -e extern/LineFormer

python src/scripts/patch_config.py extern/ChartDete/configs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py

python src/scripts/patch_lineformer.py

# Choose CUDA tag that matches your environment
# For Torch 1.13.1 + pytorch-cuda 11.7, use cu117
CUDA_TAG="${CUDA_TAG:-cu117}"

# Pins for MMDet 2.x stack
MMCV_PIN="${MMCV_PIN:-mmcv-full==1.7.2}"
MMDET_PIN="${MMDET_PIN:-mmdet==2.28.2}"
MMENGINE_PIN="${MMENGINE_PIN:-}"  # leave empty for 2.x stack

# Detect your Torch version (major.minor)
TORCH_VER="$(python - <<'PY'
import torch, re
v = torch.__version__.split('+')[0]
print(re.match(r'^\d+\.\d+', v).group(0))
PY
)"

echo "Torch version: $TORCH_VER"
echo "CUDA tag: $CUDA_TAG"

# Wheel index for prebuilt mmcv binaries
BASE_URL="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_VER}/index.html"
echo "Using wheel index: $BASE_URL"

# Optional: install mmengine if needed (for mmdet 3.x)
if [[ -n "$MMENGINE_PIN" ]]; then
  python -m pip install -U ${MMENGINE_PIN}
fi

# Install mmcv-full wheel (no compilation)
python -m pip install -U -f "$BASE_URL" ${MMCV_PIN}

# Install mmdet (2.x)
python -m pip install -U ${MMDET_PIN}

echo "Setup complete."