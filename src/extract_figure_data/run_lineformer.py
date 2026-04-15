#!/usr/bin/env python3
"""
Local LineFormer wrapper (no Modal).

Responsibilities
- Ensure LineFormer weights & config are available (via Hugging Face snapshot)
- Load the model once per process
- Run inference on a single image and write results under:
  {PLEXTRACT_BASE_OUTPUT}/{img}/lineformer/{prediction.png, coordinates.json}

Environment variables (optional)
- PLEXTRACT_BASE_INPUT:  path to {run_id}/input      (set by main.py)
- PLEXTRACT_BASE_OUTPUT: path to {run_id}/output     (set by main.py)
- LINEFORMER_HF_DIR:     directory to put HF snapshot (default: ./huggingface)
- LINEFORMER_CONFIG:     path to LineFormer config .py inside HF dir
- LINEFORMER_CKPT:       path to LineFormer checkpoint .pth inside HF dir
- LINEFORMER_DEVICE:     e.g., "cuda:0" or "cpu" (default: auto)
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from huggingface_hub import snapshot_download

import sys
import os

# Directory where LineFormer can be found
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../extern"))
sys.path.insert(0, parent_dir)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_DIR = _REPO_ROOT / "huggingface"

from LineFormer.infer import load_model, get_dataseries
from LineFormer.line_utils import draw_lines, points_to_array

def _auto_device() -> str:
    # Prefer CUDA if available, else CPU
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"

class LineFormer:
    def __init__(
        self,
        hf_repo: str = "tdsone/lineformer",
        hf_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        download_if_missing: bool = True,
    ) -> None:
        """
        Initialize and load the LineFormer model.

        Parameters
        ----------
        hf_repo : str
            Hugging Face repo id to snapshot (default: "tdsone/lineformer").
        hf_dir : str | None
            Local directory to place the snapshot. Defaults to env LINEFORMER_HF_DIR or "./huggingface".
        config_path : str | None
            Path to config file. Defaults to "<hf_dir>/lineformer_swin_t_config.py".
        ckpt_path : str | None
            Path to checkpoint. Defaults to "<hf_dir>/iter_3000.pth".
        device : str | None
            "cuda:0" or "cpu". Defaults to env LINEFORMER_DEVICE or auto-detect.
        download_if_missing : bool
            If True, snapshot repo when files not found.
        """
        self.hf_dir = Path(hf_dir or os.environ.get("LINEFORMER_HF_DIR", str(_DEFAULT_HF_DIR))).resolve()
        self.device = device or os.environ.get("LINEFORMER_DEVICE", _auto_device())


        # Default filenames to mirror the original Modal version
        default_config = self.hf_dir / "lineformer_swin_t_config.py"
        default_ckpt = self.hf_dir / "iter_3000.pth"

        self.config_path = Path(config_path or os.environ.get("LINEFORMER_CONFIG", default_config)).resolve()
        self.ckpt_path = Path(ckpt_path or os.environ.get("LINEFORMER_CKPT", default_ckpt)).resolve()

        # Ensure weights/config exist; download if missing
        if download_if_missing and (not self.config_path.exists() or not self.ckpt_path.exists()):
            self._ensure_snapshot(hf_repo)

        # If the checkpoint exists but is a Git-LFS pointer (not the real
        # binary weights), delete the pointer and re-download from HF.
        if download_if_missing and self.ckpt_path.exists() and self._is_lfs_pointer(self.ckpt_path):
            print(
                f"[LineFormer] {self.ckpt_path} is a Git-LFS pointer, not the "
                "actual weights. Re-downloading from Hugging Face …"
            )
            self.ckpt_path.unlink()
            self._ensure_snapshot(hf_repo)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"LineFormer config not found at {self.config_path}. "
                "Set LINEFORMER_CONFIG or pass config_path, and ensure the HF snapshot contains this file."
            )
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"LineFormer checkpoint not found at {self.ckpt_path}. "
                "Set LINEFORMER_CKPT or pass ckpt_path, and ensure the HF snapshot contains this file."
            )
        if self._is_lfs_pointer(self.ckpt_path):
            raise RuntimeError(
                f"Checkpoint at {self.ckpt_path} is still a Git-LFS pointer after "
                "re-download. Pull LFS objects with 'git lfs pull' or download the "
                "checkpoint manually from Hugging Face."
            )

        # Load model once
        load_model(str(self.config_path), str(self.ckpt_path), self.device)

    @staticmethod
    def _is_lfs_pointer(path: Path) -> bool:
        """Return True if *path* is a Git-LFS pointer file instead of real content."""
        try:
            if path.stat().st_size > 1024:
                return False
            with open(path, "rb") as f:
                header = f.read(64)
            return header.startswith(b"version https://git-lfs")
        except OSError:
            return False

    def _ensure_snapshot(self, hf_repo: str) -> None:
        """Download the Hugging Face snapshot into self.hf_dir (idempotent)."""
        self.hf_dir.mkdir(parents=True, exist_ok=True)
        # snapshot_download will reuse cache if already present
        snapshot_download(repo_id=hf_repo, local_dir=str(self.hf_dir), local_dir_use_symlinks=False)

    def inference(self, img: str) -> None:
        """
        Run LineFormer on one image.

        Inputs/Outputs follow the established run layout:
        - Input image:   {PLEXTRACT_BASE_INPUT}/{img}
        - Output folder: {PLEXTRACT_BASE_OUTPUT}/{img}/lineformer/
            - prediction.png
            - coordinates.json
        """
        base_input = os.environ.get("PLEXTRACT_BASE_INPUT")
        base_output = os.environ.get("PLEXTRACT_BASE_OUTPUT")

        if not base_input or not base_output:
            raise RuntimeError(
                "PLEXTRACT_BASE_INPUT and PLEXTRACT_BASE_OUTPUT must be set in the environment. "
                "If you use the provided local main.py, these are set automatically."
            )

        img_path = Path(base_input) / img
        out_dir = Path(base_output) / img / "lineformer"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Read image (BGR)
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise FileNotFoundError(f"Could not read image at {img_path}")

            # Get dataseries (list of polylines/points); to_clean=False to match original behavior
            line_dataseries, inst_masks = get_dataseries(bgr, return_masks=True, to_clean=False)

            # Visualization: overlay extracted line keypoints
            vis = draw_lines(bgr.copy(), points_to_array(line_dataseries))
            cv2.imwrite(str(out_dir / "prediction.png"), vis)

            # Raw coordinates (JSON-serializable)
            with open(out_dir / "coordinates.json", "w") as f:
                json.dump(line_dataseries, f)

            # 2) Ensure masks are uint8 (0/255) and same size as image
            H, W = bgr.shape[:2]
            masks_u8 = []
            for m in inst_masks:
                mu8 = (m > 0).astype(np.uint8) * 255
                if mu8.shape != (H, W):
                    mu8 = cv2.resize(mu8, (W, H), interpolation=cv2.INTER_NEAREST)
                masks_u8.append(mu8)

            # 3) Save per-series masks and build union
            union = np.zeros((H, W), dtype=np.uint8)
            for i, mu8 in enumerate(masks_u8):
                mp = out_dir / f"mask_series_{i}.png"
                cv2.imwrite(str(mp), mu8)
                union = cv2.bitwise_or(union, mu8)

            # 4) Save union mask
            if len(masks_u8):
                cv2.imwrite(str(out_dir / "mask.png"), union)

        except Exception as e:
            print(f"[LineFormer] Failed to make prediction for {img_path}: {e}")
