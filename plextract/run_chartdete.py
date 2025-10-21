#!/usr/bin/env python3
"""
Local ChartDete wrapper (no Modal).

Responsibilities
- Ensure ChartDete weights & config are available (via Hugging Face snapshot)
- Initialize an MMDetection model once per process (using ChartDete's fork)
- Run inference over all images in {PLEXTRACT_BASE_INPUT}
- Write results per image under:
  {PLEXTRACT_BASE_OUTPUT}/{img}/chartdete/
      predictions.jpg
      bounding_boxes.json
      cropped_xlabels_*.jpg
      cropped_ylabels_*.jpg
      label_coordinates.json

Environment variables (optional)
- PLEXTRACT_BASE_INPUT:   set by main.py
- PLEXTRACT_BASE_OUTPUT:  set by main.py
- CHARTDETE_HF_DIR:       directory for HF snapshot (default: ./huggingface)
- CHARTDETE_CONFIG:       override path to config .py inside HF dir
- CHARTDETE_CKPT:         override path to checkpoint .pth inside HF dir
- CHARTDETE_DEVICE:       "cuda:0" or "cpu" (default: auto)
- CHARTDETE_CONF_THR:     confidence threshold for cropping (default: 0.9)

Dependencies (install locally):
    # Use ChartDete's MMDet fork + mmcv-full 1.x (NOT MMDet 3.x/mmcv 2.x)
    # Example (CUDA 11.7):
    #   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    #   pip uninstall -y mmdet mmcv mmcv-full
    #   pip install -U openmim && mim install mmcv-full
    #   cd <repo-root>/ChartDete && pip install -v -e .
    # Plus:
    #   pip install huggingface_hub opencv-python-headless numpy
"""

from __future__ import annotations
import os
import sys
import json
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
from huggingface_hub import snapshot_download

# --------------------------------------------------------------------------------------
# Ensure the ChartDete fork (which contains the custom LGF head) is imported FIRST.
# ChartDete is a sibling of plextract: <repo-root>/ChartDete and <repo-root>/plextract
# --------------------------------------------------------------------------------------
chartdete_root = '/home/wschertzer3/ChartDete'

# Put ChartDete ahead of site-packages so its forked mmdet is used
sys.path.insert(0, str(chartdete_root))

# Force-register the custom CascadeRoIHead_LGF into MMDet registry
# (Import AFTER sys.path insertion, BEFORE init_detector)
# try:
#     importlib.import_module("mmdet.models.roi_heads.cascade_roi_head_lgf")
# except Exception as e:
#     raise RuntimeError(
#         "Failed to import ChartDete's custom LGF head. Ensure you've installed the ChartDete repo "
#         f"(pip install -e {chartdete_root}) with a compatible mmcv-full 1.x."
#     ) from e

# Now it's safe to import mmdet apis; they will resolve to the fork
from mmdet.apis import init_detector, inference_detector  # noqa: E402


def _auto_device() -> str:
    # try:
    #     import torch  # imported late on purpose
    #     if torch.cuda.is_available():
    #         return "cuda:0"
    # except Exception:
    #     pass
    return "cpu"


class ChartDete:
    def __init__(
        self,
        hf_repo: str = "tdsone/chartdete",
        hf_dir: str | None = None,
        config_path: str | None = None,
        ckpt_path: str | None = None,
        device: str | None = None,
        download_if_missing: bool = True,
        confidence_threshold: float | None = None,
    ) -> None:
        """
        Initialize the ChartDete detector.

        Parameters
        ----------
        hf_repo : str
            Hugging Face repo id to snapshot (default: "tdsone/chartdete").
        hf_dir : str | None
            Local directory for the snapshot. Defaults to env CHARTDETE_HF_DIR or "./huggingface".
        config_path : str | None
            Model config path. Defaults to "<hf_dir>/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py".
        ckpt_path : str | None
            Model checkpoint path. Defaults to "<hf_dir>/checkpoint.pth".
        device : str | None
            "cuda:0" or "cpu". Defaults to env CHARTDETE_DEVICE or auto-detect.
        download_if_missing : bool
            Snapshot repo when files are missing.
        confidence_threshold : float | None
            Threshold for cropping axis labels. Defaults to env CHARTDETE_CONF_THR or 0.9.
        """
        self.hf_dir = Path(hf_dir or os.environ.get("CHARTDETE_HF_DIR", "./huggingface")).resolve()
        self.device = device or os.environ.get("CHARTDETE_DEVICE", _auto_device())

        default_cfg = self.hf_dir / "cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py"
        default_ckpt = self.hf_dir / "checkpoint.pth"

        self.config_path = Path(config_path or os.environ.get("CHARTDETE_CONFIG", default_cfg)).resolve()
        self.ckpt_path = Path(ckpt_path or os.environ.get("CHARTDETE_CKPT", default_ckpt)).resolve()

        # confidence threshold for cropping labels
        self.conf_thr = float(os.environ.get("CHARTDETE_CONF_THR", confidence_threshold or 0.9))

        if download_if_missing and (not self.config_path.exists() or not self.ckpt_path.exists()):
            self._ensure_snapshot(hf_repo)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {self.config_path}. "
                "Set CHARTDETE_CONFIG or pass config_path; ensure the HF snapshot contains it."
            )
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.ckpt_path}. "
                "Set CHARTDETE_CKPT or pass ckpt_path; ensure the HF snapshot contains it."
            )

        # --- Build model once (ChartDete fork is already on sys.path; LGF is registered) ---
        # Optionally, you can add custom_imports programmatically if using mmcv Config objects.
        # For most setups, forcing the import above is sufficient.
        self.model = init_detector(str(self.config_path), str(self.ckpt_path), device=self.device)

        # Class labels order expected by the original script
        self.labels = [
            "x_title",
            "y_title",
            "plot_area",
            "other",
            "xlabel",
            "ylabel",
            "chart_title",
            "x_tick",
            "y_tick",
            "legend_patch",
            "legend_label",
            "legend_title",
            "legend_area",
            "mark_label",
            "value_label",
            "y_axis_area",
            "x_axis_area",
            "tick_grouping",
        ]

    def _ensure_snapshot(self, hf_repo: str) -> None:
        self.hf_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=hf_repo, local_dir=str(self.hf_dir), local_dir_use_symlinks=False)

    def _collect_inputs(self) -> Tuple[List[str], List[Path]]:
        base_input = os.environ.get("PLEXTRACT_BASE_INPUT")
        base_output = os.environ.get("PLEXTRACT_BASE_OUTPUT")
        if not base_input or not base_output:
            raise RuntimeError(
                "PLEXTRACT_BASE_INPUT and PLEXTRACT_BASE_OUTPUT must be set in the environment. "
                "If you use the provided local main.py, these are set automatically."
            )
        inputs = [f for f in os.listdir(base_input) if os.path.isfile(os.path.join(base_input, f))]
        img_paths = [os.path.join(base_input, img) for img in inputs]
        results_base_folders = [Path(base_output) / img / "chartdete" for img in inputs]
        for p in results_base_folders:
            p.mkdir(parents=True, exist_ok=True)
        return img_paths, results_base_folders

    def _to_label_dict(self, predictions: List[np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Convert MMDet predictions (list of arrays per class) into
        a dict[label] = [ [x1,y1,x2,y2,score], ... ] as lists.
        """
        results_labelled: Dict[str, List[List[float]]] = {}
        for arr, label in zip(predictions, self.labels):
            if isinstance(arr, np.ndarray):
                results_labelled[label] = arr.tolist()
            else:
                merged: List[List[float]] = []
                for a in arr:
                    if isinstance(a, np.ndarray):
                        merged.extend(a.tolist())
                results_labelled[label] = merged
        return results_labelled

    def _crop_bounding_boxes(self, image: np.ndarray, boxes: List[List[float]], thr: float) -> List[np.ndarray]:
        crops: List[np.ndarray] = []
        for box in boxes:
            if len(box) < 5:
                continue  # expect [x1,y1,x2,y2,score]
            x1, y1, x2, y2, conf = box
            if conf >= thr:
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                x1i, y1i = max(x1i, 0), max(y1i, 0)
                x2i, y2i = min(x2i, image.shape[1] - 1), min(y2i, image.shape[0] - 1)
                if x2i > x1i and y2i > y1i:
                    crops.append(image[y1i:y2i, x1i:x2i])
        return crops

    def inference(self, run_id: str) -> Dict[str, Any]:
        """
        Run detection on all images in PLEXTRACT_BASE_INPUT.

        Returns
        -------
        Dict mapping each input image filename to a small summary dict.
        """
        print("Running ChartDete...")
        img_paths, results_base_folders = self._collect_inputs()

        outputs: Dict[str, Any] = {}

        for img_path, results_base_folder in zip(img_paths, results_base_folders):
            img_name = Path(img_path).name
            try:
                # 1) Inference
                predictions = inference_detector(self.model, img_path)

                # 2) Save visualization
                result_path = str(results_base_folder / "predictions.jpg")
                # compatible with older MMDet 2.x fork
                self.model.show_result(img_path, predictions, out_file=result_path)

                # 3) Label -> arrays
                results_labelled = self._to_label_dict(predictions)

                # 4) bounding_boxes.json
                bb_json = str(results_base_folder / "bounding_boxes.json")
                with open(bb_json, "w") as f:
                    json.dump(results_labelled, f, indent=2)

                # 5) Crop xlabel / ylabel candidates above threshold
                image = cv2.imread(img_path)
                if image is None:
                    raise FileNotFoundError(f"Failed to read image {img_path}")

                label_coordinates: Dict[str, Any] = {}

                # pick highest-confidence plot_area (if present)
                plot_areas = sorted(
                    results_labelled.get("plot_area", []),
                    key=lambda el: el[4] if len(el) > 4 else -1,
                    reverse=True,
                )
                if plot_areas:
                    label_coordinates["plot_area"] = plot_areas[0]

                # Save cropped images and remember their coordinates by matching index
                xi = 0
                for box in results_labelled.get("xlabel", []):
                    if len(box) >= 5 and box[4] >= self.conf_thr:
                        path = results_base_folder / f"cropped_xlabels_{xi}.jpg"
                        crop = self._crop_bounding_boxes(image, [box], self.conf_thr)
                        if crop:
                            cv2.imwrite(str(path), crop[0])
                            label_coordinates[str(path)] = box
                            xi += 1

                yi = 0
                for box in results_labelled.get("ylabel", []):
                    if len(box) >= 5 and box[4] >= self.conf_thr:
                        path = results_base_folder / f"cropped_ylabels_{yi}.jpg"
                        crop = self._crop_bounding_boxes(image, [box], self.conf_thr)
                        if crop:
                            cv2.imwrite(str(path), crop[0])
                            label_coordinates[str(path)] = box
                            yi += 1
                
                # 5.4) X and Y Titles
                xti = 0
                for box in results_labelled.get("x_title", []):
                    if len(box) >= 5 and box[4] >= self.conf_thr:
                        path = results_base_folder / f"cropped_xtitle_{xti}.jpg"
                        crop = self._crop_bounding_boxes(image, [box], self.conf_thr)
                        if crop:
                            cv2.imwrite(str(path), crop[0])
                            label_coordinates[str(path)] = box
                            xti += 1

                yti = 0
                for box in results_labelled.get("y_title", []):
                    if len(box) >= 5 and box[4] >= self.conf_thr:
                        path = results_base_folder / f"cropped_ytitle_{yti}.jpg"
                        crop = self._crop_bounding_boxes(image, [box], self.conf_thr)
                        if crop:
                            cv2.imwrite(str(path), crop[0])
                            label_coordinates[str(path)] = box
                            yti += 1

                # 5.5) Legend / Mark-Label Processing
                legend_entries = []

                # helpers
                def _center(b):
                    x1, y1, x2, y2, _ = b
                    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

                def _inside(area, b):
                    ax1, ay1, ax2, ay2, *_ = area
                    x1, y1, x2, y2, *_ = b
                    return (x1 >= ax1 and y1 >= ay1 and x2 <= ax2 and y2 <= ay2)

                def _safe_crop(im, box):
                    x1, y1, x2, y2, *_ = box
                    x1i, y1i = max(0, int(x1)), max(0, int(y1))
                    x2i, y2i = min(im.shape[1] - 1, int(x2)), min(im.shape[0] - 1, int(y2))
                    if x2i > x1i and y2i > y1i:
                        return im[y1i:y2i, x1i:x2i]
                    return None

                legend_json_path = str(results_base_folder / "legend_entries.json")
                legend_dir = results_base_folder / "legend"
                legend_dir.mkdir(exist_ok=True, parents=True)

                # Prefer legend area if present
                legend_area_boxes = sorted(
                    results_labelled.get("legend_area", []),
                    key=lambda el: el[4] if len(el) > 4 else -1,
                    reverse=True,
                )
                paired = []              # (patch_bbox, label_bbox)
                used_label_idx = set()
                have_pairs = False

                if legend_area_boxes:
                    la = legend_area_boxes[0]  # best legend area
                    patches = [b for b in results_labelled.get("legend_patch", []) if len(b) >= 5 and b[4] >= 0.75]
                    labels  = [b for b in results_labelled.get("legend_label", []) if len(b) >= 5 and b[4] >= 0.75]

                    # Greedy NN pairing
                    for pi, pb in enumerate(patches):
                        pc = _center(pb)
                        dists = [
                            (li, (pc[0] - _center(lb)[0]) ** 2 + (pc[1] - _center(lb)[1]) ** 2)
                            for li, lb in enumerate(labels) if li not in used_label_idx
                        ]
                        if not dists:
                            continue
                        li, _ = min(dists, key=lambda x: x[1])
                        used_label_idx.add(li)
                        paired.append((pb, labels[li]))

                    # Save crops for any pairs we found
                    for idx, (pb, lb) in enumerate(paired):
                        pcrop = _safe_crop(image, pb)
                        lcrop = _safe_crop(image, lb)
                        patch_path = legend_dir / f"patch_{idx}.jpg"
                        label_path = legend_dir / f"label_{idx}.jpg"
                        if pcrop is not None:
                            cv2.imwrite(str(patch_path), pcrop)
                        if lcrop is not None:
                            cv2.imwrite(str(label_path), lcrop)
                        legend_entries.append({
                            "source": "legend",
                            "patch_bbox": pb,
                            "label_bbox": lb,
                            "patch_path": str(patch_path) if pcrop is not None else None,
                            "label_path": str(label_path) if lcrop is not None else None,
                            "text": None  # to be filled after OCR
                        })

                    have_pairs = len(paired) > 0
                else:
                    print('---does not contain legend area---')
                    
                # ===== Fallback to mark_label when no legend pairs/patches/labels =====

                # print(paired, have_pairs, '----test------')
                # quit()
                # quit()
                if not legend_area_boxes or not have_pairs:
                    # Try direct on-plot mark labels
                    mark_labels = [b for b in results_labelled.get("mark_label", []) if len(b) >= 5 and b[4] >= 0.5]

                    # Prefer mark labels inside the plot area if we have one
                    plot_area = label_coordinates.get("plot_area")
                    if plot_area:
                        mark_labels_in_plot = [b for b in mark_labels if _inside(plot_area, b)]
                        if mark_labels_in_plot:
                            mark_labels = mark_labels_in_plot

                    print('---fallback mark_labels exist---')

                    for mi, mb in enumerate(mark_labels):
                        lcrop = _safe_crop(image, mb)
                        label_path = legend_dir / f"mark_label_{mi}.jpg"
                        if lcrop is not None:
                            cv2.imwrite(str(label_path), lcrop)
                        legend_entries.append({
                            "source": "mark_label",
                            "patch_bbox": None,
                            "label_bbox": mb,
                            "patch_path": None,
                            "label_path": str(label_path) if lcrop is not None else None,
                            "text": None  # to be filled after OCR
                        })

                # Persist legend entries
                with open(legend_json_path, "w") as f:
                    json.dump(legend_entries, f, indent=2)

                # 6) Save label_coordinates.json
                lc_json = str(results_base_folder / "label_coordinates.json")
                with open(lc_json, "w") as f:
                    json.dump(label_coordinates, f, indent=2)

                outputs[img_name] = {
                    "predictions_image": result_path,
                    "bounding_boxes_json": bb_json,
                    "label_coordinates_json": lc_json,
                    "legend_entries_json": legend_json_path,
                }

                print(f"[ChartDete] Processed {img_name}")

            except Exception as e:
                print(f"[ChartDete] Error on {img_path}: {e}")

        return outputs