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
# Directory where ChartDete can be found
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../extern"))
sys.path.insert(0, parent_dir)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_DIR = _REPO_ROOT / "huggingface"

# Now it's safe to import mmdet apis; they will resolve to the fork
from mmdet.apis import init_detector, inference_detector  # noqa: E402


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
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
        self.hf_dir = Path(hf_dir or os.environ.get("CHARTDETE_HF_DIR", str(_DEFAULT_HF_DIR))).resolve()
        self.device = device or os.environ.get("CHARTDETE_DEVICE", _auto_device())

        default_cfg = self.hf_dir / "cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py"
        default_ckpt = self.hf_dir / "checkpoint.pth"

        self.config_path = Path(config_path or os.environ.get("CHARTDETE_CONFIG", default_cfg)).resolve()
        self.ckpt_path = Path(ckpt_path or os.environ.get("CHARTDETE_CKPT", default_ckpt)).resolve()

        # confidence threshold for cropping labels
        self.conf_thr = float(os.environ.get("CHARTDETE_CONF_THR", confidence_threshold or 0.9))

        if download_if_missing and (not self.config_path.exists() or not self.ckpt_path.exists()):
            self._ensure_snapshot(hf_repo)

        # If the checkpoint exists but is a Git-LFS pointer (not the real
        # binary weights), delete the pointer and re-download from HF.
        if download_if_missing and self.ckpt_path.exists() and self._is_lfs_pointer(self.ckpt_path):
            print(
                f"[ChartDete] {self.ckpt_path} is a Git-LFS pointer, not the "
                "actual weights. Re-downloading from Hugging Face …"
            )
            self.ckpt_path.unlink()
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
        if self._is_lfs_pointer(self.ckpt_path):
            raise RuntimeError(
                f"Checkpoint at {self.ckpt_path} is still a Git-LFS pointer after "
                "re-download. Pull LFS objects with 'git lfs pull' or download the "
                "checkpoint manually from Hugging Face."
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

    @staticmethod
    def _is_lfs_pointer(path: Path) -> bool:
        """Return True if *path* is a Git-LFS pointer file instead of real content."""
        try:
            if path.stat().st_size > 1024:
                return False  # LFS pointers are tiny; real weights are large
            with open(path, "rb") as f:
                header = f.read(64)
            return header.startswith(b"version https://git-lfs")
        except OSError:
            return False

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
        results_base_folders = [Path(base_output) / img / "chartdete" for img in inputs if img.split('.')[-1] == 'jpg']
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

    def get_chartdete_preds(self, img:str) -> None:
        """
        Run detection on all one image.

        Returns
        -------
        Dict mapping each input image filename to a small summary dict.
        """
        print("Running ChartDete Inference...")
        # img_paths, results_base_folders = self._collect_inputs()


        base_input = os.environ.get("PLEXTRACT_BASE_INPUT")
        base_output = os.environ.get("PLEXTRACT_BASE_OUTPUT")
        base_output = Path(base_output) / img / "chartdete"
        img_path = Path(base_input) / img

        try:
            # 1) Inference
            predictions = inference_detector(self.model, img_path)
            return predictions
        except Exception as e:
                print(f"[ChartDete] Error on {img_path}: {e}")
    
    def split_subfigures(self, img: str, predictions) -> None:
        """
        Split figure into subfigures, extending crops to include x_title

        Returns
        -------
        List of new figure paths if subfigures were detected
        """

        def is_vertically_stacked(a, b, H, min_x_overlap=0.7, max_gap_frac=0.2):
            """
            Check if two boxes a, b are vertically stacked:
            - significant horizontal overlap
            - box centers are separated vertically (one clearly above the other)
            - vertical gap not too huge (same column in the figure)
            a, b = [x1, y1, x2, y2, score]
            """
            x1a, y1a, x2a, y2a, _ = a
            x1b, y1b, x2b, y2b, _ = b

            # Horizontal overlap
            x_overlap = max(0, min(x2a, x2b) - max(x1a, x1b))
            min_width = min(x2a - x1a, x2b - x1b)
            if min_width <= 0:
                return False

            x_overlap_ratio = x_overlap / min_width
            if x_overlap_ratio < min_x_overlap:
                return False

            # Vertical separation (image coords: y increases downward)
            # Positive gap => non-overlapping vertically
            y_gap = max(0, max(y1a, y1b) - min(y2a, y2b))
            if y_gap <= 0:
                # They overlap vertically; treat as not "stacked"
                return False

            # Require the gap be reasonably small relative to image height
            return y_gap < max_gap_frac * H


        def nms(boxes, iou_thr=1.0):
            """NMS that allows vertically stacked subfigures"""
            keep = []
            
            def iou(a, b):
                x1 = max(a[0], b[0])
                y1 = max(a[1], b[1])
                x2 = min(a[2], b[2])
                y2 = min(a[3], b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                return inter / (area_a + area_b - inter + 1e-6)
            
            for b in boxes:
                should_keep = True
                for k in keep:
                    iou_val = iou(b, k)
                    threshold = iou_thr
                    
                    if iou_val >= threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    keep.append(b)
            
            return keep
    
        print("Checking for subfigures ...")

        PLOT_THR = 0.7          # confidence threshold
        MIN_AREA_FRAC = 0.2     # ignore tiny false plot areas
        PAD_FRAC = 0.08          # padding around plot area
        X_TITLE_CONF_THR = 0.5   # confidence threshold for x_title
        VERTICAL_IOU_THR = 0.0   # lower threshold for vertical stacking

        outputs: Dict[str, Any] = {}

        base_input = os.environ.get("PLEXTRACT_BASE_INPUT")
        base_output = os.environ.get("PLEXTRACT_BASE_OUTPUT")
        base_output = Path(base_output) / img / "chartdete"
        img_path = Path(base_input) / img

        image = cv2.imread(img_path)

        img_name = Path(img_path).name
        try:
            results_labelled = self._to_label_dict(predictions)

            # pick highest-confidence plot_area (if present)
            plot_areas = sorted(
                results_labelled.get("plot_area", []),
                key=lambda el: el[4] if len(el) > 4 else -1,
                reverse=True)
            
            # Get x_titles for extension
            x_titles = [
                (x1, y1, x2, y2, s) 
                for x1, y1, x2, y2, s in results_labelled.get("x_title", [])
                if s >= X_TITLE_CONF_THR
            ]
            
            # Filter plot areas
            candidates = []
            H, W = image.shape[:2]
            for x1, y1, x2, y2, s in plot_areas:
                area = (x2 - x1) * (y2 - y1)
                # print("====S, AREA====", s, PLOT_THR, s >= PLOT_THR, area, area >= MIN_AREA_FRAC * W * H)
                if s >= PLOT_THR and area >= MIN_AREA_FRAC * W * H:
                    candidates.append([x1, y1, x2, y2, s])

            # ---skip figures with vertically stacked plot areas ---
            has_vertical_stack = False
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if is_vertically_stacked(candidates[i], candidates[j], H):
                        print("=== Vertically stacked subfigures detected; skipping figure ===")
                        has_vertical_stack = True
                        break
                if has_vertical_stack:
                    break

            if has_vertical_stack:
                return

            # run NMS
            candidates = nms(candidates)


            # If only one plot area → nothing to split
            if len(candidates) <= 1:
                print("===NOTHING TO SPLIT===")
                return
            
            pad_x = int(PAD_FRAC * W)
            pad_y = int(PAD_FRAC * H)

            stem = img_path.stem
            suffix = img_path.suffix

            # Save auto-detected subfigures to the interventions directory so they
            # are stored alongside manually-drawn crops rather than in the plextract
            # working directory.
            base_input_path = Path(base_input)
            safe_doi = base_input_path.name
            workdir = base_input_path.parents[3]
            interventions_dir = workdir / "interventions" / safe_doi
            interventions_dir.mkdir(parents=True, exist_ok=True)

            new_imgs = []

            for i, (x1, y1, x2, y2, s) in enumerate(candidates):
                # Start with padded plot area bounds
                crop_x1 = max(0, int(x1 - pad_x))
                crop_y1 = max(0, int(y1 - pad_y))
                crop_x2 = min(W, int(x2 + pad_x))
                crop_y2 = min(H, int(y2 + pad_y))

                # Find x_titles that overlap with this plot area (horizontally aligned)
                for xt_x1, xt_y1, xt_x2, xt_y2, xt_s in x_titles:
                    # Check if x_title overlaps horizontally with plot area
                    h_overlap = not (xt_x2 < x1 or xt_x1 > x2)
                    # Check if x_title is below the plot area
                    is_below = xt_y1 >= y1

                    if h_overlap and is_below:
                        # Extend crop to include this x_title
                        crop_x1 = min(crop_x1, int(xt_x1))
                        crop_x2 = max(crop_x2, int(xt_x2))
                        crop_y2 = max(crop_y2, int(xt_y2))

                # Ensure bounds are within image
                crop_x1 = max(0, crop_x1)
                crop_y1 = max(0, crop_y1)
                crop_x2 = min(W, crop_x2)
                crop_y2 = min(H, crop_y2)

                print(f"Subfig {i}: crop bounds = ({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}), size = {crop_x2-crop_x1}x{crop_y2-crop_y1}")
                crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    print('=====CROP.SIZE==0====')
                    continue

                subfig_name = f"{stem}_subfig_{i}{suffix}"

                # Store in the interventions dir (canonical location).
                out_path = interventions_dir / subfig_name
                cv2.imwrite(str(out_path), crop)

                new_imgs.append(out_path)

            # Delete original image
            # img_path.unlink()
            print(f"[split] {img_path.name} → {len(candidates)} subfigures (with x_titles)")

            return new_imgs

        except Exception as e:
            print(f"[ChartDete] Error splitting subfigures on {img_path}: {e}")

    def inference(self, img: str, predictions=None) -> None:
        """
        Run detection on all one image.

        Parameters
        ----------
        predictions:
            Optional detections already computed for `img` (e.g. via
            `get_chartdete_preds`). Pass these in to avoid rerunning the
            detector forward pass on the same image.

        Returns
        -------
        Dict mapping each input image filename to a small summary dict.
        """
        print("Running ChartDete Inference...")
        # img_paths, results_base_folders = self._collect_inputs()

        outputs: Dict[str, Any] = {}

        base_input = os.environ.get("PLEXTRACT_BASE_INPUT")
        base_output = os.environ.get("PLEXTRACT_BASE_OUTPUT")
        base_output = Path(base_output) / img / "chartdete"
        img_path = Path(base_input) / img

        img_name = Path(img_path).name
        try:
            # 1) Inference
            if predictions is None:
                predictions = inference_detector(self.model, img_path)

            # 2) Save visualization
            result_path = str(base_output / "predictions.jpg")
            # compatible with older MMDet 2.x fork
            self.model.show_result(img_path, predictions, out_file=result_path)

            # 3) Label -> arrays
            results_labelled = self._to_label_dict(predictions)

            # 4) bounding_boxes.json
            bb_json = str(base_output / "bounding_boxes.json")
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
                    path = base_output / f"cropped_xlabels_{xi}.jpg"
                    crop = self._crop_bounding_boxes(image, [box], self.conf_thr)
                    if crop:
                        cv2.imwrite(str(path), crop[0])
                        label_coordinates[str(path)] = box
                        xi += 1

            yi = 0
            for box in results_labelled.get("ylabel", []):
                if len(box) >= 5 and box[4] >= self.conf_thr:
                    path = base_output / f"cropped_ylabels_{yi}.jpg"
                    crop = self._crop_bounding_boxes(image, [box], self.conf_thr)
                    if crop:
                        cv2.imwrite(str(path), crop[0])
                        label_coordinates[str(path)] = box
                        yi += 1
            
            # 5.4) X and Y Titles
            xti = 0
            for box in results_labelled.get("x_title", []):
                if len(box) >= 5 and box[4] >= self.conf_thr:
                    path = base_output / f"cropped_xtitle_{xti}.jpg"
                    crop = self._crop_bounding_boxes(image, [box], self.conf_thr)
                    if crop:
                        cv2.imwrite(str(path), crop[0])
                        label_coordinates[str(path)] = box
                        xti += 1

            yti = 0
            for box in results_labelled.get("y_title", []):
                if len(box) >= 5 and box[4] >= self.conf_thr:
                    path = base_output / f"cropped_ytitle_{yti}.jpg"
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

            legend_json_path = str(base_output / "legend_entries.json")
            legend_dir = base_output / "legend"
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
            lc_json = str(base_output / "label_coordinates.json")
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
