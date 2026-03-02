#!/usr/bin/env python3
"""
Minimal re-extraction pipeline for user interventions.

Compared to the full plextract.py pipeline, this script skips:
  - ChartDete  (bounding boxes / label crops already exist on disk)
  - OCR        (axis_label_texts.json, axis_title_texts.json, legend_entries.json
                already exist; axis title overrides are written directly to
                axis_title_override.json and do not need an OCR re-run)

It still runs the steps that depend on user-edited data:
  1. regenerate_coordinates_from_masks
       Rebuilds coordinates.json from mask_series_*.png (or clicked_points_*.json)
       so that subsequent steps see the user's edits.
  2. correct_coordinates
       Re-converts pixel coordinates to data-space values using the existing
       OCR results (axis_label_texts.json) and ChartDete geometry
       (label_coordinates.json).
  3. match_legend
       Re-matches legend entries to series descriptors.  Masks may have changed
       color / style, so re-running keeps the mapping accurate.
  4. export_labeled_dataset_and_plot
       Generates final_dataset.csv and final_image.png, applying:
         - intervention_overrides.json  (renamed / deleted series)
         - axis_title_override.json     (user-supplied axis titles, if present)

Usage:
  python plextract_intervention.py \\
      --input  <figures/doi_dir>  \\
      --workdir <workspace>       \\
      --run-id  <existing_run_id>

The --run-id must point to a results directory that already contains ChartDete
and LineFormer outputs (pre-populated by apply_overrides_to_plextract before
this script is called).
"""

import os
import re
import cv2
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import List

from match_legend import (
    legend_descriptors_from_json,
    series_descriptors,
    build_cost_matrix_with_fallback,
    hungarian_match,
)
from finalize_labels import export_labeled_dataset_and_plot
from correct_coordinates import correct_coordinates
from build_line_masks import regenerate_coordinates_from_masks


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", log_file: str = None):
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    root.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root


# ---------------------------------------------------------------------------
# Pipeline helpers (copied / adapted from plextract.py)
# ---------------------------------------------------------------------------

def _copy_inputs_to_run(input_dir: str, base_input: str) -> str:
    print(f"Staging {input_dir} → {base_input} ...")
    shutil.copytree(input_dir, base_input, dirs_exist_ok=True)
    return base_input


def _validate_chartdete_outputs(base_output: str, img: str) -> bool:
    """
    Return True if the required ChartDete outputs are present so we can skip
    re-running ChartDete.  Emit a warning for each missing file.
    """
    required = [
        os.path.join(base_output, img, "chartdete", "bounding_boxes.json"),
        os.path.join(base_output, img, "chartdete", "label_coordinates.json"),
        os.path.join(base_output, img, "chartdete", "axis_label_texts.json"),
    ]
    all_ok = True
    print("===required===", required)
    for p in required:
        if not os.path.exists(p):
            logger.warning(
                "[intervention] Missing required ChartDete output (run full "
                f"plextract first): {p}"
            )
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Core intervention pipeline
# ---------------------------------------------------------------------------

def run_intervention_pipeline(
    input_dir: str,
    workdir: str = ".",
    run_id: str = "0",
):
    """
    Lightweight re-extraction that reuses existing ChartDete and OCR outputs
    and only re-runs the coordinate / matching / export steps.

    Parameters
    ----------
    input_dir:
        Directory containing the figure images for one DOI
        (same layout as plextract --input).
    workdir:
        Root workspace directory (same as plextract --workdir).
    run_id:
        The existing run ID whose results directory already contains pre-
        populated lineformer masks (written by apply_overrides_to_plextract
        before this script is invoked).
    """
    workdir = os.path.abspath(workdir)
    doi = input_dir.rstrip("/").split("/")[-1]

    base_input = os.path.join(workdir, "results", run_id, "input", doi)
    base_output = os.path.join(workdir, "results", run_id, "output", doi)

    os.makedirs(base_output, exist_ok=True)

    # Export env vars so downstream modules (correct_coordinates, finalize_labels)
    # can locate the data directories without arguments.
    os.environ["PLEXTRACT_RUN_ID"] = run_id
    os.environ["PLEXTRACT_BASE_INPUT"] = base_input
    os.environ["PLEXTRACT_BASE_OUTPUT"] = base_output

    # Stage the input images (needed so correct_coordinates can list files).
    _copy_inputs_to_run(input_dir, base_input)

    input_files = [
        f for f in os.listdir(base_input)
        if os.path.isfile(os.path.join(base_input, f))
    ]

    # ------------------------------------------------------------------
    # Step 1 — Regenerate coordinates.json from user-edited masks
    # ChartDete and OCR are intentionally skipped; their outputs are reused.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Regenerating coordinates from masks (skipping ChartDete + OCR)")
    print("=" * 60)

    for img in input_files:
        if img.split(".")[-1] == "json":
            continue
        if not _validate_chartdete_outputs(base_output, img):
            logger.warning(
                f"[intervention] Skipping {img}: ChartDete outputs missing. "
                "Run the full pipeline (plextract.py) first."
            )
            continue

        lf_dir = Path(os.path.join(base_output, img, "lineformer"))
        if lf_dir.is_dir():
            print(f"  Regenerating coordinates for {img} ...")
            regenerate_coordinates_from_masks(lf_dir)
        else:
            logger.warning(
                f"[intervention] {img}: lineformer directory not found at {lf_dir}. "
                "Masks must be pre-populated before calling this script."
            )

    # ------------------------------------------------------------------
    # Step 2 — Correct coordinates  (pixel → data-space)
    # Reads: axis_label_texts.json, label_coordinates.json (ChartDete/OCR)
    #        coordinates.json (just regenerated from masks)
    # Writes: converted_datapoints/data.json
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2 — Correcting coordinates (pixel → data-space)")
    print("=" * 60)

    for img in input_files:
        if img.split(".")[-1] == "json":
            continue
        try:
            correct_coordinates(run_id, img)
        except Exception as exc:
            logger.warning(f"[correct_coordinates] Error on {img}: {exc}")

    # ------------------------------------------------------------------
    # Step 3 — Match legend entries to series
    # Reads: bounding_boxes.json, legend_entries.json (ChartDete/OCR)
    #        mask_series_*.png  (current masks — may have been edited)
    # Writes: chartdete/legend_series_mapping.json
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3 — Matching legend entries to series")
    print("=" * 60)

    for img in input_files:
        if img.split(".")[-1] == "json":
            continue

        img_path = os.path.join(base_input, img)
        out_dir = os.path.join(base_output, img, "chartdete")
        bb_json = os.path.join(out_dir, "bounding_boxes.json")
        leg_json = os.path.join(out_dir, "legend_entries.json")
        lf_dir = os.path.join(base_output, img, "lineformer")

        if not os.path.exists(bb_json):
            logger.warning(f"[match] {img}: bounding_boxes.json not found, skipping.")
            continue
        if not os.path.exists(leg_json):
            logger.warning(f"[match] {img}: legend_entries.json not found, skipping.")
            continue

        with open(bb_json) as f:
            boxes = json.load(f)

        plot_areas = sorted(
            boxes.get("plot_area", []),
            key=lambda el: el[4] if len(el) > 4 else -1,
            reverse=True,
        )
        plot_area = plot_areas[0] if plot_areas else None
        mark_boxes = boxes.get("mark_label", [])

        S = series_descriptors(img_path, plot_area, lf_dir, mark_boxes)
        L = legend_descriptors_from_json(img_path, leg_json)

        if not S:
            logger.warning(f"[match] {img}: no series descriptors found.")
        if not L:
            logger.warning(f"[match] {img}: no legend descriptors found.")

        img_full = cv2.imread(img_path)
        if img_full is None:
            logger.warning(f"[match] Cannot read {img_path}, skipping.")
            continue

        C = build_cost_matrix_with_fallback(
            img_full.shape,
            L,
            S,
            alpha_dist=0.70,
            beta_color=0.20,
            gamma_style=0.10,
        )
        matches = hungarian_match(C)

        mapping = []
        for i, j, c in matches:
            sj = S[j]
            sidx = sj.get("index")
            if sidx is None:
                mp = sj.get("mask_path", "")
                m = re.search(r"(\d+)", os.path.basename(mp)) if mp else None
                sidx = int(m.group(1)) if m else j

            mapping.append(
                {
                    "legend_index": i,
                    "legend_text": L[i]["text"],
                    "legend_patch_path": L[i]["patch_path"],
                    "series_mask_path": sj.get("mask_path"),
                    "series_index": int(sidx),
                    "cost": c,
                    "legend_desc": {
                        "hsv": L[i]["hsv"],
                        "marker": L[i]["marker"],
                        "linestyle": L[i]["linestyle"],
                    },
                    "series_desc": {
                        "hsv": sj["hsv"],
                        "marker": sj["marker"],
                        "linestyle": sj["linestyle"],
                    },
                }
            )

        mapping_path = os.path.join(out_dir, "legend_series_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"[match] {img}: matched {len(matches)} legend↔series pairs.")

    # ------------------------------------------------------------------
    # Step 4 — Export labeled dataset and final plot
    # Reads: converted_datapoints/data.json
    #        chartdete/legend_series_mapping.json
    #        chartdete/axis_title_texts.json  (or axis_title_override.json)
    #        lineformer/intervention_overrides.json  (renamed / deleted series)
    # Writes: final/final_dataset.csv, final/final_image.png
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4 — Exporting labeled dataset and final plot")
    print("=" * 60)

    for img in input_files:
        if img.split(".")[-1] == "json":
            continue
        try:
            export_labeled_dataset_and_plot(run_id, img)
        except Exception as exc:
            logger.warning(f"[finalize] Error on {img}: {exc}")

    print("=" * 60)
    print("Intervention pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    ap = argparse.ArgumentParser(
        description=(
            "Minimal re-extraction pipeline for user interventions. "
            "Skips ChartDete and OCR; reuses existing outputs. "
            "Recomputes coordinates, legend matching, and final dataset."
        )
    )
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing the figure images for one DOI.",
    )
    ap.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Root workspace directory (same as plextract.py --workdir).",
    )
    ap.add_argument(
        "--run-id",
        type=str,
        required=True,
        help=(
            "Existing run ID whose results directory already contains "
            "pre-populated lineformer masks and ChartDete/OCR outputs."
        ),
    )
    ap.add_argument(
        "--log_file",
        type=str,
        default=os.path.join(_repo_root, "src", "extract_figure_data", "results"),
        help="Directory for log files.",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    os.makedirs(args.log_file, exist_ok=True)
    log_file_path = os.path.join(
        args.log_file, f"plextract_intervention_run_{args.run_id}.log"
    )
    setup_logging(args.log_level, log_file_path)
    logger = logging.getLogger("intervention")

    entries = os.listdir(args.input)
    subdirs = [
        os.path.join(args.input, e)
        for e in entries
        if os.path.isdir(os.path.join(args.input, e))
    ]

    if not subdirs:
        # Treat --input as a single DOI directory (no subdirs)
        subdirs = [args.input]

    for subdir in subdirs:
        print(f"===== INTERVENTION PIPELINE FOR {subdir} =====")
        run_intervention_pipeline(
            input_dir=subdir,
            workdir=args.workdir,
            run_id=args.run_id,
        )

    print("===== Finished intervention data extraction =====")
