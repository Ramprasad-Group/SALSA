#!/usr/bin/env python3
"""
Local pipeline runner (no Modal).

What this does:
- Copies images from an input directory into a run-specific working directory
- Calls local LineFormer, ChartDete, and OCR OCRModel modules (non-Modal)
- Saves OCR results alongside ChartDete outputs
- Runs coordinate correction
- Prints/returns ChartDete predictions

Expected local modules (we'll adapt them in subsequent files):
- plextract.lineformer.LineFormer with .inference(run_id, img_filename)
- plextract.chartdete.ChartDete with .inference(run_id) -> predictions
- plextract.ocr.OCRModel with .inference(path_to_image) -> (path, extracted_text)

Environment handoff:
Downstream steps can read:
  PLEXTRACT_BASE_INPUT, PLEXTRACT_BASE_OUTPUT
to mirror the previous Modal volume layout.

Usage:
  python main.py --input ./input --workdir ./
"""
import os
import uuid
import json
import shutil
import argparse
from pathlib import Path

# Local imports
from match_legend import legend_descriptors_from_json, series_descriptors, build_cost_matrix, hungarian_match
from correct_coordinates import correct_coordinates
from build_line_masks import build_masks_for_image
from run_lineformer import LineFormer
from run_chartdete import ChartDete
from run_ocr import OCRModel
from typing import List
import os, shutil

def _copy_inputs_to_run(input_dir: str, base_input: str) -> List[str]:
    os.makedirs(base_input, exist_ok=True)
    input_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    if not input_files:
        raise FileNotFoundError(f"No files found in input directory: {input_dir}")

    print("Copying files from input to run workspace...")
    for file_name in input_files:
        shutil.copy(os.path.join(input_dir, file_name), base_input)
    return input_files

def _prepare_output_folders(base_output: str, input_files: List[str]) -> None:
    print("Creating output folders...")
    os.makedirs(base_output, exist_ok=True)
    for file in input_files:
        file_dir = os.path.join(base_output, file)
        os.makedirs(os.path.join(file_dir, "chartdete"), exist_ok=True)
        os.makedirs(os.path.join(file_dir, "lineformer"), exist_ok=True)

def _collect_axis_label_images(base_output: str) -> List[str]:
    axis_label_images = []
    for plot_img_dir in os.listdir(base_output):
        chartdete_dir = os.path.join(base_output, plot_img_dir, "chartdete")
        if not os.path.isdir(chartdete_dir):
            continue
        for label_img in os.listdir(chartdete_dir):
            # match previous behavior: include files containing 'label' but not .json
            if "label" in label_img and not label_img.endswith(".json"):
                axis_label_images.append(os.path.join(chartdete_dir, label_img))
    return axis_label_images

def _collect_legend_label_images(base_output: str) -> List[str]:
    legends = []
    for plot_img_dir in os.listdir(base_output):
        leg_dir = os.path.join(base_output, plot_img_dir, "chartdete", "legend")
        if not os.path.isdir(leg_dir):
            continue
        for f in os.listdir(leg_dir):
            if f.startswith("label_") and f.endswith(".jpg"):
                legends.append(os.path.join(leg_dir, f))
    return legends

def run_pipeline(input_dir: str, workdir: str = "."):
    run_id = str(uuid.uuid4())
    print(f"Processing run with run id: {run_id}...")

    # Make workdir absolute and build absolute data paths
    workdir = os.path.abspath(workdir)
    data_root = os.path.join(workdir, "data")
    base_input = os.path.join(data_root, run_id, "input")
    base_output = os.path.join(data_root, run_id, "output")

    os.makedirs(base_input, exist_ok=True)
    os.makedirs(base_output, exist_ok=True)

    # Export ABSOLUTE paths so downstream code never tries '/data/...'
    os.environ["PLEXTRACT_RUN_ID"] = run_id
    os.environ["PLEXTRACT_BASE_INPUT"] = base_input
    os.environ["PLEXTRACT_BASE_OUTPUT"] = base_output

    # 1) Stage inputs
    input_files = _copy_inputs_to_run(input_dir, base_input)

    # 2) Prepare per-image output folders
    _prepare_output_folders(base_output, input_files)

    # 3) LineFormer: extract lines from images
    print("Extracting lines from images with LineFormer...")
    lf = LineFormer()
    for img in input_files:
        # Keep signature compatible with previous code: (run_id, img_filename)
        # Downstream can read PLEXTRACT_BASE_INPUT/OUTPUT
        try:
            lf.inference(run_id, img)
        except Exception as e:
            print(f"[LineFormer] Error on {img}: {e}")
            quit()


    # 4) ChartDete: detect chart elements
    print("Detecting chart elements with ChartDete...")
    cd = ChartDete()
    try:
        chartdete_preds = cd.inference(run_id)  # should return predictions
    except Exception as e:
        print(f"[ChartDete] Error: {e}")
        quit()
        chartdete_preds = None

    # 5) OCR axis labels (TrOCR via local OCRModel)
    print("Running OCR on axis labels...")
    axis_label_images = _collect_axis_label_images(base_output)
    print('---axis label images---', axis_label_images)
    ocr = OCRModel()

    label_texts = []
    for p in axis_label_images:
        try:
            # Expect tuple (path, extracted_text) to mirror the original usage
            label_texts.append(ocr.inference(Path(p)))
        except Exception as e:
            print(f"[OCR] Error on {p}: {e}")
            quit()
    

    # Save OCR results per-figure
    for img_dir in os.listdir(base_output):
        img_dir_path = os.path.join(base_output, img_dir)
        if not os.path.isdir(img_dir_path):
            continue

        out_json = os.path.join(img_dir_path, "axis_label_texts.json")
        payload = {
            path: extracted_text
            for (path, extracted_text) in label_texts
            if img_dir in path
        }
        print(f"Saving OCR results to {out_json}...")
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
    
    # 5.5) OCR legend labels
    print("Running OCR on legend labels...")
    legend_label_images = _collect_legend_label_images(base_output)
    print('---legend label images---', legend_label_images)
    legend_label_texts = []
    for p in legend_label_images:
        try:
            legend_label_texts.append(ocr.inference(Path(p)))  # (path, text)
        except Exception as e:
            print(f"[OCR] Error on legend label {p}: {e}")
            quit()

    # Merge OCR text into legend_entries.json per figure
    legend_text_by_path = {p: t for (p, t) in legend_label_texts}
    for img_dir in os.listdir(base_output):
        leg_json = os.path.join(base_output, img_dir, "chartdete", "legend_entries.json")
        if os.path.exists(leg_json):
            with open(leg_json, "r") as f:
                entries = json.load(f)
            for ent in entries:
                lp = ent.get("label_path")
                if lp in legend_text_by_path:
                    ent["text"] = legend_text_by_path[lp]
            with open(leg_json, "w") as f:
                json.dump(entries, f, indent=2)

    # 6) Correct coordinates
    print("Correcting coordinates...")
    for img in os.listdir(base_input):
        try:
            correct_coordinates(run_id, img)
        except Exception as e:
            print(f"[correct_coordinates] Error on {img}: {e}")
            quit()

    # 7) Match legend to series and save mapping
    print("Matching legend entries to series...")
    for img in os.listdir(base_input):
        img_path = os.path.join(base_input, img)
        out_dir  = os.path.join(base_output, img, "chartdete")
        bb_json  = os.path.join(out_dir, "bounding_boxes.json")
        leg_json = os.path.join(out_dir, "legend_entries.json")
        lf_dir   = os.path.join(base_output, img, "lineformer")  # adjust if different

        if not (os.path.exists(bb_json) and os.path.exists(leg_json)):
            continue

        with open(bb_json, "r") as f:
            boxes = json.load(f)
        plot_areas = sorted(boxes.get("plot_area", []), key=lambda el: el[4] if len(el)>4 else -1, reverse=True)
        plot_area = plot_areas[0] if plot_areas else None
        mark_boxes = boxes.get("mark_label", [])

        L = legend_descriptors_from_json(img_path, leg_json)
        print('---legend_descriptors---', L)
        S = series_descriptors(img_path, plot_area, lf_dir, mark_boxes)

        if not L or not S:
            print(f"[match] Skip {img}: no legend entries or no series found.")
            continue

        C = build_cost_matrix(L, S)
        matches = hungarian_match(C)

        # Save mapping
        mapping = []
        for i,j,c in matches:
            mapping.append({
                "legend_index": i,
                "legend_text": L[i]["text"],
                "legend_patch_path": L[i]["patch_path"],
                "series_mask_path": S[j]["mask_path"],
                "cost": c,
                "legend_desc": {"hsv": L[i]["hsv"], "marker": L[i]["marker"], "linestyle": L[i]["linestyle"]},
                "series_desc": {"hsv": S[j]["hsv"], "marker": S[j]["marker"], "linestyle": S[j]["linestyle"]},
            })
        with open(os.path.join(out_dir, "legend_series_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

        print(f"[match] {img}: matched {len(matches)} legend entries to series.")

    return chartdete_preds


def parse_args():
    ap = argparse.ArgumentParser(description="Run local chart digitization pipeline.")
    ap.add_argument(
        "--input",
        type=str,
        default="input",
        help="Directory containing input images (default: ./input)",
    )
    ap.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Working directory for run data (default: current directory)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preds = run_pipeline(input_dir='/home/wschertzer3/extract-line-chart-data/input', workdir='/home/wschertzer3/extract-line-chart-data')
    print("\n=== ChartDete predictions ===")
    print(preds)