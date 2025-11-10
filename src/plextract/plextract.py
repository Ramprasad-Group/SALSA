#!/usr/bin/env python3
"""
Local pipeline runner (no Modal).

What this does:
- Copies images from an input directory into a run-specific working directory
- Calls local LineFormer, ChartDete, and OCR OCRModel modules (non-Modal)
- Saves OCR results alongside ChartDete outputs
- Runs coordinate correction
- Prints/returns ChartDete predictions

Expected local modules:
- LineFormer with .inference(run_id, img_filename)
- ChartDete with .inference(run_id) -> predictions

Environment handoff:
Downstream steps can read:
  PLEXTRACT_BASE_INPUT, PLEXTRACT_BASE_OUTPUT
to mirror the previous Modal volume layout.

Usage:
  python main.py --input ./input --workdir ./
"""
import os
import re
import cv2
import json
import shutil
import argparse
from typing import List
from pathlib import Path

# Local imports
from match_legend import legend_descriptors_from_json, series_descriptors, build_cost_matrix_with_fallback, hungarian_match
from finalize_labels import export_labeled_dataset_and_plot
from correct_coordinates import correct_coordinates
from build_line_masks import build_masks_for_image
from run_lineformer import LineFormer
from run_chartdete import ChartDete
from run_ocr import OCRModel

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
            if "label" in label_img and not label_img.endswith(".json"):
                axis_label_images.append(os.path.join(chartdete_dir, label_img))
    return axis_label_images

def _collect_axis_title_images(base_output: str) -> List[str]:
    titles = []
    for plot_img_dir in os.listdir(base_output):
        temp_titles = []
        chartdete_dir = os.path.join(base_output, plot_img_dir, "chartdete")
        if not os.path.isdir(chartdete_dir):
            continue
        for f in os.listdir(chartdete_dir):
            if "title" in f.lower() and f.lower().endswith(".jpg"):
                temp_titles.append(os.path.join(chartdete_dir, f))
                titles.append(os.path.join(chartdete_dir, f))
        if len(temp_titles) != 2: # check if found one x and one y axis title
            print(f'ERROR: len of extracted titles: {len(titles)}')
            print(f'TITLES: {titles}')
            quit()

    return titles

def _collect_legend_label_images(base_output: str) -> List[str]:
    legends = []
    for plot_img_dir in os.listdir(base_output):
        leg_dir = os.path.join(base_output, plot_img_dir, "chartdete", "legend")
        if not os.path.isdir(leg_dir):
            continue

        # First try files starting with "label_"
        label_imgs = [os.path.join(leg_dir, f) for f in os.listdir(leg_dir)
                      if f.startswith("label_") and f.endswith(".jpg")]

        # If none found, try "mark_label_"
        if not label_imgs:
            label_imgs = [os.path.join(leg_dir, f) for f in os.listdir(leg_dir)
                          if f.startswith("mark_label_") and f.endswith(".jpg")]

        # If still none found, print warning
        if not label_imgs:
            print(f"---No label was found in {leg_dir}---")

        legends.extend(label_imgs)

    return legends

def find_next_number(workdir: str) -> int:
    """
    Scans the given workdir for directories named with integers
    and returns the next available integer run_id.
    """
    existing_numbers = []

    # Ensure directory exists
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # Iterate through items in workdir
    for name in os.listdir(workdir):
        full_path = os.path.join(workdir, name)
        if os.path.isdir(full_path):
            # Match pure integer directory names (e.g., "1", "2", "003")
            match = re.fullmatch(r'\d+', name)
            if match:
                existing_numbers.append(int(name))

    # If no directories yet, start from 0
    print(existing_numbers)
    next_number = max(existing_numbers) + 1 if existing_numbers else 0
    return str(next_number)

def run_pipeline(input_dir: str, workdir: str = "."):
    # Make workdir absolute and build absolute data paths
    workdir = os.path.abspath(workdir)
    run_id = find_next_number(os.path.join(workdir, 'results'))
    print(f"Processing run with run id: {run_id}...")
    data_root = os.path.join(workdir, "results")
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
        try:
            lf.inference(run_id, img)
        except Exception as e:
            print(f"[LineFormer] Error on {img}: {e}")
            quit()


    # 4) ChartDete: detect chart elements
    print("Detecting chart elements with ChartDete...")
    cd = ChartDete()
    try:
        cd.inference(run_id)
    except Exception as e:
        print(f"[ChartDete] Error: {e}")
        quit()

    # 5) OCR axis labels (TrOCR via local OCRModel)
    print("Running OCR on axis labels...")
    axis_label_images = _collect_axis_label_images(base_output)
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

        out_json = os.path.join(img_dir_path, "chartdete", "axis_label_texts.json")
        payload = {
            path: extracted_text
            for (path, extracted_text) in label_texts
            if img_dir in path
        }
        print(f"Saving axis label OCR results to {out_json}...")
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

    # 5.4 OCR X and Y axis titles
    print("Running OCR on axis titles...")
    axis_title_images = _collect_axis_title_images(base_output)
    axis_title_texts = []
    for p in axis_title_images:
        try:
            axis_title_texts.append(ocr.inference(Path(p)))
        except Exception as e:
            print(f"[OCR] Error on axis title {p}: {e}")
            quit()

    # Save OCR results per-figure
    for img_dir in os.listdir(base_output):
        img_dir_path = os.path.join(base_output, img_dir)
        if not os.path.isdir(img_dir_path):
            continue

        out_json = os.path.join(img_dir_path, "chartdete", "axis_title_texts.json")
        payload = {
            path: extracted_text
            for (path, extracted_text) in axis_title_texts
            if img_dir in path
        }
        print(f"Saving axis title OCR results to {out_json}...")
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)



    # 5.5) OCR legend labels
    print("Running OCR on legend labels...")
    legend_label_images = _collect_legend_label_images(base_output)
    legend_label_texts = []
    for p in legend_label_images:
        try:
            legend_label_texts.append(ocr.inference(Path(p)))
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
        print('----img----', img)
        img_path = os.path.join(base_input, img)
        out_dir  = os.path.join(base_output, img, "chartdete")
        bb_json  = os.path.join(out_dir, "bounding_boxes.json")
        leg_json = os.path.join(out_dir, "legend_entries.json")
        lf_dir   = os.path.join(base_output, img, "lineformer")  # adjust if different

        if not os.path.exists(bb_json):
            print(f'[match] {img}: no boundary bbox')
        if not os.path.exists(leg_json):
            print(f'[match] {img}: no legend entries')

        with open(bb_json, "r") as f:
            boxes = json.load(f)
        plot_areas = sorted(boxes.get("plot_area", []), key=lambda el: el[4] if len(el)>4 else -1, reverse=True)
        plot_area = plot_areas[0] if plot_areas else None
        mark_boxes = boxes.get("mark_label", [])

        S = series_descriptors(img_path, plot_area, lf_dir, mark_boxes)
        L = legend_descriptors_from_json(img_path, leg_json)

        if not S:
            print(f"[match] {img}: no series descriptors found.")
            # continue

        if not L:
            print(f"[match] {img}: no legend descriptors found.")
            # continue

        # print('---L---', L)
        # print('-----S----', S)

        img_full = cv2.imread(img_path)
        if img_full is None:
            print(f"[match] Skip {img}: cannot read image.")
            continue

        C = build_cost_matrix_with_fallback(
            img_full.shape,
            L,
            S,
            alpha_dist=0.70,  # emphasize proximity for mark-label fallback
            beta_color=0.20,
            gamma_style=0.10
        )
        matches = hungarian_match(C)

        # print('----COST----', C)

        # print('---matches---', matches)

        # Save mapping
        mapping = []
        for i, j, c in matches:
            sj = S[j]
            sidx = sj.get("index")
            if sidx is None:
                # try parse from mask filename
                mp = sj.get("mask_path", "")
                m = re.search(r"(\d+)", os.path.basename(mp)) if mp else None
                sidx = int(m.group(1)) if m else j  # fallback to j

            mapping.append({
                "legend_index": i,
                "legend_text": L[i]["text"],
                "legend_patch_path": L[i]["patch_path"],
                "series_mask_path": sj.get("mask_path"),
                "series_index": int(sidx),                     # <-- now always present
                "cost": c,
                "legend_desc": {"hsv": L[i]["hsv"], "marker": L[i]["marker"], "linestyle": L[i]["linestyle"]},
                "series_desc": {"hsv": sj["hsv"], "marker": sj["marker"], "linestyle": sj["linestyle"]},
            })
        with open(os.path.join(out_dir, "legend_series_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

        print(f"[match] {img}: matched {len(matches)} legend entries to series.")

    # 9) Export labeled datasets and plots

    print("Exporting labeled datasets + plots...")
    for img in os.listdir(base_input):
        try:
            export_labeled_dataset_and_plot(run_id, img)
        except Exception as e:
            print(f"[finalize] Error on {img}: {e}")

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
    run_pipeline(input_dir='src/input', workdir='src/')
    # print("\n=== ChartDete predictions ===")
    # print(preds)
    print('=====Pipeline Finished Running=====')