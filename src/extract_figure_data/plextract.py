#!/usr/bin/env python3
"""
Local pipeline runner (no Modal).

What this does:
- Copies sub dirs from an input directory into a run-specific working directory
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
import datetime
import argparse
import logging
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


def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def _copy_inputs_to_run(input_dir: str, base_input: str) -> List[str]:

    print(f"Copying dir {input_dir} from paper-downloader to plextract base input destination {base_input}...")
    shutil.copytree(input_dir, base_input, dirs_exist_ok=True)
    return base_input

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
            if "title" in f.lower() and not f.lower().endswith(".json"):
                temp_titles.append(os.path.join(chartdete_dir, f))
                titles.append(os.path.join(chartdete_dir, f))
        if len(temp_titles) != 2: # check if found one x and one y axis title
            logger.warning(f'ERROR: len of extracted titles: {len(titles)}')
            logger.warning(f'TITLES: {titles}')
            

    return titles

def _collect_legend_label_images(base_output: str) -> List[str]:
    legends = []
    for plot_img_dir in os.listdir(base_output):
        leg_dir = os.path.join(base_output, plot_img_dir, "chartdete", "legend")
        if not os.path.isdir(leg_dir):
            continue

        # First try files starting with "label_"
        label_imgs = [os.path.join(leg_dir, f) for f in os.listdir(leg_dir)
                      if f.startswith("label_") and not f.endswith(".json")]

        # If none found, try "mark_label_"
        if not label_imgs:
            label_imgs = [os.path.join(leg_dir, f) for f in os.listdir(leg_dir)
                          if f.startswith("mark_label_") and not f.endswith(".json")]

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

def run_pipeline(input_dir: str, workdir: str = ".", run_id: int = 0, skip_lineformer: bool = False):
    # Make workdir absolute and build absolute data paths
    workdir = os.path.abspath(workdir)
    print(f"Processing run with run id: {run_id}...")
    data_root = os.path.join(workdir, "results")
    doi = input_dir.split('/')[-1]
    base_input = os.path.join(data_root, run_id, "input", doi)
    base_output = os.path.join(data_root, run_id, "output", doi)

    os.makedirs(base_output, exist_ok=True)

    # Export ABSOLUTE paths so downstream code never tries '/data/...'
    os.environ["PLEXTRACT_RUN_ID"] = run_id
    os.environ["PLEXTRACT_BASE_INPUT"] = base_input
    os.environ["PLEXTRACT_BASE_OUTPUT"] = base_output

    # Stage inputs
    input_dir = _copy_inputs_to_run(input_dir, base_input)

    # # 2) LineFormer: extract lines from images
    # print("Extracting lines from images with LineFormer...")
    # lf = LineFormer()
    # input_files = [f for f in os.listdir(input_dir)
    #                if os.path.isfile(os.path.join(base_input, f))]
    # for img in input_files:
    #     extension = img.split('.')[-1]
    #     if extension != 'json':
    #         try:
    #             lf.inference(img)
    #         except Exception as e:
    #             logger.warning(f"[LineFormer] Error on {img}: {e}")


    # # 3) ChartDete: detect chart elements
    # print("Detecting chart elements with ChartDete...")
    # cd = ChartDete()
    # for img in input_files:
    #     extension = img.split('.')[-1]
    #     if extension != 'json':
    #         try:
    #             cd.inference(img)
    #         except Exception as e:
    #             logger.warning(f"[ChartDete] Error: {e}")




    # try doing chartdete first

    # get input files
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(base_input, f))]


    # ChartDete: detect chart elements (and split subfigures)
    print("Detecting chart elements with ChartDete...")
    cd = ChartDete()
    for img in input_files:
        extension = img.split('.')[-1]
        if extension != 'json':
            try:
                predictions = cd.get_chartdete_preds(img)

                new_imgs = cd.split_subfigures(img, predictions)
                print("=====NEW_IMGS=====", new_imgs)
                if new_imgs:
                    # og image
                    cd.inference(img)
                    # get and save predictions on new images
                    for new_img in new_imgs:
                        print('=====NEW_IMG====', new_img)
                        cd.inference(Path(new_img).name)
                else:
                    # get and save predicitons on original image
                    cd.inference(img)



            except Exception as e:
                logger.warning(f"[ChartDete] Error: {e}")

    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(base_input, f))]


    # LineFormer: extract lines from images
    if skip_lineformer:
        print("Skipping LineFormer (using existing masks)...")
    else:
        print("Extracting lines from images with LineFormer...")
        lf = LineFormer()
        for img in input_files:
            extension = img.split('.')[-1]
            if extension != 'json':
                try:
                    lf.inference(img)
                except Exception as e:
                    logger.warning(f"[LineFormer] Error on {img}: {e}")


    # OCR axis labels (TrOCR via local OCRModel)
    print("Running OCR on axis labels...")
    axis_label_images = _collect_axis_label_images(base_output)
    ocr = OCRModel()

    label_texts = []
    for p in axis_label_images:
        try:
            # Expect tuple (path, extracted_text) to mirror the original usage
            label_texts.append(ocr.inference(Path(p)))
        except Exception as e:
            logger.warning(f"[OCR] Error on {p}: {e}")
    

    # Save label_texts OCR results per-figure
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
            logger.warning(f"[OCR] Error on axis title {p}: {e}")

    # Save axis_title_texts OCR results per-figure
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
            logger.warning(f"[OCR] Error on legend label {p}: {e}")

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
        extension = img.split('.')[-1]
        if extension != 'json':
            try:
                correct_coordinates(run_id, img)
            except Exception as e:
                logger.warning(f"[correct_coordinates] Error on {img}: {e}")

    # 7) Match legend to series and save mapping
    print("Matching legend entries to series...")
    for img in os.listdir(base_input):
        extension = img.split('.')[-1]
        if extension != 'json':
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
        extension = img.split('.')[-1]
        if extension != 'json':
            try:
                export_labeled_dataset_and_plot(run_id, img)
            except Exception as e:
                print(f"[finalize] Error on {img}: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description="Run local chart digitization pipeline.")
    # Resolve repo root so defaults work regardless of cwd
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    ap.add_argument(
        "--input",
        type=str,
        default=os.path.join(_repo_root, "src", "paper-downloader", "paper_downloader", "logs", "figures_tables", "figures"),
        help="Directory containing input images",
    )
    ap.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Working directory for run data",
    )

    ap.add_argument(
        "--log_file",
        type=str,
        default=os.path.join(_repo_root, "src", "extract_figure_data", "results"),
        help="default location for log file"
    )

    ap.add_argument(
        '--log_level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)')

    ap.add_argument(
        '--skip-lineformer',
        action='store_true',
        default=False,
        help='Skip LineFormer mask generation and use existing masks')

    ap.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Force a specific run ID instead of auto-incrementing. '
             'Used when lineformer masks have been pre-populated into a results directory.')

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Setup logging
    # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # log_file_path = os.path.join(args.log_file, f'plextract_{timestamp}.log')
    # logger = setup_logging(args.log_level, log_file_path)
    # logger = logging.getLogger('extraction')  # Use consistent logger name\

    # find doi dirs
    entries = os.listdir(args.input)
    subdirs = [os.path.join(args.input, e) for e in entries if os.path.isdir(os.path.join(args.input, e))]
    workdir = os.path.abspath(args.workdir)
    run_id = args.run_id if args.run_id is not None else find_next_number(os.path.join(workdir, 'results'))

    # Setup logging (stable name, workspace-scoped)
    os.makedirs(args.log_file, exist_ok=True)
    log_file_path = os.path.join(args.log_file, f'plextract_run_{run_id}.log')
    logger = setup_logging(args.log_level, log_file_path)
    logger = logging.getLogger('extraction')

    # exdtract data from figures for each doi
    for subdir in subdirs:
        print(f"=====RUN PIPELINE FOR {subdir}=====")
        run_pipeline(input_dir=subdir, workdir=args.workdir, run_id=run_id, skip_lineformer=args.skip_lineformer)

    print('=====Finished extracting figure data=====')