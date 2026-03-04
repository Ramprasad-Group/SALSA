import json


import os
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from PIL import Image
import glob, re, numpy as np, cv2


base_input_dir = os.environ.get("PLEXTRACT_BASE_INPUT")
base_output_dir = os.environ.get("PLEXTRACT_BASE_OUTPUT")


def _load_manual_pix2val_override(base_output_dir: str, img: str):
    """
    Return a user-supplied calibration override (if present) for this image.

    The Streamlit intervention UI writes axis-value calibrations to:
      chartdete/pix2val_override.json
    Those values are in *image pixel coordinates* and should take precedence
    over OCR-derived tick calibration when re-running extraction.
    """
    override_path = Path(base_output_dir) / img / "chartdete" / "pix2val_override.json"
    print('---override path---', override_path)
    if not override_path.exists():
        return None
    with open(override_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "x" not in data or "y" not in data:
        raise ValueError(f"Invalid pix2val_override format: {override_path}")
    return data


def _axis_map_value(pixel_value: float, axis_map: Dict[str, Any]) -> float:
    """Map a pixel coordinate to data value using a pix2val axis mapping."""
    a = float(axis_map.get("a", 1.0))
    b = float(axis_map.get("b", 0.0))
    axis_type = str(axis_map.get("type", "linear")).lower()
    t = a * pixel_value + b
    if axis_type == "linear":
        return t
    if axis_type == "log10":
        return 10 ** t
    # Unknown type fallback: keep linear behavior instead of crashing.
    return t

def _xy_from_coord(c: Any) -> Tuple[float, float]:
    """
    Accept either a point (x, y) or a bbox [x1, y1, x2, y2]; return a single plotting point.
    For bboxes we take the center; for points we pass through.
    """
    if isinstance(c, (list, tuple)):
        if len(c) >= 4:
            x1, y1, x2, y2 = c[:4]
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        elif len(c) == 2:
            return float(c[0]), float(c[1])
    raise ValueError(f"Unrecognized coordinate format: {c}")

def plot_axis_points_on_image(
    run_id: str,
    img: str,
    label_coordinates: Dict[str, Any],
    axis_label_texts: Dict[str, str],
    image_path: str = None,
):
    base_input_dir = os.environ.get("PLEXTRACT_BASE_INPUT")
    base_output_dir = os.environ.get("PLEXTRACT_BASE_OUTPUT")
    """
    Draw:
      - the plot area rectangle
      - points for each x/y axis tick label location
      - text annotations with the OCR'd numeric values
    Saves an overlay PNG alongside ChartDete outputs.
    """
    # Try to auto-locate the original figure if not provided
    if image_path is None:
        # Common locations in your pipeline; adjust if your input path differs
        candidates = [
            f"{base_input_dir}/{img}",
            f"{base_output_dir}/{img}/predictions.jpg",  # sometimes saved by detectors
            f"{base_output_dir}/{img}/{img}",            # fallback
        ]
        # print('---candidates---', candidates)
        # quit()
        for p in candidates:
            if os.path.exists(p):
                image_path = p
                break
        if image_path is None:
            raise FileNotFoundError("Could not locate base image; pass image_path explicitly.")

    # Where to save the overlay
    out_dir = f"{base_output_dir}/{img}/chartdete"
    os.makedirs(out_dir, exist_ok=True)
    out_png = f"{out_dir}/axis_debug_overlay.png"

    # Load image
    im = Image.open(image_path).convert("RGB")
    w, h = im.size

    # Prepare figure (origin='upper' so pixel coords align with image space)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.imshow(im, origin='upper')
    ax.set_title("Axis label coordinates overlay")
    ax.set_xticks([]); ax.set_yticks([])

    # 1) Draw plot area if present
    if "plot_area" in label_coordinates and len(label_coordinates["plot_area"]) >= 4:
        x1, y1, x2, y2 = label_coordinates["plot_area"][:4]
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], lw=2, alpha=0.8)

    # 2) Separate keys + annotate
    x_keys = [k for k in axis_label_texts.keys() if "xlabel" in k]
    y_keys = [k for k in axis_label_texts.keys() if "ylabel" in k]

    # Helpful: sort by pixel position to make drawing order intuitive
    # x: left->right; y: top->bottom (image y increases downward)
    x_sorted = sorted(x_keys, key=lambda k: _xy_from_coord(label_coordinates[k])[0])
    y_sorted = sorted(y_keys, key=lambda k: _xy_from_coord(label_coordinates[k])[1])

    # 3) Draw X labels
    for i, k in enumerate(x_sorted):
        x, y = _xy_from_coord(label_coordinates[k])
        txt = axis_label_texts[k]
        ax.scatter([x], [y], s=40, marker='o')  # default color
        ax.text(x, y - 6, f"x{i}:{txt}", ha='center', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

    # 4) Draw Y labels
    for i, k in enumerate(y_sorted):
        x, y = _xy_from_coord(label_coordinates[k])
        txt = axis_label_texts[k]
        ax.scatter([x], [y], s=40, marker='s')  # default color, different marker
        ax.text(x + 6, y, f"y{i}:{txt}", ha='left', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

    # 5) Add a tiny legend guide
    ax.text(8, 16, "• xlabels (circles)\n■ ylabels (squares)",
            ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[overlay] Wrote: {out_png}")

def sort_and_check_labels(label_coordinates: dict, axis_label_texts: dict, img_key: str):
    print("\tNow starting to sort and check labels...")

    # Convert all OCR texts to floats (skip plot_area)
    ocr_values = {k: float(v) for k, v in axis_label_texts.items() if "plot_area" not in k}

    yvalues = {k: v for k, v in ocr_values.items() if "ylabel" in k}
    xvalues = {k: v for k, v in ocr_values.items() if "xlabel" in k}

    # Use CENTER POINTS for coordinates
    y_coords = {k: _xy_from_coord(label_coordinates[k]) for k in yvalues.keys()}
    x_coords = {k: _xy_from_coord(label_coordinates[k]) for k in xvalues.keys()}

    # Sort labels by numeric value
    sorted_ylabels = sorted(yvalues.items(), key=lambda item: item[1])  # low->high
    sorted_xlabels = sorted(xvalues.items(), key=lambda item: item[1])

    # Sort coords by pixel position (x: left->right, y: top->bottom)
    sorted_y_coords = sorted(y_coords.items(), key=lambda item: item[1][1], reverse=True)  # top->bottom
    sorted_x_coords = sorted(x_coords.items(), key=lambda item: item[1][0])

    # Pair i-th pixel-position-sorted tick with i-th value-sorted tick.
    # This is the standard chart assumption (left/bottom pixel = smallest value).
    # If the key orders don't match it means OCR produced an unexpected value for
    # a tick label (e.g. a misread digit), but we proceed with the positional
    # pairing rather than raising an exception so the calibration can still run.
    xaggr, yaggr = {}, {}
    for (k1, v1), (k2, v2) in zip(sorted_x_coords, sorted_xlabels):
        xaggr[k1] = {"coord": v1, "val": v2}

    for (k1, v1), (k2, v2) in zip(sorted_y_coords, sorted_ylabels):
        yaggr[k1] = {"coord": v1, "val": v2}

    return {"xs": xaggr, "ys": yaggr}

def calc_conversion(coord_val_map: dict, plot_area: list):
    """Compute pixel->value linear maps for x and y in a consistent frame.
       X: use center-x in plot frame (left-positive).
       Y: use y_up = plot_y2 - center-y (upward-positive).
    """
    from scipy.stats import linregress

    x1, y1, x2, y2 = plot_area[:4]     # image coords
    plot_w = x2 - x1
    plot_h = y2 - y1
    assert plot_w > 0 and plot_h > 0

    def get_best_fit_x(coord_map: dict):
        pts = []
        for _, v in coord_map.items():
            cx, cy = v["coord"]  # already center from Patch 1
            x_local = cx - x1    # plot-area local x (left-positive)
            pts.append((x_local, v["val"]))
        xs, vals = zip(*pts)
        slope, intercept, *_ = linregress(xs, vals)
        return {"slope": slope, "intercept": intercept}

    def get_best_fit_y(coord_map: dict):
        pts = []
        for _, v in coord_map.items():
            cx, cy = v["coord"]      # center
            y_up = y2 - cy           # plot-area local upward-positive
            pts.append((y_up, v["val"]))
        ys, vals = zip(*pts)
        slope, intercept, *_ = linregress(ys, vals)
        return {"slope": slope, "intercept": intercept}

    xpix2val = get_best_fit_x(coord_val_map["xs"])
    ypix2val = get_best_fit_y(coord_val_map["ys"])
    return {"x": xpix2val, "y": ypix2val}

def convert_data_points(conversions, run_id: str, img: str, label_coordinates: dict):
    base_input_dir = os.environ.get("PLEXTRACT_BASE_INPUT")
    base_output_dir = os.environ.get("PLEXTRACT_BASE_OUTPUT")
    print("Converting data points now...")

    coordinates_path = Path(f"{base_output_dir}/{img}/lineformer/coordinates.json")
    if not coordinates_path.exists():
        print(f"\tWarning: coordinates.json not found at {coordinates_path}")
        print(f"\tSkipping data point conversion for {img}")
        return

    with open(coordinates_path, "r") as f:
        all_lineseries = json.load(f)

    x1, y1, x2, y2 = label_coordinates["plot_area"][:4]
    plot_area_height = y2 - y1
    plot_area_width  = x2 - x1
    assert plot_area_height > 0 and plot_area_width > 0

    def convertx(x_img):
        # Manual overrides use direct image pixel coordinates (x_img).
        if "a" in conversions.get("x", {}):
            return _axis_map_value(float(x_img), conversions["x"])
        # OCR-derived fallback uses plot-local coordinates.
        x_local = x_img - x1
        return conversions["x"]["slope"] * x_local + conversions["x"]["intercept"]

    def converty(y_img):
        # Manual overrides use direct image pixel coordinates (y_img).
        if "a" in conversions.get("y", {}):
            return _axis_map_value(float(y_img), conversions["y"])
        # OCR-derived fallback uses upward-positive plot-local coordinates.
        y_up = y2 - y_img
        return conversions["y"]["slope"] * y_up + conversions["y"]["intercept"]

    converted_lineseries = {}

    for series_index, lineseries in enumerate(all_lineseries):
        converted_points = []
        for point in lineseries:
            # If your LineFormer points are plot-local, change to:
            # x_img = point["x"] + x1
            # y_img = point["y"] + y1
            x_img = point["x"]
            y_img = point["y"]
            converted_points.append({
                "x": convertx(x_img),
                "y": converty(y_img),
            })
        converted_lineseries[f"series_{series_index}"] = converted_points

    dest_folder = f"{base_output_dir}/{img}/converted_datapoints"
    os.makedirs(dest_folder, exist_ok=True)
    with open(f"{dest_folder}/data.json", "w") as json_file:
        json.dump(converted_lineseries, json_file, indent=4)

    # === NEW: write a series catalog that maps series_i -> mask_path (and centroid) ===
    catalog_path = f"{base_output_dir}/{img}/lineformer/series_catalog.json"
    catalog = []

    # Look for masks written by LineFormer (e.g., mask_0.png, mask_1.png, ...)
    # If your filenames differ, adjust the glob/pattern accordingly.

    mask_files = sorted(glob.glob(f"{base_output_dir}/{img}/lineformer/*.png"))

    # Try to map masks to indices by number in filename; fallback to natural order
    def _extract_idx(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else None

    by_idx = { _extract_idx(p): p for p in mask_files if _extract_idx(p) is not None }

    for si, points in enumerate(all_lineseries):
        # pick a mask path for si
        mask_path = by_idx.get(si, by_idx.get(si))  # si if present; else arbitrary
        centroid = None
        if mask_path and os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                ys, xs = np.where(m > 127)
                if len(xs):
                    centroid = [float(xs.mean()), float(ys.mean())]
        catalog.append({
            "series_index": si,
            "mask_path": mask_path,
            "centroid": centroid
        })

    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"[lineformer] Wrote series catalog: {catalog_path}")

    plt.figure(figsize=(10, 6))
    for series_name, points in converted_lineseries.items():
        x_vals = [p["x"] for p in points]
        y_vals = [p["y"] for p in points]
        plt.plot(x_vals, y_vals, label=series_name)
    plt.xlabel("x-axis"); plt.ylabel("y-axis"); plt.legend()
    plt.savefig(f"{dest_folder}/plot.png")

def correct_coordinates(run_id: str, img: str):

    base_input_dir = os.environ.get("PLEXTRACT_BASE_INPUT")
    base_output_dir = os.environ.get("PLEXTRACT_BASE_OUTPUT")

    print("\tCorrecting coordinates for:", run_id, img)

    with open(
        f"{base_output_dir}/{img}/chartdete/label_coordinates.json", "r"
    ) as f:
        label_coordinates = json.load(f)

    with open(f"{base_output_dir}/{img}/chartdete/axis_label_texts.json", "r") as f:
        axis_label_texts = json.load(f)

    try:
        coord_val_map = sort_and_check_labels(
            label_coordinates=label_coordinates,
            axis_label_texts=axis_label_texts,
            img_key=img,
        )

        # Visual overlay to verify placement:
        plot_axis_points_on_image(
            run_id=run_id,
            img=img,
            label_coordinates=label_coordinates,
            axis_label_texts=axis_label_texts,
            # image_path=<optional explicit path to the original figure if needed>
        )


        manual_override = _load_manual_pix2val_override(base_output_dir, img)
        if manual_override is not None:
            print("\tUsing manual pix2val_override.json for coordinate conversion")
            conversions = {"x": manual_override["x"], "y": manual_override["y"]}
            print("conversions: ", conversions)
        else:
            conversions = calc_conversion(coord_val_map, plot_area=label_coordinates["plot_area"])
        convert_data_points(conversions, run_id, img, label_coordinates)

    except Exception as e:
        print(f"\tCorrecting coordinates of {img} did not work!")
        print(e)
