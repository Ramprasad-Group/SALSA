import json


import os
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from PIL import Image

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
    """
    Draw:
      - the plot area rectangle
      - points for each x/y axis tick label location
      - text annotations with the OCR'd numeric values
    Saves an overlay PNG alongside ChartDete outputs.
    """
    base_dir = f"/home/wschertzer3/extract-line-chart-data/data/{run_id}"
    # Try to auto-locate the original figure if not provided
    if image_path is None:
        # Common locations in your pipeline; adjust if your input path differs
        candidates = [
            f"{base_dir}/input/{img}",
            f"{base_dir}/output/{img}/predictions.jpg",  # sometimes saved by detectors
            f"{base_dir}/output/{img}/{img}",            # fallback
        ]
        for p in candidates:
            if os.path.exists(p):
                image_path = p
                break
        if image_path is None:
            raise FileNotFoundError("Could not locate base image; pass image_path explicitly.")

    # Where to save the overlay
    out_dir = f"{base_dir}/output/{img}/chartdete"
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
    # (We only use this to check pairing consistency)
    sorted_y_coords = sorted(y_coords.items(), key=lambda item: item[1][1], reverse=True)  # top->bottom
    sorted_x_coords = sorted(x_coords.items(), key=lambda item: item[1][0])

    assert [label for label, _ in sorted_xlabels] == [coord for coord, _ in sorted_x_coords], \
        "The keys of sorted_xlabels and sorted_x_coords are not in the same order."

    assert [label for label, _ in sorted_ylabels] == [coord for coord, _ in sorted_y_coords], \
        "The keys of sorted_ylabels and sorted_y_coords are not in the same order."

    xaggr, yaggr = {}, {}
    for (k1, v1), (k2, v2) in zip(sorted_x_coords, sorted_xlabels):
        assert k1 == k2
        xaggr[k1] = {"coord": v1, "val": v2}

    for (k1, v1), (k2, v2) in zip(sorted_y_coords, sorted_ylabels):
        assert k1 == k2
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
    print("Converting data points now...")

    with open(f"/home/wschertzer3/extract-line-chart-data/data/{run_id}/output/{img}/lineformer/coordinates.json", "r") as f:
        all_lineseries = json.load(f)

    x1, y1, x2, y2 = label_coordinates["plot_area"][:4]
    plot_area_height = y2 - y1
    plot_area_width  = x2 - x1
    assert plot_area_height > 0 and plot_area_width > 0

    def convertx(x_img):
        # if your LineFormer x is already in image coords, use x_img directly;
        # if it's plot-local, add x1 first. Below I assume image coords:
        x_local = x_img - x1  # plot-area local (left-positive)
        return conversions["x"]["slope"] * x_local + conversions["x"]["intercept"]

    def converty(y_img):
        # Same: assume y_img is image coords. If plot-local, add y1 first.
        y_up = y2 - y_img     # plot-area local upward-positive
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

    dest_folder = f"/home/wschertzer3/extract-line-chart-data/data/{run_id}/output/{img}/converted_datapoints"
    os.makedirs(dest_folder, exist_ok=True)
    with open(f"{dest_folder}/data.json", "w") as json_file:
        json.dump(converted_lineseries, json_file, indent=4)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for series_name, points in converted_lineseries.items():
        x_vals = [p["x"] for p in points]
        y_vals = [p["y"] for p in points]
        plt.plot(x_vals, y_vals, label=series_name)
    plt.xlabel("x-axis"); plt.ylabel("y-axis"); plt.legend()
    plt.savefig(f"{dest_folder}/plot.png")

def correct_coordinates(run_id: str, img: str):

    print("\tCorrecting coordinates for:", run_id, img)

    with open(
        f"/home/wschertzer3/extract-line-chart-data/data/{run_id}/output/{img}/chartdete/label_coordinates.json", "r"
    ) as f:
        label_coordinates = json.load(f)

    with open(f"/home/wschertzer3/extract-line-chart-data/data/{run_id}/output/{img}/axis_label_texts.json", "r") as f:
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


        conversions = calc_conversion(coord_val_map, plot_area=label_coordinates["plot_area"])
        convert_data_points(conversions, run_id, img, label_coordinates)

    except Exception as e:
        print(f"\tCorrecting coordinates of {img} did not work!")
        print(e)
