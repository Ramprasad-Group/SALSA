#!/usr/bin/env python3
"""
extract_series_xy.py

Purpose
-------
For each input figure in a finished run, produce (x,y) pairs for every series,
tagged with the matched legend/marker name.

EXPECTS (per image under {PLEXTRACT_BASE_OUTPUT}/{img}/chartdete/):
- bounding_boxes.json : has "plot_area": [[x1,y1,x2,y2,score], ...]
- legend_series_mapping.json : list of mappings
    [
      {
        "legend_index": i,
        "legend_text": "...",              # may be ""
        "legend_patch_path": ".../patch_0.jpg" or null,
        "series_mask_path": ".../lineformer/mask_series_002.png",
        "cost": 0.123,
        "legend_desc": {...},
        "series_desc": {...}
      },
      ...
    ]
- pix2val.json : axis calibration produced by correct_coordinates(), e.g.
    {
      "x": {"type": "linear", "a": 0.0123, "b": -5.6},
      "y": {"type": "linear", "a": -0.034, "b": 9.9},   # note inverted slope OK
      "domain": {"x": [xmin,xmax], "y": [ymin,ymax]}    # optional
    }
  Supported types: "linear", "log10" (meaning value = 10**(a*px + b)).

Also EXPECTS the series masks under {PLEXTRACT_BASE_OUTPUT}/{img}/lineformer/.
If your LineFormer uses a single "mask.png", connected components will be split.

Output
------
Writes CSV per image at:
{PLEXTRACT_BASE_OUTPUT}/{img}/series_xy.csv with columns:
series_name, series_mask_path, x, y, px, py, order
"""

import os, json, csv, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

def load_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def read_plot_area(bb_json: Path) -> Optional[List[float]]:
    bbs = load_json(bb_json)
    areas = sorted(bbs.get("plot_area", []), key=lambda el: el[4] if len(el)>4 else -1, reverse=True)
    return areas[0] if areas else None

def read_pix2val(pix2val_json: Path) -> Dict[str, Any]:
    data = load_json(pix2val_json)
    # Sanity defaults if something is missing
    for ax in ("x","y"):
        if ax not in data:
            data[ax] = {"type": "linear", "a": 1.0, "b": 0.0}
    return data

def px_to_val(px: float, mapping: Dict[str, float], axis_type: str) -> float:
    a = float(mapping.get("a", 1.0))
    b = float(mapping.get("b", 0.0))
    t = a*px + b
    if axis_type == "linear":
        return t
    elif axis_type == "log10":
        # mapping produced log10(value) = a*px + b, so invert:
        return 10.0 ** t
    else:
        # Fallback: treat as linear
        return t

def skeletonize_mask(m: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-ish skeleton via morphological thinning."""
    # OpenCV doesn't ship Zhang-Suen thinning; emulate with iterative erosion on edges
    # For our purpose (ordering pixels by x), extracting the contour is enough and robust.
    contours, _ = cv2.findContours((m>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(m)
    # Take the largest contour as the series centerline proxy
    cnt = max(contours, key=cv2.contourArea)
    skel = np.zeros_like(m)
    cv2.drawContours(skel, [cnt], -1, 255, 1)  # 1 px stroke
    return skel

def mask_points_sorted_by_x(m: np.ndarray, plot_area: Tuple[int,int,int,int]) -> np.ndarray:
    """Collect foreground pixels inside plot area and sort by x (then y)."""
    x1,y1,x2,y2 = plot_area
    sub = m[y1:y2, x1:x2]
    ys, xs = np.where(sub > 0)
    if xs.size == 0:
        return np.empty((0,2), dtype=np.int32)
    # Convert back to full-image coords
    xs = xs + x1
    ys = ys + y1
    # Sort by x, then y to get a left-to-right polyline
    order = np.lexsort((ys, xs))
    pts = np.stack([xs[order], ys[order]], axis=1)
    # Downsample to something reasonable
    if len(pts) > 2000:
        idx = np.linspace(0, len(pts)-1, 2000).astype(int)
        pts = pts[idx]
    return pts.astype(np.int32)

def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    # binarize
    return (m > 127).astype(np.uint8)

def infer_series_name(mapping_item: Dict[str,Any]) -> str:
    txt = (mapping_item.get("legend_text") or "").strip()
    if txt:
        return txt
    # Fallbacks for mark-labels or missing OCR
    lp = mapping_item.get("legend_desc", {})
    src = lp.get("source", None) or mapping_item.get("legend_source", None)
    if src == "mark_label":
        return "mark_label"
    # Last resort: basename of patch or mask
    patch = mapping_item.get("legend_patch_path") or ""
    if patch:
        return Path(patch).stem
    return Path(mapping_item.get("series_mask_path","series")).stem

def extract_for_image(base_out_dir: Path, img_name: str) -> Optional[Path]:
    img_dir = base_out_dir / img_name
    chartdete = img_dir / "chartdete"
    lf_dir   = img_dir / "lineformer"
    out_csv  = img_dir / "series_xy.csv"

    # required inputs
    bb_json  = chartdete / "bounding_boxes.json"
    map_json = chartdete / "legend_series_mapping.json"
    pix2val  = chartdete / "pix2val.json"

    if not bb_json.exists() or not map_json.exists() or not pix2val.exists():
        print(f"[extract] Missing inputs for {img_name} (need bounding_boxes.json, legend_series_mapping.json, pix2val.json)")
        return None

    plot_area = read_plot_area(bb_json)
    if not plot_area:
        print(f"[extract] No plot_area for {img_name}")
        return None
    x1,y1,x2,y2,_ = plot_area
    plot_box = (int(x1), int(y1), int(x2), int(y2))

    cal = read_pix2val(pix2val)
    xmap, ymap = cal.get("x", {}), cal.get("y", {})
    xtype, ytype = xmap.get("type", "linear"), ymap.get("type", "linear")

    mapping = load_json(map_json)

    rows = []
    order_counter = 0

    for mitem in mapping:
        series_name = infer_series_name(mitem)
        mask_path = Path(mitem["series_mask_path"])
        m = load_mask(mask_path)
        if m is None:
            print(f"[extract] Mask missing: {mask_path}")
            continue

        # Restrict to plot area and get an ordered polyline via contour
        skel = skeletonize_mask(m)
        pts = mask_points_sorted_by_x(skel, plot_box)
        if pts.shape[0] == 0:
            # try using raw mask if skeleton had holes
            pts = mask_points_sorted_by_x(m, plot_box)
        if pts.shape[0] == 0:
            print(f"[extract] No points in plot area for {mask_path}")
            continue

        for (px, py) in pts:
            xv = px_to_val(px, xmap, xtype)
            yv = px_to_val(py, ymap, ytype)
            rows.append({
                "series_name": series_name,
                "series_mask_path": str(mask_path),
                "x": xv,
                "y": yv,
                "px": int(px),
                "py": int(py),
                "order": order_counter
            })
            order_counter += 1

    # sort within each series by x (value space) for cleanliness
    rows.sort(key=lambda r: (r["series_name"], r["x"]))

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series_name","series_mask_path","x","y","px","py","order"])
        w.writeheader()
        w.writerows(rows)

    print(f"[extract] Wrote {out_csv}")
    return out_csv

def main():
    base_out = os.environ.get("PLEXTRACT_BASE_OUTPUT")
    if not base_out:
        raise RuntimeError("PLEXTRACT_BASE_OUTPUT not set. Run via your main pipeline so env is exported.")
    base_out = Path(base_out)
    for img_name in os.listdir(base_out):
        if not (base_out / img_name).is_dir(): 
            continue
        extract_for_image(base_out, img_name)

if __name__ == "__main__":
    main()