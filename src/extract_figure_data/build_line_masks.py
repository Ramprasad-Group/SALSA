# build_line_masks.py
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def _img_size_from_predictions(pred_img_path: Path) -> Optional[Tuple[int,int]]:
    im = cv2.imread(str(pred_img_path))
    if im is None: return None
    h, w = im.shape[:2]
    return (h, w)

def _extract_polylines(obj: Any) -> List[List[Tuple[float,float]]]:
    """
    Be liberal in what we accept. Handles:
      - {"polylines": [ [[x,y],...], ... ]}
      - {"lines": [ {"points":[[x,y],...]}, ... ]}
      - [{"points":[[x,y],...]}, ...]
      - [[[x,y],...], ...]
    Returns: list of polyline as list of (x,y) floats.
    """
    def as_polyline(x) -> Optional[List[Tuple[float,float]]]:
        if isinstance(x, dict):
            for key in ("points","polyline","coords","xy","vertices"):
                if key in x and isinstance(x[key], (list, tuple)):
                    pts = x[key]
                    if len(pts) >= 2 and isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 2:
                        return [(float(p[0]), float(p[1])) for p in pts]
            # Some tools store as {"x":[...], "y":[...]}
            if "x" in x and "y" in x and isinstance(x["x"], list) and isinstance(x["y"], list):
                if len(x["x"]) == len(x["y"]) and len(x["x"]) >= 2:
                    return [(float(a), float(b)) for a,b in zip(x["x"], x["y"])]
            return None
        elif isinstance(x, (list, tuple)):
            if len(x) >= 2 and isinstance(x[0], (list, tuple)) and len(x[0]) >= 2:
                # already a list of [x,y]
                try:
                    return [(float(p[0]), float(p[1])) for p in x]
                except Exception:
                    return None
        return None

    cand_lists = []
    # dict with a known top-level key
    if isinstance(obj, dict):
        for k in ("polylines","lines","segments","shapes","data"):
            if k in obj and isinstance(obj[k], (list, tuple)):
                cand_lists.append(obj[k])
        # sometimes everything is at top-level list-like
        if not cand_lists:
            for v in obj.values():
                if isinstance(v, (list, tuple)):
                    cand_lists.append(v)

    elif isinstance(obj, (list, tuple)):
        cand_lists.append(obj)

    polylines: List[List[Tuple[float,float]]] = []
    for L in cand_lists:
        for item in L:
            pl = as_polyline(item)
            if pl is not None and len(pl) >= 2:
                polylines.append(pl)

    # De-duplicate identical polylines
    seen = set()
    unique = []
    for pl in polylines:
        key = tuple((round(x,2), round(y,2)) for x,y in pl[:5])  # cheap hash
        if key not in seen:
            seen.add(key)
            unique.append(pl)
    return unique

def _estimate_thickness(polylines: List[List[Tuple[float,float]]], default: int = 2) -> int:
    # If we had stroke width in JSON, we could use it; otherwise use a conservative 2 px.
    return int(default)

def _draw_polyline_mask(h: int, w: int, poly: List[Tuple[float,float]], thickness: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x,y in poly], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(m, [pts], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return m

def _merge_close_polylines(polylines: List[List[Tuple[float,float]]], join_thresh: float = 8.0) -> List[List[Tuple[float,float]]]:
    """
    Optional: merge fragmented pieces if the end of one is near the start of another and directions align.
    Very light-weight; good enough for plots where a series is broken at tick/legend overlays.
    """
    if not polylines:
        return polylines

    def endpoints(pl):
        return np.array(pl[0]), np.array(pl[-1])

    used = [False]*len(polylines)
    merged: List[List[Tuple[float,float]]] = []

    for i, pi in enumerate(polylines):
        if used[i]: continue
        cur = pi[:]
        used[i] = True
        changed = True
        while changed:
            changed = False
            s0, e0 = endpoints(cur)
            for j, pj in enumerate(polylines):
                if used[j] or i == j: continue
                s1, e1 = endpoints(pj)
                # try four combinations to see if we can concatenate
                pairs = [
                    (np.linalg.norm(e0 - s1), "e0->s1", False),
                    (np.linalg.norm(s0 - e1), "e1->s0", True),
                    (np.linalg.norm(s0 - s1), "s0->s1", "rev_cur"),
                    (np.linalg.norm(e0 - e1), "e0->e1", "rev_pj"),
                ]
                dmin, mode, flag = min(pairs, key=lambda x: x[0])
                if dmin <= join_thresh:
                    # concatenate with direction fixes
                    cj = pj[:]
                    if mode == "e0->s1":
                        cur = cur + cj
                    elif mode == "e1->s0":
                        cur = cj + cur
                    elif mode == "s0->s1":
                        cur = list(reversed(cur)) + cj
                    elif mode == "e0->e1":
                        cur = cur + list(reversed(cj))
                    used[j] = True
                    changed = True
                    break
        merged.append(cur)
    return merged

def build_masks_for_image(
    image_path: Path,
    lineformer_dir: Path,
    coords_json: Path,
    predictions_png: Optional[Path] = None,
    write_union: bool = True,
    merge_fragments: bool = True,
    thickness_px: Optional[int] = None
) -> List[Path]:
    """
    Returns list of per-series mask paths written to lineformer_dir.
    """
    _ensure_dir(lineformer_dir)
    obj = _load_json(coords_json)
    polylines = _extract_polylines(obj)
    if not polylines:
        raise RuntimeError(f"No polylines found in {coords_json}")

    # Determine canvas size
    size = None
    if predictions_png and predictions_png.exists():
        size = _img_size_from_predictions(predictions_png)
    if size is None:
        # fall back to reading the original image
        im = cv2.imread(str(image_path))
        if im is None:
            raise RuntimeError(f"Cannot read image to infer size: {image_path}")
        size = im.shape[:2]
    H, W = size

    # Optional merge of fragments
    if merge_fragments:
        polylines = _merge_close_polylines(polylines)

    thick = thickness_px if thickness_px is not None else _estimate_thickness(polylines, default=2)

    mask_paths: List[Path] = []
    union = np.zeros((H, W), dtype=np.uint8)
    for i, poly in enumerate(polylines):
        m = _draw_polyline_mask(H, W, poly, thickness=thick)
        outp = lineformer_dir / f"mask_series_{i}.png"
        cv2.imwrite(str(outp), m)
        mask_paths.append(outp)
        union = cv2.bitwise_or(union, m)

    if write_union:
        cv2.imwrite(str(lineformer_dir / "mask.png"), union)

    # optional debug overlay if you want
    # ov = cv2.imread(str(image_path))
    # ov = cv2.addWeighted(ov, 0.7, cv2.cvtColor(union, cv2.COLOR_GRAY2BGR), 0.3, 0)
    # cv2.imwrite(str(lineformer_dir / "mask_overlay.png"), ov)

    return mask_paths


def regenerate_coordinates_from_masks(lineformer_dir: Path) -> None:
    """
    Rebuild ``coordinates.json`` from ``mask_series_*.png`` files.

    For each mask the function extracts the centerline by averaging y for
    every unique x column, producing an ordered polyline that
    ``correct_coordinates`` can consume.  This is needed when masks have
    been edited by the user and LineFormer is skipped (``--skip-lineformer``).
    """
    import glob as _glob
    import re as _re

    mask_files = sorted(_glob.glob(str(lineformer_dir / "mask_series_*.png")))
    if not mask_files:
        return

    # Write a dict keyed by the filename-derived series index so that
    # correct_coordinates assigns the right series_<N> key even when indices
    # are non-consecutive (e.g. a newly created series with index 5 among
    # existing indices 0 and 1).
    series_dict: Dict[str, List[Dict[str, float]]] = {}
    for mask_path in mask_files:
        mask_p = Path(mask_path)

        # Extract the numeric index from the filename (mask_series_<N>.png).
        _m = _re.search(r"(\d+)", mask_p.stem)
        series_key = _m.group(1) if _m else str(len(series_dict))

        # If the user picked exact points via the UI, a clicked_points_<i>.json
        # file sits alongside the mask.  Use those coordinates directly instead
        # of deriving them from the rasterised mask.
        clicked_pts_file = mask_p.parent / mask_p.name.replace("mask_series_", "clicked_points_").replace(".png", ".json")
        if clicked_pts_file.exists():
            try:
                pts = json.loads(clicked_pts_file.read_text())
                pts = [{"x": float(p["x"]), "y": float(p["y"])} for p in pts]
                pts.sort(key=lambda p: p["x"])
                series_dict[series_key] = pts
                continue
            except Exception:
                pass  # fall through to mask-extraction on error

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            series_dict[series_key] = []
            continue

        ys, xs = np.where(m > 127)
        if len(xs) == 0:
            series_dict[series_key] = []
            continue

        # For each unique x column, average the y values to get the centerline
        unique_xs = np.unique(xs)
        points: List[Dict[str, float]] = []
        for ux in unique_xs:
            mean_y = float(np.mean(ys[xs == ux]))
            points.append({"x": float(ux), "y": mean_y})

        points.sort(key=lambda p: p["x"])

        # Downsample if too many points (keeps file size manageable)
        if len(points) > 2000:
            indices = np.linspace(0, len(points) - 1, 2000).astype(int)
            points = [points[i] for i in indices]

        series_dict[series_key] = points

    coords_path = lineformer_dir / "coordinates.json"
    with open(coords_path, "w") as f:
        json.dump(series_dict, f)
    print(f"[lineformer] Regenerated coordinates.json from masks: {coords_path}")