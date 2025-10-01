# match_legend.py
import cv2, json, math, numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

import cv2, numpy as np, math

def dominant_hsv(img_or_pixels):
    """
    Accepts:
      - HxWx3 BGR image,
      - Nx3 list of BGR pixels (e.g., img[mask]),
      - HxW (grayscale) image.

    Returns (hue_degrees, mean_s, mean_v).
    """
    arr = np.asarray(img_or_pixels)
    if arr is None or arr.size == 0:
        return (0.0, 0.0, 0.0)

    # Normalize to an array with 3 channels so cvtColor is happy
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = arr
    elif arr.ndim == 2:
        # Grayscale image -> make 3 channels
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        # Nx3 flat pixel list -> reshape to 1xNx3
        bgr = arr.reshape(1, -1, 3)
    else:
        # Try best-effort: if last dim is 3, treat as pixels
        if arr.ndim >= 1 and arr.shape[-1] == 3:
            bgr = arr.reshape(1, -1, 3)
        else:
            # Fallback: duplicate as 3 channels
            arr2 = arr.astype(np.uint8)
            if arr2.ndim == 1:
                arr2 = arr2[:, None]
            bgr = np.repeat(arr2[..., None], 3, axis=-1)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Flatten to a list of HSV pixels
    if hsv.ndim == 3:
        H = hsv[..., 0].ravel()
        S = hsv[..., 1].ravel()
        V = hsv[..., 2].ravel()
    else:
        H = hsv[:, 0]
        S = hsv[:, 1]
        V = hsv[:, 2]

    # Filter out near-white/near-black to avoid background bias
    mask = (V > 40) & (S > 30)
    if np.count_nonzero(mask) >= 30:
        Hf = H[mask]
        Sf = S[mask]
        Vf = V[mask]
    else:
        Hf, Sf, Vf = H, S, V

    if Hf.size == 0:
        return (0.0, float(S.mean()) if S.size else 0.0, float(V.mean()) if V.size else 0.0)

    hist = np.bincount(Hf.astype(np.int32), minlength=180)
    mode = int(np.argmax(hist))  # 0..179
    return float(mode * 2), float(Sf.mean()), float(Vf.mean())

def detect_marker_shape(img: np.ndarray) -> str:
    """Return one of {'circle','square','triangle','star','line','unknown'}."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 50, 150)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return "unknown"
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 20: return "unknown"
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
    k = len(approx)
    if k <= 2: return "line"
    if k == 3: return "triangle"
    if k == 4:
        # square vs rectangle by aspect
        x,y,w,h = cv2.boundingRect(approx)
        ar = w/float(h)
        return "square" if 0.8 <= ar <= 1.25 else "rect"
    if k == 5: return "star"  # rough
    # circle-ish test
    (x,y),r = cv2.minEnclosingCircle(cnt)
    circ_area = math.pi*r*r
    if area / (circ_area + 1e-6) > 0.6:
        return "circle"
    return "unknown"

def estimate_linestyle_along_mask(img: np.ndarray, mask: np.ndarray) -> str:
    """
    mask: binary uint8 (0/1) of the series pixels (from LineFormer)
    We'll check run-length along skeleton to guess solid vs dashed.
    """
    # Skeletonize cheaply by thinning edges
    # Simple approach: project to a 1D profile along major axis
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return "unknown"
    # PCA to get principal direction
    X = np.stack([xs, ys], axis=1).astype(np.float32)
    X -= X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    dir_vec = Vt[0]  # principal axis
    # Project points
    t = X @ dir_vec
    t_sorted = np.sort(t)
    # Run-length: count gaps between consecutive pixels
    diffs = np.diff(t_sorted)
    # Large gaps indicate dashes
    gap_thresh = np.percentile(diffs, 95)
    large_gaps = (diffs > gap_thresh).sum()
    return "dashed" if large_gaps > max(2, len(diffs)*0.02) else "solid"

def color_distance(hsv1: Tuple[float,float,float], hsv2: Tuple[float,float,float]) -> float:
    # circular hue distance + weighted S,V
    dh = min(abs(hsv1[0]-hsv2[0]), 360-abs(hsv1[0]-hsv2[0]))/180.0  # 0..1
    ds = abs(hsv1[1]-hsv2[1])/255.0
    dv = abs(hsv1[2]-hsv2[2])/255.0
    return 0.6*dh + 0.25*ds + 0.15*dv

def shape_cost(a: str, b: str) -> float:
    if a == "unknown" or b == "unknown": return 0.3
    return 0.0 if a == b else 1.0

def style_cost(a: str, b: str) -> float:
    if a == "unknown" or b == "unknown": return 0.2
    return 0.0 if a == b else 0.8

def build_cost_matrix(legend_desc: List[Dict[str,Any]], series_desc: List[Dict[str,Any]]) -> np.ndarray:
    M = np.zeros((len(legend_desc), len(series_desc)), dtype=np.float32)
    for i, L in enumerate(legend_desc):
        for j, S in enumerate(series_desc):
            c = color_distance(L["hsv"], S["hsv"]) \
                + 0.5*shape_cost(L["marker"], S["marker"]) \
                + 0.4*style_cost(L["linestyle"], S["linestyle"])
            M[i,j] = c
    return M

def hungarian_match(cost: np.ndarray) -> List[Tuple[int,int,float]]:
    ri, cj = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i,j])) for i,j in zip(ri,cj)]


def legend_descriptors_from_json(image_path: str, legend_json_path: str) -> List[Dict[str,Any]]:
    entries = []
    with open(legend_json_path, "r") as f:
        ents = json.load(f)
    for ent in ents:
        patch = cv2.imread(ent["patch_path"])
        if patch is None: continue
        hsv = dominant_hsv(patch)
        marker = detect_marker_shape(patch)
        # Many legends also hint line style as a small line segment; try to detect dashed vs solid:
        line_style = "unknown"
        # Heuristic: look for long thin components inside patch
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Run-length in central row
        mid = edges.shape[0]//2
        row = edges[mid,:] > 0
        runs = np.diff(np.where(np.concatenate(([row[0]],
                                                row[:-1] != row[1:],
                                                [True])))[0])[::2]
        if len(runs) >= 3 and np.median(runs) < 0.5* (np.max(runs)+1e-6):
            line_style = "dashed"
        elif row.sum() > edges.shape[1]*0.05:
            line_style = "solid"

        entries.append({
            "text": (ent.get("text") or "").strip(),
            "patch_path": ent["patch_path"],
            "hsv": hsv,
            "marker": marker,
            "linestyle": line_style
        })
    return entries

def _mask_from_path(p: str) -> Optional[np.ndarray]:
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    return (m > 127).astype(np.uint8)

def series_descriptors(image_path: str, plot_area: List[float], lineformer_dir: str, mark_boxes: List[List[float]]) -> List[Dict[str,Any]]:
    """
    - image_path: original figure
    - plot_area: [x1,y1,x2,y2,score]
    - lineformer_dir: folder with masks (mask_series_*.png or mask.png)
    - mark_boxes: ChartDete 'mark_label' (optional)
    """
    img = cv2.imread(image_path)
    H,W = img.shape[:2]
    x1,y1,x2,y2,_ = plot_area if plot_area else [0,0,W,H,1.0]
    roi = img[int(y1):int(y2), int(x1):int(x2)]
    series = []

    # Try per-series masks
    lf_dir = Path(lineformer_dir)
    cand = sorted(lf_dir.glob("mask_series_*.png"))
    masks = []
    if cand:
        for p in cand:
            m = _mask_from_path(str(p))
            if m is not None:
                masks.append((str(p), m))
    else:
        # Fallback single mask
        m = _mask_from_path(str(lf_dir / "mask.png"))
        if m is not None:
            # Connected components to split
            num, lab = cv2.connectedComponents(m)
            for k in range(1, num):
                mk = (lab == k).astype(np.uint8)
                masks.append((f"{lf_dir}/mask_cc_{k}.png", mk))

    # Compute marker hints from mark_boxes (shape + local color)
    box_markers = []
    for bb in mark_boxes:
        if len(bb) < 5: continue
        bx1,by1,bx2,by2,_ = bb
        bx1i,by1i,bx2i,by2i = map(int, [bx1,by1,bx2,by2])
        bx1i,by1i = max(bx1i,0), max(by1i,0)
        bx2i,by2i = min(bx2i,W-1), min(by2i,H-1)
        if bx2i>bx1i and by2i>by1i:
            crop = img[by1i:by2i, bx1i:bx2i]
            box_markers.append((bb, detect_marker_shape(crop), dominant_hsv(crop)))

    # Build descriptors per series mask
    for (mpath, m) in masks:
        # restrict to plot area
        sub = m[int(y1):int(y2), int(x1):int(x2)]
        if sub.sum() < 50:
            continue
        hsv = dominant_hsv(roi[sub.astype(bool)])
        linestyle = estimate_linestyle_along_mask(roi, sub)
        # optional marker inference: pick nearest mark box to series centroid
        ys, xs = np.where(sub>0)
        cx, cy = int(x1)+int(xs.mean()), int(y1)+int(ys.mean())
        nearest_marker = ("unknown", (0.,0.,0.))
        if box_markers:
            d2, best = 1e18, None
            for bb, shape, mhsv in box_markers:
                bx1,by1,bx2,by2,_ = bb
                mx, my = 0.5*(bx1+bx2), 0.5*(by1+by2)
                dd = (mx-cx)**2 + (my-cy)**2
                if dd < d2:
                    d2, best = dd, (shape, mhsv)
            if best: nearest_marker = best

        series.append({
            "mask_path": mpath,
            "hsv": hsv,
            "linestyle": linestyle,
            "marker": nearest_marker[0]  # might be 'unknown'
        })
    return series