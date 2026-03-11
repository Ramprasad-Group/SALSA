# match_legend.py
import os
import glob, re
import cv2, json, math, numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

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

def hsv_from_mask(img_bgr: np.ndarray, mask_bin: np.ndarray,
                  k_dilate: int = 1, min_keep: int = 50) -> Tuple[Tuple[float,float,float], float]:
    """
    Return (H,S,V) using only high-chroma pixels near the mask, plus a confidence [0..1].
    """
    mask = (mask_bin > 0).astype(np.uint8)
    if k_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k_dilate+1, 2*k_dilate+1))
        mask = cv2.dilate(mask, k, iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0), 0.0

    # sample 3x3 around each mask pixel; pick the pixel with max saturation
    H, W = img_bgr.shape[:2]
    picks = []
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y-1), min(H, y+2)
        x0, x1 = max(0, x-1), min(W, x+2)
        patch = img_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # flatten and take the pixel with maximum S
        idx = np.argmax(hsv[...,1].reshape(-1))
        hh = hsv[...,0].reshape(-1)[idx]
        ss = hsv[...,1].reshape(-1)[idx]
        vv = hsv[...,2].reshape(-1)[idx]
        picks.append((hh, ss, vv))

    if len(picks) < min_keep:
        # try a bit more dilation if too few samples
        return hsv_from_mask(img_bgr, cv2.dilate(mask, np.ones((3,3),np.uint8), 1), 0, 0)

    arr = np.asarray(picks, dtype=np.float32)
    # keep only reasonably colored pixels
    keep = (arr[:,2] > 40) & (arr[:,1] > 30)
    conf = float(np.mean(keep))  # fraction of colored pixels
    arr2 = arr[keep] if np.any(keep) else arr

    # dominant hue (mode over 0..179), mean S,V
    hist = np.bincount(arr2[:,0].astype(np.int32), minlength=180)
    mode = int(np.argmax(hist))
    Hdeg = float(mode * 2)
    Smean = float(arr2[:,1].mean()) if len(arr2) else float(arr[:,1].mean())
    Vmean = float(arr2[:,2].mean()) if len(arr2) else float(arr[:,2].mean())
    return (Hdeg, Smean, Vmean), conf

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

# def build_cost_matrix(legend_desc: List[Dict[str,Any]], series_desc: List[Dict[str,Any]]) -> np.ndarray:
#     M = np.zeros((len(legend_desc), len(series_desc)), dtype=np.float32)
#     for i, L in enumerate(legend_desc):
#         for j, S in enumerate(series_desc):
#             c = color_distance(L["hsv"], S["hsv"]) \
#                 + 0.5*shape_cost(L["marker"], S["marker"]) \
#                 + 0.4*style_cost(L["linestyle"], S["linestyle"])
#             M[i,j] = c
#     return M

def hungarian_match(cost: np.ndarray) -> List[Tuple[int,int,float]]:
    ri, cj = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i,j])) for i,j in zip(ri,cj)]

def _norm_diag(image_wh: Tuple[int,int]) -> float:
    W, H = image_wh
    return max(1.0, math.hypot(W, H))

def build_cost_matrix_with_fallback(
    image_shape: Tuple[int,int],
    legend_desc: List[Dict[str,Any]],
    series_desc: List[Dict[str,Any]],
    alpha_dist: float = 0.70,
    beta_color: float = 0.20,
    gamma_style: float = 0.10
) -> np.ndarray:
    """
    If legend entry is a real legend patch -> use your color/shape/style costs (as before).
    If legend entry is a mark_label (no patch) -> prioritize proximity to series centroid,
    optionally blending color/style if available.
    """
    H, W = image_shape[:2]
    dnorm = _norm_diag((W, H))
    M = np.zeros((len(legend_desc), len(series_desc)), dtype=np.float32)

    for i, L in enumerate(legend_desc):
        L_is_mark = (L.get("source") == "mark_label")
        Lbbox = L.get("label_bbox")
        Lhsv = L.get("hsv", (0., 0., 0.))
        for j, S in enumerate(series_desc):
            sc = float(S.get("color_conf", 0.0))
            # color weight between 0.3 and 0.9 depending on confidence
            w_color = 0.3 + 0.6*sc
            if not L_is_mark:
                c = w_color*color_distance(Lhsv, S["hsv"]) \
                    + 0.15*shape_cost(L["marker"], S["marker"]) \
                    + 0.15*style_cost(L["linestyle"], S["linestyle"])
                M[i, j] = c
            else:
                # Fallback: distance from label center to series centroid
                if Lbbox and "centroid" in S:
                    x1, y1, x2, y2, *_ = Lbbox
                    lc = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
                    sc = S["centroid"]
                    dist = math.hypot(lc[0] - sc[0], lc[1] - sc[1]) / dnorm  # 0..~1
                else:
                    dist = 1.0  # worst if we somehow lack geometry

                # Blend in weak color/style when available (they can help)
                ccol = color_distance(Lhsv, S["hsv"])
                csty = style_cost(L.get("linestyle", "unknown"), S.get("linestyle", "unknown"))

                # M[i, j] = float(alpha_dist * dist + beta_color * ccol + gamma_style * csty)
                M[i, j] = float(alpha_dist * dist + 0.5*w_color*ccol + gamma_style*csty)
    return M

def legend_descriptors_from_json(image_path: str, legend_json_path: str) -> List[Dict[str,Any]]:
    entries: List[Dict[str,Any]] = []

    if not os.path.exists(legend_json_path):
        return []

    with open(legend_json_path, "r") as f:
        ents = json.load(f)

    for ent in ents:
        src = (ent.get("source") or "legend").strip()
        patch_path = ent.get("patch_path")
        label_path = ent.get("label_path")
        label_bbox = ent.get("label_bbox")  # [x1,y1,x2,y2,score] or None

        hsv = (0.0, 0.0, 0.0)
        marker = "unknown"
        line_style = "unknown"

        if patch_path and os.path.exists(patch_path):
            patch = cv2.imread(patch_path)
            if patch is not None:
                hsv = dominant_hsv(patch)
                marker = detect_marker_shape(patch)
                # crude linestyle heuristic (as you had)
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                mid = edges.shape[0] // 2
                row = edges[mid, :] > 0
                runs = np.diff(np.where(np.concatenate(([row[0]],
                                                        row[:-1] != row[1:],
                                                        [True])))[0])[::2]
                if len(runs) >= 3 and np.median(runs) < 0.5 * (np.max(runs) + 1e-6):
                    line_style = "dashed"
                elif row.sum() > edges.shape[1] * 0.05:
                    line_style = "solid"
        else:
            # Fallback: sample color from the label crop if present
            if label_path and os.path.exists(label_path):
                lab_crop = cv2.imread(label_path)
                if lab_crop is not None:
                    hsv = dominant_hsv(lab_crop)
            # leave marker, line_style as "unknown"

        entries.append({
            "text": (ent.get("text") or "").strip(),
            "patch_path": patch_path,
            "label_path": label_path,
            "label_bbox": label_bbox,     # for proximity fallback
            "source": src,                # "legend" or "mark_label"
            "hsv": hsv,
            "marker": marker,
            "linestyle": line_style
        })
    return entries

def _mask_from_path(p: str) -> Optional[np.ndarray]:
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    return (m > 127).astype(np.uint8)

def series_descriptors(image_path: str,
                       plot_area: List[float],
                       lineformer_dir: str,
                       mark_boxes: List[List[float]]) -> List[Dict[str,Any]]:
    """
    Build descriptors for each extracted series.

    Returns a list of dicts with at least:
      - "index": stable int (matches series_i in coordinates.json)
      - "mask_path": str
      - "centroid": (x,y) in IMAGE coords
      - "hsv": tuple
      - "marker": str
      - "linestyle": str
    """
    entries: List[Dict[str,Any]] = []

    # Load base image (for HSV sampling)
    img = cv2.imread(image_path)
    if img is None:
        return entries

    # Load series catalog if present (written by convert_data_points)
    catalog_path = os.path.join(lineformer_dir, "series_catalog.json")
    catalog = []
    if os.path.exists(catalog_path):
        with open(catalog_path, "r") as f:
            catalog = json.load(f)

    # Discover masks (adjust patterns if yours differ)
    mask_files = sorted(glob.glob(os.path.join(lineformer_dir, "*.png")))
    # Prefer obvious mask names
    mask_files = [p for p in mask_files if re.search(r"(mask|series|line)\D*\d+\.png$", os.path.basename(p))] or mask_files

    # Utility: extract an index from filename if possible
    def _idx_from_name(p: str) -> Optional[int]:
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else None

    # Build a quick lookup from mask_path basename -> catalog item
    by_basename = {os.path.basename(c.get("mask_path","")): c for c in catalog if c.get("mask_path")}

    # # Plot area clip (optional)
    x1=y1=x2=y2=None
    if plot_area and len(plot_area)>=4:
        x1,y1,x2,y2 = map(float, plot_area[:4])

    for k, mpath in enumerate(mask_files):
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        # centroid
        ys, xs = np.where(m > 127)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        centroid = (float(cx), float(cy))

        # clip pixels to plot area if available (makes HSV more reliable)
        if x1 is not None:
            xs_clip = (xs >= x1) & (xs <= x2)
            ys_clip = (ys >= y1) & (ys <= y2)
            keep = xs_clip & ys_clip
            if np.count_nonzero(keep) >= 10:
                xs2, ys2 = xs[keep], ys[keep]
            else:
                xs2, ys2 = xs, ys
        else:
            xs2, ys2 = xs, ys

        img_h, img_w = img.shape[:2]
        if len(xs2):
            valid = (xs2 < img_w) & (ys2 < img_h)
            xs2, ys2 = xs2[valid], ys2[valid]
        if len(xs2):
            pixels = img[ys2, xs2]
        else:
            valid = (xs < img_w) & (ys < img_h)
            pixels = img[ys[valid], xs[valid]]

        mask_bin = (m > 127).astype(np.uint8)
        hsv, color_conf = hsv_from_mask(img, mask_bin, k_dilate=1)
        
        # linestyle
        linestyle = estimate_linestyle_along_mask(img, (m>127).astype(np.uint8))

        # marker is hard from polyline-only masks; leave unknown (color/line help a lot)
        marker = "unknown"

        # choose index
        base = os.path.basename(mpath)
        idx = None
        if base in by_basename and "series_index" in by_basename[base]:
            idx = int(by_basename[base]["series_index"])
        if idx is None:
            idx = _idx_from_name(base)
        if idx is None:
            print('---last resort index generator---')
            idx = k  # last-resort stable order

        entries.append({
            "index": int(idx),
            "mask_path": mpath,
            "centroid": centroid,
            "hsv": hsv,
            "marker": marker,
            "linestyle": linestyle,
            "hsv": hsv,
            "color_conf": float(color_conf)
        })

    # Sort by index to keep things deterministic
    entries.sort(key=lambda d: d["index"])
    return entries