import os, json, csv, re
from pathlib import Path
from typing import Tuple, Optional, Dict
from plot_final_dataset import plot_final_dataset

# -----------------------------
# Unit parsing & conversion
# -----------------------------

_UNIT_ALIASES = {
    # time
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "days": "day", "day": "day",

    # conductivity
    "s/cm": "S/cm", "S/cm": "S/cm",
    "ms/cm": "mS/cm", "mS/cm": "mS/cm",

}

# Map canonical unit to a "dimension" so we know how to pick a target unit
_UNIT_DIMENSIONS = {
    # time
    "s": "time", "min": "time", "h": "time",
    # conductivity
    "S/cm": "conductivity", "mS/cm": "conductivity", "S/m": "conductivity", "mS/m": "conductivity",
}

# Choose target canonical units per dimension
_CANONICAL_TARGET = {
    "time": "h",
    "conductivity": "mS/cm",
}

def _clean_label(s: str) -> str:
    if not s: return "unlabeled"
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[;,:.\u200b]+$", "", s).strip()
    return s if s else "unlabeled"

def _normalize_unit_text(u: str) -> Optional[str]:
    if u is None: return None
    t = u.strip().replace(" ", "")
    t = t.replace("·","/")  # normalize "S·cm^-1" style to slash where possible (heuristic)
    t = t.replace("^2", "2")  # normalize exponent formatting for matching keys like cm^2
    t = t.replace("CM", "cm").replace("Cm", "cm")
    t = t.replace("MS","mS").replace("MS","mS")
    lower = t.lower()

    # direct alias hit
    return _UNIT_ALIASES.get(lower, u.strip())

def _parse_label_and_unit(text: str) -> Tuple[str, Optional[str]]:
    """
    Extracts label and unit from strings like:
    "Hydroxide Conductivity [mS/cm]", "Time (h)"
    Returns (label, unit or None).
    """
    if not text: return "unlabeled", None
    s = _clean_label(text)
    m = re.match(r"^(.*?)[\s]*[\[\(]([\s\S]*?)[\]\)]\s*$", s)
    if m:
        name = _clean_label(m.group(1))
        unit = _normalize_unit_text(m.group(2))
        return name if name else "unlabeled", unit
    # no bracketed unit found
    return s, None

def _dimension_of(unit: Optional[str]) -> Optional[str]:
    if not unit: return None
    return _UNIT_DIMENSIONS.get(unit)

def _convert_value(value: float, unit_in: Optional[str], dim: Optional[str], unknown_bucket: Dict[str, int]) -> Tuple[float, Optional[str], Optional[str]]:
    """
    Convert 'value' from unit_in to the canonical unit selected for 'dim'.
    Returns (value_converted, unit_out, unit_in_normalized).
    unknown_bucket accumulates unknown unit tokens for later emission.
    """
    if dim is None or unit_in is None:
        return value, unit_in, unit_in

    unit_in_norm = unit_in
    target = _CANONICAL_TARGET.get(dim)
    if target is None or unit_in_norm == target:
        return value, unit_in_norm, unit_in

    # ---- time
    if dim == "time":
        if unit_in_norm == "s" and target == "h":
            return value / 3600.0, target, unit_in
        if unit_in_norm == "min" and target == "h":
            return value / 60.0, target, unit_in
        if unit_in_norm == "h" and target == "h":
            return value, target, unit_in
        if unit_in_norm == "h" and target == "min":
            return value * 60.0, target, unit_in
        if unit_in_norm == "min" and target == "s":
            return value * 60.0, target, unit_in
        if unit_in_norm == "s" and target == "min":
            return value / 60.0, target, unit_in

    # ---- conductivity
    if dim == "conductivity":
        # targets mS/cm
        if unit_in_norm == "S/cm" and target == "mS/cm":
            return value * 1000.0, target, unit_in
        if unit_in_norm == "S/m" and target == "mS/cm":
            # 1 S/m = 0.01 S/cm = 10 mS/cm
            return value * 10.0, target, unit_in
        if unit_in_norm == "mS/m" and target == "mS/cm":
            # 1 mS/m = 0.01 mS/cm
            return value * 0.01, target, unit_in
        if unit_in_norm == "mS/cm" and target == "mS/cm":
            return value, target, unit_in

    # ---- current density
    if dim == "current_density":
        # target mA/cm^2
        if unit_in_norm == "A/cm^2" and target == "mA/cm^2":
            return value * 1000.0, target, unit_in
        if unit_in_norm == "mA/cm^2" and target == "mA/cm^2":
            return value, target, unit_in


    # If we got here, the unit is unknown or not implemented
    key = f"{dim}:{unit_in_norm}->{_CANONICAL_TARGET.get(dim)}"
    unknown_bucket[key] = unknown_bucket.get(key, 0) + 1
    return value, unit_in_norm, unit_in

def _infer_axis_roles_from_titles_dict(titles_dict: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given {path: text, ...}, try to pick x_title and y_title
    using filename hints ('x' vs 'y' in basename). Returns (x_text, y_text).
    """
    x_text = None
    y_text = None
    for p, t in titles_dict.items():
        b = os.path.basename(p).lower()
        if "x_" in b or "x-" in b or b.startswith("x") or "_x." in b:
            x_text = t
        elif "y_" in b or "y-" in b or b.startswith("y") or "_y." in b:
            y_text = t
    # fallback: if exactly two titles, arbitrarily assign in sorted order (stable but warn)
    if (x_text is None or y_text is None) and len(titles_dict) == 2:
        items = list(titles_dict.items())
        # deterministic assignment by path sort
        items.sort(key=lambda kv: kv[0])
        if x_text is None: x_text = items[0][1]
        if y_text is None: y_text = items[1][1]
    return x_text, y_text

def _validate_or_flag_unit(unit_raw: Optional[str], axis_name: Optional[str], axis_prefix: str, unknown_units: Dict[str, int]):
    """
    Try to normalize + dimension the unit. If we can't, flag it as unknown and return (None, None).
    Returns (unit_norm, dimension).
    """
    if not unit_raw:
        return None, None

    unit_norm = _normalize_unit_text(unit_raw)
    dim = _dimension_of(unit_norm)

    if unit_norm is None or dim is None:
        key = f"{axis_prefix}:unrecognized:{(axis_name or '').strip()}:{(unit_raw or '').strip()}"
        unknown_units[key] = unknown_units.get(key, 0) + 1
        return None, None

    return unit_norm, dim

# -----------------------------
# Main function
# -----------------------------

def export_labeled_dataset_and_plot(run_id: str, img: str):
    base_output_dir = os.environ.get("PLEXTRACT_BASE_OUTPUT")
    img_dir = Path(base_output_dir) / img
    unknown_units = {}

    # Inputs
    data_json = img_dir / "converted_datapoints" / "data.json"
    mapping_json = img_dir / "chartdete" / "legend_series_mapping.json"
    axis_titles_json = img_dir / "chartdete" / "axis_title_texts.json"

    if not data_json.exists():
        print(f"[finalize] Missing {data_json}, skipping {img}")
        return
    if not mapping_json.exists():
        print(f"[finalize] Missing {mapping_json}, skipping {img}")
        return

    with open(data_json, "r") as f:
        series_xy = json.load(f)
    with open(mapping_json, "r") as f:
        mapping = json.load(f)

    # --- Axis titles & units (optional but recommended)
    x_name, x_unit_raw, y_name, y_unit_raw = None, None, None, None
    if axis_titles_json.exists():
        with open(axis_titles_json, "r") as f:
            titles_dict = json.load(f)  # {path: text}
        x_text, y_text = _infer_axis_roles_from_titles_dict(titles_dict)

        if x_text:
            xn, xu = _parse_label_and_unit(x_text)
            x_name = _clean_label(xn)
            x_unit_raw = xu
            x_unit_norm, x_dim = _validate_or_flag_unit(x_unit_raw, x_name, "x", unknown_units)
        if y_text:
            yn, yu = _parse_label_and_unit(y_text)
            y_name = _clean_label(yn)
            y_unit_raw = yu
            y_unit_norm, y_dim = _validate_or_flag_unit(y_unit_raw, y_name, "y", unknown_units)

    # Build index->label map (legend to series)
    idx_to_label = {}
    for m in mapping:
        idx = m.get("series_index")
        lbl = _clean_label(m.get("legend_text", ""))
        if idx is None:
            mp = m.get("series_mask_path") or ""
            mm = re.search(r"(\d+)", os.path.basename(mp))
            if mm: idx = int(mm.group(1))
        if idx is not None:
            idx_to_label[idx] = lbl

    # Determine unit "dimensions" from raw units
    x_dim = _dimension_of(x_unit_raw) if x_unit_raw else None
    y_dim = _dimension_of(y_unit_raw) if y_unit_raw else None

    # Canonical unit targets for the axes (if determinable)
    x_unit_target = _CANONICAL_TARGET.get(x_dim) if x_dim else None
    y_unit_target = _CANONICAL_TARGET.get(y_dim) if y_dim else None

    # Flatten rows (with conversion)
    rows = []
    for key, pts in series_xy.items():
        mm = re.match(r"series_(\d+)$", key)
        if not mm:
            continue
        idx = int(mm.group(1))
        label = idx_to_label.get(idx, f"series_{idx}")
        for k, p in enumerate(pts):
            x_raw = float(p["x"])
            y_raw = float(p["y"])

            # Convert if units are known
            # Convert only if we have a dimension + normalized unit
            x_conv, x_unit_out, _ = _convert_value(x_raw, x_unit_norm, x_dim, unknown_units) if x_dim else (x_raw, None, x_unit_raw)
            y_conv, y_unit_out, _ = _convert_value(y_raw, y_unit_norm, y_dim, unknown_units) if y_dim else (y_raw, None, y_unit_raw)

            # x_conv, x_unit_out, x_unit_in = _convert_value(x_raw, x_unit_raw, x_dim, unknown_units)
            # y_conv, y_unit_out, y_unit_in = _convert_value(y_raw, y_unit_raw, y_dim, unknown_units)

            rows.append({
                "run_id": run_id,
                "image": img,
                "series_index": idx,
                "series_label": label,
                "point_index": k,

                # raw values from coordinate correction (no unit applied)
                "x_raw": x_raw,
                "y_raw": y_raw,

                "x_name": x_name or "x",
                "x_unit_raw": x_unit_raw,
                "x_unit": x_unit_out,     # None if unknown
                "x_raw": x_raw,
                "x_value": x_conv,        # stays raw if unknown

                # canonicalized units and converted values
                "y_name": y_name or "y",
                "y_unit_raw": y_unit_raw,
                "y_unit": y_unit_out,     # None if unknown
                "y_raw": y_raw,
                "y_value": y_conv,        # stays raw if unknown
                })

    # Output dirs
    out_dir = img_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = out_dir / "final_dataset.csv"
    save_path = out_dir / "final_image.png"

    # Write CSV (stable field order)
    fieldnames = [
        "run_id","image","series_index","series_label","point_index",
        "x_name","x_unit_raw","x_unit","x_raw","x_value",
        "y_name","y_unit_raw","y_unit","y_raw","y_value",
    ]
    if rows:
        with open(per_image_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
        print(f"[finalize] Wrote {per_image_csv}")
    else:
        print(f"[finalize] No rows to write for {img}")
        return

    # Emit unknown units (if any) to help you extend converters
    if unknown_units:
        with open(out_dir / "unknown_units.json", "w") as f:
            json.dump(unknown_units, f, indent=2)
        print(f"[finalize] Unknown units encountered. See {out_dir/'unknown_units.json'}")

    # Plot with your existing helper (update if it expects different columns)
    # You might adapt plot_final_dataset to use x_value/y_value and x_name/y_name/x_unit
    plot_final_dataset(str(per_image_csv), str(save_path))