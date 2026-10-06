"""CLI that lets an agent perform SALSA's human-in-the-loop curation steps.

Run with the SALSA conda env from ``src/``::

    python -m agent_curation <command> --run <run_dir> ...

Every command prints JSON (or a short summary plus file paths) so agents can
chain them. Images written for visual inspection go to
``<run_dir>/agent_curation/<safe_doi>/views/``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The UI's plextract subprocess calls bare "python"; make sure it is this env's.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

from ui.intervention import (  # noqa: E402
    apply_overrides_to_plextract,
    intervention_dir,
    load_mapping,
    load_series_catalog,
    override_lineformer_dir,
    save_mapping,
)
from ui.run_plextract_manual import run_plextract_on_crops  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".gif")
HUMAN_ONLY_DIRS = ["interventions", "plextract/interventions", "aggregated"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _out(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def safe(doi: str) -> str:
    return doi.replace("/", "@")


def plain(doi: str) -> str:
    return doi.replace("@", "/")


def agent_dir(run: Path, doi: str) -> Path:
    d = run / "agent_curation" / safe(doi)
    (d / "views").mkdir(parents=True, exist_ok=True)
    return d


def ledger(run: Path, doi: str, event: Dict[str, Any]) -> None:
    event = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "doi": plain(doi), **event}
    p = run / "agent_curation" / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _font(size: int = 12):
    for cand in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def _numeric(text: str) -> Optional[float]:
    t = str(text).strip().replace("−", "-").replace("–", "-").replace("O", "0").replace("o", "0")
    t = t.replace(",", ".")
    m = re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", t, re.IGNORECASE)
    if m:
        return float(t)
    m = re.fullmatch(r"10\^?([-+]?\d+)", t)
    if m:
        return 10.0 ** int(m.group(1))
    return None


class Target:
    """Resolves all paths for one figure or subfigure image of a paper."""

    def __init__(self, run: Path, doi: str, fig_id: str):
        self.run = run
        self.doi = safe(doi)
        self.fig_id = Path(fig_id).stem if Path(fig_id).suffix.lower() in IMG_EXTS else fig_id
        self.parent_fig = self.fig_id.split("_subfig_")[0]
        self.is_subfig = "_subfig_" in self.fig_id
        self.iv_dir = intervention_dir(run, self.doi, self.fig_id)
        self.lf_override = override_lineformer_dir(run, self.doi, self.fig_id)
        self.cd_override = self.iv_dir / "new_chartdete"

        # Image: an agent crop takes precedence over the automatic subfigure split.
        self.agent_crop = None
        crop_dir = intervention_dir(run, self.doi, self.parent_fig)
        for ext in IMG_EXTS:
            p = crop_dir / f"{self.fig_id}{ext}"
            if self.is_subfig and p.exists():
                self.agent_crop = p
        self.image = self.agent_crop or self._auto_input_image()

        # Base (pre-intervention) plextract output for this image.
        self.base_out = None
        if self.agent_crop is not None:
            self.base_out = self._first_full_run_output()
        else:
            cand = run / "plextract" / "results" / "0" / "output" / self.doi / self.image.name if self.image else None
            if cand is not None and cand.exists():
                self.base_out = cand

    def _auto_input_image(self) -> Optional[Path]:
        d = self.run / "plextract" / "results" / "0" / "input" / self.doi
        for ext in IMG_EXTS:
            if (d / f"{self.fig_id}{ext}").exists():
                return d / f"{self.fig_id}{ext}"
        d2 = self.run / "figures_tables" / "figures" / self.doi
        for ext in IMG_EXTS:
            if (d2 / f"{self.fig_id}{ext}").exists():
                return d2 / f"{self.fig_id}{ext}"
        return None

    def _results(self) -> List[Path]:
        root = self.iv_dir / "plextract_interventions" / "results"
        if not root.exists():
            return []
        runs = sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        out = []
        for r in runs:
            o = r / "output" / self.doi / self.image.name
            if o.exists():
                out.append(o)
        return out

    def _first_full_run_output(self) -> Optional[Path]:
        for o in self._results():
            if (o / "chartdete" / "bounding_boxes.json").exists() and (o / "lineformer").exists():
                return o
        return None

    def latest_out(self) -> Optional[Path]:
        for o in reversed(self._results()):
            if (o / "final" / "final_dataset.csv").exists():
                return o
        return self.base_out


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def _tick_labels(out_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    cd = out_dir / "chartdete"
    res: Dict[str, List[Dict[str, Any]]] = {"x": [], "y": []}
    try:
        coords = json.loads((cd / "label_coordinates.json").read_text())
        texts = json.loads((cd / "axis_label_texts.json").read_text())
    except (OSError, json.JSONDecodeError):
        return res
    for key, box in coords.items():
        name = Path(key).name
        axis = "x" if name.startswith("cropped_xlabels") else "y" if name.startswith("cropped_ylabels") else None
        if axis is None:
            continue
        txt = next((v for k, v in texts.items() if Path(k).name == name), None)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        res[axis].append({"label": name, "ocr": txt, "value": _numeric(txt) if txt is not None else None,
                          "center_px": [round(cx, 1), round(cy, 1)], "box": [round(b, 1) for b in box[:4]]})
    return res


def _plot_area(out_dir: Optional[Path]) -> Optional[List[float]]:
    if out_dir is None:
        return None
    try:
        pa = json.loads((out_dir / "chartdete" / "label_coordinates.json").read_text()).get("plot_area")
        return [float(v) for v in pa[:4]] if pa else None
    except (OSError, json.JSONDecodeError):
        return None


def _fit(pairs: List[Tuple[float, float]], typ: str) -> Optional[Dict[str, Any]]:
    pts = [(p, v) for p, v in pairs if v is not None and (typ == "linear" or v > 0)]
    if len(pts) < 2 or len({p for p, _ in pts}) < 2:
        return None
    u = np.array([p for p, _ in pts], dtype=float)
    v = np.array([vv if typ == "linear" else math.log10(vv) for _, vv in pts], dtype=float)
    a, b = np.polyfit(u, v, 1)
    resid = v - (a * u + b)
    return {"type": typ, "a": float(a), "b": float(b), "max_abs_resid": float(np.max(np.abs(resid))),
            "n_points": len(pts)}


def current_calibration(t: Target) -> Dict[str, Any]:
    """Calibration that the next rerun will use: agent override, else OCR tick fit."""
    if (t.cd_override / "pix2val_override.json").exists():
        d = json.loads((t.cd_override / "pix2val_override.json").read_text())
        return {"source": "override", "x": d["x"], "y": d["y"]}
    out = t.latest_out()
    ticks = _tick_labels(out) if out else {"x": [], "y": []}
    types = {"x": "linear", "y": "linear"}
    for d in (t.cd_override, out / "chartdete" if out else None):
        if d and (d / "axis_type_override.json").exists():
            at = json.loads((d / "axis_type_override.json").read_text())
            types = {"x": at.get("x_axis_type", "linear"), "y": at.get("y_axis_type", "linear")}
            break
    fx = _fit([(tk["center_px"][0], tk["value"]) for tk in ticks["x"]], types["x"])
    fy = _fit([(tk["center_px"][1], tk["value"]) for tk in ticks["y"]], types["y"])
    return {"source": "ocr_tick_fit (approximation of SALSA's automatic calibration)", "x": fx, "y": fy}


def _p2v(px: float, m: Dict[str, Any]) -> float:
    t = m["a"] * px + m["b"]
    return 10 ** t if m.get("type") == "log10" else t


def _v2p(val: float, m: Dict[str, Any]) -> Optional[float]:
    if m.get("type") == "log10":
        if val <= 0:
            return None
        val = math.log10(val)
    if m["a"] == 0:
        return None
    return (val - m["b"]) / m["a"]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (128, 128, 0), (0, 0, 128), (170, 110, 40)]


def ruler(img: Image.Image, scale: int, origin: Tuple[int, int] = (0, 0), step: Optional[int] = None) -> Image.Image:
    """Upscale ``img`` and add pixel rulers (in ORIGINAL image coordinates) on all sides."""
    w, h = img.size
    big = img.convert("RGB").resize((w * scale, h * scale), Image.NEAREST)
    pad = 34
    canvas = Image.new("RGB", (big.width + 2 * pad, big.height + 2 * pad), "white")
    canvas.paste(big, (pad, pad))
    dr = ImageDraw.Draw(canvas)
    f = _font(10)
    if step is None:
        step = max(5, int(round(50 / scale / 5.0)) * 5) if scale > 1 else 50
        step = {1: 50, 2: 25, 3: 20, 4: 10}.get(scale, step) if scale <= 4 else 5
    ox, oy = origin
    for x in range((-(ox % step)) % step, w + 1, step):
        X = pad + x * scale
        dr.line([(X, pad - 6), (X, pad)], fill="black")
        dr.line([(X, pad + big.height), (X, pad + big.height + 6)], fill="black")
        dr.line([(X, pad), (X, pad + big.height)], fill=(255, 0, 255, 60) if False else (230, 230, 255), width=1) if False else None
        dr.text((X - 8, 4), str(x + ox), fill="black", font=f)
        dr.text((X - 8, pad + big.height + 8), str(x + ox), fill="black", font=f)
    for y in range((-(oy % step)) % step, h + 1, step):
        Y = pad + y * scale
        dr.line([(pad - 6, Y), (pad, Y)], fill="black")
        dr.line([(pad + big.width, Y), (pad + big.width + 6, Y)], fill="black")
        dr.text((2, Y - 6), str(y + oy), fill="black", font=f)
        dr.text((pad + big.width + 8, Y - 6), str(y + oy), fill="black", font=f)
    return canvas


def _grid_overlay(img: Image.Image) -> Image.Image:
    """Faint pixel grid every 10 px (bold every 50) on top of the figure."""
    im = img.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    for x in range(0, im.width, 10):
        dr.line([(x, 0), (x, im.height)], fill=(0, 120, 255, 90 if x % 50 == 0 else 28))
    for y in range(0, im.height, 10):
        dr.line([(0, y), (im.width, y)], fill=(0, 120, 255, 90 if y % 50 == 0 else 28))
    return Image.alpha_composite(im, layer).convert("RGB")


def _nice_ticks(lo: float, hi: float, n: int = 6) -> List[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi == lo:
        return []
    span = abs(hi - lo)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.ceil(min(lo, hi) / step) * step
    out, v = [], start
    while v <= max(lo, hi) + 1e-9 and len(out) < 40:
        out.append(round(v, 10))
        v += step
    return out


def render_check(t: Target, out_dir: Optional[Path], dest: Path, scale: int = 2) -> Dict[str, Any]:
    """Overlay: calibration gridlines (data-space ticks re-projected to pixels) + extracted points."""
    img = Image.open(t.image).convert("RGB")
    W, H = img.size
    over = img.copy()
    dr = ImageDraw.Draw(over, "RGBA")
    f = _font(11)
    cal = current_calibration(t)
    pa = _plot_area(out_dir) or [0, 0, W, H]
    info: Dict[str, Any] = {"calibration": cal, "plot_area": pa}
    if cal.get("x") and cal.get("y"):
        x0, x1 = _p2v(pa[0], cal["x"]), _p2v(pa[2], cal["x"])
        y0, y1 = _p2v(pa[3], cal["y"]), _p2v(pa[1], cal["y"])
        for v in _nice_ticks(x0, x1):
            px = _v2p(v, cal["x"])
            if px is not None and 0 <= px < W:
                dr.line([(px, 0), (px, H)], fill=(0, 160, 0, 110))
                dr.text((px + 2, H - 14), f"{v:g}", fill=(0, 120, 0, 255), font=f)
        for v in _nice_ticks(y0, y1):
            py = _v2p(v, cal["y"])
            if py is not None and 0 <= py < H:
                dr.line([(0, py), (W, py)], fill=(0, 160, 0, 110))
                dr.text((2, py - 13), f"{v:g}", fill=(0, 120, 0, 255), font=f)
    dr.rectangle(pa, outline=(255, 140, 0, 200))

    series_summary = []
    final_csv = out_dir / "final" / "final_dataset.csv" if out_dir else None
    if final_csv and final_csv.exists() and cal.get("x") and cal.get("y"):
        rows = list(csv.DictReader(final_csv.open()))
        by: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            by.setdefault(f"{r['series_index']}|{r['series_label']}", []).append(r)
        for k, (key, rs) in enumerate(sorted(by.items())):
            idx, label = key.split("|", 1)
            col = PALETTE[int(idx) % len(PALETTE)] if idx.isdigit() else PALETTE[k % len(PALETTE)]
            xs, ys = [], []
            for r in rs:
                try:
                    xv, yv = float(r["x_value"]), float(r["y_value"])
                except (TypeError, ValueError):
                    continue
                xs.append(xv)
                ys.append(yv)
                px, py = _v2p(xv, cal["x"]), _v2p(yv, cal["y"])
                if px is None or py is None:
                    continue
                if len(rs) > 40:
                    dr.point((px, py), fill=col + (200,))
                else:
                    dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col + (230,), width=2)
            if xs:
                lx, ly = _v2p(xs[-1], cal["x"]), _v2p(ys[-1], cal["y"])
                if lx is not None and ly is not None:
                    dr.text((lx + 6, ly - 6), f"s{idx}:{label}", fill=col + (255,), font=f)
            series_summary.append({"series_index": idx, "label": label, "n_points": len(xs),
                                   "x_range": [min(xs), max(xs)] if xs else None,
                                   "y_range": [min(ys), max(ys)] if ys else None,
                                   "overlay_color_rgb": col})
    info["extracted_series"] = series_summary
    ruler(over, scale).save(dest)
    info["check_image"] = str(dest)
    info["note"] = ("Green lines = data values re-projected through the calibration in effect; they must sit "
                    "exactly on the figure's own tick marks. Circles = extracted points drawn at their DATA "
                    "values; they must sit on the markers/line of the matching series.")
    return info


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_prepare(a) -> None:
    src, dst = Path(a.src).resolve(), Path(a.dst).resolve()
    if dst.exists():
        sys.exit(f"{dst} already exists")
    ignore_top = {d.split("/")[0] for d in HUMAN_ONLY_DIRS if "/" not in d}

    def _ignore(dirpath, names):
        rel = Path(dirpath).resolve().relative_to(src).as_posix()
        out = set()
        for n in names:
            relp = f"{rel}/{n}" if rel != "." else n
            if relp in HUMAN_ONLY_DIRS or (rel == "." and n in ignore_top):
                out.add(n)
            if rel == "complete_dataset/row_cache" and n.startswith("emit_rows_figure_"):
                out.add(n)  # figure rows only exist after human intervention
            if rel == "complete_dataset" and n.endswith("_dataset.csv"):
                out.add(n)
        return out

    shutil.copytree(src, dst, ignore=_ignore, copy_function=shutil.copy2)
    man = dst / "run_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        m["run_id"] = dst.name
        m["cloned_from"] = src.name
        man.write_text(json.dumps(m, indent=2))
    (dst / "agent_curation").mkdir(exist_ok=True)
    if a.dois:
        (dst / "agent_curation" / "queue.json").write_text(json.dumps([plain(d) for d in a.dois], indent=2))
    _out({"run": str(dst), "excluded": HUMAN_ONLY_DIRS + ["row_cache/emit_rows_figure_*", "*_dataset.csv"]})


def _caption(meta: Path) -> str:
    try:
        return json.loads(meta.read_text()).get("caption", "")
    except (OSError, json.JSONDecodeError):
        return ""


def _final_summary(out: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not out or not (out / "final" / "final_dataset.csv").exists():
        return None
    rows = list(csv.DictReader((out / "final" / "final_dataset.csv").open()))
    ser: Dict[str, int] = {}
    for r in rows:
        ser[f"s{r['series_index']}:{r['series_label']}"] = ser.get(f"s{r['series_index']}:{r['series_label']}", 0) + 1
    r0 = rows[0] if rows else {}
    return {"x": f"{r0.get('x_name')} [{r0.get('x_unit_raw')}]", "y": f"{r0.get('y_name')} [{r0.get('y_unit_raw')}]",
            "series_points": ser}


def cmd_paper(a) -> None:
    run, doi = Path(a.run), safe(a.doi)
    ad = agent_dir(run, doi)
    lines: List[str] = []
    comp = run / "complete_dataset" / f"{doi}_complete.json"
    ft = None
    if comp.exists():
        ft = json.loads(comp.read_text()).get("full_text")
    if ft is None:
        pj = run / "reparsed" / doi / "parsed.json"
        if pj.exists():
            ft = json.loads(pj.read_text())
    title = (ft or {}).get("Title", "?")
    lines += [f"# {title}", f"DOI: {plain(doi)}", f"Journal: {(ft or {}).get('Journal', '?')}", ""]
    if ft:
        lines += ["## Abstract", str(ft.get("Abstract", "")), ""]
        for s in ft.get("Sections", []):
            if str(s.get("name", "")).lower().startswith("reference"):
                continue
            lines.append(f"## {s.get('name')}")

            def _flat(c):
                if isinstance(c, str):
                    return [c]
                if isinstance(c, dict):
                    out = [f"### {c.get('name')}"] if c.get("name") else []
                    return out + _flat(c.get("content", []))
                if isinstance(c, list):
                    return [x for i in c for x in _flat(i)]
                return [str(c)]
            lines += _flat(s.get("content", []))
            lines.append("")
    (ad / "fulltext.md").write_text("\n".join(lines), encoding="utf-8")

    figs = []
    fdir = run / "figures_tables" / "figures" / doi
    for img in sorted(fdir.glob("*")) if fdir.exists() else []:
        if img.suffix.lower() not in IMG_EXTS:
            continue
        cands = []
        for sub in sorted((run / "plextract" / "results" / "0" / "input" / doi).glob(f"{img.stem}*")):
            if sub.suffix.lower() in IMG_EXTS and (sub.stem == img.stem or sub.stem.startswith(img.stem + "_subfig_")):
                t = Target(run, doi, sub.stem)
                cands.append({"id": sub.stem, "image": str(t.image), "auto_extraction": _final_summary(t.base_out)})
        figs.append({"figure": img.stem, "image": str(img), "caption": _caption(fdir / f"{img.stem}_metadata.json"),
                     "candidates": cands})
    tables = []
    tdir = run / "figures_tables" / "tables" / doi
    for tcsv in sorted(tdir.glob("*.csv")) if tdir.exists() else []:
        meta = {}
        try:
            meta = json.loads((tdir / f"{tcsv.stem}_metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
        caches = sorted((run / "complete_dataset" / "row_cache").glob(f"emit_rows_table_{_key(doi)}_{tcsv.stem}_*"))
        tables.append({"table": tcsv.stem, "csv": str(tcsv), "caption": meta.get("caption"), "notes": meta.get("notes"),
                       "auto_rows_file": str(caches[-1]) if caches else None})
    chem = run / "chemical_composition" / doi
    scheme_imgs = sorted(str(p) for p in (chem / "figures").glob("*") if p.suffix.lower() in IMG_EXTS) if (chem / "figures").exists() else []
    ctx = {"doi": plain(doi), "title": title, "fulltext": str(ad / "fulltext.md"), "figures": figs, "tables": tables,
           "chemistry": {"samples_csv": str(chem / "samples.csv") if (chem / "samples.csv").exists() else None,
                         "structure_images": scheme_imgs},
           "agent_dir": str(ad)}
    (ad / "context.json").write_text(json.dumps(ctx, indent=2, ensure_ascii=False))
    _out(ctx)


def _key(doi_safe: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", plain(doi_safe))


def cmd_inspect(a) -> None:
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)
    if t.image is None:
        sys.exit(f"image not found for {a.fig}")
    ad = agent_dir(run, a.doi)
    img = Image.open(t.image).convert("RGB")
    grid_path = ad / "views" / f"{t.fig_id}_grid.png"
    ruler(_grid_overlay(img), a.scale).save(grid_path)
    out = t.latest_out()
    info: Dict[str, Any] = {"figure_id": t.fig_id, "image": str(t.image), "size_px": list(img.size),
                            "is_agent_crop": t.agent_crop is not None, "grid_image": str(grid_path),
                            "extraction_dir": str(out) if out else None}
    if out:
        info["tick_labels_ocr"] = _tick_labels(out)
        cd = out / "chartdete"
        for name in ("axis_title_texts.json", "legend_entries.json"):
            p = cd / name
            if p.exists():
                d = json.loads(p.read_text())
                if isinstance(d, dict):
                    d = {Path(k).name: v for k, v in d.items()}
                elif isinstance(d, list):
                    d = [{k: v for k, v in e.items() if not str(k).endswith("_path")} for e in d]
                info[name.replace(".json", "")] = d
        cat = load_series_catalog(out / "lineformer") if (out / "lineformer").exists() else []
        arr = np.array(img)
        series = []
        for s in cat:
            entry: Dict[str, Any] = {"series_index": s.series_index, "centroid_px": [round(c) for c in s.centroid]}
            try:
                m = np.array(Image.open(s.mask_path).convert("L").resize(img.size)) > 127
                if m.any():
                    ys, xs = np.where(m)
                    entry["x_px_range"] = [int(xs.min()), int(xs.max())]
                    entry["median_rgb_under_mask"] = [int(v) for v in np.median(arr[m], axis=0)]
            except OSError:
                pass
            series.append(entry)
        info["lineformer_series"] = series
        check = render_check(t, out, ad / "views" / f"{t.fig_id}_check.png", a.scale)
        info.update(check)
    mp = load_mapping(run, t.doi, t.fig_id)
    if mp:
        info["current_mapping"] = mp
    _out(info)


def cmd_zoom(a) -> None:
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)
    img = Image.open(t.image).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in a.box.split(",")]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)
    crop = _grid_overlay(img).crop((x1, y1, x2, y2)) if a.grid else img.crop((x1, y1, x2, y2))
    dest = agent_dir(run, a.doi) / "views" / f"{t.fig_id}_zoom_{x1}_{y1}_{x2}_{y2}.png"
    ruler(crop, a.scale, origin=(x1, y1)).save(dest)
    _out({"zoom_image": str(dest), "note": "ruler numbers are ORIGINAL image pixel coordinates"})


def cmd_crop(a) -> None:
    run = Path(a.run)
    doi = safe(a.doi)
    src = run / "figures_tables" / "figures" / doi
    img_path = next((src / f"{a.fig}{e}" for e in IMG_EXTS if (src / f"{a.fig}{e}").exists()), None)
    if img_path is None:
        sys.exit(f"figure {a.fig} not found in {src}")
    im = Image.open(img_path).convert("RGB")
    d = intervention_dir(run, doi, a.fig)
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    for k, box in enumerate(a.box):
        x1, y1, x2, y2 = [int(v) for v in box.split(",")]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        p = d / f"{a.fig}_subfig_{a.start + k}.png"
        im.crop((x1, y1, x2, y2)).save(p)
        saved.append({"figure_id": p.stem, "image": str(p), "box": [x1, y1, x2, y2]})
    ledger(run, doi, {"action": "crop", "figure": a.fig, "crops": saved, "reason": a.reason})
    _out({"crops": saved, "next": "run `rerun` on each new figure_id (runs full extraction on the crop)"})


def cmd_calibrate(a) -> None:
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)

    def parse(items):
        return [(float(s.split("=")[0]), float(s.split("=")[1])) for s in items]
    xs, ys = parse(a.x), parse(a.y)
    fx, fy = _fit(xs, a.xtype), _fit(ys, a.ytype)
    if not fx or not fy:
        sys.exit("need >=2 distinct points per axis (log axes need positive values)")
    pa = _plot_area(t.latest_out())
    pix2val = {
        "x": {"type": a.xtype, "a": fx["a"], "b": fx["b"]},
        "y": {"type": a.ytype, "a": fy["a"], "b": fy["b"]},
        "domain": {"x": [_p2v(pa[0], fx), _p2v(pa[2], fx)] if pa else [min(v for _, v in xs), max(v for _, v in xs)],
                   "y": [_p2v(pa[3], fy), _p2v(pa[1], fy)] if pa else [min(v for _, v in ys), max(v for _, v in ys)]},
        "meta": {"source": "agent_tick_points", "image_used": str(t.image),
                 "x_points": xs, "y_points": ys, "plot_area": pa,
                 "fit_max_abs_resid": {"x": fx["max_abs_resid"], "y": fy["max_abs_resid"]}},
    }
    t.cd_override.mkdir(parents=True, exist_ok=True)
    (t.cd_override / "pix2val_override.json").write_text(json.dumps(pix2val, indent=2))
    (t.cd_override / "axis_type_override.json").write_text(
        json.dumps({"x_axis_type": a.xtype, "y_axis_type": a.ytype}, indent=2))
    ledger(run, t.doi, {"action": "calibrate", "figure": t.fig_id, "x": xs, "y": ys, "reason": a.reason})
    dest = agent_dir(run, a.doi) / "views" / f"{t.fig_id}_check.png"
    _out(render_check(t, t.latest_out(), dest))


def cmd_titles(a) -> None:
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)
    t.cd_override.mkdir(parents=True, exist_ok=True)
    (t.cd_override / "axis_title_override.json").write_text(
        json.dumps({"x_title": a.x or "none", "y_title": a.y or "none"}, indent=2))
    ledger(run, t.doi, {"action": "axis_titles", "figure": t.fig_id, "x": a.x, "y": a.y})
    _out({"written": str(t.cd_override / "axis_title_override.json")})


def _hex_or_rgb(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore
    return tuple(int(v) for v in s.split(","))  # type: ignore


def cmd_markers(a) -> None:
    """Detect marker centroids of one colour inside the plot area."""
    import cv2
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)
    img = np.array(Image.open(t.image).convert("RGB")).astype(int)
    H, W = img.shape[:2]
    if a.at:
        x, y = [int(v) for v in a.at.split(",")]
        rgb = tuple(int(v) for v in img[y, x])
    else:
        rgb = _hex_or_rgb(a.rgb)
    dist = np.sqrt(((img - np.array(rgb)) ** 2).sum(axis=2))
    mask = (dist <= a.tol).astype(np.uint8)
    region = [int(v) for v in a.region.split(",")] if a.region else [int(round(v)) for v in (_plot_area(t.latest_out()) or [0, 0, W, H])]
    keep = np.zeros_like(mask)
    keep[max(0, region[1]):region[3], max(0, region[0]):region[2]] = 1
    mask &= keep
    for ex in a.exclude or []:
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in ex.split(",")]
        except ValueError:
            sys.exit(f"--exclude expects numeric x1,y1,x2,y2 pixels (got {ex!r}); take the legend box from `inspect`")
        mask[y1:y2, x1:x2] = 0
    if a.fill_holes:
        # JPEG artefacts break thin rings; close small gaps before filling.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        filled = mask.copy() * 255
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cnts, -1, 255, thickness=-1)
        mask = (filled > 0).astype(np.uint8)
    if a.open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.open + 1, 2 * a.open + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if a.min_area <= area <= a.max_area:
            found.append({"x": round(float(cents[i][0]), 1), "y": round(float(cents[i][1]), 1), "area": area})
    found.sort(key=lambda p: p["x"])
    ad = agent_dir(run, a.doi)
    vis = Image.open(t.image).convert("RGB")
    over = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    ys, xs = np.where(mask > 0)
    for xx, yy in zip(xs[::1], ys[::1]):
        od.point((int(xx), int(yy)), fill=(255, 255, 0, 110))
    f = _font(10)
    for k, p in enumerate(found):
        od.ellipse([p["x"] - 4, p["y"] - 4, p["x"] + 4, p["y"] + 4], outline=(255, 0, 0, 255), width=2)
        od.text((p["x"] + 4, p["y"] - 14), str(k), fill=(255, 0, 0, 255), font=f)
    od.rectangle(region, outline=(255, 140, 0, 200))
    vis = Image.alpha_composite(vis.convert("RGBA"), over).convert("RGB")
    dest = ad / "views" / f"{t.fig_id}_markers_{'%02x%02x%02x' % tuple(rgb)}.png"
    ruler(vis, a.scale).save(dest)
    (ad / "last_markers.json").write_text(json.dumps({"fig": t.fig_id, "points": found}))
    _out({"rgb": list(rgb), "tol": a.tol, "region": region, "n_found": len(found), "points": found,
          "preview": str(dest),
          "hint": "Check the preview. Merge/drop/add points as needed, then pass them to `series --points`."})


def _points_mask(size: Tuple[int, int], pts: List[Tuple[float, float]], r: int = 3) -> Image.Image:
    m = Image.new("L", size, 0)
    dr = ImageDraw.Draw(m)
    for x, y in pts:
        dr.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return m


def cmd_series(a) -> None:
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)
    mapping = load_mapping(run, t.doi, t.fig_id) or {}
    mapping.setdefault("series_names", {})
    mapping.setdefault("deleted", [])
    t.lf_override.mkdir(parents=True, exist_ok=True)
    changed: Dict[str, Any] = {}
    for spec in a.name or []:
        idx, name = spec.split("=", 1)
        mapping["series_names"][str(int(idx))] = name
        changed.setdefault("named", {})[idx] = name
    for idx in a.delete or []:
        if int(idx) not in mapping["deleted"]:
            mapping["deleted"].append(int(idx))
        mapping["series_names"].pop(str(int(idx)), None)
        changed.setdefault("deleted", []).append(int(idx))
    for idx in a.undelete or []:
        if int(idx) in mapping["deleted"]:
            mapping["deleted"].remove(int(idx))
    if a.points is not None:
        if a.index is None:
            sys.exit("--points requires --index")
        if a.points == "last-markers":
            pts = [(p["x"], p["y"]) for p in json.loads((agent_dir(run, a.doi) / "last_markers.json").read_text())["points"]]
        else:
            pts = [tuple(float(v) for v in p.split(",")) for p in a.points.replace(";", " ").split()]
        size = Image.open(t.image).size
        (t.lf_override / f"clicked_points_{a.index}.json").write_text(
            json.dumps([{"x": x, "y": y} for x, y in pts], indent=2))
        _points_mask(size, pts).save(t.lf_override / f"mask_series_{a.index}.png")
        if int(a.index) in mapping["deleted"]:
            mapping["deleted"].remove(int(a.index))
        changed["points"] = {"index": a.index, "n": len(pts)}
    if a.clear_points is not None:
        for fn in (f"clicked_points_{a.clear_points}.json", f"mask_series_{a.clear_points}.png"):
            p = t.lf_override / fn
            if p.exists():
                p.unlink()
        changed["cleared_points"] = a.clear_points
    save_mapping(run, t.doi, t.fig_id, mapping)
    # The UI keeps a copy of the names next to the masks.
    (t.lf_override / "intervention_overrides.json").write_text(
        json.dumps({k: v for k, v in mapping.items() if v}, indent=2))
    ledger(run, t.doi, {"action": "series", "figure": t.fig_id, **changed, "reason": a.reason})
    _out({"mapping": mapping, "override_dir": str(t.lf_override)})


def cmd_rerun(a) -> None:
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)
    if t.image is None:
        sys.exit("image not found")
    log = agent_dir(run, a.doi) / f"rerun_{t.fig_id}.log"
    with log.open("a") as lf:
        sys_stdout = os.dup(1), os.dup(2)
        os.dup2(lf.fileno(), 1)
        os.dup2(lf.fileno(), 2)
        try:
            if t.base_out is None:
                # brand-new crop: full pipeline (ChartDete + OCR + LineFormer) on the crop
                run_plextract_on_crops(scripts_root=SRC, run_dir=run, safe_doi=t.doi, image_paths=[t.image],
                                       figure_id=t.fig_id, intervention=False)
                t = Target(run, a.doi, a.fig)
            if t.lf_override.exists() or t.cd_override.exists() or load_mapping(run, t.doi, t.fig_id):
                apply_overrides_to_plextract(t.base_out, run, t.doi, t.fig_id)
                run_plextract_on_crops(
                    scripts_root=SRC, run_dir=run, safe_doi=t.doi, image_paths=[t.image],
                    override_lineformer_dirs={t.image.name: t.lf_override},
                    override_chartdete_dirs={t.image.name: t.base_out / "chartdete"},
                    figure_id=t.fig_id, intervention=True)
        finally:
            os.dup2(sys_stdout[0], 1)
            os.dup2(sys_stdout[1], 2)
    out = t.latest_out()
    ledger(run, t.doi, {"action": "rerun", "figure": t.fig_id, "output": str(out)})
    info = render_check(t, out, agent_dir(run, a.doi) / "views" / f"{t.fig_id}_check.png", a.scale)
    info["final_dataset_csv"] = str(out / "final" / "final_dataset.csv") if out else None
    info["log"] = str(log)
    _out(info)


def cmd_draft_rows(a) -> None:
    """Turn the latest extraction into draft rows in the run's output schema."""
    run = Path(a.run)
    t = Target(run, a.doi, a.fig)
    out = t.latest_out()
    if not out or not (out / "final" / "final_dataset.csv").exists():
        sys.exit("no extraction output; run `rerun` first")
    ci = json.loads((run / "custom_instructions.json").read_text()) if (run / "custom_instructions.json").exists() else {}
    cols = ci.get("output_columns") or ["Sample Name", "time(h)", "relative humidity(%)", "solvent type",
                                        "solvent molarity (M)", "degradation test temp (C)", "prop test temp (C)",
                                        "property", "value", "DOI", "source_kind", "source_name"]
    defaults = ci.get("missing_value_defaults", {})
    consts = dict(c.split("=", 1) for c in (a.const or []))
    props = dict(p.split("=", 1) for p in (a.property or []))
    rows = []
    for r in csv.DictReader((out / "final" / "final_dataset.csv").open()):
        try:
            xv, yv = float(r["x_value"]), float(r["y_value"])
        except (TypeError, ValueError):
            continue
        if a.x_transform == "inverse_temp_1000_K":
            xv = 1000.0 / xv - 273.15
        elif a.x_transform == "days_to_h":
            xv *= 24
        if a.x_snap:
            xv = round(xv / a.x_snap) * a.x_snap
        if a.x_round is not None:
            xv = round(xv, a.x_round)
        row = {c: defaults.get(c, "") for c in cols}
        row.update({"Sample Name": r["series_label"], "DOI": plain(t.doi), "source_kind": "figure",
                    "source_name": t.fig_id, "property": props.get(r["series_index"], a.default_property or r["y_name"]),
                    "value": f"{yv:.{a.sig}g}"})
        if a.x_col:
            row[a.x_col] = f"{xv:g}"
        row.update(consts)
        rows.append(row)
    dest = agent_dir(run, a.doi) / f"rows_{t.fig_id}.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    _out({"draft": str(dest), "n_rows": len(rows),
          "next": "Review/edit the CSV (samples, property names, conditions), then `emit` it."})


def _row_cache_files(run: Path, doi: str, kind: str, source: str) -> List[Path]:
    rc = run / "complete_dataset" / "row_cache"
    stem = Path(source).stem if kind == "figure" else source
    out = []
    for p in rc.glob(f"emit_rows_{kind}_{_key(safe(doi))}_*.json"):
        m = re.match(rf"emit_rows_{kind}_{re.escape(_key(safe(doi)))}_(.+?)_[0-9a-f]{{40}}__", p.name)
        name = m.group(1) if m else None
        if name is not None and Path(name).stem == stem:
            out.append(p)
    return out


def _retire(run: Path, files: List[Path]) -> List[str]:
    dest = run / "complete_dataset" / "row_cache_superseded"
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in files:
        shutil.move(str(p), str(dest / p.name))
        moved.append(p.name)
    return moved


def cmd_emit(a) -> None:
    run = Path(a.run)
    doi = safe(a.doi)
    ci = json.loads((run / "custom_instructions.json").read_text()) if (run / "custom_instructions.json").exists() else {}
    cols = ci.get("output_columns")
    src = Path(a.rows)
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig"))) if src.suffix == ".csv" else json.loads(src.read_text())
    problems = []
    for i, r in enumerate(rows):
        r["DOI"] = plain(doi)
        r["source_kind"] = a.kind
        r["source_name"] = a.source
        if cols:
            missing = [c for c in cols if c not in r]
            extra = [c for c in r if c not in cols]
            if missing or extra:
                problems.append(f"row {i}: missing={missing} extra={extra}")
        try:
            float(str(r.get("value", "")).split("±")[0].strip())
        except ValueError:
            problems.append(f"row {i}: non-numeric value {r.get('value')!r}")
    if problems and not a.force:
        _out({"error": "validation failed (use --force to override)", "problems": problems[:30]})
        sys.exit(1)
    retired = _retire(run, _row_cache_files(run, doi, a.kind, a.source))
    key = f"emit_rows::{a.kind}::{plain(doi)}::{a.source}::agent"
    h40 = hashlib.sha1(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    h20 = hashlib.sha256(key.encode()).hexdigest()[:20]
    dest = run / "complete_dataset" / "row_cache" / f"emit_rows_{a.kind}_{_key(doi)}_{a.source}_{h40}__{h20}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"rows": rows, "emitted_by": "agent"}, indent=2, ensure_ascii=False))
    ledger(run, doi, {"action": "emit", "kind": a.kind, "source": a.source, "n_rows": len(rows),
                      "retired": retired, "reason": a.reason})
    _out({"written": str(dest), "n_rows": len(rows), "retired_previous": retired, "warnings": problems})


def cmd_reject(a) -> None:
    run = Path(a.run)
    doi = safe(a.doi)
    retired = []
    if a.kind:
        retired = _retire(run, _row_cache_files(run, doi, a.kind, a.source))
    ledger(run, doi, {"action": "reject", "kind": a.kind or "figure", "source": a.source, "reason": a.reason,
                      "retired": retired})
    _out({"rejected": a.source, "retired_rows_files": retired})


def cmd_verdict(a) -> None:
    run = Path(a.run)
    doi = safe(a.doi)
    retired = []
    if a.decision == "exclude":
        for p in (run / "complete_dataset" / "row_cache").glob(f"emit_rows_*_{_key(doi)}_*.json"):
            retired += _retire(run, [p])
    v = {"doi": plain(doi), "decision": a.decision, "reason": a.reason}
    (agent_dir(run, doi) / "verdict.json").write_text(json.dumps(v, indent=2))
    ledger(run, doi, {"action": "paper_verdict", **v, "retired": retired})
    _out(v)


def cmd_chem(a) -> None:
    """Run RDKit checks (in a separate env that has RDKit) and render structures."""
    py = os.environ.get("SALSA_RDKIT_PYTHON") or shutil.which("python3")
    args = [py, str(Path(__file__).with_name("chem_check.py")), "--run", a.run, "--doi", a.doi]
    if a.csv:
        args += ["--csv", a.csv]
    r = subprocess.run(args, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)


def cmd_chem_apply(a) -> None:
    run = Path(a.run)
    doi = safe(a.doi)
    d = run / "chemical_composition" / doi
    d.mkdir(parents=True, exist_ok=True)
    cur = d / "samples.csv"
    backup = d / "samples.salsa.csv"
    if cur.exists() and not backup.exists():
        shutil.copy2(cur, backup)
    shutil.copy2(a.csv, cur)
    ledger(run, doi, {"action": "chem_apply", "csv": str(a.csv), "reason": a.reason})
    _out({"written": str(cur), "salsa_original": str(backup) if backup.exists() else None})


def cmd_aggregate(a) -> None:
    from run_aggregation.aggregate import aggregate_run
    res = aggregate_run(Path(a.run))
    _out({k: (str(v) if isinstance(v, Path) else v) for k, v in getattr(res, "__dict__", {"result": res}).items()})


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_curation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, help_, fig=True, doi=True):
        s = sp.add_parser(name, help=help_)
        s.add_argument("--run", required=True)
        if doi:
            s.add_argument("--doi", required=True)
        if fig:
            s.add_argument("--fig", required=True, help="figure id, e.g. Figure_4 or Figure_4_subfig_1")
        s.set_defaults(fn=fn)
        return s

    s = sp.add_parser("prepare", help="clone a run without human interventions")
    s.add_argument("--src", required=True)
    s.add_argument("--dst", required=True)
    s.add_argument("--dois", nargs="*")
    s.set_defaults(fn=cmd_prepare)

    p("paper", cmd_paper, "write paper context (full text, figures, tables, chemistry)", fig=False)
    s = p("inspect", cmd_inspect, "grid image + OCR/legend/series/calibration check for a figure")
    s.add_argument("--scale", type=int, default=2)
    s = p("zoom", cmd_zoom, "enlarged region with pixel rulers")
    s.add_argument("--box", required=True, help="x1,y1,x2,y2 in image pixels")
    s.add_argument("--scale", type=int, default=4)
    s.add_argument("--grid", action="store_true")
    s = p("crop", cmd_crop, "save subfigure crops of a full figure (like the UI's crop tool)")
    s.add_argument("--box", action="append", required=True, help="x1,y1,x2,y2 (repeatable)")
    s.add_argument("--start", type=int, default=0, help="first subfig index")
    s.add_argument("--reason", default="")
    s = p("calibrate", cmd_calibrate, "write a pix2val axis-calibration override")
    s.add_argument("--x", action="append", required=True, help="PIXEL_X=VALUE (>=2)")
    s.add_argument("--y", action="append", required=True, help="PIXEL_Y=VALUE (>=2)")
    s.add_argument("--xtype", default="linear", choices=["linear", "log10"])
    s.add_argument("--ytype", default="linear", choices=["linear", "log10"])
    s.add_argument("--reason", default="")
    s = p("titles", cmd_titles, "override axis titles")
    s.add_argument("--x")
    s.add_argument("--y")
    s = p("markers", cmd_markers, "detect marker centroids of one colour")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--rgb", help="R,G,B or #rrggbb")
    g.add_argument("--at", help="X,Y pixel to sample the colour from")
    s.add_argument("--tol", type=float, default=60)
    s.add_argument("--region", help="x1,y1,x2,y2 (default: detected plot area)")
    s.add_argument("--exclude", action="append", help="x1,y1,x2,y2 to ignore, e.g. legend (repeatable)")
    s.add_argument("--open", type=int, default=0, help="morphological opening radius (removes thin lines)")
    s.add_argument("--fill-holes", action="store_true", help="fill hollow markers before opening")
    s.add_argument("--min-area", type=int, default=8)
    s.add_argument("--max-area", type=int, default=600)
    s.add_argument("--scale", type=int, default=2)
    s = p("series", cmd_series, "name/delete series, or set exact points for a series")
    s.add_argument("--name", action="append", help="INDEX=Sample name (repeatable)")
    s.add_argument("--delete", action="append", help="series index to delete (repeatable)")
    s.add_argument("--undelete", action="append")
    s.add_argument("--index", help="series index for --points (new index creates a new series)")
    s.add_argument("--points", help="'x,y x,y ...' in image pixels, or 'last-markers'")
    s.add_argument("--clear-points", help="remove agent points/mask for this series index")
    s.add_argument("--reason", default="")
    s = p("rerun", cmd_rerun, "apply overrides and re-run extraction (like 'Rerun OCR data extraction')")
    s.add_argument("--scale", type=int, default=2)
    s = p("draft-rows", cmd_draft_rows, "draft output-schema rows from the latest extraction")
    s.add_argument("--x-col", help="output column that receives the x value, e.g. 'time(h)' or 'prop test temp (C)'")
    s.add_argument("--x-transform", choices=["inverse_temp_1000_K", "days_to_h"])
    s.add_argument("--x-round", type=int, help="decimal places for x (not a step size; see --x-snap)")
    s.add_argument("--x-snap", type=float, help="snap x to multiples of this step, e.g. 24 for daily sampling")
    s.add_argument("--default-property")
    s.add_argument("--property", action="append", help="SERIES_INDEX=property (repeatable)")
    s.add_argument("--const", action="append", help="COLUMN=VALUE applied to all rows (repeatable)")
    s.add_argument("--sig", type=int, default=4, help="significant digits for value")
    s = p("emit", cmd_emit, "write final rows for one source (replaces earlier rows for it)", fig=False)
    s.add_argument("--kind", required=True, choices=["figure", "table"])
    s.add_argument("--source", required=True, help="e.g. Figure_4_subfig_0 or Table_2")
    s.add_argument("--rows", required=True, help="CSV or JSON rows file")
    s.add_argument("--force", action="store_true")
    s.add_argument("--reason", default="")
    s = p("reject", cmd_reject, "record that a figure/table is not used (removes its rows)", fig=False)
    s.add_argument("--source", required=True)
    s.add_argument("--kind", choices=["figure", "table"])
    s.add_argument("--reason", required=True)
    s = p("verdict", cmd_verdict, "paper-level include/exclude decision", fig=False)
    s.add_argument("--decision", required=True, choices=["include", "exclude"])
    s.add_argument("--reason", required=True)
    s = p("chem", cmd_chem, "RDKit-check the samples (or a candidate CSV) and render structures", fig=False)
    s.add_argument("--csv", help="candidate samples CSV (default: chemical_composition/<doi>/samples.csv)")
    s = p("chem-apply", cmd_chem_apply, "replace samples.csv with the verified version", fig=False)
    s.add_argument("--csv", required=True)
    s.add_argument("--reason", default="")
    p("aggregate", cmd_aggregate, "run run_aggregation on the run", fig=False, doi=False)

    a = ap.parse_args(argv)
    if getattr(a, "run", None):
        a.run = str(Path(a.run).resolve())
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
