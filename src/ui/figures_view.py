"""The Figures view: per-figure review and the intervention workflow.

Extracted from `streamlit_app.py`, which had grown a ~1,400-line inline block for
this view. Figures and subfigures are rendered by one code path (`FigureTarget` +
`render_figure_panel`) instead of the two near-duplicate blocks that existed
before, and the interventions are laid out as tabs so only one drawable canvas is
mounted at a time.

The view needs a handful of callables that still live in `streamlit_app.py`
(LLM step configuration, emit-row lookup, the plextract runner). Those arrive via
`FiguresViewDeps` rather than an import, so this module stays importable on its
own and `streamlit_app` remains the only entry point.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from canvas_geometry import fabric_circle_center
from cropping import save_crops
from image_render import is_valid_image_path as _is_valid_image_path
from image_render import open_image_safe as _open_image_safe
from intervention import (
    apply_overrides_to_plextract,
    get_lineformer_dir,
    latest_plextract_run_dir,
    list_override_masks,
    load_mapping,
    load_series_catalog,
    override_lineformer_dir,
    save_mapping,
    save_mask_override,
    write_series_catalog_override,
)
from ui_common import (
    bump_canvas_version_on_width_change,
    canvas_version,
    load_and_resize_rgba,
    load_csv as _load_csv,
    load_emit_rows,
    section_dropdown,
    st_canvas,
    st_image_auto,
    st_resizable_image,
)

# ---------------------------------------------------------------------------
# Shared section renderers (used by both figures and subfigures)
# ---------------------------------------------------------------------------

def _section(title: str, *, expanded: bool, wrap: bool):
    """An expander, or a transparent container when the caller supplies its own."""
    if wrap:
        return section_dropdown(title, expanded=expanded)
    return nullcontext()


def render_ocr_dropdown(*, csv_path: Optional[Path], key_prefix: str, axis_override_dir: Optional[Path] = None, title: str = "OCR-Extracted Data"):
    """Render OCR-extracted data using the shared dropdown structure."""
    with section_dropdown(title, expanded=False):
        if not (csv_path and csv_path.exists()):
            st.info("No extracted dataset found.")
            return
        try:
            df = _load_csv(str(csv_path))
            if axis_override_dir is not None:
                df = _apply_axis_title_override_df(df, axis_override_dir)
            st.data_editor(df, use_container_width=True, hide_index=True, key=f"{key_prefix}_ocr")
            st.caption(str(csv_path))
        except Exception as e:
            st.warning(f"Failed to load extracted dataset:\n`{csv_path}`\n\n{e}")

def render_llm_dropdown(*, run_dir: Path, emit_rows_path: Optional[Path], key_prefix: str, file_name_prefix: str, title: str = "LLM-Generated Dataset"):
    """Render LLM-generated dataset using the shared dropdown structure."""
    with section_dropdown(title, expanded=False):
        if not emit_rows_path:
            st.info("No LLM-generated rows found.")
            return
        df_emit = load_emit_rows(emit_rows_path)
        if df_emit is None or df_emit.empty:
            st.info("LLM-generated row cache exists but contains no rows.")
            return
        st.data_editor(df_emit, use_container_width=True, hide_index=True, key=f"{key_prefix}_llm")
        st.download_button(
            label="Download emit_rows CSV",
            data=df_emit.to_csv(index=False),
            file_name=f"{file_name_prefix}_emit_rows.csv",
            mime="text/csv",
            key=f"{key_prefix}_llm_download",
        )

def render_metadata_dropdown(*, metadata_path: Optional[Path], empty_message: str = "No metadata found."):
    """Render metadata using the shared dropdown structure."""
    with section_dropdown("Metadata", expanded=False):
        if metadata_path and metadata_path.exists():
            st.json(json.loads(metadata_path.read_text()))
        else:
            st.info(empty_message)

def render_figure_image_dropdown(
    *,
    image: Optional[Path],
    image_caption: Optional[str] = None,
    figure_caption: Optional[str] = None,
    title: str,
    key: str,
    expanded: bool = False,
):
    """Render the figure image inside a dropdown, shared between figures and subfigures."""
    with section_dropdown(title, expanded=expanded):
        if figure_caption:
            st.markdown(f"**Caption:** {figure_caption}")
        if image and _is_valid_image_path(image):
            st_resizable_image(image, caption=image_caption, key=key)
        else:
            st.warning("No source image found for this figure.")

def render_axis_configuration_dropdown(
    *,
    chartdete_dir: Optional[Path],
    source_img: Optional[Path],
    key_prefix: str,
    expanded: bool = True,
    override_dir: Optional[Path] = None,
    wrap: bool = True,
):
    """Render the full Axis Configuration section, shared between figures and subfigures.

    chartdete_dir: directory containing the read-only plextract source files
                   (axis_title_texts.json, pix2val.json, label_coordinates.json).
    override_dir:  directory where user overrides are read from and written to
                   (axis_title_override.json, pix2val_override.json, axis_type_override.json).
                   Defaults to chartdete_dir when not supplied.
    wrap:          wrap the section in an expander. Pass False when it already
                   sits inside a tab, so the content is not double-nested.
    """
    # Determine where overrides are stored; separate from the read-only source dir so
    # we never write back into the original plextract output.
    _override_base: Optional[Path] = override_dir if override_dir is not None else chartdete_dir

    with _section("Axis Configuration", expanded=expanded, wrap=wrap):
        if chartdete_dir is None:
            if _override_base is None:
                st.info("No plextract output found for this figure. Run plextract extraction first.")
                return
            chartdete_dir = _override_base
            st.info("No plextract axis output found yet. You can still calibrate from clicks below.")
        chartdete_dir.mkdir(parents=True, exist_ok=True)
        if _override_base is not None:
            _override_base.mkdir(parents=True, exist_ok=True)

        def _render_axis_titles() -> None:
            st.markdown("**Axis Titles**")
            title_override_path = _override_base / "axis_title_override.json"
            title_ocr_path = chartdete_dir / "axis_title_texts.json"

            def _load_axis_titles_default() -> tuple[str, str]:
                if title_override_path.exists():
                    try:
                        d = json.loads(title_override_path.read_text())
                        return d.get("x_title", "") or "", d.get("y_title", "") or ""
                    except Exception:
                        return "", ""
                if title_ocr_path.exists():
                    try:
                        d = json.loads(title_ocr_path.read_text())
                        x_title, y_title = "", ""
                        for ap, txt in d.items():
                            bn = Path(ap).name.lower()
                            if any(t in bn for t in ("x_", "x-", "_x.", "xtitle", "xlabel")):
                                x_title = txt
                            elif any(t in bn for t in ("y_", "y-", "_y.", "ytitle", "ylabel")):
                                y_title = txt
                        if not x_title and not y_title and len(d) == 2:
                            vals = list(d.values())
                            x_title, y_title = vals[0], vals[1]
                        return x_title or "", y_title or ""
                    except Exception:
                        return "", ""
                return "", ""

            x_title_default, y_title_default = _load_axis_titles_default()
            x_title_key = f"{key_prefix}_ax_x_title"
            y_title_key = f"{key_prefix}_ax_y_title"
            if x_title_key not in st.session_state:
                st.session_state[x_title_key] = x_title_default
            if y_title_key not in st.session_state:
                st.session_state[y_title_key] = y_title_default

            col_xt, col_yt = st.columns(2)
            with col_xt:
                new_x_title = st.text_input("X axis title", key=x_title_key)
            with col_yt:
                new_y_title = st.text_input("Y axis title", key=y_title_key)

            if st.button("Save axis titles", key=f"{key_prefix}_ax_save_titles"):
                title_override_path.write_text(json.dumps({"x_title": new_x_title, "y_title": new_y_title}, indent=2))
                st.success("Axis title overrides saved.")
                st.rerun()

        # ========================
        # Axis Calibration
        # ========================
        st.markdown("**Axis Calibration**")
        if not _is_valid_image_path(source_img):
            st.info("No source image available for axis calibration.")
            _render_axis_titles()
            return

        pix2val_override_path = _override_base / "pix2val_override.json"
        pix2val_orig_path = chartdete_dir / "pix2val.json"
        axis_type_path = _override_base / "axis_type_override.json"
        label_coords_path = chartdete_dir / "label_coordinates.json"

        # Plot area bbox (optional but helpful)
        plot_area = None
        if label_coords_path.exists():
            try:
                _lc = json.loads(label_coords_path.read_text())
                if "plot_area" in _lc and len(_lc["plot_area"]) >= 4:
                    plot_area = list(map(float, _lc["plot_area"][:4]))
            except Exception:
                pass

        if plot_area is None:
            st.info(
                "No plot_area found (`chartdete/label_coordinates.json`). "
                "You can still calibrate from clicks; domain will use your entered min/max."
            )

        pil_img = _open_image_safe(source_img, mode="RGB")
        if pil_img is None:
            st.warning(f"Could not open source image for axis calibration: {source_img}")
            _render_axis_titles()
            return
        img_w, img_h = pil_img.size

        width_key = f"{key_prefix}_ax_calib_w"
        if width_key not in st.session_state:
            st.session_state[width_key] = min(1100, img_w)
        _slider_min = min(400, max(1, img_w))
        _slider_max = max(_slider_min, 1800)
        st.session_state[width_key] = max(
            _slider_min,
            min(_slider_max, int(st.session_state[width_key])),
        )
        disp_w = st.slider(
            "Calibration view width (px)", min_value=_slider_min, max_value=_slider_max,
            step=50, key=width_key,
        )
        disp_h = int(round(img_h * (disp_w / img_w)))

        bg = pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS).convert("RGB")
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h

        endpoint_key = f"{key_prefix}_ax_calib_endpoint"
        endpoint = st.radio("Click target", ["Xmin", "Xmax", "Ymin", "Ymax"], key=endpoint_key, horizontal=True)

        cv_ver_key = f"{key_prefix}_ax_calib_ver"
        bump_canvas_version_on_width_change(
            last_width_key=f"{key_prefix}_ax_calib_last_w",
            version_key=cv_ver_key,
            width=disp_w,
        )

        c1_ax, _c2_ax = st.columns([1, 3])
        with c1_ax:
            if st.button("Clear markers", key=f"{key_prefix}_ax_clear_markers"):
                st.session_state[cv_ver_key] += 1
                for k in ("xmin", "xmax", "ymin", "ymax"):
                    st.session_state.pop(f"{key_prefix}_ax_pt_{k}", None)
                st.rerun()

        st.write("**Click on the image to place the selected endpoint.**")

        canvas_key = (
            f"{key_prefix}_ax_calib_canvas_{Path(source_img).name}"
            f"_v{canvas_version(cv_ver_key)}"
        )
        canvas = st_canvas(
            background_image=bg,
            drawing_mode="circle",
            update_streamlit=True,
            height=disp_h, width=disp_w,
            stroke_width=2, stroke_color="#00ff00", fill_color="rgba(0,255,0,0.35)",
            key=canvas_key,
        )

        def _last_circle_center(json_data):
            if not json_data:
                return None
            objs = json_data.get("objects") or []
            circles = [o for o in objs if o.get("type") == "circle"]
            if not circles:
                return None
            return fabric_circle_center(circles[-1])

        click = _last_circle_center(getattr(canvas, "json_data", None))

        def _store_endpoint(ep_name: str, cx_disp: float, cy_disp: float):
            st.session_state[f"{key_prefix}_ax_pt_{ep_name.lower()}"] = {
                "disp": [float(cx_disp), float(cy_disp)],
                "img": [float(cx_disp * scale_x), float(cy_disp * scale_y)],
                "img_wh": [int(img_w), int(img_h)],
                "shown_wh": [int(disp_w), int(disp_h)],
                "source_image": str(source_img),
            }

        if click is not None:
            _store_endpoint(endpoint, click[0], click[1])

        def _get_pt(ep_name: str):
            return st.session_state.get(f"{key_prefix}_ax_pt_{ep_name.lower()}")

        pt_xmin = _get_pt("Xmin")
        pt_xmax = _get_pt("Xmax")
        pt_ymin = _get_pt("Ymin")
        pt_ymax = _get_pt("Ymax")

        show_debug = st.checkbox("Show click debug", value=False, key=f"{key_prefix}_ax_show_debug")
        if show_debug:
            st.write("Latest click (disp):", click)
            st.write("Xmin:", pt_xmin)
            st.write("Xmax:", pt_xmax)
            st.write("Ymin:", pt_ymin)
            st.write("Ymax:", pt_ymax)

        vxmin_key = f"{key_prefix}_ax_val_xmin"
        vxmax_key = f"{key_prefix}_ax_val_xmax"
        vymin_key = f"{key_prefix}_ax_val_ymin"
        vymax_key = f"{key_prefix}_ax_val_ymax"
        for k in (vxmin_key, vxmax_key, vymin_key, vymax_key):
            if k not in st.session_state:
                st.session_state[k] = ""

        vcols = st.columns(4)
        with vcols[0]:
            vxmin = st.text_input("Xmin value", key=vxmin_key, disabled=pt_xmin is None)
            if pt_xmin is None:
                st.caption("Click Xmin on image")
        with vcols[1]:
            vxmax = st.text_input("Xmax value", key=vxmax_key, disabled=pt_xmax is None)
            if pt_xmax is None:
                st.caption("Click Xmax on image")
        with vcols[2]:
            vymin = st.text_input("Ymin value", key=vymin_key, disabled=pt_ymin is None)
            if pt_ymin is None:
                st.caption("Click Ymin on image")
        with vcols[3]:
            vymax = st.text_input("Ymax value", key=vymax_key, disabled=pt_ymax is None)
            if pt_ymax is None:
                st.caption("Click Ymax on image")

        def _parse_float(s):
            s = (s or "").strip()
            return None if not s else float(s)

        nxmin = _parse_float(vxmin)
        nxmax = _parse_float(vxmax)
        nymin = _parse_float(vymin)
        nymax = _parse_float(vymax)

        # Axis type (also saved to axis_type_override.json for consistency)
        ax_type_default = {"x_axis_type": "linear", "y_axis_type": "linear"}
        if axis_type_path.exists():
            try:
                _atd = json.loads(axis_type_path.read_text())
                ax_type_default["x_axis_type"] = _atd.get("x_axis_type", "linear")
                ax_type_default["y_axis_type"] = _atd.get("y_axis_type", "linear")
            except Exception:
                pass
        elif pix2val_override_path.exists():
            try:
                _pv = json.loads(pix2val_override_path.read_text())
                ax_type_default["x_axis_type"] = _pv.get("x", {}).get("type", "linear")
                ax_type_default["y_axis_type"] = _pv.get("y", {}).get("type", "linear")
            except Exception:
                pass

        xtype_key = f"{key_prefix}_ax_xtype"
        ytype_key = f"{key_prefix}_ax_ytype"
        if xtype_key not in st.session_state:
            st.session_state[xtype_key] = ax_type_default["x_axis_type"]
        if ytype_key not in st.session_state:
            st.session_state[ytype_key] = ax_type_default["y_axis_type"]

        tcols = st.columns(2)
        with tcols[0]:
            new_xtype = st.selectbox("X axis type", ["linear", "log10"], key=xtype_key)
        with tcols[1]:
            new_ytype = st.selectbox("Y axis type", ["linear", "log10"], key=ytype_key)

        def _ab_from_two_points(p1_img, v1, p2_img, v2, axis: str, typ: str):
            u1, u2 = (float(p1_img[0]), float(p2_img[0])) if axis == "x" else (float(p1_img[1]), float(p2_img[1]))
            if u2 == u1:
                raise ValueError(f"{axis} endpoints have identical pixel coordinate; cannot calibrate.")
            if typ == "linear":
                a = (v2 - v1) / (u2 - u1)
                return float(a), float(v1 - a * u1)
            if v1 <= 0 or v2 <= 0:
                raise ValueError(f"{axis}-axis log10 requires positive values.")
            lv1, lv2 = math.log10(v1), math.log10(v2)
            a = (lv2 - lv1) / (u2 - u1)
            return float(a), float(lv1 - a * u1)

        ready = (
            pt_xmin and pt_xmax and pt_ymin and pt_ymax and
            None not in (nxmin, nxmax, nymin, nymax)
        )
        bcols = st.columns([1, 1, 2])
        with bcols[0]:
            if st.button(
                "Save calibration override from clicks",
                disabled=not ready,
                key=f"{key_prefix}_ax_save_calib",
            ):
                try:
                    xa, xb = _ab_from_two_points(pt_xmin["img"], nxmin, pt_xmax["img"], nxmax, axis="x", typ=new_xtype)
                    ya, yb = _ab_from_two_points(pt_ymin["img"], nymin, pt_ymax["img"], nymax, axis="y", typ=new_ytype)
                    if plot_area is not None:
                        px1, py1, px2, py2 = plot_area
                        x_min = xa * px1 + xb if new_xtype == "linear" else 10 ** (xa * px1 + xb)
                        x_max = xa * px2 + xb if new_xtype == "linear" else 10 ** (xa * px2 + xb)
                        y_min = ya * py2 + yb if new_ytype == "linear" else 10 ** (ya * py2 + yb)
                        y_max = ya * py1 + yb if new_ytype == "linear" else 10 ** (ya * py1 + yb)
                    else:
                        x_min, x_max = nxmin, nxmax
                        y_min, y_max = nymin, nymax
                    pix2val = {
                        "x": {"type": new_xtype, "a": xa, "b": xb},
                        "y": {"type": new_ytype, "a": ya, "b": yb},
                        "domain": {"x": [x_min, x_max], "y": [y_min, y_max]},
                        "meta": {
                            "source": "ui_click_endpoints",
                            "image_used": str(source_img),
                            "x_clicks": {"min": pt_xmin["img"], "max": pt_xmax["img"]},
                            "y_clicks": {"min": pt_ymin["img"], "max": pt_ymax["img"]},
                            "x_values": {"min": nxmin, "max": nxmax},
                            "y_values": {"min": nymin, "max": nymax},
                            "plot_area": plot_area,
                        },
                    }
                    pix2val_override_path.write_text(json.dumps(pix2val, indent=2))
                    axis_type_path.write_text(json.dumps({"x_axis_type": new_xtype, "y_axis_type": new_ytype}, indent=2))
                    st.success(f"Saved calibration override: {pix2val_override_path}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save click calibration: {e}")

        with bcols[1]:
            if st.button("Clear calibration override", key=f"{key_prefix}_ax_clear_calib"):
                if pix2val_override_path.exists():
                    pix2val_override_path.unlink()
                st.success("Cleared calibration override.")
                st.rerun()

        with bcols[2]:
            show_json = st.checkbox("Show current calibration JSON", value=False, key=f"{key_prefix}_ax_show_json")
            if show_json:
                if pix2val_override_path.exists():
                    st.code(pix2val_override_path.read_text(), language="json")
                elif pix2val_orig_path.exists():
                    st.code(pix2val_orig_path.read_text(), language="json")
                else:
                    st.info("No pix2val JSON exists yet. Save calibration from clicks first.")

        st.divider()
        _render_axis_titles()

def _resolve_series_source_image(
    *,
    run_dir: Path,
    safe_doi: str,
    fig_id: str,
    fig_dir: Optional[Path],
    source_img: Optional[Path],
) -> Optional[Path]:
    """Find a renderable image for Series Management from UI, plextract, or fallback outputs."""
    candidates: list[Path] = []

    def _add(path: Optional[Path]) -> None:
        if path is not None:
            candidates.append(Path(path))

    _add(source_img)

    figures_root = run_dir / "figures_tables" / "figures" / safe_doi
    if figures_root.exists():
        fig_stem = Path(fig_id).stem
        for ext in ("", ".png", ".jpg", ".jpeg", ".webp", ".gif"):
            _add(figures_root / f"{fig_id}{ext}")
            _add(figures_root / f"{fig_stem}{ext}")
        for fp in figures_root.iterdir():
            if fp.is_file() and Path(fp.name).stem == fig_stem:
                _add(fp)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        _add(fig_dir / "final" / "final_image.png")

        if fig_dir.parent.name == safe_doi and fig_dir.parent.parent.name == "output":
            plextract_run_dir = fig_dir.parent.parent.parent
            input_dir = plextract_run_dir / "input" / safe_doi
            if input_dir.exists():
                fig_stem = Path(fig_dir.name).stem
                _add(input_dir / fig_dir.name)
                for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                    _add(input_dir / f"{fig_stem}{ext}")
                for fp in input_dir.iterdir():
                    if fp.is_file() and Path(fp.name).stem == fig_stem:
                        _add(fp)

        if "plextract_interventions" in fig_dir.parts:
            try:
                intervention_crop_base = fig_dir.parents[5]
                for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                    _add(intervention_crop_base.with_suffix(ext))
            except IndexError:
                pass

    for candidate in candidates:
        if _is_valid_image_path(candidate):
            return candidate
    return None

def render_series_management_dropdown(
    *,
    run_dir: Path,
    safe_doi: str,
    fig_id: str,
    fig_dir: Optional[Path],
    lf_dir: Optional[Path],
    lf_override_dir: Path,
    source_img: Optional[Path],
    key_prefix: str,
    expanded: bool = True,
    wrap: bool = True,
):
    """Render the full Series Management section, shared between figures and subfigures.

    `wrap=False` skips the surrounding expander for callers that already provide
    their own container (e.g. a tab).
    """
    with _section("Series Management", expanded=expanded, wrap=wrap):
        source_img = _resolve_series_source_image(
            run_dir=run_dir,
            safe_doi=safe_doi,
            fig_id=fig_id,
            fig_dir=fig_dir,
            source_img=source_img,
        )

        # Ensure we have a usable lineformer dir
        if lf_dir is None or not lf_dir.exists():
            lf_override_dir.mkdir(parents=True, exist_ok=True)
            lf_cat = lf_override_dir / "series_catalog.json"
            if not lf_cat.exists():
                lf_cat.write_text("[]")
            lf_dir = lf_override_dir
            st.info("No plextract lineformer output found for this figure. You can still create new series below.")

        series_items = load_series_catalog(lf_dir)
        mapping = load_mapping(run_dir, safe_doi, fig_id)
        names = mapping.get("series_names", {})
        deleted = set(int(x) for x in mapping.get("deleted", []))
        override_masks = list_override_masks(run_dir, safe_doi, fig_id)

        # Include newly-created series from override masks
        orig_indices = {s.series_index for s in series_items}
        for idx, mask_path in sorted(override_masks.items()):
            if idx not in orig_indices:
                try:
                    mimg = _open_image_safe(Path(mask_path), mode="L")
                    if mimg is None:
                        raise ValueError(f"Could not open series mask: {mask_path}")
                    marr = np.array(mimg)
                    ys, xs = np.where(marr > 0)
                    cx = float(np.mean(xs)) if len(xs) > 0 else 0.0
                    cy = float(np.mean(ys)) if len(ys) > 0 else 0.0
                except Exception:
                    cx, cy = 0.0, 0.0
                from intervention import SeriesItem as _SeriesItem
                series_items.append(_SeriesItem(series_index=idx, mask_path=mask_path, centroid=(cx, cy)))

        SERIES_COLORS = [
            (255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 165, 0),
            (148, 0, 211), (0, 206, 209), (255, 20, 147), (128, 128, 0),
            (0, 255, 127), (100, 149, 237),
        ]

        # Load series labels from plextract final_dataset.csv
        plextract_labels: dict = {}
        final_csv_sm = (fig_dir / "final" / "final_dataset.csv") if fig_dir else None
        if final_csv_sm and final_csv_sm.exists():
            try:
                fdf = pd.read_csv(final_csv_sm)
                if "series_label" in fdf.columns:
                    for sl in fdf["series_label"].dropna().unique():
                        sl_str = str(sl).strip()
                        try:
                            si = int(sl_str)
                            plextract_labels[si] = sl_str
                        except ValueError:
                            si = len(plextract_labels)
                            plextract_labels[si] = sl_str
            except Exception:
                pass

        def _display_name(idx):
            if str(idx) in names and names[str(idx)] != f"Series {idx}":
                return names[str(idx)]
            if idx in plextract_labels:
                return plextract_labels[idx]
            return f"Series {idx}"

        if _is_valid_image_path(source_img):
            base_img = _open_image_safe(source_img, mode="RGBA")
            if base_img is None:
                st.warning(f"Could not open source image for series overlay: {source_img}")
                base_img = None
                w, h = 600, 400
            else:
                w, h = base_img.size
        else:
            st.warning("No source image for series overlay.")
            base_img = None
            w, h = 600, 400

        cv_ver_key_sm = f"{key_prefix}_series_canvas_ver"
        canvas_version(cv_ver_key_sm)

        # --- Display width slider ---
        width_key_sm = f"{key_prefix}_series_view_w"
        if width_key_sm not in st.session_state:
            # default to something reasonable for the figure, capped like before
            st.session_state[width_key_sm] = min(900, w)
        _slider_min_sm = min(400, max(1, w))
        _slider_max_sm = max(_slider_min_sm, 1800)
        st.session_state[width_key_sm] = max(
            _slider_min_sm,
            min(_slider_max_sm, int(st.session_state[width_key_sm])),
        )

        disp_w = st.slider(
            "Series view width (px)",
            min_value=_slider_min_sm,
            max_value=_slider_max_sm,
            step=50,
            key=width_key_sm,
        )

        bump_canvas_version_on_width_change(
            last_width_key=f"{key_prefix}_series_last_w",
            version_key=cv_ver_key_sm,
            width=disp_w,
        )

        # disp_w = min(disp_w, w)  # don't exceed native width
        disp_h = int(round(h * (disp_w / w)))

        # Build overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        legend_lines = []
        for s in sorted(series_items, key=lambda x: x.series_index):
            i = s.series_index
            if i in deleted:
                continue
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            mp = override_masks.get(i, s.mask_path)
            try:
                mask = _open_image_safe(Path(mp), mode="L")
                if mask is None:
                    raise ValueError(f"Could not open overlay mask: {mp}")
                marr2 = np.array(mask)
                cl = np.zeros((*marr2.shape, 4), dtype=np.uint8)
                cl[marr2 > 0] = (*color, 140)
                overlay = Image.alpha_composite(overlay, Image.fromarray(cl))
            except Exception:
                pass
            hex_c = "#{:02x}{:02x}{:02x}".format(*color)
            legend_lines.append(f'<span style="color:{hex_c}; font-weight:bold;">&#9632;</span> {_display_name(i)}')

        if base_img is not None:
            composite = Image.alpha_composite(base_img, overlay).convert("RGB")
            composite_disp = composite.resize((disp_w, disp_h))
        else:
            composite_disp = None

        # Edit or Create Series Mask
        st.write("**Edit or Create Series Mask**")
        active_series = [s for s in series_items if s.series_index not in deleted]
        existing_indices = [s.series_index for s in series_items]
        next_idx = max(existing_indices) + 1 if existing_indices else 0

        mode_options = ["Edit existing series", "Create new series"]
        mode_key = f"{key_prefix}_series_mode"
        if not active_series:
            st.session_state[mode_key] = "Create new series"
        elif st.session_state.get(mode_key) not in mode_options:
            st.session_state[mode_key] = mode_options[0]
        mode = st.radio("Mode", mode_options, key=mode_key, horizontal=True)
        is_creating = (mode == "Create new series")
        cur_mask_img = None

        if is_creating:
            selected_idx = next_idx
            series_name = st.text_input(
                "Name for new series",
                value=f"Series {next_idx}",
                key=f"{key_prefix}_new_series_name",
            )
        else:
            if not active_series:
                st.info("No active series to edit.")
                selected_idx = None
                series_name = ""
            else:
                sel_opts = [f"{_display_name(s.series_index)}" for s in active_series]
                sel_ui = st.selectbox(
                    "Select series to edit",
                    range(len(active_series)),
                    format_func=lambda i: sel_opts[i],
                    key=f"{key_prefix}_mask_edit_sel",
                )
                sel_s = active_series[sel_ui]
                selected_idx = sel_s.series_index
                series_name = _display_name(selected_idx)
                orig_mask_path = next(s.mask_path for s in series_items if s.series_index == selected_idx)
                cur_mask_path = override_masks.get(selected_idx, orig_mask_path)
                if _is_valid_image_path(Path(cur_mask_path)):
                    _mask_img = _open_image_safe(Path(cur_mask_path), mode="L")
                    if _mask_img is not None:
                        cur_mask_img = _mask_img.resize((disp_w, disp_h))
                    else:
                        st.warning(
                            f"Series mask could not be opened for {series_name}: {cur_mask_path}. "
                            "You can still draw a replacement mask and save it."
                        )
                else:
                    st.warning(
                        f"Series mask not found for {series_name}: {cur_mask_path}. "
                        "You can still draw a replacement mask and save it."
                    )

        if selected_idx is not None:
            sel_color = SERIES_COLORS[selected_idx % len(SERIES_COLORS)]
            hex_color = "#{:02x}{:02x}{:02x}".format(*sel_color)
            st.markdown(
                f'**Drawing with color:** <span style="color:{hex_color}; font-size:1.8em; font-weight:bold;">&#9632;</span> {series_name}',
                unsafe_allow_html=True,
            )
            stroke_rgba = f"rgba({sel_color[0]},{sel_color[1]},{sel_color[2]},1.0)"
            scale_x_s = w / disp_w
            scale_y_s = h / disp_h

            input_mode_key = f"{key_prefix}_input_mode_{selected_idx}_{mode}"
            if input_mode_key not in st.session_state:
                st.session_state[input_mode_key] = "Pick exact points"
            input_mode = st.radio(
                "Input mode", ["Freehand drawing", "Pick exact points"],
                horizontal=True,
                key=input_mode_key,
            )
            c1_sm, c2_sm = st.columns([2.2, 1.5])

            if input_mode == "Freehand drawing":
                with c1_sm:
                    st.write("Draw on the colored series image below")
                    canvas_key_freehand = (
                        f"{key_prefix}_mask_canvas_{selected_idx}_{mode}"
                        f"_v{canvas_version(cv_ver_key_sm)}"
                    )

                    mask_canvas = None
                    if composite_disp is None:
                        st.warning("Series image could not be rendered.")
                    else:
                        mask_canvas = st_canvas(
                            background_image=composite_disp,
                            drawing_mode="freedraw",
                            stroke_width=5,
                            stroke_color=stroke_rgba,
                            fill_color=stroke_rgba,
                            update_streamlit=True,
                            height=disp_h,
                            width=disp_w,
                            key=canvas_key_freehand,
                        )
                if legend_lines:
                    st.markdown(" &nbsp; ".join(legend_lines), unsafe_allow_html=True)

                if mask_canvas is not None and mask_canvas.image_data is not None:
                    rgba = mask_canvas.image_data.astype(np.uint8)
                    r, g, b = sel_color
                    color_match = (
                        (np.abs(rgba[..., 0].astype(int) - r) < 30) &
                        (np.abs(rgba[..., 1].astype(int) - g) < 30) &
                        (np.abs(rgba[..., 2].astype(int) - b) < 30) &
                        (rgba[..., 3] > 0)
                    )
                    new_mask_small = color_match.astype(np.uint8) * 255
                    if np.any(new_mask_small > 0):
                        new_mask_full = cv2.resize(new_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
                        new_mask_pil = Image.fromarray(new_mask_full).convert("L")
                        colA, colB = st.columns(2)
                        with colA:
                            save_label = "Create new series" if is_creating else "Save edited mask"
                            if st.button(save_label, type="primary", key=f"{key_prefix}_save_mask_{selected_idx}_{mode}"):
                                outp = save_mask_override(run_dir, safe_doi, fig_id, selected_idx, new_mask_pil)
                                if is_creating:
                                    map_upd = load_mapping(run_dir, safe_doi, fig_id)
                                    if "series_names" not in map_upd:
                                        map_upd["series_names"] = {}
                                    map_upd["series_names"][str(selected_idx)] = series_name
                                    save_mapping(run_dir, safe_doi, fig_id, map_upd)
                                    marr3 = np.array(new_mask_pil)
                                    yc, xc = np.where(marr3 > 0)
                                    cx3 = float(np.mean(xc)) if len(xc) > 0 else 0.0
                                    cy3 = float(np.mean(yc)) if len(yc) > 0 else 0.0
                                    from intervention import SeriesItem as _SI3
                                    series_items.append(_SI3(series_index=selected_idx, mask_path=outp, centroid=(cx3, cy3)))
                                    st.success(f"Created new series '{series_name}'")
                                else:
                                    st.success(f"Saved override mask for {series_name}")
                                om_upd = list_override_masks(run_dir, safe_doi, fig_id)
                                write_series_catalog_override(run_dir, safe_doi, fig_id, series_items, om_upd)
                                st.rerun()
                        with colB:
                            if st.button("Reset canvas", key=f"{key_prefix}_reset_mask_{selected_idx}_{mode}"):
                                st.rerun()

            else:  # Pick exact points
                lf_ov_dir = override_lineformer_dir(run_dir, safe_doi, fig_id)
                pts_file = lf_ov_dir / f"clicked_points_{selected_idx}.json"
                existing_pts: list = []
                if pts_file.exists():
                    try:
                        existing_pts = json.loads(pts_file.read_text())
                    except Exception:
                        pass
                with c1_sm:
                    dot_size = 1
                    if existing_pts:
                        st.caption(f"Currently saved: **{len(existing_pts)} point(s)**. Placing new points replaces them.")
                    else:
                        st.caption("Click on the figure to place data points.")
                    canvas_key_points = (
                        f"{key_prefix}_point_canvas_{selected_idx}_{mode}"
                        f"_v{canvas_version(cv_ver_key_sm)}"
                    )

                    point_canvas = st_canvas(
                        background_image=composite_disp,
                        drawing_mode="point",
                        point_display_radius=dot_size,
                        stroke_color=stroke_rgba,
                        fill_color=stroke_rgba,
                        update_streamlit=True,
                        height=disp_h,
                        width=disp_w,
                        key=canvas_key_points,
                    ) if composite_disp is not None else None
                    if composite_disp is None:
                        st.warning("Series image could not be rendered.")
                if legend_lines:
                    st.markdown(" &nbsp; ".join(legend_lines), unsafe_allow_html=True)

                new_pts: list = []
                if point_canvas is not None and point_canvas.json_data and "objects" in point_canvas.json_data:
                    for obj2 in point_canvas.json_data["objects"]:
                        if obj2.get("type") == "circle":
                            cxd, cyd = fabric_circle_center(obj2)
                            new_pts.append({"x": cxd * scale_x_s, "y": cyd * scale_y_s})
                new_pts.sort(key=lambda p: p["x"])

                if new_pts:
                    st.caption(f"Placed **{len(new_pts)} new point(s)** — click Save to apply.")
                    colA2, colB2 = st.columns(2)
                    with colA2:
                        save_label2 = "Create new series" if is_creating else "Save points"
                        if st.button(save_label2, type="primary", key=f"{key_prefix}_save_pts_{selected_idx}_{mode}"):
                            lf_ov_dir.mkdir(parents=True, exist_ok=True)
                            pts_file.write_text(json.dumps(new_pts, indent=2))
                            dot_r = max(3, int(5 * scale_x_s))
                            mask_arr = np.zeros((h, w), dtype=np.uint8)
                            for pt in new_pts:
                                cv2.circle(mask_arr, (int(round(pt["x"])), int(round(pt["y"]))), dot_r, 255, -1)
                            mask_pil = Image.fromarray(mask_arr).convert("L")
                            outp2 = save_mask_override(run_dir, safe_doi, fig_id, selected_idx, mask_pil)
                            if is_creating:
                                map_upd2 = load_mapping(run_dir, safe_doi, fig_id)
                                if "series_names" not in map_upd2:
                                    map_upd2["series_names"] = {}
                                map_upd2["series_names"][str(selected_idx)] = series_name
                                save_mapping(run_dir, safe_doi, fig_id, map_upd2)
                                from intervention import SeriesItem as _SI4
                                cx4 = float(np.mean([p["x"] for p in new_pts]))
                                cy4 = float(np.mean([p["y"] for p in new_pts]))
                                series_items.append(_SI4(series_index=selected_idx, mask_path=outp2, centroid=(cx4, cy4)))
                                st.success(f"Created new series '{series_name}' with {len(new_pts)} points")
                            else:
                                st.success(f"Saved {len(new_pts)} points for {series_name}")
                            om_upd2 = list_override_masks(run_dir, safe_doi, fig_id)
                            write_series_catalog_override(run_dir, safe_doi, fig_id, series_items, om_upd2)
                            st.rerun()
                    with colB2:
                        if st.button("Reset", key=f"{key_prefix}_reset_pts_{selected_idx}_{mode}"):
                            st.rerun()

            with c2_sm:
                if cur_mask_img is not None:
                    st.write("Current mask")
                    st_image_auto(cur_mask_img, full_width=True)
                else:
                    st.info("Drawing new series")

        # Rename / delete section
        st.write("**Rename or delete** each series below.")
        new_names = {}
        new_deleted = set(deleted)
        for s2 in sorted(series_items, key=lambda x: x.series_index):
            i2 = s2.series_index
            if i2 in deleted:
                continue
            color2 = SERIES_COLORS[i2 % len(SERIES_COLORS)]
            hex2 = "#{:02x}{:02x}{:02x}".format(*color2)
            cols = st.columns([0.5, 3, 1])
            with cols[0]:
                st.markdown(f'<span style="color:{hex2}; font-size:1.4em; font-weight:bold;">&#9632;</span>', unsafe_allow_html=True)
            with cols[1]:
                new_names[str(i2)] = st.text_input(
                    f"Name for series {i2}", value=_display_name(i2),
                    key=f"{key_prefix}_series_name_{i2}",
                    label_visibility="collapsed",
                )
            with cols[2]:
                del_flag = st.checkbox("Delete", value=False, key=f"{key_prefix}_series_del_{i2}")
                if del_flag:
                    new_deleted.add(i2)

        if st.button("Save series mapping", type="primary", key=f"{key_prefix}_save_mapping"):
            map2 = {"series_names": new_names, "deleted": sorted(list(new_deleted))}
            p = save_mapping(run_dir, safe_doi, fig_id, map2)
            st.success(f"Saved mapping: {p}")
            st.rerun()


# ---------------------------------------------------------------------------
# Axis-title overrides
# ---------------------------------------------------------------------------

def _parse_axis_title_for_override(text: str):
    if not text:
        return None, None
    text = text.strip()
    m = re.match(r"^(.*?)[\s]*[\[\(]([\s\S]*?)[\]\)]\s*$", text)
    if m:
        _name = m.group(1).strip() or None
        _unit = m.group(2).strip() or None
        return _name, _unit
    return text or None, None

def _apply_axis_title_override_df(df: pd.DataFrame, figure_dir: Path) -> pd.DataFrame:
    _override_path = figure_dir / "chartdete" / "axis_title_override.json"
    if not _override_path.exists() or df is None or df.empty:
        return df
    try:
        _ov = json.loads(_override_path.read_text())
    except Exception:
        return df

    _x_title = (_ov.get("x_title") or "").strip()
    _y_title = (_ov.get("y_title") or "").strip()
    if not _x_title and not _y_title:
        return df

    _out = df.copy()
    if _x_title:
        _x_name, _x_unit = _parse_axis_title_for_override(_x_title)
        if _x_name is not None and "x_name" in _out.columns:
            _out["x_name"] = _x_name
        if _x_unit is not None and "x_unit" in _out.columns:
            _out["x_unit"] = _x_unit
    if _y_title:
        _y_name, _y_unit = _parse_axis_title_for_override(_y_title)
        if _y_name is not None and "y_name" in _out.columns:
            _out["y_name"] = _y_name
        if _y_unit is not None and "y_unit" in _out.columns:
            _out["y_unit"] = _y_unit
    return _out



# ---------------------------------------------------------------------------
# Subfigure discovery
# ---------------------------------------------------------------------------

def iter_runs_newest_first(workdir: Path) -> list:
    results = workdir / "results"
    if not results.exists():
        return []
    runs = [p for p in results.iterdir() if p.is_dir()]
    # Sort by mtime desc (newest first)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def collect_subfig_dirs_across_runs(workdir: Path, safe_doi: str, fig_stem: str) -> dict[str, Path]:
    subfig_map: dict[str, Path] = {}
    for run in iter_runs_newest_first(workdir):
        out_base = run / "output" / safe_doi
        if not out_base.exists():
            continue
        for d in out_base.iterdir():
            if not d.is_dir():
                continue
            if not d.name.startswith(fig_stem + "_subfig_"):
                continue
            # newest run wins (we're iterating newest->oldest)
            if d.name not in subfig_map:
                subfig_map[d.name] = d
    return subfig_map

def resolve_source_img_for_subfig_dir(subfig_dir: Path, safe_doi: str):
    """
    Given a subfig output directory:
      .../results/<run_id>/output/<safe_doi>/<subfig_name>
    find the corresponding input image from the SAME run:
      .../results/<run_id>/input/<safe_doi>/<subfig_name>(.<ext>)
    """
    run_dir = subfig_dir.parents[2]  # .../results/<run_id>
    input_dir = run_dir / "input" / safe_doi

    def _valid_candidate(path: Path, caption: str):
        if _is_valid_image_path(path):
            return path, caption
        return None

    if input_dir.exists():
        # Try exact name first (in case it includes extension)
        p = input_dir / subfig_dir.name
        found = _valid_candidate(p, "Cropped image")
        if found:
            return found

        # Try matching by stem (handles missing/extra extension)
        stem = Path(subfig_dir.name).stem
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p2 = input_dir / f"{stem}{ext}"
            found = _valid_candidate(p2, "Cropped image (run input)")
            if found:
                return found

        # Last resort: scan for any file with matching stem
        for fp in input_dir.iterdir():
            if fp.is_file() and Path(fp.name).stem == stem:
                found = _valid_candidate(fp, "Cropped image (run input)")
                if found:
                    return found

    # Intervention runs often keep the original crop as a sibling image next to
    # the intervention directory. Prefer that over empty placeholder files in
    # results/<n>/input/.
    if "plextract_interventions" in subfig_dir.parts:
        intervention_crop_base = subfig_dir.parents[5]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            found = _valid_candidate(
                intervention_crop_base.with_suffix(ext),
                "Intervention source crop",
            )
            if found:
                return found

    # Fallback: recreated image
    final_img = subfig_dir / "final" / "final_image.png"
    found = _valid_candidate(final_img, "Recreated image (crop not found)")
    if found:
        return found

    return None, None



# ---------------------------------------------------------------------------
# Figure / subfigure panel
# ---------------------------------------------------------------------------

@dataclass
class FiguresViewDeps:
    """Callables and constants this view needs from `streamlit_app`.

    Injected rather than imported so `figures_view` has no import cycle back to
    the Streamlit entry script (which executes as `__main__`).
    """

    run_dir: Path
    scripts_root: Path
    selected_doi: str
    safe_doi: str
    find_emit_rows_for_source: Callable[..., Optional[Path]]
    step_has_configured_model: Callable[[str], bool]
    missing_openai_api_key_steps: Callable[[Sequence[str]], list]
    complete_dataset_env: Callable[[], dict]
    run_plextract_on_crops: Callable[..., Any]
    llm_step_complete_dataset: str


@dataclass
class FigureTarget:
    """A single figure or subfigure that the panel can render and intervene on.

    Parent figures and subfigures differ only in a few fields, so both flow
    through the same renderer instead of the two near-identical blocks that used
    to exist.
    """

    fig_id: str
    display_name: str
    key_prefix: str
    emit_source_name: str
    fig_stem: str
    source_img: Optional[Path] = None
    image_caption: Optional[str] = None
    figure_caption: Optional[str] = None
    plextract_dir: Optional[Path] = None
    chartdete_dir: Optional[Path] = None
    final_csv: Optional[Path] = None
    crops_dir: Optional[Path] = None
    is_subfigure: bool = False
    # Where axis-title overrides for the OCR table are read from. Parent figures
    # store them beside the plextract final_dataset.csv; subfigures use their own
    # plextract output dir.
    ocr_override_dir: Optional[Path] = None
    ocr_title: str = "OCR-Extracted Data"


def _rect_to_bbox_canvas(obj: dict) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of a Fabric.js rect, in canvas coordinates."""
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    width = float(obj.get("width", 0.0))
    height = float(obj.get("height", 0.0))
    scale_x = float(obj.get("scaleX", 1.0))
    scale_y = float(obj.get("scaleY", 1.0))
    angle = float(obj.get("angle", 0.0))

    scaled_w = width * scale_x
    scaled_h = height * scale_y

    if abs(angle) < 1e-6:
        return left, top, left + scaled_w, top + scaled_h

    cx = left + scaled_w / 2.0
    cy = top + scaled_h / 2.0
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = scaled_w / 2.0, scaled_h / 2.0

    xs, ys = [], []
    for px, py in ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)):
        xs.append(cx + cos_t * px - sin_t * py)
        ys.append(cy + sin_t * px + cos_t * py)
    return min(xs), min(ys), max(xs), max(ys)


def _canvas_rects_to_image_boxes(
    canvas,
    *,
    disp_w: int,
    disp_h: int,
    orig_w: int,
    orig_h: int,
) -> list:
    """Convert drawn rectangles to de-duplicated pixel boxes on the original image."""
    objects = (canvas.json_data or {}).get("objects", []) if canvas is not None else []
    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    boxes: list = []
    for obj in objects:
        if obj.get("type") != "rect":
            continue
        x1c, y1c, x2c, y2c = _rect_to_bbox_canvas(obj)

        # clamp to canvas bounds
        x1c = max(0.0, min(float(disp_w), x1c))
        x2c = max(0.0, min(float(disp_w), x2c))
        y1c = max(0.0, min(float(disp_h), y1c))
        y2c = max(0.0, min(float(disp_h), y2c))
        if x2c <= x1c or y2c <= y1c:
            continue

        # scale back to original image coords, then clamp to original bounds
        x1 = max(0, min(orig_w - 1, int(round(x1c * scale_x))))
        x2 = max(1, min(orig_w, int(round(x2c * scale_x))))
        y1 = max(0, min(orig_h - 1, int(round(y1c * scale_y))))
        y2 = max(1, min(orig_h, int(round(y2c * scale_y))))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return list(dict.fromkeys(boxes))


def _render_crop_tab(target: FigureTarget) -> None:
    """Draw-rectangles-to-crop tool. Only meaningful for parent figures."""
    if target.is_subfigure or target.crops_dir is None:
        st.info("Cropping is available on the parent figure, not on a subfigure.")
        return
    if not _is_valid_image_path(target.source_img):
        st.warning("No source image found for this figure.")
        return

    pil_img = _open_image_safe(target.source_img, mode="RGB")
    if pil_img is None:
        st.warning(f"Could not open source image for cropping: {target.source_img}")
        return
    native_w = pil_img.size[0]

    width_key = f"{target.key_prefix}_crop_view_w"
    if width_key not in st.session_state:
        st.session_state[width_key] = min(1100, native_w)
    slider_min = min(400, max(1, native_w))
    slider_max = max(slider_min, 1800)
    st.session_state[width_key] = max(slider_min, min(slider_max, int(st.session_state[width_key])))

    disp_w_target = st.slider(
        "Crop view width (px)",
        min_value=slider_min,
        max_value=slider_max,
        step=50,
        key=width_key,
    )

    orig_w, orig_h, disp_w, disp_h, bg_img = load_and_resize_rgba(
        str(target.source_img), int(disp_w_target)
    )

    draw_mode = st.radio(
        "Mode",
        ["rect", "transform"],
        format_func=lambda x: "Draw rectangles" if x == "rect" else "Move/resize rectangles",
        key=f"{target.key_prefix}_crop_mode",
        horizontal=True,
    )

    ver_key = f"{target.key_prefix}_crop_canvas_ver"
    if st.button("Clear drawn rectangles", key=f"{target.key_prefix}_crop_clear"):
        st.session_state[ver_key] = canvas_version(ver_key) + 1

    bump_canvas_version_on_width_change(
        last_width_key=f"{target.key_prefix}_crop_last_w",
        version_key=ver_key,
        width=disp_w,
    )

    canvas = st_canvas(
        background_image=bg_img,
        drawing_mode=draw_mode,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        stroke_width=3,
        stroke_color="#ff0000",
        fill_color="rgba(255, 0, 0, 0.20)",
        key=f"{target.key_prefix}_crop_canvas_v{canvas_version(ver_key)}",
    )

    rects = _canvas_rects_to_image_boxes(
        canvas, disp_w=disp_w, disp_h=disp_h, orig_w=orig_w, orig_h=orig_h
    )
    st.write(f"Rectangles detected: **{len(rects)}**")

    if st.button(
        "Save crops",
        disabled=len(rects) == 0,
        type="primary",
        key=f"{target.key_prefix}_save_crops",
    ):
        save_crops(target.source_img, rects, target.crops_dir, prefix=target.fig_stem)
        st.rerun()

    existing_crops = (
        sorted(target.crops_dir.glob(f"{target.fig_stem}_subfig_*.*"))
        if target.crops_dir.exists()
        else []
    )
    if not existing_crops:
        st.info("No crops saved yet. Draw rectangles above to create crops.")
        return

    st.write(f"**Existing crops:** {len(existing_crops)}")
    crop_cols = st.columns(min(len(existing_crops), 4))
    for i, crop_path in enumerate(existing_crops):
        with crop_cols[i % len(crop_cols)]:
            st_image_auto(str(crop_path), full_width=True)
            st.caption(crop_path.name)

    if st.button("Clear all crops", key=f"{target.key_prefix}_clear_crops"):
        for crop_path in existing_crops:
            crop_path.unlink(missing_ok=True)
        st.success("Cleared existing crops.")
        st.rerun()


def _safe_cache_filename(value: str) -> str:
    """Reproduce row_emitter._safe_filename so stale cache entries can be cleared."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)[:180]


def _clear_emit_row_cache(run_dir: Path, doi: str, source_name: str) -> None:
    """Drop cached LLM rows for one figure so a rerun always re-calls the model."""
    cache_dir = run_dir / "complete_dataset" / "row_cache"
    if not cache_dir.exists():
        return
    prefix = _safe_cache_filename(f"emit_rows::figure::{doi}::{source_name}::")
    for cache_file in list(cache_dir.iterdir()):
        if cache_file.suffix == ".json" and (
            cache_file.stem.startswith(prefix) or cache_file.name.startswith(prefix)
        ):
            cache_file.unlink(missing_ok=True)


def _run_llm_dataset_generation(
    *,
    deps: FiguresViewDeps,
    target_figure: str,
    extracted_figures_root: Path,
) -> None:
    """Regenerate completed-dataset rows for one figure via complete_dataset.py."""
    run_dir = deps.run_dir
    index_csv = run_dir / "reparsed" / "index.csv"
    if not index_csv.exists():
        message = f"**Required file not found:** `{index_csv.relative_to(run_dir)}`\n\n"
        if not (run_dir / "reparsed").exists():
            message += "The `reparsed/` directory doesn't exist. "
        else:
            message += "The `reparsed/` directory exists but `index.csv` is missing. "
        message += "This file is created during the 'Download DOIs' stage."
        st.error(message)
        return

    if not deps.step_has_configured_model(deps.llm_step_complete_dataset):
        st.error(
            "Configure an LLM for Complete dataset in the sidebar before rerunning "
            "LLM dataset generation."
        )
        return
    if deps.missing_openai_api_key_steps([deps.llm_step_complete_dataset]):
        st.error("Enter an OpenAI API key in the sidebar before rerunning LLM dataset generation.")
        return

    tmp_path = None
    try:
        _clear_emit_row_cache(run_dir, deps.selected_doi, target_figure)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(f"DOI\n{deps.selected_doi}\n")
            tmp_path = tmp.name

        complete_script = Path(__file__).parent.parent / "complete_dataset" / "complete_dataset.py"
        cmd = [
            "python", str(complete_script),
            "--file_path", tmp_path,
            "--output_dir", str(run_dir / "complete_dataset"),
            "--index_csv_path", str(index_csv),
            "--figures_tables_dir", str(run_dir / "figures_tables"),
            "--extracted_figures_root", str(extracted_figures_root),
            "--target_figure", target_figure,
        ]

        with st.spinner(f"Running LLM dataset generation for {target_figure}..."):
            result = subprocess.run(
                cmd, stdout=None, stderr=None, text=True, timeout=300,
                env=deps.complete_dataset_env(),
            )
        if result.returncode == 0:
            st.success(f"LLM dataset generation complete for {target_figure}!")
            st.rerun()
        else:
            st.error(f"LLM dataset generation failed:\n```\n{result.stderr}\n```")
    except subprocess.TimeoutExpired:
        st.error("LLM dataset generation timed out (>5 minutes)")
    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _crop_lineformer_dir(run_dir: Path, safe_doi: str, crop: Path, fallback_workdirs: list) -> Optional[Path]:
    """Find a LineFormer output dir belonging to one specific crop.

    Never falls back to the parent figure's masks: those are in the parent's
    coordinate space and are wrong for the (smaller) crop image.
    """
    per_crop_wd = run_dir / "interventions" / safe_doi / crop.stem / "plextract_interventions"
    for workdir in [per_crop_wd] + fallback_workdirs:
        if not workdir.exists():
            continue
        latest = latest_plextract_run_dir(workdir)
        if not latest:
            continue
        candidate = latest / "output" / safe_doi / crop.name / "lineformer"
        if candidate.exists():
            return candidate
    return None


def _rerun_ocr_extraction(*, target: FigureTarget, deps: FiguresViewDeps) -> None:
    """Apply mask/axis overrides, then re-run plextract for this target."""
    run_dir = deps.run_dir
    safe_doi = deps.safe_doi
    lf_override_dir = override_lineformer_dir(run_dir, safe_doi, target.fig_id)

    with st.spinner("Applying mask overrides..."):
        # Writes prepared masks + catalog + overrides into the intervention dir
        # only; the original plextract output is never modified.
        apply_overrides_to_plextract(target.plextract_dir, run_dir, safe_doi, target.fig_id)

    crops = (
        sorted(target.crops_dir.glob(f"{target.fig_stem}_subfig_*.png"))
        if (target.crops_dir and target.crops_dir.exists() and not target.is_subfigure)
        else []
    )

    with st.spinner("Re-running data extraction..."):
        if target.is_subfigure:
            if not _is_valid_image_path(target.source_img):
                st.error("Could not locate subfigure source image for rerun.")
                return
            lf_map = (
                {target.source_img.name: lf_override_dir} if lf_override_dir.exists() else {}
            )
            cd_map = (
                {target.source_img.name: target.chartdete_dir}
                if target.chartdete_dir and target.chartdete_dir.exists()
                else {}
            )
            deps.run_plextract_on_crops(
                scripts_root=deps.scripts_root,
                run_dir=run_dir,
                safe_doi=safe_doi,
                image_paths=[target.source_img],
                override_lineformer_dirs=lf_map,
                override_chartdete_dirs=cd_map,
                figure_id=target.fig_id,
                intervention=True,
            )
        elif crops:
            # Each crop is an independent image with its own coordinate space.
            fallback_workdirs = [
                run_dir / "interventions" / safe_doi / target.fig_id / "plextract_interventions",
                run_dir / "interventions" / "interventions_plextract",
                run_dir / "plextract",
            ]
            for crop in crops:
                crop_lf = _crop_lineformer_dir(run_dir, safe_doi, crop, fallback_workdirs)
                if crop_lf:
                    # Prior results exist: intervention pipeline, skipping LineFormer.
                    deps.run_plextract_on_crops(
                        scripts_root=deps.scripts_root,
                        run_dir=run_dir,
                        safe_doi=safe_doi,
                        image_paths=[crop],
                        override_lineformer_dirs={crop.name: crop_lf},
                        figure_id=crop.stem,
                        intervention=True,
                    )
                else:
                    # No crop-specific masks yet: full pipeline so LineFormer
                    # generates them in crop-coordinate space.
                    deps.run_plextract_on_crops(
                        scripts_root=deps.scripts_root,
                        run_dir=run_dir,
                        safe_doi=safe_doi,
                        image_paths=[crop],
                        figure_id=crop.stem,
                        intervention=False,
                    )
        elif _is_valid_image_path(target.source_img):
            lf_map = (
                {target.source_img.name: lf_override_dir} if lf_override_dir.exists() else {}
            )
            deps.run_plextract_on_crops(
                scripts_root=deps.scripts_root,
                run_dir=run_dir,
                safe_doi=safe_doi,
                image_paths=[target.source_img],
                override_lineformer_dirs=lf_map,
                figure_id=target.fig_id,
                intervention=True,
            )
        else:
            st.error("Could not locate a source image for rerun.")
            return

    st.success("Data extraction complete!")
    st.rerun()


def _llm_rerun_options(*, target: FigureTarget, deps: FiguresViewDeps) -> list:
    """Figures/subfigures the LLM rerun can target, as (name, extracted_figures_root).

    When user-created crops exist, their intervention-plextract results are
    preferred over the parent figure's original plextract output.
    """
    run_dir = deps.run_dir
    safe_doi = deps.safe_doi
    options: list = []

    if not target.is_subfigure and target.crops_dir and target.crops_dir.exists():
        for crop in sorted(target.crops_dir.glob(f"{target.fig_stem}_subfig_*.png")):
            crop_wd = run_dir / "interventions" / safe_doi / crop.name / "plextract_interventions"
            if not crop_wd.exists():
                continue
            latest = latest_plextract_run_dir(crop_wd)
            if not latest:
                continue
            if (latest / "output" / safe_doi / crop.name).is_dir():
                options.append((crop.name, latest / "output"))

    if not options and target.plextract_dir and target.plextract_dir.is_dir():
        # `.../output/<safe_doi>/<name>` → extracted_figures_root is `.../output`
        options.append((target.plextract_dir.name, target.plextract_dir.parent.parent))

    return options


def _render_rerun_tab(*, target: FigureTarget, deps: FiguresViewDeps) -> None:
    """Re-run OCR extraction and LLM dataset generation for this target."""
    st.markdown("**Re-run OCR data extraction**")
    crops = (
        sorted(target.crops_dir.glob(f"{target.fig_stem}_subfig_*.png"))
        if (target.crops_dir and target.crops_dir.exists() and not target.is_subfigure)
        else []
    )
    if crops:
        st.caption(f"Will run on **{len(crops)} cropped image(s)**.")
    elif _is_valid_image_path(target.source_img):
        st.caption("Will run on the original figure image.")

    if st.button("Rerun OCR data extraction", type="primary", key=f"{target.key_prefix}_rerun_ocr"):
        try:
            _rerun_ocr_extraction(target=target, deps=deps)
        except Exception as exc:
            st.error(f"Failed: {exc}")

    st.divider()
    st.markdown("**Run LLM dataset generation**")
    options = _llm_rerun_options(target=target, deps=deps)
    if not options:
        st.info("No plextract output found. Run OCR data extraction first.")
        return

    names = [name for name, _ in options]
    if len(names) > 1:
        selected = st.selectbox(
            "Select figure/subfigure for LLM rerun",
            names,
            key=f"{target.key_prefix}_llm_sel",
        )
    else:
        selected = names[0]
    efr = dict(options)[selected]

    if st.button("Run LLM dataset generation", type="primary", key=f"{target.key_prefix}_rerun_llm"):
        _run_llm_dataset_generation(
            deps=deps, target_figure=selected, extracted_figures_root=efr
        )


def _render_intervention_results(*, target: FigureTarget, deps: FiguresViewDeps) -> None:
    """Show OCR/LLM data produced by this target's intervention plextract runs."""
    workdir = (
        deps.run_dir / "interventions" / deps.safe_doi / target.fig_id / "plextract_interventions"
    )
    if not workdir.exists():
        return
    latest = latest_plextract_run_dir(workdir)
    if not latest:
        return
    out_base = latest / "output" / deps.safe_doi
    if not out_base.exists():
        return

    for result_dir in sorted(out_base.iterdir()):
        if not result_dir.is_dir():
            continue
        if not target.is_subfigure and result_dir.name.startswith(target.fig_stem + "_subfig_"):
            # Subfigure results are shown under their own panel.
            continue
        if Path(result_dir.name).stem != Path(target.fig_id).stem:
            continue

        render_ocr_dropdown(
            csv_path=result_dir / "final" / "final_dataset.csv",
            key_prefix=f"{target.key_prefix}_int_ocr_{result_dir.name}",
            axis_override_dir=result_dir,
            title=f"New OCR-Extracted Data: {result_dir.name}",
        )
        render_llm_dropdown(
            run_dir=deps.run_dir,
            emit_rows_path=deps.find_emit_rows_for_source(
                deps.run_dir, deps.selected_doi, "figure", result_dir.name
            ),
            key_prefix=f"{target.key_prefix}_int_llm_{result_dir.name}",
            file_name_prefix=result_dir.name,
            title=f"New LLM-Generated Dataset: {result_dir.name}",
        )


def render_figure_panel(*, target: FigureTarget, deps: FiguresViewDeps, image_expanded: bool = True) -> None:
    """Render one figure or subfigure: image, extracted data, and interventions."""
    render_figure_image_dropdown(
        image=target.source_img,
        image_caption=target.image_caption,
        figure_caption=target.figure_caption,
        title=target.display_name,
        key=f"{target.key_prefix}_img",
        expanded=image_expanded,
    )

    render_ocr_dropdown(
        csv_path=target.final_csv if (target.final_csv and target.final_csv.exists()) else None,
        key_prefix=f"{target.key_prefix}_ocr",
        axis_override_dir=target.ocr_override_dir,
        title=target.ocr_title,
    )
    render_llm_dropdown(
        run_dir=deps.run_dir,
        emit_rows_path=deps.find_emit_rows_for_source(
            deps.run_dir, deps.selected_doi, "figure", target.emit_source_name
        ),
        key_prefix=f"{target.key_prefix}_llm",
        file_name_prefix=target.emit_source_name,
    )

    show_key = f"{target.key_prefix}_show_interventions"
    if show_key not in st.session_state:
        st.session_state[show_key] = False
    if st.button(
        "Hide interventions" if st.session_state[show_key] else "Load interventions",
        type="primary",
        key=f"{target.key_prefix}_intervene_toggle",
    ):
        st.session_state[show_key] = not st.session_state[show_key]
        st.rerun()

    if st.session_state[show_key]:
        # A rerun-triggering selector instead of st.tabs. st.tabs hides inactive
        # tab content with CSS but keeps it mounted in the DOM, and switching
        # tabs is purely client-side — no Python rerun happens. A canvas that
        # mounts (or receives a new background) while its tab is display:none
        # doesn't get painted by the browser until something else forces a
        # redraw, which is why moving the width slider or clicking a button
        # "fixed" it. Rendering only the selected section means at most one
        # canvas exists at a time, and it always mounts into a visible
        # container because selecting it is a real widget interaction.
        section_options = ["Crop", "Axes", "Series", "Re-run"]
        section_key = f"{target.key_prefix}_intervention_section"
        if st.session_state.get(section_key) not in section_options:
            st.session_state[section_key] = section_options[0]
        if hasattr(st, "segmented_control"):
            section = st.segmented_control(
                "Intervention section", section_options, key=section_key,
                label_visibility="collapsed",
            )
        else:
            section = st.radio(
                "Intervention section", section_options, key=section_key,
                horizontal=True, label_visibility="collapsed",
            )
        section = section or st.session_state[section_key]

        if section == "Crop":
            _render_crop_tab(target)
        elif section == "Axes":
            render_axis_configuration_dropdown(
                chartdete_dir=target.chartdete_dir,
                source_img=target.source_img,
                key_prefix=f"{target.key_prefix}_ax",
                override_dir=(
                    deps.run_dir / "interventions" / deps.safe_doi / target.fig_id / "new_chartdete"
                ),
                wrap=False,
            )
        elif section == "Series":
            render_series_management_dropdown(
                run_dir=deps.run_dir,
                safe_doi=deps.safe_doi,
                fig_id=target.fig_id,
                fig_dir=target.plextract_dir,
                lf_dir=get_lineformer_dir(target.plextract_dir) if target.plextract_dir else None,
                lf_override_dir=override_lineformer_dir(deps.run_dir, deps.safe_doi, target.fig_id),
                source_img=target.source_img,
                key_prefix=f"{target.key_prefix}_sm",
                wrap=False,
            )
        else:
            _render_rerun_tab(target=target, deps=deps)

    _render_intervention_results(target=target, deps=deps)


# ---------------------------------------------------------------------------
# Building targets from a run
# ---------------------------------------------------------------------------

def _find_plextract_dir(run_dir: Path, safe_doi: str, fig_id: str) -> Optional[Path]:
    """Locate this figure's plextract output dir.

    Output dirs carry the image extension (e.g. `Figure_9.jpg/`), so match on
    stem rather than exact name.
    """
    latest = latest_plextract_run_dir(run_dir / "plextract")
    if not latest:
        return None
    out_base = latest / "output" / safe_doi
    if not out_base.exists():
        return None
    for candidate in out_base.iterdir():
        if candidate.is_dir() and Path(candidate.name).stem == fig_id:
            return candidate
    return None


def build_figure_target(*, figure, rr, deps: FiguresViewDeps, find_source_image: Callable) -> FigureTarget:
    """Build the panel target for a parent figure in the run."""
    safe_doi = deps.safe_doi
    fig_id = figure.id
    fig_stem = Path(fig_id).stem

    source_img = figure.figure_path
    if source_img is None or not source_img.exists():
        source_img = find_source_image(
            rr.run_dir / "figures_tables" / "figures", safe_doi, fig_id
        )

    image_caption = None
    if not _is_valid_image_path(source_img) and _is_valid_image_path(figure.figure_path):
        source_img = figure.figure_path
        image_caption = str(figure.figure_path)

    extra = getattr(figure, "extra", None) or {}
    final_csv = Path(extra["final_dataset_csv"]) if extra.get("final_dataset_csv") else None
    ocr_override_dir = (
        final_csv.parent.parent if (final_csv and final_csv.exists()) else None
    )

    plextract_dir = _find_plextract_dir(rr.run_dir, safe_doi, fig_id)

    # Auto-detected subfigure crops are stored flat under interventions/<safe_doi>/.
    crops_dir = rr.run_dir / "interventions" / safe_doi

    return FigureTarget(
        fig_id=fig_id,
        display_name=fig_id,
        key_prefix=f"fig_{safe_doi}_{fig_id}",
        emit_source_name=fig_id,
        fig_stem=fig_stem,
        source_img=source_img,
        image_caption=image_caption,
        figure_caption=figure.caption,
        plextract_dir=plextract_dir,
        chartdete_dir=(plextract_dir / "chartdete") if plextract_dir else None,
        final_csv=final_csv,
        crops_dir=crops_dir,
        is_subfigure=False,
        ocr_override_dir=ocr_override_dir,
        ocr_title=(
            "LLM-Extracted Figure Data"
            if extra.get("figure_data_source") == "llm_direct"
            else "OCR-Extracted Data"
        ),
    )


def build_subfigure_target(
    *, subfig_dir: Path, parent: FigureTarget, deps: FiguresViewDeps
) -> FigureTarget:
    """Build the panel target for one cropped subfigure."""
    safe_doi = deps.safe_doi
    source_img, image_caption = resolve_source_img_for_subfig_dir(subfig_dir, safe_doi)
    if source_img is None:
        recreated = subfig_dir / "final" / "final_image.png"
        if _is_valid_image_path(recreated):
            source_img = recreated
            image_caption = "Recreated image (crop not found)"

    # Use the stem so intervention dirs don't collide with the crop image file
    # stored at interventions/<safe_doi>/<name>.
    fig_id = subfig_dir.stem

    return FigureTarget(
        fig_id=fig_id,
        display_name=subfig_dir.name,
        key_prefix=f"subfig_{safe_doi}_{subfig_dir.name}",
        emit_source_name=subfig_dir.name,
        fig_stem=parent.fig_stem,
        source_img=source_img,
        image_caption=image_caption,
        plextract_dir=subfig_dir,
        chartdete_dir=subfig_dir / "chartdete",
        final_csv=subfig_dir / "final" / "final_dataset.csv",
        crops_dir=None,
        is_subfigure=True,
        ocr_override_dir=subfig_dir,
    )


def find_subfigure_dirs(*, parent: FigureTarget, deps: FiguresViewDeps) -> list:
    """Collect this figure's subfigure output dirs, highest-priority workdir first."""
    run_dir = deps.run_dir
    safe_doi = deps.safe_doi
    fig_stem = parent.fig_stem

    workdirs: list = []

    # 1. Per-figure intervention workdir.
    per_fig = run_dir / "interventions" / safe_doi / parent.fig_id / "plextract_interventions"
    if per_fig.exists():
        workdirs.append(per_fig)

    # 1b. Per-crop workdirs: each crop has its own plextract_interventions dir.
    if parent.crops_dir and parent.crops_dir.exists():
        for crop in sorted(parent.crops_dir.glob(f"{fig_stem}_subfig_*.png")):
            crop_wd = run_dir / "interventions" / safe_doi / crop.stem / "plextract_interventions"
            if crop_wd.exists():
                workdirs.append(crop_wd)

    # 2. Legacy shared interventions_plextract (backward compatibility).
    workdirs.append(run_dir / "interventions" / "interventions_plextract")

    # 3. Original plextract output.
    workdirs.append(run_dir / "plextract")

    subfig_map: dict = {}
    for workdir in workdirs:
        if not workdir or not workdir.exists():
            continue
        # Priority order: only fill missing names so the higher-priority workdir wins.
        for name, path in collect_subfig_dirs_across_runs(workdir, safe_doi, fig_stem).items():
            subfig_map.setdefault(name, path)

    return sorted(subfig_map.values(), key=lambda d: d.name)


# ---------------------------------------------------------------------------
# View entry point
# ---------------------------------------------------------------------------

def render_figures_view(*, rr, doi_item, deps: FiguresViewDeps, find_source_image: Callable) -> None:
    """Render the Figures view for the selected DOI.

    Exactly one figure is rendered at a time. Rendering every figure on every
    rerun (Streamlit executes collapsed expander bodies too) was the dominant
    cost of this view and the main source of cross-figure widget-key collisions.
    """
    figures = list(doi_item.figures or [])
    if not figures:
        st.warning("No figures found.")
        return

    # Drop cached axis-config widget values when the loaded run changes, so the
    # new run's override files are what gets displayed.
    run_dir_key = "_axis_cfg_last_run_dir"
    if st.session_state.get(run_dir_key) != str(rr.run_dir):
        stale_patterns = (
            "_ax_x_title", "_ax_y_title",
            "_ax_xtype", "_ax_ytype",
            "_ax_val_xmin", "_ax_val_xmax", "_ax_val_ymin", "_ax_val_ymax",
            "_ax_pt_xmin", "_ax_pt_xmax", "_ax_pt_ymin", "_ax_pt_ymax",
            "_ax_calib_ver", "_ax_calib_w",
        )
        for key in list(st.session_state.keys()):
            if any(pattern in key for pattern in stale_patterns):
                del st.session_state[key]
        st.session_state[run_dir_key] = str(rr.run_dir)

    names = [f.id for f in figures]
    select_key = f"figures_view_selected_{deps.safe_doi}"
    if st.session_state.get(select_key) not in names:
        st.session_state[select_key] = names[0]

    prev_col, select_col, count_col, next_col = st.columns([1, 5, 2, 1])
    index = names.index(st.session_state[select_key])
    with prev_col:
        if st.button("← Prev", disabled=index == 0, use_container_width=True, key=f"{select_key}_prev"):
            st.session_state[select_key] = names[index - 1]
            st.rerun()
    with next_col:
        if st.button(
            "Next →",
            disabled=index >= len(names) - 1,
            use_container_width=True,
            key=f"{select_key}_next",
        ):
            st.session_state[select_key] = names[index + 1]
            st.rerun()
    with select_col:
        # The label must stay constant across reruns. Streamlit factors a
        # widget's label into the identity it uses to match an incoming
        # selection against the widget that produced it; a label that embeds
        # the current index (as this used to) changes on every selection, so
        # the match fails and the widget silently resets to its first option
        # — the fix is one selection "late" because the stale label from the
        # broken run happens to already match the target by the second try.
        st.selectbox("Figure", names, key=select_key, label_visibility="collapsed")
    with count_col:
        st.markdown(
            f'<div style="padding-top:0.5rem;">Figure {index + 1} of {len(names)}</div>',
            unsafe_allow_html=True,
        )

    figure = figures[names.index(st.session_state[select_key])]
    target = build_figure_target(
        figure=figure, rr=rr, deps=deps, find_source_image=find_source_image
    )

    subfig_dirs = find_subfigure_dirs(parent=target, deps=deps)
    st.markdown(f"### {target.display_name}")
    render_figure_panel(target=target, deps=deps, image_expanded=not subfig_dirs)

    if not subfig_dirs:
        return

    st.divider()
    st.markdown("**Cropped subfigures**")
    for subfig_dir in subfig_dirs:
        st.markdown(f"#### {subfig_dir.name}")
        render_figure_panel(
            target=build_subfigure_target(subfig_dir=subfig_dir, parent=target, deps=deps),
            deps=deps,
            image_expanded=True,
        )
        if subfig_dir is not subfig_dirs[-1]:
            st.divider()
