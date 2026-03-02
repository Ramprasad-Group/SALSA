# AEM_OCR/src/ui/streamlit_app.py
from __future__ import annotations
import json
import inspect
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pipeline import run_full_pipeline
from discover import discover_run
from schemas import RunResult
from intervention import find_interventions, latest_plextract_run_dir
from intervention import (
    get_lineformer_dir,
    load_series_catalog,
    load_mapping,
    save_mapping,
    save_mask_override,
    write_series_catalog_override,
    list_override_masks,
    apply_overrides_to_plextract,
    override_lineformer_dir,
)
from run_plextract_manual import run_plextract_intervention_on_crops
from custom_instructions import (
    default_instructions,
    list_presets,
    save_preset,
    load_preset,
    delete_preset,
)
import numpy as np
import cv2
from cropping import save_crops
from run_plextract_manual import run_plextract_on_manual_crops
from axis_config import (
    load_axis_config, save_axis_config, reset_axis_config,
    list_presets as list_axis_presets,
    save_preset as save_axis_preset,
    load_preset as load_axis_preset,
    delete_preset as delete_axis_preset,
)
import math
import os
import sys
import subprocess
import sys
import base64
from typing import List, Optional
import re
from datetime import datetime
from typing import Callable, Sequence, Optional, Any

# Make cost module importable
_cost_dir = os.path.join(os.path.dirname(__file__), "..", "cost")
if _cost_dir not in sys.path:
    sys.path.insert(0, _cost_dir)
from sum_cost import summarise_usage

#-------------------
#-----WORKSPACE-----
#-------------------

_WORKSPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

def _list_subdirs(p: Path) -> list[str]:
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])

def _is_safe_workspace_name(name: str) -> bool:
    """
    Restrict workspace names so users can't do path traversal like ../../etc.
    """
    name = (name or "").strip()
    if not name:
        return False
    if "/" in name or "\\" in name:
        return False
    return bool(_WORKSPACE_RE.match(name))

def _ensure_workspace(root: Path, name: str) -> Path:
    """
    Create (if needed) and return <root>/<name>.
    """
    ws = (root / name).resolve()
    # Enforce that ws is inside root even after resolve()
    root_res = root.resolve()
    if root_res not in ws.parents and ws != root_res:
        raise ValueError(f"Workspace path escapes root: {ws}")
    ws.mkdir(parents=True, exist_ok=True)
    return ws

def _default_workspace_name() -> str:
    # nice default when user hasn't chosen anything yet
    return datetime.now().strftime("workspace_%Y%m%d")

def _default_ollama_models_dir() -> Path:
    """
    Programmatically discover the Ollama models directory.

    Resolution order:
    1. ``OLLAMA_MODELS`` environment variable (explicit override).
    2. Ask the running ``ollama`` daemon via ``ollama show --modelpath`` (most
       authoritative – reflects the daemon's *actual* storage location).
    3. Parse the systemd unit file for an ``Environment=OLLAMA_MODELS=…`` line.
    4. Probe well-known filesystem candidates.
    5. Fall back to ``~/.ollama/models`` and create it if it doesn't exist.
    """
    # 1. Environment variable
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        p = Path(env).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2. Ask the CLI – ``ollama show --modelpath <any-model>`` prints the
    #    blob-storage root.  We don't need a real model name; even when the
    #    command fails it can reveal the base path in its error text.  A
    #    lighter alternative is to inspect ``ollama list`` output and derive
    #    the path, but ``OLLAMA_MODELS`` from the daemon env is more direct.
    try:
        # ``ollama`` respects OLLAMA_HOST, so this talks to the right daemon.
        out = subprocess.check_output(
            ["ollama", "list"], text=True, stderr=subprocess.STDOUT, timeout=5
        )
        # If ``ollama list`` succeeds the daemon is up; the daemon's storage
        # dir is authoritative.  Try to read it from the process environment.
        models_dir = _ollama_models_dir_from_proc()
        if models_dir and models_dir.is_dir():
            return models_dir
    except Exception:
        pass

    # 3. systemd unit file (common on Linux when Ollama was installed via the
    #    official installer script).
    for unit_path in (
        Path("/etc/systemd/system/ollama.service"),
        Path("/lib/systemd/system/ollama.service"),
    ):
        try:
            if unit_path.is_file():
                text = unit_path.read_text()
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("Environment=") and "OLLAMA_MODELS" in line:
                        # e.g. Environment="OLLAMA_MODELS=/data/ollama/models"
                        _, _, val = line.partition("OLLAMA_MODELS=")
                        val = val.strip().strip('"').strip("'")
                        if val:
                            p = Path(val)
                            if p.is_dir():
                                return p
        except Exception:
            pass

    # 4. Well-known candidates
    candidates = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
        Path("/opt/ollama/models"),
    ]
    for c in candidates:
        try:
            if c.exists():
                return c
        except (PermissionError, OSError):
            continue

    # 5. Fallback – create the user-local directory
    fallback = Path.home() / ".ollama" / "models"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

def _ollama_models_dir_from_proc() -> Optional[Path]:
    """
    Try to read ``OLLAMA_MODELS`` from the running ``ollama`` daemon process's
    environment on Linux (``/proc/<pid>/environ``).  Returns *None* on failure.
    """
    try:
        pids = subprocess.check_output(
            ["pgrep", "-x", "ollama"], text=True, timeout=3
        ).split()
        for pid in pids:
            env_file = Path(f"/proc/{pid}/environ")
            if env_file.exists():
                env_bytes = env_file.read_bytes()
                for entry in env_bytes.split(b"\x00"):
                    if entry.startswith(b"OLLAMA_MODELS="):
                        val = entry.split(b"=", 1)[1].decode()
                        p = Path(val)
                        if p.is_dir():
                            return p
    except Exception:
        pass
    return None

def _default_ollama_host() -> str:
    """
    Programmatically discover the Ollama host endpoint.

    Resolution order:
    1. ``OLLAMA_HOST`` environment variable.
    2. ``OLLAMA_HOST`` from the running daemon's process environment (Linux).
    3. ``OLLAMA_HOST`` from the systemd unit file.
    4. Default: ``http://127.0.0.1:11434`` (standard Ollama port).
    """
    # 1. Environment variable
    env = os.environ.get("OLLAMA_HOST")
    if env:
        host = env.strip()
        if not host.startswith("http"):
            host = "http://" + host
        return host

    # 2. Daemon process environment (Linux)
    try:
        pids = subprocess.check_output(
            ["pgrep", "-x", "ollama"], text=True, timeout=3
        ).split()
        for pid in pids:
            env_file = Path(f"/proc/{pid}/environ")
            if env_file.exists():
                env_bytes = env_file.read_bytes()
                for entry in env_bytes.split(b"\x00"):
                    if entry.startswith(b"OLLAMA_HOST="):
                        val = entry.split(b"=", 1)[1].decode().strip()
                        if not val.startswith("http"):
                            val = "http://" + val
                        return val
    except Exception:
        pass

    # 3. systemd unit file
    for unit_path in (
        Path("/etc/systemd/system/ollama.service"),
        Path("/lib/systemd/system/ollama.service"),
    ):
        try:
            if unit_path.is_file():
                text = unit_path.read_text()
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("Environment=") and "OLLAMA_HOST" in line:
                        _, _, val = line.partition("OLLAMA_HOST=")
                        val = val.strip().strip('"').strip("'")
                        if val:
                            if not val.startswith("http"):
                                val = "http://" + val
                            return val
        except Exception:
            pass

    # 4. Standard default
    return "http://127.0.0.1:11434"

def _ollama_list_models_via_cli() -> List[str]:
    """
    Best source of truth: `ollama list` returns the models actually available to the ollama daemon.
    """
    try:
        # Note: OLLAMA_HOST can be set in env to point to a non-default daemon
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not lines:
            return []
        # Expect header like: NAME ID SIZE MODIFIED
        # Models in first column, e.g. "llama3:8b"
        models = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []

def _ollama_list_models_from_dir(models_dir: Path) -> List[str]:
    """
    Fallback: attempt to infer installed models from the local models directory.
    This is less reliable than `ollama list`, but helps when CLI isn't available.
    """
    try:
        # Typical structure includes "manifests" + "blobs"
        manifests = models_dir / "manifests"
        if not manifests.exists():
            return []
        # manifests can be nested; filenames represent tags.
        # We'll return relative leaf paths that look like repo/model:tag-ish.
        found = []
        for p in manifests.rglob("*"):
            if p.is_file():
                rel = p.relative_to(manifests).as_posix()
                # Heuristic: keep only reasonably short “tag” paths
                if len(rel.split("/")) >= 2:
                    found.append(rel)
        # Deduplicate + shorten presentation
        found = sorted(set(found))
        return found
    except Exception:
        return []

def st_image_auto(img, *, full_width=True, **kwargs):
    """
    Streamlit image compat:
      - newer streamlit: use_container_width
      - older streamlit: use_column_width
    Avoid passing both.
    """
    sig = inspect.signature(st.image)
    kwargs.pop("use_container_width", None)
    kwargs.pop("use_column_width", None)
    if "use_container_width" in sig.parameters:
        return st.image(img, use_container_width=full_width, **kwargs)
    return st.image(img, use_column_width=full_width, **kwargs)

def section_dropdown(title: str, *, expanded: bool = False):
    """
    Small UI helper so both main figures and subfigures create section
    dropdowns (expanders) in exactly the same way.
    """
    return st.expander(title, expanded=expanded)

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
        if image and image.exists():
            st_resizable_image(image, caption=image_caption, key=key)
        else:
            st.warning("No source image found for this figure.")

def render_axis_configuration_dropdown(
    *,
    chartdete_dir: Optional[Path],
    source_img: Optional[Path],
    key_prefix: str,
    expanded: bool = False,
):
    """Render the full Axis Configuration dropdown, shared between figures and subfigures."""
    with section_dropdown("Axis Configuration", expanded=expanded):
        if chartdete_dir is None:
            st.info("No plextract output found for this figure. Run plextract extraction first.")
            return
        chartdete_dir.mkdir(parents=True, exist_ok=True)

        # ========================
        # Axis Titles
        # ========================
        st.markdown("**Axis Titles**")
        title_override_path = chartdete_dir / "axis_title_override.json"
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
        if source_img is None or not source_img.exists():
            st.info("No source image available for axis calibration.")
            return

        st.caption(
            "Click the image to set Xmin/Xmax/Ymin/Ymax pixel locations, then enter the numeric values. "
            "This writes `chartdete/pix2val_override.json` for this figure."
        )
        st.caption(f"Calibrating on image: `{source_img}`")

        pix2val_override_path = chartdete_dir / "pix2val_override.json"
        pix2val_orig_path = chartdete_dir / "pix2val.json"
        axis_type_path = chartdete_dir / "axis_type_override.json"
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

        pil_img = Image.open(source_img).convert("RGB")
        img_w, img_h = pil_img.size

        width_key = f"{key_prefix}_ax_calib_w"
        if width_key not in st.session_state:
            st.session_state[width_key] = min(1100, img_w)
        disp_w = st.slider(
            "Calibration view width (px)", min_value=400, max_value=1800,
            value=int(st.session_state[width_key]), step=50, key=width_key,
        )
        disp_h = int(round(img_h * (disp_w / img_w)))

        bg = pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS).convert("RGBA")
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h

        endpoint_key = f"{key_prefix}_ax_calib_endpoint"
        endpoint = st.radio("Click target", ["Xmin", "Xmax", "Ymin", "Ymax"], key=endpoint_key, horizontal=True)

        cv_ver_key = f"{key_prefix}_ax_calib_ver"
        if cv_ver_key not in st.session_state:
            st.session_state[cv_ver_key] = 0

        c1_ax, _c2_ax = st.columns([1, 3])
        with c1_ax:
            if st.button("Clear markers", key=f"{key_prefix}_ax_clear_markers"):
                st.session_state[cv_ver_key] += 1
                for k in ("xmin", "xmax", "ymin", "ymax"):
                    st.session_state.pop(f"{key_prefix}_ax_pt_{k}", None)
                st.rerun()

        st.write("**Click on the image to place the selected endpoint.**")

        canvas_key = (
            f"{key_prefix}_ax_calib_canvas_{Path(source_img).name}_{disp_w}x{disp_h}"
            f"_v{st.session_state[cv_ver_key]}"
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
            o = circles[-1]
            left = float(o.get("left", 0.0))
            top = float(o.get("top", 0.0))
            r = float(o.get("radius", 0.0)) * float(o.get("scaleX", 1.0))
            return left + r, top + r

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

        st.markdown("**Endpoint values** (enter the numeric value for each clicked point)")
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
    expanded: bool = False,
):
    """Render the full Series Management dropdown, shared between figures and subfigures."""
    with section_dropdown("Series Management", expanded=expanded):
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
                    mimg = Image.open(mask_path).convert("L")
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

        if source_img and source_img.exists():
            base_img = Image.open(source_img).convert("RGBA")
            w, h = base_img.size
        else:
            st.warning("No source image for series overlay.")
            base_img = None
            w, h = 600, 400

        disp_w = min(w, 900)
        disp_h = int(h * (disp_w / w))

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
                mask = Image.open(mp).convert("L")
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
        mode = st.radio("Mode", mode_options, key=f"{key_prefix}_series_mode", horizontal=True)
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
                cur_mask_img = Image.open(cur_mask_path).convert("L").resize((disp_w, disp_h))

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

            input_mode = st.radio(
                "Input mode", ["Freehand drawing", "Pick exact points"],
                horizontal=True,
                key=f"{key_prefix}_input_mode_{selected_idx}_{mode}",
            )
            c1_sm, c2_sm = st.columns([3, 1])

            if input_mode == "Freehand drawing":
                with c1_sm:
                    st.write("Draw on the colored series image below")
                    brush = st.slider("Brush width", 1, 40, 10, key=f"{key_prefix}_brush_{selected_idx}")
                    mask_canvas = st_canvas(
                        background_image=composite_disp,
                        drawing_mode="freedraw",
                        stroke_width=brush, stroke_color=stroke_rgba,
                        fill_color=stroke_rgba, update_streamlit=True,
                        height=disp_h, width=disp_w,
                        key=f"{key_prefix}_mask_canvas_{selected_idx}_{mode}",
                    )
                if legend_lines:
                    st.markdown(" &nbsp; ".join(legend_lines), unsafe_allow_html=True)

                if mask_canvas.image_data is not None:
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
                    dot_size = st.slider(
                        "Point display size", 1, 5, 1,
                        key=f"{key_prefix}_dot_size_{selected_idx}_{mode}",
                    )
                    if existing_pts:
                        st.caption(f"Currently saved: **{len(existing_pts)} point(s)**. Placing new points replaces them.")
                    else:
                        st.caption("Click on the figure to place data points.")
                    point_canvas = st_canvas(
                        background_image=composite_disp,
                        drawing_mode="point", point_display_radius=dot_size,
                        stroke_color=stroke_rgba, fill_color=stroke_rgba,
                        update_streamlit=True, height=disp_h, width=disp_w,
                        key=f"{key_prefix}_point_canvas_{selected_idx}_{mode}",
                    )
                if legend_lines:
                    st.markdown(" &nbsp; ".join(legend_lines), unsafe_allow_html=True)

                new_pts: list = []
                if point_canvas.json_data and "objects" in point_canvas.json_data:
                    for obj2 in point_canvas.json_data["objects"]:
                        if obj2.get("type") == "circle":
                            r2 = float(obj2.get("radius", dot_size))
                            cxd = float(obj2.get("left", 0)) + r2
                            cyd = float(obj2.get("top", 0)) + r2
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
                    st_image_auto(cur_mask_img, full_width=True, use_container_width=True)
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

def st_resizable_image(image_path: Path, *, caption: Optional[str] = None, key: str = ""):
    """
    Render an image inside a container that can be resized by dragging
    the bottom-right corner. This removes the need for explicit width sliders.
    """
    if not image_path.exists():
        st.warning(f"Image not found: {image_path}")
        return
    try:
        mime = f"image/{image_path.suffix.lower().lstrip('.') or 'png'}"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <div id="resizable-img-{key}" style="
                width:min(100%, 700px);
                max-width:100%;
                resize:horizontal;
                overflow:auto;
                border:1px solid rgba(128,128,128,0.35);
                border-radius:6px;
                padding:4px;
                margin-bottom:0.4rem;
            ">
                <img src="data:{mime};base64,{b64}" style="width:100%;height:auto;display:block;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        if caption:
            st.caption(caption)
    except Exception as _img_ex:
        st.warning(f"Failed to render resizable image: {_img_ex}")
        st_image_auto(str(image_path), full_width=True)

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

def doi_to_safe(doi: str) -> str:
    return doi.replace("/", "@").strip()

def _render_cost_summary(run_dir: Path):
    """Display cost summary for a run if usage.jsonl exists."""
    usage_path = run_dir / "cost" / "usage.jsonl"
    if not usage_path.exists():
        return
    summary = summarise_usage(str(usage_path))
    if not summary["models"]:
        return
    with st.expander("Cost Summary", expanded=False):
        for model_name, info in summary["models"].items():
            label = model_name if model_name != "unknown" else "unknown model"
            st.markdown(
                f"**{label}**: {info['calls']} calls, "
                f"{info['input']:,} input tokens, "
                f"{info['cached']:,} cached, "
                f"{info['output']:,} output tokens "
                f"— **${info['cost']:.4f}**"
            )
        st.markdown(f"### Total: **${summary['grand_total']:.4f}**")

@st.cache_data(show_spinner=False)
def _cached_flagged(run_dir_str: str):
    # cache intervention scan so it doesn't re-run constantly
    run_dir = Path(run_dir_str)
    return find_interventions(run_dir)

@st.cache_data(show_spinner=False)
def _load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def completed_dataset_csv_path(run_dir: Path, doi: str) -> Path:
    """
    complete-dataset.py writes:
      <run_dir>/complete_dataset/<modified_doi>_dataset.csv
    where modified_doi is DOI with '/' -> '@'
    """
    modified = doi_to_safe(doi)
    return run_dir / "complete_dataset" / f"{modified}_dataset.csv"

def completed_dataset_json_path(run_dir: Path, doi: str) -> Path:
    """
    Also written by complete-dataset.py:
      <run_dir>/complete_dataset/<modified_doi>_complete.json
    """
    modified = doi_to_safe(doi)
    return run_dir / "complete_dataset" / f"{modified}_complete.json"

def find_emit_rows_for_source(
    run_dir: Path,
    doi: str,
    source_kind: str,
    source_name: str
) -> Path | None:
    """
    Find the emit_rows cache file for a specific table/figure.

    Cache key format used by row_emitter.py:
      emit_rows::{source_kind}::{doi}::{source_name}::{hash}

    The _safe_filename() in row_emitter replaces non-alphanumeric chars with '_',
    then _cache_path appends '__<sha256[:20]>.json'.  So the on-disk name is:
      emit_rows__{source_kind}__{sanitized_doi}_{sanitized_source_name}__{sha256}__{sha256}.json
    (double underscores between cache_key segments, single underscores within segments)

    Returns path to cache file if found, None otherwise.
    """
    cache_dir = run_dir / "complete_dataset" / "row_cache"
    if not cache_dir.exists():
        return None

    # Build the exact prefix that row_emitter._safe_filename would produce from
    # the cache key "emit_rows::{source_kind}::{doi}::{source_name}::".
    # _safe_filename replaces any non-[a-zA-Z0-9._-] with '_', so :: -> __ and / -> _
    def _safe_filename_compat(s: str) -> str:
        """Reproduce row_emitter._safe_filename logic."""
        import re as _re
        return _re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:180]

    # The cache key always starts with "emit_rows::{kind}::{doi}::{name}::"
    # After _safe_filename this becomes "emit_rows__{kind}__{doi_safe}__{name_safe}__"
    cache_key_prefix = f"emit_rows::{source_kind}::{doi}::{source_name}::"
    safe_prefix = _safe_filename_compat(cache_key_prefix)

    candidates = []
    for cache_file in cache_dir.iterdir():
        if cache_file.suffix != ".json":
            continue
        # Files are named: <safe_prefix><hash>__<sha256>.json
        # Check that the filename starts with our safe_prefix
        if cache_file.stem.startswith(safe_prefix) or cache_file.name.startswith(safe_prefix):
            candidates.append(cache_file)

    # Return most recent if multiple found
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None

def load_emit_rows(cache_path: Path | None) -> pd.DataFrame | None:
    """
    Load emit_rows from cache file and return as DataFrame.

    Cache file format: {"rows": [...], "metadata": {...}}
    """
    if not cache_path or not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)

        # Support both {"rows": [...]} and [...]
        if isinstance(data, dict) and "rows" in data:
            rows = data["rows"]
        elif isinstance(data, list):
            rows = data
        else:
            return None

        if not rows:
            return None

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Failed to load emit_rows: {e}")
        return None

def collect_all_emit_rows(run_dir: Path, doi: str, doi_item) -> pd.DataFrame:
    """
    Collect and concatenate all LLM-generated emit_rows from row_cache
    for all tables, figures, and subfigures for the given DOI.

    This creates a reactive dataset that updates whenever emit_rows are regenerated.
    """
    all_dfs = []
    safe_doi = doi.replace("/", "@").strip()
    seen_source_names: set[str] = set()

    def _add_emit(source_kind: str, source_name: str):
        if source_name in seen_source_names:
            return
        seen_source_names.add(source_name)
        emit_path = find_emit_rows_for_source(run_dir, doi, source_kind, source_name)
        if emit_path:
            df = load_emit_rows(emit_path)
            if df is not None and not df.empty:
                all_dfs.append(df)

    # Collect emit_rows from all tables
    for table in doi_item.tables:
        _add_emit("table", table.id)

    # Collect emit_rows from all figures
    for figure in doi_item.figures:
        _add_emit("figure", figure.id)

    # Collect emit_rows from subfigures in both plextract workdirs.
    # Subfigures can appear in:
    #   1. <run_dir>/plextract/results/<n>/output/<safe_doi>/<subfig_name>/
    #   2. <run_dir>/interventions/interventions_plextract/results/<n>/output/<safe_doi>/<subfig_name>/
    _workdirs_to_scan = [
        run_dir / "plextract",
        run_dir / "interventions" / "interventions_plextract",
    ]
    for wd in _workdirs_to_scan:
        if not wd.exists():
            continue
        latest = latest_plextract_run_dir(wd)
        if not latest:
            continue
        out_base = latest / "output" / safe_doi
        if not out_base.exists():
            continue
        for subfig_dir in out_base.iterdir():
            if subfig_dir.is_dir() and "_subfig_" in subfig_dir.name:
                _add_emit("figure", subfig_dir.name)

    # Concatenate all dataframes
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove exact duplicates
    combined = combined.drop_duplicates()

    return combined

def build_doi_stats(rr: RunResult, *, include_interventions: bool, doi_order: list[str] | None = None) -> pd.DataFrame:
    rows = []
    flagged = []
    if include_interventions:
        flagged = _cached_flagged(str(rr.run_dir))

    # count interventions per safe_doi
    flagged_counts = {}
    if flagged:
        for x in flagged:
            flagged_counts[x.safe_doi] = flagged_counts.get(x.safe_doi, 0) + 1

    for d in rr.dois:
        safe = doi_to_safe(d.doi)
        rows.append({
            "DOI": d.doi,
            "Figures": len(d.figures or []),
            "Tables": len(d.tables or []),
            "Interventions Needed": flagged_counts.get(safe, 0) if include_interventions else None,
            "_doi_safe": safe,  # hidden helper
        })

    df = pd.DataFrame(rows)
    # --- preserve paste order if provided ---
    if doi_order:
        order_map = {doi: i for i, doi in enumerate(doi_order)}
        df["_order"] = df["DOI"].map(order_map).fillna(10**9).astype(int)
        df = df.sort_values("_order", kind="stable").drop(columns=["_order"]).reset_index(drop=True)

    
    return df

def _ollama_models() -> List[str]:
    models = _ollama_list_models_via_cli()
    if not models:
        models = _ollama_list_models_from_dir(Path(str(default_ollama_dir)))
    return models

def iter_runs_newest_first(workdir: Path) -> list[Path]:
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
    if input_dir.exists():
        # Try exact name first (in case it includes extension)
        p = input_dir / subfig_dir.name
        if p.exists():
            return p, "Cropped image (run input)"

        # Try matching by stem (handles missing/extra extension)
        stem = Path(subfig_dir.name).stem
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p2 = input_dir / f"{stem}{ext}"
            if p2.exists():
                return p2, "Cropped image (run input)"

        # Last resort: scan for any file with matching stem
        for fp in input_dir.iterdir():
            if fp.is_file() and Path(fp.name).stem == stem:
                return fp, "Cropped image (run input)"

    # Fallback: recreated image
    final_img = subfig_dir / "final" / "final_image.png"
    if final_img.exists():
        return final_img, "Recreated image (crop not found)"

    return None, None

def select_with_preview(
    label: str,
    options: Sequence[str],
    key: str,
    preview_path_for_option: Callable[[str], Optional[Path]],
    *,
    default_to_first: bool = True,
    preview_width: Optional[int] = None,
):
    if not options:
        st.info("No options available.")
        return None

    # Ensure a deterministic default *before* creating the widget
    if default_to_first:
        st.session_state.setdefault(key, options[0])
        # If options changed and current value is no longer valid, reset
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[0]

    sel = st.selectbox(
        label,
        options,
        index=options.index(st.session_state[key]) if st.session_state.get(key) in options else 0,
        key=key,
    )

    # Preview renders immediately because `sel` is never None now
    p = preview_path_for_option(sel)
    if p and Path(p).exists():
        st.image(str(p), width=preview_width)
        st.caption(Path(p).name)
    else:
        st.warning("Preview image not found for the current selection.")

    return sel

@st.cache_data(show_spinner=False)
def load_and_resize_rgba(path: str, disp_w: int):
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    dh = int(round(oh * (disp_w / ow)))
    bg = img.resize((disp_w, dh), Image.Resampling.LANCZOS).convert("RGBA")
    return ow, oh, disp_w, dh, bg

# SET OLLAMA DIR AND HOST (computed outside expander for reuse)
default_ollama_dir = _default_ollama_models_dir()
default_ollama_host = _default_ollama_host()

_openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini", "gpt-5-nano", "gpt-5"]



st.set_page_config(page_title="AEM_OCR UI", layout="wide")
st.title("AEM_OCR")

st.markdown("""
<style>
/* Make the expander content scrollable and resizable */
div[data-testid="stExpanderDetails"] {
    max-height: 800px;
    overflow-y: auto;
    padding-right: 10px;
    resize: vertical;
    min-height: 100px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:

    # Set workspace variables – resolve relative to this file, not cwd
    scripts_root = (Path(__file__).resolve().parent / "..").resolve()
    workspaces_root = Path(str(scripts_root / "ui_runs"))
    workspaces_root.mkdir(parents=True, exist_ok=True)

    st.divider()

    # Choose/create workspace
    st.header("Workspace")

    # List existing workspaces (filter out axis_presets and instruction_presets)
    existing_workspaces = [
        ws for ws in _list_subdirs(workspaces_root)
        if ws not in ("axis_presets", "instruction_presets")
    ]

    # Apply pending selection (but DON'T write workspace_name)
    pending = st.session_state.pop("pending_workspace_name", None)

    # Determine what we want selected this run
    desired = pending or st.session_state.get("workspace_name")
    if desired not in existing_workspaces:
        desired = existing_workspaces[0] if existing_workspaces else _default_workspace_name()

    options = existing_workspaces if existing_workspaces else [desired]
    index = options.index(desired) if desired in options else 0

    st.selectbox(
        "Active workspace",
        options=options,
        index=index,
        key="workspace_name",
    )


    # Allow creation of a new workspace
    new_ws = st.text_input("Create new workspace", value="", key="new_workspace_name")
    col_ws_a, col_ws_b = st.columns([1, 1])

    with col_ws_a:
        if st.button("Create workspace"):
            if not _is_safe_workspace_name(new_ws):
                st.error("Invalid workspace name. Use letters/numbers plus _ . - (max 64 chars).")
            else:
                _ensure_workspace(workspaces_root, new_ws)
                st.session_state["pending_workspace_name"] = new_ws
                st.rerun()

    with col_ws_b:
        if st.button("Refresh workspaces"):
            st.rerun()

    # Now this is always the true selected value
    active_workspace_name = st.session_state["workspace_name"]
    active_workspace_dir = _ensure_workspace(workspaces_root, active_workspace_name)
    run_root = active_workspace_dir

    st.divider()
    st.header("Select Stages")
    with st.expander("Stages", expanded=False):
        run_download_doi = st.checkbox("Download DOIs", value=True)
        run_extract_figures = st.checkbox("Extract figures", value=True)
        run_extract_tables = st.checkbox("Extract tables", value=True)
        run_chemical_composition = st.checkbox("Extract chemical composition", value=True)
        run_filter_figures = st.checkbox("Filter figures", value=True)
        run_filter_tables = st.checkbox("Filter tables", value=True)
        run_plextract = st.checkbox("Extract data from figures", value=True)
        run_complete_dataset = st.checkbox("Complete dataset", value=True)
        exclude_full_text = st.checkbox("Exclude full text from context", value=True)

    st.divider()
    st.header("Select LLM(s)")
    with st.expander("LLM", expanded=False):
        # --- General LLM (used for filtering, figure planning, etc.) ---
        st.subheader("General LLM")
        st.caption("Used for figure/table filtering, figure planning, and other pre-processing steps.")

        provider = st.radio(
            "Provider",
            options=["Ollama", "Chat GPT"],
            index=0,
            key="llm_provider",
        )

        if provider == "Ollama":
            ollama_models_dir = Path(str(default_ollama_dir))
            ollama_host = default_ollama_host

            models = _ollama_models()
            if models:
                # Clear stale value from a previous provider so index=0 takes effect.
                if st.session_state.get("llm_model") not in models:
                    st.session_state.pop("llm_model", None)
                selected_model = st.selectbox("Model", options=models, index=0, key="llm_model")
            else:
                selected_model = st.text_input(
                    "Model (no models detected)",
                    value="",
                    key="llm_model",
                )
                st.caption("Tip: run `ollama pull <model>` then refresh.")
        else:
            # Read OpenAI config from environment (not displayed in UI)
            if "openai_base_url" not in st.session_state:
                st.session_state["openai_base_url"] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if "openai_api_key" not in st.session_state:
                st.session_state["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
            # Clear stale value from a previous provider so index=0 (gpt-4o) takes effect.
            if st.session_state.get("llm_model") not in _openai_models:
                st.session_state.pop("llm_model", None)
            selected_model = st.selectbox("Model", options=_openai_models, index=0, key="llm_model")

        # --- Data Processing LLM ---
        st.subheader("Data Processing LLM")
        st.caption("Used for dataset completion and chemical extraction. Can differ from the general LLM.")

        use_separate_processing_llm = st.checkbox(
            "Use a different LLM for data processing",
            value=False,
            key="use_separate_processing_llm",
        )

        if use_separate_processing_llm:
            processing_provider = st.radio(
                "Processing provider",
                options=["Ollama", "Chat GPT"],
                index=1,
                key="processing_llm_provider",
            )

            if processing_provider == "Ollama":
                models = _ollama_models()
                if models:
                    if st.session_state.get("processing_llm_model") not in models:
                        st.session_state.pop("processing_llm_model", None)
                    st.selectbox("Processing model", options=models, index=0, key="processing_llm_model")
                else:
                    st.text_input("Processing model (no models detected)", value="", key="processing_llm_model")
            else:
                # Read processing OpenAI config from environment (not displayed in UI)
                if "processing_openai_base_url" not in st.session_state:
                    st.session_state["processing_openai_base_url"] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                if "processing_openai_api_key" not in st.session_state:
                    st.session_state["processing_openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
                if st.session_state.get("processing_llm_model") not in _openai_models:
                    st.session_state.pop("processing_llm_model", None)
                st.selectbox("Processing model", options=_openai_models, index=0, key="processing_llm_model")

    st.divider()
    st.header("Custom Instructions")
    if "custom_instructions" not in st.session_state:
        st.session_state["custom_instructions"] = default_instructions()
        st.session_state["current_preset_name"] = None

    ci = st.session_state["custom_instructions"]

    # --- Load presets ---
    existing_presets = list_presets(workspaces_root)
    with st.expander("Load Preset Custom Instructions", expanded = False):
        if existing_presets:
            col_load_a, col_load_b = st.columns([3, 1])
            with col_load_a:
                preset_choice = st.selectbox(
                    "Select preset to load",
                    options=[""] + existing_presets,
                    key="ci_preset_choice",
                )
            with col_load_b:
                if st.button("Load", type="secondary") and preset_choice:
                    loaded = load_preset(workspaces_root, preset_choice)
                    st.session_state["custom_instructions"] = loaded
                    st.session_state["current_preset_name"] = preset_choice
                    st.success(f"Loaded preset: {preset_choice}")
                    st.rerun()
        else:
            st.info("No presets saved yet. Create your first preset below.")

        # Show currently loaded preset
        if st.session_state.get("current_preset_name"):
            st.caption(f"Currently editing: **{st.session_state['current_preset_name']}**")
        else:
            st.caption("Currently editing: **Unsaved changes (default)**")

    # --- Instruction editors ---
    with st.expander("Table extraction rules", expanded=False):
        st.caption(
            "Extra rules for how raw HTML/XML tables are parsed into structured data. "
            "E.g. how merged cells or multi-level headers should be handled."
        )
        st.text_area(
            "Table extraction extra rules",
            value=ci.get("table_extraction_rules", ""),
            height=120,
            key="ci_table_extraction",
        )

    with st.expander("Figure filtering criteria", expanded=False):
        st.caption(
            "Criteria the LLM uses to decide whether to keep or discard a figure. "
            "E.g. 'only keep line charts showing conductivity vs time'."
        )
        st.text_area(
            "Figure filter criteria",
            value=ci.get("figure_filter_criteria", ""),
            height=150,
            key="ci_figure_filter",
        )

    with st.expander("Table filtering criteria", expanded=False):
        st.caption(
            "Criteria the LLM uses to decide whether to keep or discard a table."
        )
        st.text_area(
            "Table filter criteria",
            value=ci.get("table_filter_criteria", ""),
            height=150,
            key="ci_table_filter",
        )

    with st.expander("Data processing rules", expanded=False):
        st.caption(
            "Extra rules applied when the LLM processes data (converts tables/figures into "
            "final dataset rows, extracts chemical compositions). E.g. 'before = time 0'."
        )
        st.text_area(
            "Data processing extra rules",
            value=ci.get("data_processing_rules", ""),
            height=120,
            key="ci_data_processing",
        )

    with st.expander("Output columns", expanded=False):
        st.caption(
            "One column name per line for the final dataset. "
            "Change this to capture different properties."
        )
        st.text_area(
            "Output columns (one per line)",
            value="\n".join(ci.get("output_columns", [])),
            height=160,
            key="ci_output_columns",
        )

    with st.expander("Missing-value defaults", expanded=False):
        st.caption(
            "JSON dict of column name to default value when not found. "
            "E.g. {\"time(h)\": \"0 Not Found\"}"
        )
        st.text_area(
            "Missing-value defaults (JSON)",
            value=json.dumps(ci.get("missing_value_defaults", {}), indent=2),
            height=160,
            key="ci_missing_defaults",
        )

    with st.expander("Save Custom Instructions", expanded=False):

        st.subheader("Save Preset")
        col_name, col_save_btn, col_del_btn = st.columns([3, 1, 1])

        with col_name:
            # Default to current preset name if editing an existing one
            default_name = st.session_state.get("current_preset_name", "")
            preset_name = st.text_input(
                "Preset name",
                value=default_name,
                key="ci_new_preset_name",
                placeholder="Enter a name for this preset"
            )

        with col_save_btn:
            st.write("")  # spacing
            st.write("")  # spacing
            if st.button("Save", type="primary") and preset_name.strip():
                # Read all fields from session state
                _ci_to_save = {}
                _ci_to_save["table_extraction_rules"] = st.session_state.get("ci_table_extraction", "")
                _ci_to_save["figure_filter_criteria"] = st.session_state.get("ci_figure_filter", "")
                _ci_to_save["table_filter_criteria"] = st.session_state.get("ci_table_filter", "")
                _ci_to_save["data_processing_rules"] = st.session_state.get("ci_data_processing", "")

                raw_cols = st.session_state.get("ci_output_columns", "")
                _ci_to_save["output_columns"] = [c.strip() for c in raw_cols.splitlines() if c.strip()]
                try:
                    _ci_to_save["missing_value_defaults"] = json.loads(
                        st.session_state.get("ci_missing_defaults", "{}")
                    )
                except json.JSONDecodeError:
                    st.error("Missing-value defaults must be valid JSON.")
                    _ci_to_save["missing_value_defaults"] = {}

                save_preset(workspaces_root, preset_name.strip(), _ci_to_save)
                st.session_state["current_preset_name"] = preset_name.strip()
                st.success(f"Saved preset: {preset_name.strip()}")
                st.rerun()

        with col_del_btn:
            st.write("")  # spacing
            st.write("")  # spacing
            if existing_presets and st.button("Delete", type="secondary") and st.session_state.get("ci_preset_choice"):
                delete_preset(workspaces_root, st.session_state["ci_preset_choice"])
                if st.session_state.get("current_preset_name") == st.session_state["ci_preset_choice"]:
                    st.session_state["current_preset_name"] = None
                st.success(f"Deleted preset: {st.session_state['ci_preset_choice']}")
                st.rerun()

        # --- Reset to defaults button ---
        if st.button("Reset to defaults", type="secondary"):
            st.session_state["custom_instructions"] = default_instructions()
            st.session_state["current_preset_name"] = None
            st.rerun()

    st.divider()
    st.header("Axis & Units Configuration")

    # Initialize axis config in session state
    if "axis_config" not in st.session_state:
        # Try to load from active workspace if available from session state
        _active_rr = st.session_state.get("rr")
        if _active_rr and hasattr(_active_rr, 'run_dir') and _active_rr.run_dir:
            st.session_state["axis_config"] = load_axis_config(_active_rr.run_dir)
        else:
            st.session_state["axis_config"] = load_axis_config(workspaces_root)
        st.session_state["current_axis_preset_name"] = None

    _ax_cfg = st.session_state["axis_config"]

    # --- Load axis presets ---
    _active_rr = st.session_state.get("rr")
    workspace_for_axis = (_active_rr.run_dir if (_active_rr and hasattr(_active_rr, 'run_dir') and _active_rr.run_dir)
                          else workspaces_root)
    existing_axis_presets = list_axis_presets(workspace_for_axis)

    with st.expander("Load Axis Preset", expanded=False):
        if existing_axis_presets:
            col_ax_load_a, col_ax_load_b = st.columns([3, 1])
            with col_ax_load_a:
                axis_preset_choice = st.selectbox(
                    "Select axis preset to load",
                    options=[""] + existing_axis_presets,
                    key="axis_preset_choice",
                )
            with col_ax_load_b:
                if st.button("Load", type="secondary", key="load_axis_preset_btn") and axis_preset_choice:
                    loaded = load_axis_preset(workspace_for_axis, axis_preset_choice)
                    if loaded:
                        st.session_state["axis_config"] = loaded
                        st.session_state["current_axis_preset_name"] = axis_preset_choice
                        st.success(f"Loaded axis preset: {axis_preset_choice}")
                        st.rerun()
        else:
            st.info("No axis presets saved yet. Create your first preset below.")

        # Show currently loaded preset
        if st.session_state.get("current_axis_preset_name"):
            st.caption(f"Currently editing: **{st.session_state['current_axis_preset_name']}**")
        else:
            st.caption("Currently editing: **Unsaved changes (default)**")

    # --- Axis config editors in expanders ---
    with st.expander("Unit Aliases", expanded=False):
        st.caption("Map raw unit strings (e.g. `sec`, `hrs`) to their canonical form (e.g. `s`, `h`).")
        _ua = _ax_cfg.get("unit_aliases", {})
        _ua_rows = [{"raw": k, "canonical": v} for k, v in sorted(_ua.items())]
        _ua_df = pd.DataFrame(_ua_rows) if _ua_rows else pd.DataFrame(columns=["raw", "canonical"])
        _ua_edited = st.data_editor(
            _ua_df,
            num_rows="dynamic",
            use_container_width=True,
            key="sidebar_ax_unit_aliases_editor",
        )
        # Store edited data in session state
        if st.button("Apply changes", key="apply_unit_aliases"):
            new_ua = {}
            for _, row in _ua_edited.iterrows():
                raw = str(row.get("raw", "")).strip()
                can = str(row.get("canonical", "")).strip()
                if raw and can:
                    new_ua[raw] = can
            st.session_state["axis_config"]["unit_aliases"] = new_ua
            st.success("Applied unit alias changes")

    with st.expander("Unit Dimensions", expanded=False):
        st.caption("Map canonical unit strings to a dimension name (e.g. `s` -> `time`, `mS/cm` -> `conductivity`).")
        _ud = _ax_cfg.get("unit_dimensions", {})
        _ud_rows = [{"unit": k, "dimension": v} for k, v in sorted(_ud.items())]
        _ud_df = pd.DataFrame(_ud_rows) if _ud_rows else pd.DataFrame(columns=["unit", "dimension"])
        _ud_edited = st.data_editor(
            _ud_df,
            num_rows="dynamic",
            use_container_width=True,
            key="sidebar_ax_unit_dims_editor",
        )
        if st.button("Apply changes", key="apply_unit_dims"):
            new_ud = {}
            for _, row in _ud_edited.iterrows():
                unit = str(row.get("unit", "")).strip()
                dim = str(row.get("dimension", "")).strip()
                if unit and dim:
                    new_ud[unit] = dim
            st.session_state["axis_config"]["unit_dimensions"] = new_ud
            st.success("Applied unit dimension changes")

    with st.expander("Canonical Targets", expanded=False):
        st.caption("For each dimension, choose the canonical target unit that values are converted to.")
        _ct = _ax_cfg.get("canonical_targets", {})
        _ct_rows = [{"dimension": k, "target_unit": v} for k, v in sorted(_ct.items())]
        _ct_df = pd.DataFrame(_ct_rows) if _ct_rows else pd.DataFrame(columns=["dimension", "target_unit"])
        _ct_edited = st.data_editor(
            _ct_df,
            num_rows="dynamic",
            use_container_width=True,
            key="sidebar_ax_canon_targets_editor",
        )
        if st.button("Apply changes", key="apply_canon_targets"):
            new_ct = {}
            for _, row in _ct_edited.iterrows():
                dim = str(row.get("dimension", "")).strip()
                target = str(row.get("target_unit", "")).strip()
                if dim and target:
                    new_ct[dim] = target
            st.session_state["axis_config"]["canonical_targets"] = new_ct
            st.success("Applied canonical target changes")

    # --- Save axis preset ---
    with st.expander("Save Axis Preset", expanded=False):
        st.subheader("Save Preset")
        col_ax_name, col_ax_save, col_ax_del = st.columns([3, 1, 1])

        with col_ax_name:
            default_ax_name = st.session_state.get("current_axis_preset_name", "")
            axis_preset_name = st.text_input(
                "Axis preset name",
                value=default_ax_name,
                key="axis_new_preset_name",
                placeholder="Enter a name for this axis preset"
            )

        with col_ax_save:
            st.write("")  # spacing
            st.write("")  # spacing
            if st.button("Save", type="primary", key="save_axis_preset_btn") and axis_preset_name.strip():
                save_axis_preset(workspace_for_axis, axis_preset_name.strip(), st.session_state["axis_config"])
                # Also save to the workspace
                save_axis_config(workspace_for_axis, st.session_state["axis_config"])
                st.session_state["current_axis_preset_name"] = axis_preset_name.strip()
                st.success(f"Saved axis preset: {axis_preset_name.strip()}")
                st.rerun()

        with col_ax_del:
            st.write("")  # spacing
            st.write("")  # spacing
            if existing_axis_presets and st.button("Delete", type="secondary", key="delete_axis_preset_btn") and st.session_state.get("axis_preset_choice"):
                delete_axis_preset(workspace_for_axis, st.session_state["axis_preset_choice"])
                if st.session_state.get("current_axis_preset_name") == st.session_state["axis_preset_choice"]:
                    st.session_state["current_axis_preset_name"] = None
                st.success(f"Deleted axis preset: {st.session_state['axis_preset_choice']}")
                st.rerun()

        # --- Reset to defaults button ---
        if st.button("Reset axis config to defaults", type="secondary", key="reset_axis_btn"):
            st.session_state["axis_config"] = reset_axis_config(workspace_for_axis)
            st.session_state["current_axis_preset_name"] = None
            st.success("Reset axis configuration to defaults")
            st.rerun()

    st.divider()
    with st.expander("Load an existing run", expanded=False):

        # Choose which workspace to browse when loading runs (filter out axis_presets and instruction_presets)
        load_ws_options = [
            ws for ws in _list_subdirs(workspaces_root)
            if ws not in ("axis_presets", "instruction_presets")
        ]

        if load_ws_options:
            # ensure state is valid
            if st.session_state.get("load_workspace") not in load_ws_options:
                st.session_state["load_workspace"] = (
                    active_workspace_name if active_workspace_name in load_ws_options else load_ws_options[0]
                )

            load_workspace_name = st.selectbox(
                "Browse workspace",
                options=load_ws_options,
                key="load_workspace",   # <-- single source of truth
            )

            load_workspace_dir = _ensure_workspace(workspaces_root, load_workspace_name)

            existing_runs = sorted([p.name for p in load_workspace_dir.iterdir() if p.is_dir()], reverse=True)
            pick = st.selectbox("Run ID", options=[""] + existing_runs, key="load_run_id")

            if st.button("Load run") and pick:
                rr = discover_run(load_workspace_dir / pick)
                st.session_state["rr"] = rr
        else:
            st.info("No workspaces exist yet. Create one above, then run the pipeline.")

# ---------------- Run pipeline ----------------
st.subheader("Input DOIs")
doi_text = st.text_area("One DOI per line", height=140)

st.subheader("Run configuration")
default_run_id = st.session_state.get("pending_run_id", "")
run_id_input = st.text_input(
    "Run ID (optional)",
    value=default_run_id,
    help="Leave blank to auto-generate timestamp-based run ID",
)

st.session_state["pending_run_id"] = run_id_input


if st.button("Run pipeline", type="primary"):
    dois = [d.strip() for d in doi_text.splitlines() if d.strip()]
    st.session_state["doi_order"] = dois
    if not dois:
        st.error("Please paste at least one DOI.")
    else:
        with st.spinner("Running your pipeline end-to-end…"):
            custom_run_id = run_id_input.strip() or None

            # --- Apply general LLM configuration for this run ---
            provider = st.session_state.get("llm_provider", "Ollama")
            model = st.session_state.get("llm_model", "")

            if provider == "Ollama":
                os.environ["AEM_OCR_LLM_PROVIDER"] = "ollama"
                os.environ["AEM_OCR_LLM_MODEL"] = model
                os.environ["OLLAMA_HOST"] = st.session_state.get("ollama_host", _default_ollama_host())
                os.environ["OLLAMA_MODELS"] = st.session_state.get("ollama_models_dir", str(_default_ollama_models_dir()))
            else:
                os.environ["AEM_OCR_LLM_PROVIDER"] = "openai"
                os.environ["AEM_OCR_LLM_MODEL"] = model
                os.environ["OPENAI_BASE_URL"] = st.session_state.get("openai_base_url", "https://api.openai.com/v1")
                if st.session_state.get("openai_api_key"):
                    os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]

            # --- Apply data processing LLM configuration ---
            if st.session_state.get("use_separate_processing_llm"):
                p_provider = st.session_state.get("processing_llm_provider", "Chat GPT")
                p_model = st.session_state.get("processing_llm_model", "")
                if p_provider == "Ollama":
                    os.environ["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "ollama"
                    os.environ["AEM_OCR_PROCESSING_LLM_MODEL"] = p_model
                else:
                    os.environ["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "openai"
                    os.environ["AEM_OCR_PROCESSING_LLM_MODEL"] = p_model
                    p_base = st.session_state.get("processing_openai_base_url", "https://api.openai.com/v1")
                    p_key = st.session_state.get("processing_openai_api_key", "")
                    if p_base:
                        os.environ["AEM_OCR_PROCESSING_OPENAI_BASE_URL"] = p_base
                    if p_key:
                        os.environ["AEM_OCR_PROCESSING_OPENAI_API_KEY"] = p_key
            else:
                # Clear processing-specific env vars so processing uses the general LLM
                for k in ["AEM_OCR_PROCESSING_LLM_PROVIDER", "AEM_OCR_PROCESSING_LLM_MODEL",
                           "AEM_OCR_PROCESSING_OPENAI_BASE_URL", "AEM_OCR_PROCESSING_OPENAI_API_KEY"]:
                    os.environ.pop(k, None)

            if custom_run_id:
                target = run_root / custom_run_id
                if target.exists():
                    st.warning(f"Run directory already exists and may be overwritten: {target}")


            # --- Save custom instructions for this run ---
            _run_ci = st.session_state.get("custom_instructions", default_instructions())
            # Parse output_columns if still a string
            if isinstance(_run_ci.get("output_columns"), str):
                _run_ci["output_columns"] = [c.strip() for c in _run_ci["output_columns"].splitlines() if c.strip()]
            if isinstance(_run_ci.get("missing_value_defaults"), str):
                try:
                    _run_ci["missing_value_defaults"] = json.loads(_run_ci["missing_value_defaults"])
                except Exception:
                    pass

            # Debug: show what preset is being used
            preset_name = st.session_state.get("current_preset_name", "defaults")
            st.info(f"Using custom instructions: **{preset_name}**")
            with st.expander("View custom instructions being used", expanded=False):
                st.json(_run_ci)

            rr = run_full_pipeline(
                dois=dois,
                run_root=run_root,
                scripts_root=scripts_root,
                run_id=custom_run_id,
                run_extract_figures=run_extract_figures,
                run_extract_tables=run_extract_tables,
                run_filter_figures=run_filter_figures,
                run_filter_tables=run_filter_tables,
                run_plextract=run_plextract,
                run_complete_dataset=run_complete_dataset,
                run_chemical_composition=run_chemical_composition,
                exclude_full_text=exclude_full_text,
                custom_instructions=_run_ci,
            )

        st.session_state["rr"] = rr
        st.success(f"Done. Run ID: {rr.run_id}")
        _render_cost_summary(rr.run_dir)

rr: RunResult | None = st.session_state.get("rr")
if not rr:
    st.stop()

st.info(f"Run ID: **{rr.run_id}**")
st.info(f"Workspace: **{rr.run_dir}**")
_render_cost_summary(rr.run_dir)

# ---------------- DOI summary table + DOI selector ----------------
st.subheader("Processed DOIs")

# Let user decide whether to compute interventions (can be expensive)
colA, colB = st.columns([1, 3])
with colA:
    compute_interventions = st.checkbox("Include intervention counts", value=False)
with colB:
    if compute_interventions:
        st.caption("Intervention counts are cached per run directory once computed.")
    else:
        st.caption("Turn on to identify figures needing manual intervention.")

df_stats = build_doi_stats(rr, include_interventions=compute_interventions)

# Render as interactive table (row click -> selects DOI when supported)
df_view = df_stats.drop(columns=["_doi_safe"], errors="ignore")

# Streamlit row selection support varies by version; use feature detection.
df_selected_doi = None
try:
    sig = inspect.signature(st.dataframe)
    if "on_select" in sig.parameters and "selection_mode" in sig.parameters:
        evt = st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if evt and "selection" in evt and evt["selection"].get("rows"):
            i = evt["selection"]["rows"][0]
            df_selected_doi = df_stats.loc[i, "DOI"]
    else:
        st.dataframe(df_view, use_container_width=True, hide_index=True)
except Exception:
    st.dataframe(df_view, use_container_width=True, hide_index=True)

# Keep selected DOI in session_state so table + dropdown stay in sync
doi_options = [d.doi for d in rr.dois]

if not doi_options:
    st.warning("No DOIs found in the loaded run. The run may be empty or incomplete.")
    st.stop()

if "selected_doi" not in st.session_state:
    st.session_state["selected_doi"] = doi_options[0]

if df_selected_doi:
    st.session_state["selected_doi"] = df_selected_doi

selected_doi = st.selectbox(
    "Select DOI",
    options=doi_options,
    index=doi_options.index(st.session_state["selected_doi"]) if st.session_state["selected_doi"] in doi_options else 0,
)
st.session_state["selected_doi"] = selected_doi

doi_item = next((d for d in rr.dois if d.doi == selected_doi), None)

if doi_item is None:
    st.error(f"Could not find DOI '{selected_doi}' in the loaded run. The data may have changed since this session was saved.")
    st.stop()

# ---------------- Tabs (under DOI dropdown) ----------------
tab_text, tab_figs, tab_tables, tab_chem, tab_xy = st.tabs(
    ["Parsed text", "Figures", "Tables", "Chemical Composition", "Dataset"]
)

with tab_text:
    if doi_item.parsed_json_path and doi_item.parsed_json_path.exists():
        st.json(json.loads(doi_item.parsed_json_path.read_text()))
    else:
        st.warning("No parsed.json found for this DOI in the run workspace.")

with tab_figs:
    if not doi_item.figures:
        st.warning("No figures found.")
    for idx, f in enumerate(doi_item.figures):
        # Add divider between figures
        if idx > 0:
            st.divider()

        # Figure header and basic info (not in expander)
        st.markdown(f"### {f.id}")

        # ---- Per-figure variable setup ----
        _safe_doi_er = doi_to_safe(selected_doi)
        _fig_id_er = f.id
        _fig_stem_er = Path(f.id).stem
        _img_name_er = Path(f.id).name

        # Find source image
        _source_img_er = f.figure_path
        if _source_img_er is None or not _source_img_er.exists():
            _filtered_fig_root_er = rr.run_dir / "figures_tables" / "figures"
            from intervention import _try_find_source_image as _tfs_er
            _source_img_er = _tfs_er(_filtered_fig_root_er, _safe_doi_er, _fig_id_er)

        # Find plextract figure_dir; plextract output dirs include the file extension
        # (e.g. "Figure_9.jpg/") so we match by stem, not exact name.
        _plextract_fig_dir_er = None
        _latest_plextract_er = latest_plextract_run_dir(rr.run_dir / "plextract")
        if _latest_plextract_er:
            _plex_out_base_er = _latest_plextract_er / "output" / _safe_doi_er
            if _plex_out_base_er.exists():
                for _cand_er in _plex_out_base_er.iterdir():
                    if _cand_er.is_dir() and Path(_cand_er.name).stem == _fig_id_er:
                        _plextract_fig_dir_er = _cand_er
                        break

        # Per-figure intervention plextract workdir
        _per_fig_int_wd_er = rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions"
        _int_fig_dir_er = None
        if _per_fig_int_wd_er.exists():
            _lat_int_er = latest_plextract_run_dir(_per_fig_int_wd_er)
            if _lat_int_er:
                _int_out_base_er = _lat_int_er / "output" / _safe_doi_er
                if _int_out_base_er.exists():
                    for _cand2_er in _int_out_base_er.iterdir():
                        if _cand2_er.is_dir() and Path(_cand2_er.name).stem == _fig_id_er:
                            _int_fig_dir_er = _cand2_er
                            break

        # crops output directory
        _crops_out_dir_er = rr.run_dir / "interventions" / _safe_doi_er / _img_name_er

        # ---- Collect axis config subfig dirs ----
        # Match by exact name, by stem (handles extension-bearing plextract dirs),
        # or by subfigure prefix.
        _ax_subfig_options_er: list = []
        _ax_subfig_dirs_er: list = []
        for _ax_wd_er in [
            rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions",
            rr.run_dir / "interventions" / "interventions_plextract",
            rr.run_dir / "plextract",
        ]:
            if not _ax_wd_er.exists():
                continue
            _ax_lat_er = latest_plextract_run_dir(_ax_wd_er)
            if not _ax_lat_er:
                continue
            _ax_out_base_er = _ax_lat_er / "output" / _safe_doi_er
            if not _ax_out_base_er.exists():
                continue
            for _ax_d_er in sorted(_ax_out_base_er.iterdir()):
                if _ax_d_er.is_dir() and (
                    _ax_d_er.name == _fig_id_er or
                    Path(_ax_d_er.name).stem == _fig_id_er or
                    _ax_d_er.name.startswith(_fig_stem_er + "_subfig_")
                ):
                    if _ax_d_er.name not in _ax_subfig_options_er:
                        _ax_subfig_options_er.append(_ax_d_er.name)
                        _ax_subfig_dirs_er.append(_ax_d_er)

        # ========================
        # SECTION: Figure expander
        # ========================
        _fig_display_img = None
        _fig_display_img_caption = None
        if _source_img_er and _source_img_er.exists():
            _fig_display_img = _source_img_er
        elif f.figure_path and f.figure_path.exists():
            _fig_display_img = f.figure_path
            _fig_display_img_caption = str(f.figure_path)
        render_figure_image_dropdown(
            image=_fig_display_img,
            image_caption=_fig_display_img_caption,
            figure_caption=f.caption,
            title=f"Figure: {f.id}",
            key=f"fig_{_safe_doi_er}_{_fig_id_er}_img",
            expanded=True,
        )


        # ========================
        # SECTION: Cropping
        # ========================
        # NOTE: We deliberately avoid st.expander here. streamlit-drawable-canvas mounts its
        # Fabric.js canvas immediately when the page renders — even inside a *collapsed* expander.
        # Fabric.js needs a visible, non-zero-size container to load the background image correctly.
        # When mounted hidden it silently fails, producing a blank canvas that only "fixes" itself
        # after the user clicks "Clear drawn rectangles" (which changes the canvas key, forcing a
        # remount while the expander is open). The session_state toggle below ensures the canvas
        # is only ever mounted when it is actually visible.
        _show_interventions_key = f"er_show_interventions_{_safe_doi_er}_{_fig_id_er}"
        if _show_interventions_key not in st.session_state:
            st.session_state[_show_interventions_key] = False

        intervene_button = "Load Interventions" if st.session_state[_show_interventions_key] else "▶ Crop Figure"
        if st.button(intervene_button, key=f"er_intervene_toggle_{_safe_doi_er}_{_fig_id_er}"):
            st.session_state[_show_interventions_key] = not st.session_state[_show_interventions_key]
            st.rerun()

        if st.session_state[_show_interventions_key]:
            if not (_source_img_er and _source_img_er.exists()):
                st.warning("No source image found for this figure.")
            else:
                # ---- Load image ----
                _pil_img_er = Image.open(_source_img_er).convert("RGB")
                _orig_w_er, _orig_h_er = _pil_img_er.size

                # # ---- Display sizing ----
                _max_display_w_er = 1100
                _disp_w_er = min(_orig_w_er, _max_display_w_er)
                _disp_h_er = int(round(_orig_h_er * (_disp_w_er / _orig_w_er)))
                
                # IMPORTANT: streamlit-drawable-canvas draws correctly when the background image
                # is a *PIL Image* (RGB/RGBA) that matches the canvas width/height exactly.
                
                _bg_img_er = _pil_img_er.resize((_disp_w_er, _disp_h_er), Image.Resampling.LANCZOS).convert("RGBA")

                # ---- Controls ----
                st.write("**Draw crop rectangles directly on top of the image.**")

                _mode_key_er = f"er_canvas_mode_{_safe_doi_er}_{_fig_id_er}"
                _canvas_key_er = f"er_canvas_{_safe_doi_er}_{_fig_id_er}_{_disp_w_er}x{_disp_h_er}"

                _draw_mode_er = st.radio(
                    "Mode",
                    ["rect", "transform"],
                    format_func=lambda x: "Draw rectangles" if x == "rect" else "Move/resize rectangles",
                    key=_mode_key_er,
                    horizontal=True,
                )

                _btn_cols_er = st.columns([1, 1, 2])
                with _btn_cols_er[0]:
                    _clear_canvas_clicked = st.button("Clear drawn rectangles", key=f"er_clear_canvas_{_safe_doi_er}_{_fig_id_er}")
                with _btn_cols_er[1]:
                    _show_json = st.checkbox("Show raw canvas JSON", value=False, key=f"er_show_canvas_json_{_safe_doi_er}_{_fig_id_er}")

                # ---- Canvas version tracking (used to force remount on "Clear drawn rectangles") ----
                # streamlit-drawable-canvas stores state by key; bumping the version suffix forces remount.
                _cv_ver_key = f"er_canvas_ver_{_safe_doi_er}_{_fig_id_er}"
                if _cv_ver_key not in st.session_state:
                    st.session_state[_cv_ver_key] = 0
                if _clear_canvas_clicked:
                    st.session_state[_cv_ver_key] += 1
                    st.rerun()

                _canvas_key_er = f"{_canvas_key_er}_v{st.session_state[_cv_ver_key]}"

                # ---- Canvas ----
                # NOTE: The most common reason you "can't crop anything" is the canvas is in a mode
                # that doesn't create rectangles, or the object parsing ignores transforms.
                # We set drawing_mode to rect/transform and parse *all* rect transforms properly.
                _canvas_er = st_canvas(
                    background_image=_bg_img_er,
                    drawing_mode=_draw_mode_er,          # "rect" draws; "transform" edits
                    update_streamlit=True,
                    height=_disp_h_er,
                    width=_disp_w_er,
                    stroke_width=3,
                    stroke_color="#ff0000",
                    fill_color="rgba(255, 0, 0, 0.20)",
                    key=_canvas_key_er,
                )

                if _show_json and _canvas_er.json_data is not None:
                    st.code(json.dumps(_canvas_er.json_data, indent=2), language="json")

                # ---- Extract rectangles from canvas JSON ----
                # Fabric.js rect coords are in canvas space; need to account for:
                # - left/top
                # - width/height
                # - scaleX/scaleY
                # - (optional) angle: we will convert rotated rects to an axis-aligned bounding box
                def _rect_to_bbox_canvas(obj: dict):
                    # base dims
                    left = float(obj.get("left", 0.0))
                    top = float(obj.get("top", 0.0))
                    w = float(obj.get("width", 0.0))
                    h = float(obj.get("height", 0.0))
                    sx = float(obj.get("scaleX", 1.0))
                    sy = float(obj.get("scaleY", 1.0))
                    angle = float(obj.get("angle", 0.0))

                    # If not rotated, easy:
                    if abs(angle) < 1e-6:
                        w2 = w * sx
                        h2 = h * sy
                        x1 = left
                        y1 = top
                        x2 = left + w2
                        y2 = top + h2
                        return x1, y1, x2, y2

                    # Rotated: compute axis-aligned bbox of the rotated rectangle
                    import math
                    w2 = w * sx
                    h2 = h * sy

                    # Fabric rect is defined with origin at top-left by default in this canvas,
                    # and rotation happens around its center.
                    cx = left + w2 / 2.0
                    cy = top + h2 / 2.0

                    theta = math.radians(angle)
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)

                    # corners relative to center
                    dx = w2 / 2.0
                    dy = h2 / 2.0
                    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]

                    xs, ys = [], []
                    for (px, py) in corners:
                        rx = cos_t * px - sin_t * py
                        ry = sin_t * px + cos_t * py
                        xs.append(cx + rx)
                        ys.append(cy + ry)

                    return min(xs), min(ys), max(xs), max(ys)

                _scale_x_er = _orig_w_er / _disp_w_er
                _scale_y_er = _orig_h_er / _disp_h_er

                _rects_er = []
                _objs = (_canvas_er.json_data or {}).get("objects", []) if _canvas_er is not None else []
                for _obj_er in _objs:
                    if _obj_er.get("type") != "rect":
                        continue
                    x1c, y1c, x2c, y2c = _rect_to_bbox_canvas(_obj_er)

                    # clamp to canvas bounds
                    x1c = max(0.0, min(float(_disp_w_er), x1c))
                    x2c = max(0.0, min(float(_disp_w_er), x2c))
                    y1c = max(0.0, min(float(_disp_h_er), y1c))
                    y2c = max(0.0, min(float(_disp_h_er), y2c))

                    # ensure ordering
                    if x2c <= x1c or y2c <= y1c:
                        continue

                    # scale back to original image coords
                    x1 = int(round(x1c * _scale_x_er))
                    y1 = int(round(y1c * _scale_y_er))
                    x2 = int(round(x2c * _scale_x_er))
                    y2 = int(round(y2c * _scale_y_er))

                    # clamp to original bounds
                    x1 = max(0, min(_orig_w_er - 1, x1))
                    x2 = max(1, min(_orig_w_er, x2))
                    y1 = max(0, min(_orig_h_er - 1, y1))
                    y2 = max(1, min(_orig_h_er, y2))

                    if x2 > x1 and y2 > y1:
                        _rects_er.append((x1, y1, x2, y2))

                # de-duplicate exact duplicates (can happen when editing)
                _rects_er = list(dict.fromkeys(_rects_er))

                st.write(f"Rectangles detected: **{len(_rects_er)}**")

                # ---- Save crops ----
                _cc1_er, _cc2_er = st.columns(2)
                with _cc1_er:
                    if st.button(
                        "Save crops",
                        disabled=len(_rects_er) == 0,
                        key=f"er_save_crops_{_safe_doi_er}_{_fig_id_er}",
                    ):
                        _saved_er = save_crops(_source_img_er, _rects_er, _crops_out_dir_er, prefix=_fig_stem_er)
                        st.success(f"Saved {len(_saved_er)} crops to `{_crops_out_dir_er}`")

                        if _saved_er:
                            _sc = st.columns(min(len(_saved_er), 4))
                            for _si, _p in enumerate(_saved_er):
                                with _sc[_si % len(_sc)]:
                                    st.image(str(_p), width=200)
                                    st.caption(_p.name)

                # ---- Existing crops ----
                _existing_crops_er = sorted(_crops_out_dir_er.glob(f"{_fig_stem_er}_subfig_*.png")) if _crops_out_dir_er.exists() else []
                if _existing_crops_er:
                    st.write(f"**Existing crops:** {len(_existing_crops_er)}")
                    _crop_cols_er = st.columns(min(len(_existing_crops_er), 6))
                    for _ci, _cp in enumerate(_existing_crops_er):
                        with _crop_cols_er[_ci % len(_crop_cols_er)]:
                            st.image(str(_cp), width=150)
                            st.caption(_cp.name)

                    _cl1_er, _cl2_er = st.columns(2)
                    with _cl1_er:
                        if st.button("Clear all crops", key=f"er_clear_crops_{_safe_doi_er}_{_fig_id_er}"):
                            for _cp in _existing_crops_er:
                                _cp.unlink(missing_ok=True)
                            st.success("Cleared existing crops.")
                            st.rerun()
                else:
                    st.info("No crops saved yet. Draw rectangles above to create crops.")

        # ========================
        # SECTION: Axis Configuration
        # ========================
        # Determine chartdete dir and source image for calibration
        _ax_chartdete_er: Path | None = None
        _ax_sel_name_er = "figure"
        if _ax_subfig_dirs_er:
            _ax_sel_dir_er = _ax_subfig_dirs_er[0]
            _ax_sel_name_er = _ax_subfig_options_er[0] if _ax_subfig_options_er else "figure"
            _ax_chartdete_er = _ax_sel_dir_er / "chartdete"
        _calib_img_path_er: Path | None = None
        if _source_img_er and Path(_source_img_er).exists():
            _calib_img_path_er = Path(_source_img_er)
        if _calib_img_path_er is None:
            _base_input = os.environ.get("PLEXTRACT_BASE_INPUT", "")
            if _base_input:
                _base_input_p = Path(_base_input)
                _tokens = [str(_fig_id_er).lower(), str(_fig_stem_er).lower(), str(_img_name_er).lower()]
                _tokens = [t for t in _tokens if t]
                def _score(p: Path) -> int:
                    s = p.as_posix().lower()
                    score = sum(10 for t in _tokens if t in s)
                    score -= len(p.parts)
                    return score
                _all_imgs: list[Path] = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                    _all_imgs.extend(_base_input_p.rglob(ext))
                _all_imgs = sorted(set(_all_imgs), key=_score, reverse=True)
                if _all_imgs:
                    _calib_img_path_er = _all_imgs[0]
        render_axis_configuration_dropdown(
            chartdete_dir=_ax_chartdete_er,
            source_img=_calib_img_path_er,
            key_prefix=f"er_{_safe_doi_er}_{_ax_sel_name_er}",
        )

        # ========================
        # SECTION: Series Management
        # ========================
        _sm_int_wd_er = rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions"
        _sm_fig_dir_er = _plextract_fig_dir_er
        _sm_fig_id_er = _fig_id_er
        _sm_source_img_er = _source_img_er
        _sm_workdirs_to_search_er = [
            _sm_int_wd_er,
            rr.run_dir / "interventions" / "interventions_plextract",
            rr.run_dir / "plextract",
        ]
        _sm_lf_dir_er = get_lineformer_dir(_sm_fig_dir_er) if _sm_fig_dir_er else None
        _sm_lf_override_dir_er = override_lineformer_dir(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
        render_series_management_dropdown(
            run_dir=rr.run_dir,
            safe_doi=_safe_doi_er,
            fig_id=_sm_fig_id_er,
            fig_dir=_sm_fig_dir_er,
            lf_dir=_sm_lf_dir_er,
            lf_override_dir=_sm_lf_override_dir_er,
            source_img=_sm_source_img_er,
            key_prefix=f"er_{_safe_doi_er}_{_sm_fig_id_er}",
        )


        # ========================
        # SECTION: OCR + LLM Data
        # ========================
        final_csv = None
        if isinstance(getattr(f, "extra", None), dict):
            final_csv = f.extra.get("final_dataset_csv")
        _final_csv_path = Path(final_csv) if final_csv else None
        _figure_dir_for_overrides = _final_csv_path.parent.parent if _final_csv_path and _final_csv_path.exists() else None
        render_ocr_dropdown(
            csv_path=_final_csv_path,
            key_prefix=f"fig_{selected_doi}_{f.id}",
            axis_override_dir=_figure_dir_for_overrides,
        )

        emit_rows_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "figure", f.id)
        render_llm_dropdown(
            run_dir=rr.run_dir,
            emit_rows_path=emit_rows_path,
            key_prefix=f"fig_{selected_doi}_{f.id}",
            file_name_prefix=f.id,
        )
        safe_doi = doi_to_safe(selected_doi)
        # ========================
        # SECTION: Rerun Data Extraction
        # ========================
        _is_subfig_sel_er = _sm_fig_id_er != _fig_id_er
        _int_crops_er = sorted(_crops_out_dir_er.glob(f"{_fig_stem_er}_subfig_*.png")) if _crops_out_dir_er.exists() else []
        # if _is_subfig_sel_er:
        #     st.caption(f"Will run on selected subfigure: **{_sm_fig_id_er}**")
        if _int_crops_er:
            st.caption(f"Will run on **{len(_int_crops_er)} cropped image(s)**.")
        elif _source_img_er and _source_img_er.exists():
            st.caption("Will run on the original figure image.")

        if st.button("Rerun OCR data extraction", type="primary", key=f"er_rerun_from_series_{_safe_doi_er}_{_sm_fig_id_er}"):
            try:
                _rerun_images_er: list = []
                _rerun_lf_er: dict = {}

                with st.spinner("Applying mask overrides..."):
                    # Writes prepared masks + catalog + overrides into the
                    # intervention dir only; the original plextract output
                    # is never modified (item_figure_dir is read-only).
                    apply_overrides_to_plextract(_sm_fig_dir_er, rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                # Always source lineformer from the intervention override dir
                # so the original plextract results remain untouched.
                _lf_dir_rerun_er = _sm_lf_override_dir_er

                if _is_subfig_sel_er:
                    _rerun_images_er = [_sm_source_img_er]
                    if _lf_dir_rerun_er and _lf_dir_rerun_er.exists():
                        _rerun_lf_er[_sm_source_img_er.name] = _lf_dir_rerun_er
                elif _int_crops_er:
                    _rerun_images_er = list(_int_crops_er)
                    for _crop_er in _int_crops_er:
                        _crop_lf_er = None
                        # Search all workdirs for this crop's lineformer dir
                        for _sm_wd_r_er in _sm_workdirs_to_search_er:
                            if not _sm_wd_r_er.exists():
                                continue
                            _lat2_er = latest_plextract_run_dir(_sm_wd_r_er)
                            if _lat2_er:
                                _cand_lf_er = _lat2_er / "output" / _safe_doi_er / _crop_er.name / "lineformer"
                                if _cand_lf_er.exists():
                                    _crop_lf_er = _cand_lf_er
                                    break
                        _lf_to_use_er = _crop_lf_er or (_lf_dir_rerun_er if _lf_dir_rerun_er and _lf_dir_rerun_er.exists() else None)
                        if _lf_to_use_er:
                            _rerun_lf_er[_crop_er.name] = _lf_to_use_er
                else:
                    if _source_img_er and _source_img_er.exists():
                        _rerun_images_er = [_source_img_er]
                        if _lf_dir_rerun_er and _lf_dir_rerun_er.exists():
                            _rerun_lf_er[_source_img_er.name] = _lf_dir_rerun_er
                with st.spinner("Re-running data extraction..."):
                    run_plextract_intervention_on_crops(
                        scripts_root=scripts_root, run_dir=rr.run_dir,
                        safe_doi=_safe_doi_er, image_paths=_rerun_images_er,
                        override_lineformer_dirs=_rerun_lf_er,
                        figure_id=_fig_id_er,
                    )
                st.success("Data extraction complete!")
                st.rerun()
            except Exception as _e2_er:
                st.error(f"Failed: {_e2_er}")

        # ========================
        # SECTION: Rerun LLM Dataset Generation
        # ========================

        # Find available figures/subfigures for LLM rerun
        _llm_fig_options_er: list = []
        _llm_fig_dirs_er: list = []
        for _llm_wd_er in [
            # rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions",
            # rr.run_dir / "interventions" / "interventions_plextract",
            Path(f"{rr.run_dir}/plextract/results/0/output/{_safe_doi_er}/{_fig_id_er}.jpg"),
            Path(f"{rr.run_dir}/plextract/results/0/output/{_safe_doi_er}/{_fig_id_er}.png")
        ]:
            if not _llm_wd_er.exists():
                continue
            if _llm_wd_er.is_dir() and (_fig_stem_er == _fig_id_er or
                _llm_wd_er.name.startswith(_fig_stem_er + "_subfig_")):
                    if _llm_wd_er.name not in _llm_fig_options_er:
                        _llm_fig_options_er.append(_llm_wd_er.name)
                        _llm_fig_dirs_er.append(_llm_wd_er)

        if not _llm_fig_dirs_er:
            st.info("No plextract output found. Run plextract extraction first.")
        else:
            if len(_llm_fig_dirs_er) > 1:
                _llm_sel_er = st.selectbox(
                    "Select figure/subfigure for LLM rerun",
                    _llm_fig_options_er,
                    key=f"er_llm_sel_{_safe_doi_er}_{_fig_id_er}"
                )
                _llm_sel_dir_er = _llm_fig_dirs_er[_llm_fig_options_er.index(_llm_sel_er)]
            else:
                _llm_sel_er = _llm_fig_options_er[0]
                _llm_sel_dir_er = _llm_fig_dirs_er[0]

            if st.button("Rerun LLM dataset generation", type="primary", key=f"er_rerun_llm_{_safe_doi_er}_{_fig_id_er}"):
                _reparsed_dir_er = rr.run_dir / "reparsed"
                _index_csv_er = _reparsed_dir_er / "index.csv"
                if not _index_csv_er.exists():
                    _err_msg_er = f"**Required file not found:** `{_index_csv_er.relative_to(rr.run_dir)}`\n\n"
                    if not _reparsed_dir_er.exists():
                        _err_msg_er += "The `reparsed/` directory doesn't exist. "
                    else:
                        _err_msg_er += "The `reparsed/` directory exists but `index.csv` is missing. "
                    _err_msg_er += "This file is created during the 'Download DOIs' stage."
                    st.error(_err_msg_er)
                else:
                    try:
                        # Delete any existing cache files for this figure so the
                        # LLM is always re-called (avoids stale cache on re-run).
                        _row_cache_dir_er = rr.run_dir / "complete_dataset" / "row_cache"
                        if _row_cache_dir_er.exists():
                            import re as _re_er
                            _safe_fn_er = lambda s: _re_er.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:180]
                            _cache_pfx_er = _safe_fn_er(f"emit_rows::figure::{selected_doi}::{_llm_sel_er}::")
                            for _cf_er in list(_row_cache_dir_er.iterdir()):
                                if _cf_er.suffix == ".json" and (
                                    _cf_er.stem.startswith(_cache_pfx_er) or
                                    _cf_er.name.startswith(_cache_pfx_er)
                                ):
                                    _cf_er.unlink(missing_ok=True)

                        import tempfile as _tmpmod_er
                        with _tmpmod_er.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as _tmp_er:
                            _tmp_er.write(f"DOI\n{selected_doi}\n")
                            _tmp_path_er = _tmp_er.name

                        _complete_script_er = Path(__file__).parent.parent / "complete_dataset" / "complete_dataset.py"
                        _efr_er = _llm_sel_dir_er.parent.parent  # .../output
                        _cmd_er = [
                            "python", str(_complete_script_er),
                            "--file_path", _tmp_path_er,
                            "--output_dir", str(rr.run_dir / "complete_dataset"),
                            "--index_csv_path", str(_index_csv_er),
                            "--figures_tables_dir", str(rr.run_dir / "figures_tables"),
                            "--extracted_figures_root", str(_efr_er),
                            "--target_figure", _llm_sel_er,
                        ]

                        _env_er = os.environ.copy()
                        _provider_er = st.session_state.get("llm_provider", "Ollama")
                        _model_er = st.session_state.get("llm_model", "")
                        if _provider_er == "Ollama":
                            _env_er["AEM_OCR_LLM_PROVIDER"] = "ollama"
                            _env_er["AEM_OCR_LLM_MODEL"] = _model_er
                            _env_er["OLLAMA_HOST"] = st.session_state.get("ollama_host", "http://127.0.0.1:11434")
                            _om_dir_er = st.session_state.get("ollama_models_dir", "")
                            if _om_dir_er:
                                _env_er["OLLAMA_MODELS"] = _om_dir_er
                        else:
                            _env_er["AEM_OCR_LLM_PROVIDER"] = "openai"
                            _env_er["AEM_OCR_LLM_MODEL"] = _model_er
                            _env_er["OPENAI_BASE_URL"] = st.session_state.get("openai_base_url", "https://api.openai.com/v1")
                            if st.session_state.get("openai_api_key"):
                                _env_er["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
                        if st.session_state.get("use_separate_processing_llm"):
                            _pp_er = st.session_state.get("processing_llm_provider", "Chat GPT")
                            _pm_er = st.session_state.get("processing_llm_model", "")
                            if _pp_er == "Ollama":
                                _env_er["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "ollama"
                                _env_er["AEM_OCR_PROCESSING_LLM_MODEL"] = _pm_er
                            else:
                                _env_er["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "openai"
                                _env_er["AEM_OCR_PROCESSING_LLM_MODEL"] = _pm_er
                                _pb_er = st.session_state.get("processing_openai_base_url", "https://api.openai.com/v1")
                                _pk_er = st.session_state.get("processing_openai_api_key", "")
                                if _pb_er:
                                    _env_er["AEM_OCR_PROCESSING_OPENAI_BASE_URL"] = _pb_er
                                if _pk_er:
                                    _env_er["AEM_OCR_PROCESSING_OPENAI_API_KEY"] = _pk_er
                        else:
                            for _k_er in ["AEM_OCR_PROCESSING_LLM_PROVIDER", "AEM_OCR_PROCESSING_LLM_MODEL",
                                            "AEM_OCR_PROCESSING_OPENAI_BASE_URL", "AEM_OCR_PROCESSING_OPENAI_API_KEY"]:
                                _env_er.pop(_k_er, None)

                        if not _model_er and not st.session_state.get("processing_llm_model"):
                            st.error("No LLM model configured. Configure LLM settings in the sidebar first.")
                            Path(_tmp_path_er).unlink(missing_ok=True)
                        else:
                            _ci_path_er = rr.run_dir.parent / "custom_instructions" / "default.json"
                            if _ci_path_er.exists():
                                _env_er["AEM_OCR_CUSTOM_INSTRUCTIONS"] = str(_ci_path_er)
                            _env_er["AEM_OCR_RUN_ID"] = rr.run_id
                            _env_er["AEM_OCR_RUN_DIR"] = str(rr.run_dir)

                            with st.spinner(f"Running LLM dataset generation for {_llm_sel_er}..."):
                                _result_er = subprocess.run(_cmd_er, stdout=None, stderr=None, text=True, timeout=300, env=_env_er)
                            Path(_tmp_path_er).unlink(missing_ok=True)
                            if _result_er.returncode == 0:
                                st.success(f"LLM dataset generation complete for {_llm_sel_er}!")
                                st.rerun()
                            else:
                                st.error(f"LLM dataset generation failed:\n```\n{_result_er.stderr}\n```")
                    except subprocess.TimeoutExpired:
                        st.error("LLM dataset generation timed out (>5 minutes)")
                    except Exception as _e4_er:
                        st.error(f"Error: {_e4_er}")

        # ========================
        # SECTION: Intervention Results
        # Show new OCR and LLM data once a per-figure plextract_interventions
        # run exists (i.e. the user has clicked "Rerun data extraction").
        # ========================
        fig_stem = Path(f.id).stem
        _int_doi_dir = rr.run_dir / "interventions" / safe_doi
        _per_fig_plextract: Path | None = None
        if _int_doi_dir.exists():
            for _int_fig_d in _int_doi_dir.iterdir():
                if _int_fig_d.is_dir() and Path(_int_fig_d.name).stem == fig_stem:
                    _candidate = _int_fig_d / "plextract_interventions"
                    if _candidate.exists():
                        _per_fig_plextract = _candidate
                    break

        if _per_fig_plextract and _per_fig_plextract.exists():
            _int_latest_run = latest_plextract_run_dir(_per_fig_plextract)
            _int_out_base = (
                (_int_latest_run / "output" / safe_doi) if _int_latest_run else None
            )
            def _stem(s: str) -> str:
                return Path(s).stem
            if _int_out_base and _int_out_base.exists():
                for _idir in sorted(_int_out_base.iterdir()):
                    if not _idir.is_dir():
                        continue
                    # show only the parent figure output (exclude *_subfig_*)
                    if _idir.name.startswith(_stem(_fig_id_er) + "_subfig_"):
                        continue
                    if _stem(_idir.name) != _stem(_fig_id_er):
                        continue                    
                    if _idir.is_dir():
                        render_ocr_dropdown(
                            csv_path=_idir / "final" / "final_dataset.csv",
                            key_prefix=f"int_ocr_{safe_doi}_{f.id}_{_idir.name}",
                            axis_override_dir=_idir,
                            title=f"New OCR-Extracted Data: {_idir.name}",
                        )
                        _int_emit_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "figure", _idir.name)
                        render_llm_dropdown(
                            run_dir=rr.run_dir,
                            emit_rows_path=_int_emit_path,
                            key_prefix=f"int_llm_{safe_doi}_{f.id}_{_idir.name}",
                            file_name_prefix=_idir.name,
                            title=f"New LLM-Generated Dataset: {_idir.name}",
                        )
        # ========================
        # SECTION: Display metadata
        # ========================
        render_metadata_dropdown(metadata_path=f.metadata_path)
        _subfig_map: dict[str, Path] = {} 
        _workdirs_priority: list[Path] = []


        if _per_fig_plextract is not None:
            _workdirs_priority.append(_per_fig_plextract)

        # 2. Legacy shared interventions_plextract (backward compatibility).
        _workdirs_priority.append(rr.run_dir / "interventions" / "interventions_plextract")

        # 3. Original plextract output.
        _workdirs_priority.append(rr.run_dir / "plextract")

        for _wd in _workdirs_priority:
            if not _wd or not _wd.exists():
                continue
            subfigs_here = collect_subfig_dirs_across_runs(_wd, safe_doi, fig_stem)
            # priority order: only fill missing names so higher-priority workdir wins
            for name, path in subfigs_here.items():
                if name not in _subfig_map:
                    _subfig_map[name] = path

        subfig_dirs = sorted(_subfig_map.values(), key=lambda d: d.name)

        # Set latest_manual / out_base for use by the data completion button and
        # the original-image lookup below.  Prefer the per-figure workdir, then
        # fall back to the shared interventions_plextract, then plain plextract.
        # latest_manual = None
        # out_base = None
        # for _wd in [
        #     _per_fig_plextract,
        #     rr.run_dir / "interventions" / "interventions_plextract",
        #     rr.run_dir / "plextract",
        # ]:
        #     if _wd is None or not _wd.exists():
        #         continue
        #     _lat = latest_plextract_run_dir(_wd)
        #     if _lat:
        #         _ob = _lat / "output" / safe_doi
        #         if _ob.exists():
        #             latest_manual = _lat
        #             out_base = _ob
        #             break

        #### SUBFIGURE SECTION #####
        if subfig_dirs:
                # Header for subfigures section
                st.markdown("**Cropped Subfigures**")
                for subfig_dir in subfig_dirs:
                    st.markdown(f"### {subfig_dir.name}")
                    # Resolve source image before expander so it's available for rerun buttons
                    # _subfig_source_img = None
                    # _subfig_image_caption = None
                    _subfig_source_img, _subfig_image_caption = resolve_source_img_for_subfig_dir(subfig_dir, safe_doi)
                    # if latest_manual:
                    #     input_dir = latest_manual / "input" / safe_doi
                    #     original_img = input_dir / subfig_dir.name
                    #     if original_img.exists():
                    #         _subfig_source_img = original_img
                    #         _subfig_image_caption = "Cropped image"
                    # if _subfig_source_img is None:
                    #     _crops_dir = rr.run_dir / "interventions" / safe_doi
                    #     for _fig_d in (_crops_dir.iterdir() if _crops_dir.exists() else []):
                    #         _candidate = _fig_d / subfig_dir.name
                    #         if _candidate.exists():
                    #             _subfig_source_img = _candidate
                    #             _subfig_image_caption = "Cropped image"
                    #             break
                    if _subfig_source_img is None:
                        _subfig_final_dir = subfig_dir / "final"
                        if _subfig_final_dir.exists():
                            _subfig_final_img = _subfig_final_dir / "final_image.png"
                            if _subfig_final_img.exists():
                                _subfig_source_img = _subfig_final_img
                                _subfig_image_caption = "Recreated image (crop not found)"
                    render_figure_image_dropdown(
                        image=_subfig_source_img,
                        image_caption=_subfig_image_caption,
                        title=subfig_dir.name,
                        key=f"subfig_{safe_doi}_{subfig_dir.name}_img",
                        expanded=True,
                    )
                    _subfig_lf_dir = get_lineformer_dir(subfig_dir)
                    _subfig_override_lf = override_lineformer_dir(rr.run_dir, safe_doi, subfig_dir.name)
                    render_axis_configuration_dropdown(
                        chartdete_dir=subfig_dir / "chartdete",
                        source_img=_subfig_source_img,
                        key_prefix=f"subfig_{safe_doi}_{subfig_dir.name}",
                    )
                    render_series_management_dropdown(
                        run_dir=rr.run_dir,
                        safe_doi=safe_doi,
                        fig_id=subfig_dir.name,
                        fig_dir=subfig_dir,
                        lf_dir=_subfig_lf_dir,
                        lf_override_dir=_subfig_override_lf,
                        source_img=_subfig_source_img,
                        key_prefix=f"subfig_{safe_doi}_{subfig_dir.name}",
                    )

                    final_dir = subfig_dir / "final"
                    final_csv = final_dir / "final_dataset.csv"
                    render_ocr_dropdown(
                        csv_path=final_csv if final_csv.exists() else None,
                        key_prefix=f"subfig_{selected_doi}_{subfig_dir.name}",
                        axis_override_dir=subfig_dir,
                    )

                    emit_rows_path_subfig = find_emit_rows_for_source(
                        rr.run_dir, selected_doi, "figure", subfig_dir.name
                    )
                    render_llm_dropdown(
                        run_dir=rr.run_dir,
                        emit_rows_path=emit_rows_path_subfig,
                        key_prefix=f"subfig_{selected_doi}_{subfig_dir.name}",
                        file_name_prefix=subfig_dir.name,
                    )

                    _subfig_metadata = final_dir / "final_metadata.json"
                    render_metadata_dropdown(
                        metadata_path=_subfig_metadata if _subfig_metadata.exists() else None,
                    )

                    # ========================
                    # SECTION: Rerun buttons (at subfigure level, matching figure structure)
                    # ========================
                    if st.button("Rerun OCR data extraction", type="primary", key=f"subfig_rerun_ocr_{safe_doi}_{subfig_dir.name}"):
                        try:
                            with st.spinner("Applying mask overrides..."):
                                apply_overrides_to_plextract(subfig_dir, rr.run_dir, safe_doi, subfig_dir.name)

                            if not (_subfig_source_img and _subfig_source_img.exists()):
                                st.error("Could not locate subfigure source image for rerun.")
                            else:
                                _subfig_lf_override = override_lineformer_dir(rr.run_dir, safe_doi, subfig_dir.name)
                                _lf_map = {}
                                if _subfig_lf_override.exists():
                                    _lf_map[_subfig_source_img.name] = _subfig_lf_override
                                _cd_map = {}
                                _subfig_cd_dir = subfig_dir / "chartdete"
                                if _subfig_cd_dir.exists():
                                    _cd_map[_subfig_source_img.name] = _subfig_cd_dir
                                target_id = subfig_dir.name
                                with st.spinner("Re-running data extraction..."):
                                    run_plextract_intervention_on_crops(
                                        scripts_root=scripts_root,
                                        run_dir=rr.run_dir,
                                        safe_doi=safe_doi,
                                        image_paths=[_subfig_source_img],
                                        override_lineformer_dirs=_lf_map,
                                        figure_id=target_id,
                                        override_chartdete_dirs=_cd_map,
                                    )
                                st.success(f"Data extraction complete for {subfig_dir.name}!")
                                st.rerun()
                        except Exception as _subfig_ocr_e:
                            st.error(f"Failed: {_subfig_ocr_e}")

                    if st.button("Rerun LLM dataset generation", type="primary", key=f"subfig_rerun_llm_{safe_doi}_{subfig_dir.name}"):
                        _reparsed_dir_sf = rr.run_dir / "reparsed"
                        _index_csv_sf = _reparsed_dir_sf / "index.csv"
                        if not _index_csv_sf.exists():
                            st.error(f"Required file not found: `{_index_csv_sf.relative_to(rr.run_dir)}`")
                        else:
                            try:
                                _row_cache_dir_sf = rr.run_dir / "complete_dataset" / "row_cache"
                                if _row_cache_dir_sf.exists():
                                    import re as _re_sf
                                    _safe_fn_sf = lambda s: _re_sf.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:180]
                                    _cache_pfx_sf = _safe_fn_sf(f"emit_rows::figure::{selected_doi}::{subfig_dir.name}::")
                                    for _cf_sf in list(_row_cache_dir_sf.iterdir()):
                                        if _cf_sf.suffix == ".json" and (_cf_sf.stem.startswith(_cache_pfx_sf) or _cf_sf.name.startswith(_cache_pfx_sf)):
                                            _cf_sf.unlink(missing_ok=True)

                                import tempfile as _tmpmod_sf
                                with _tmpmod_sf.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as _tmp_sf:
                                    _tmp_sf.write(f"DOI\n{selected_doi}\n")
                                    _tmp_path_sf = _tmp_sf.name

                                _complete_script_sf = Path(__file__).parent.parent / "complete_dataset" / "complete_dataset.py"
                                _cmd_sf = [
                                    "python", str(_complete_script_sf),
                                    "--file_path", _tmp_path_sf,
                                    "--output_dir", str(rr.run_dir / "complete_dataset"),
                                    "--index_csv_path", str(_index_csv_sf),
                                    "--figures_tables_dir", str(rr.run_dir / "figures_tables"),
                                    "--extracted_figures_root", str(subfig_dir.parent.parent),
                                    "--target_figure", subfig_dir.name,
                                ]

                                _env_sf = os.environ.copy()
                                _provider_sf = st.session_state.get("llm_provider", "Ollama")
                                _model_sf = st.session_state.get("llm_model", "")
                                if _provider_sf == "Ollama":
                                    _env_sf["AEM_OCR_LLM_PROVIDER"] = "ollama"
                                    _env_sf["AEM_OCR_LLM_MODEL"] = _model_sf
                                    _env_sf["OLLAMA_HOST"] = st.session_state.get("ollama_host", "http://127.0.0.1:11434")
                                else:
                                    _env_sf["AEM_OCR_LLM_PROVIDER"] = "openai"
                                    _env_sf["AEM_OCR_LLM_MODEL"] = _model_sf
                                    _env_sf["OPENAI_BASE_URL"] = st.session_state.get("openai_base_url", "https://api.openai.com/v1")
                                    if st.session_state.get("openai_api_key"):
                                        _env_sf["OPENAI_API_KEY"] = st.session_state["openai_api_key"]

                                _env_sf["AEM_OCR_RUN_ID"] = rr.run_id
                                _env_sf["AEM_OCR_RUN_DIR"] = str(rr.run_dir)

                                with st.spinner(f"Running LLM dataset generation for {subfig_dir.name}..."):
                                    _result_sf = subprocess.run(_cmd_sf, stdout=None, stderr=None, text=True, timeout=300, env=_env_sf)
                                Path(_tmp_path_sf).unlink(missing_ok=True)
                                if _result_sf.returncode == 0:
                                    st.success(f"LLM dataset generation complete for {subfig_dir.name}!")
                                    st.rerun()
                                else:
                                    st.error(f"LLM dataset generation failed:\n```\n{_result_sf.stderr}\n```")
                            except subprocess.TimeoutExpired:
                                st.error("LLM dataset generation timed out (>5 minutes)")
                            except Exception as _subfig_llm_e:
                                st.error(f"Error: {_subfig_llm_e}")

                    # subfigure-specific intervention workdir
                    _subfig_int_wd = rr.run_dir / "interventions" / safe_doi / subfig_dir.name / "plextract_interventions"
                    if _subfig_int_wd.exists():
                        _lat = latest_plextract_run_dir(_subfig_int_wd)
                        if _lat:
                            _out = _lat / "output" / safe_doi / subfig_dir.name
                            if _out.exists():
                                render_ocr_dropdown(
                                    csv_path=_out / "final" / "final_dataset.csv",
                                    key_prefix=f"subfig_int_ocr_{safe_doi}_{subfig_dir.name}",
                                    axis_override_dir=_out,
                                    title=f"New OCR-Extracted Data: {subfig_dir.name}",
                                )
                                _emit = find_emit_rows_for_source(rr.run_dir, selected_doi, "figure", subfig_dir.name)
                                render_llm_dropdown(
                                    run_dir=rr.run_dir,
                                    emit_rows_path=_emit,
                                    key_prefix=f"subfig_int_llm_{safe_doi}_{subfig_dir.name}",
                                    file_name_prefix=subfig_dir.name,
                                    title=f"New LLM-Generated Dataset: {subfig_dir.name}",
                                )

                    # Add separator between subfigures if there are multiple
                    if subfig_dir != subfig_dirs[-1]:
                        st.markdown("---")

with tab_tables:
    if not doi_item.tables:
        st.warning("No tables found.")
    for idx, t in enumerate(doi_item.tables):
        # Add divider between tables
        if idx > 0:
            st.divider()

        # Table header and basic info (not in expander)
        st.markdown(f"### {t.id}")
        with st.expander(f"### {t.id}"):
            if t.caption:
                st.markdown(f"**Caption:** {t.caption}")
            if t.table_path and t.table_path.exists():
                suf = t.table_path.suffix.lower()
                if suf in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                    st_image_auto(str(t.table_path), full_width=True)
                elif suf == ".csv":
                    df = pd.read_csv(t.table_path)
                    st.data_editor(df, use_container_width=True, key=f"table_{selected_doi}_{t.id}")
                else:
                    st.code(t.table_path.read_text(errors="ignore")[:20000])
                st.caption(str(t.table_path))

        # --- Display LLM-generated emit_rows from row_cache ---
        emit_rows_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "table", t.id)
        if emit_rows_path:
            df_emit = load_emit_rows(emit_rows_path)
            if df_emit is not None and not df_emit.empty:
                with section_dropdown("LLM-Generated Dataset", expanded=False):
                    st.caption(f"**Source:** `{emit_rows_path.relative_to(rr.run_dir)}`")
                    st.data_editor(df_emit, use_container_width=True, hide_index=True, key=f"table_emit_{selected_doi}_{t.id}")

                    csv_data = df_emit.to_csv(index=False)
                    st.download_button(
                        label="Download emit_rows CSV",
                        data=csv_data,
                        file_name=f"{t.id}_emit_rows.csv",
                        mime="text/csv",
                    )
        else:
            # Check if row_cache exists but no data found for this table
            row_cache_dir = rr.run_dir / "complete_dataset" / "row_cache"
            if row_cache_dir.exists():
                with section_dropdown("LLM-Generated Dataset", expanded=False):
                    st.info(f"No row_cache data found for {t.id}. Cache may not have been generated yet.")

        # Metadata expander
        if t.metadata_path and t.metadata_path.exists():
            with st.expander("Metadata", expanded=False):
                st.json(json.loads(t.metadata_path.read_text()))

with tab_xy:
    st.subheader("Completed dataset")

    # Collect all emit_rows from row_cache (reactive to changes)
    df_combined = collect_all_emit_rows(rr.run_dir, selected_doi, doi_item)

    if not df_combined.empty:
        st.info(f"Dataset contains {len(df_combined)} rows from {len(df_combined.columns)} columns (concatenated from all tables, figures, and subfigures)")
        st.data_editor(df_combined, use_container_width=True, hide_index=True, key=f"dataset_{selected_doi}")

        # Download button
        csv_data = df_combined.to_csv(index=False)
        st.download_button(
            label="Download completed dataset CSV",
            data=csv_data,
            file_name=f"{doi_to_safe(selected_doi)}_dataset.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No dataset rows found in row_cache for this DOI.\n\n"
            "To generate dataset rows: enable **Complete dataset (LLM)** and run the pipeline, "
            "or use the per-figure/per-subfigure 'Rerun LLM dataset generation' controls."
        )

    # Show legacy dataset CSV if it exists (for reference)
    ds_csv = completed_dataset_csv_path(rr.run_dir, selected_doi)
    ds_json = completed_dataset_json_path(rr.run_dir, selected_doi)

    if ds_csv.exists():
        with st.expander("Legacy completed dataset CSV (from complete_dataset.py)", expanded=False):
            try:
                df_legacy = _load_csv(str(ds_csv))
                st.dataframe(df_legacy, use_container_width=True, hide_index=True)
                st.caption(f"CSV path: {ds_csv}")
            except Exception as e:
                st.warning(f"Failed to load legacy dataset CSV: {e}")

with tab_chem:
    st.subheader("Chemical Composition")

    # --- Add src/ to path for chemical_composition imports ---
    _src_for_chem = (Path(__file__).resolve().parent / "..").resolve()
    if str(_src_for_chem) not in sys.path:
        sys.path.insert(0, str(_src_for_chem))

    from chemical_composition.extract_composition import (
        identify_relevant_content as _cc_identify,
        extract_samples_and_compositions as _cc_extract,
        run_chemical_composition_extraction as _cc_run,
    )
    from chemical_composition.drawing_tool import (
        render_jsme_editor_local,
        load_drawn_structures,
        save_drawn_structure,
        delete_drawn_structure,
    )

    _cc_safe = doi_to_safe(selected_doi)

    # Show previously extracted results if they exist
    cc_json_path = doi_item.extra.get("chemical_composition_json")
    cc_csv_path = doi_item.extra.get("chemical_composition_csv")

    if cc_json_path and Path(cc_json_path).exists():
        cc_data = json.loads(Path(cc_json_path).read_text())

        # --- Samples table ---
        with st.expander("Extracted Samples & Compositions", expanded=True):
            if cc_csv_path and Path(cc_csv_path).exists():
                try:
                    df_cc = pd.read_csv(cc_csv_path)
                    st.dataframe(df_cc, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download samples CSV",
                        data=Path(cc_csv_path).read_bytes(),
                        file_name=f"{_cc_safe}_samples.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(f"Failed to load samples CSV: {e}")
            else:
                # Fall back to JSON display
                samples = cc_data.get("samples", [])
                if samples:
                    for s in samples:
                        with st.expander(f"Sample: {s.get('sample_name', 'Unknown')}"):
                            if s.get("structure_class"):
                                st.markdown(f"**Structure class:** {s['structure_class']}")
                            if s.get("notes"):
                                st.caption(s["notes"])
                            comps = s.get("components", [])
                            if comps:
                                comp_rows = []
                                for c in comps:
                                    comp_rows.append({
                                        "Component": c.get("name", ""),
                                        "Role": c.get("role", ""),
                                        "Formula": c.get("formula", ""),
                                        "SMILES": c.get("smiles", ""),
                                        "Amount": c.get("amount", ""),
                                    })
                                st.dataframe(
                                    pd.DataFrame(comp_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                else:
                    st.info("No samples extracted.")

        # --- Relevant content (with full table/figure rendering) ---
        with st.expander("Relevant Content", expanded=False):
            relevant = cc_data.get("relevant_content", [])
            if relevant:
                # First priority: chemical_composition directory (preserved copies)
                _rc_chem_figures_dir = rr.run_dir / "chemical_composition" / _cc_safe / "figures"
                _rc_chem_tables_dir = rr.run_dir / "chemical_composition" / _cc_safe / "tables"
                # Fallback: figures_tables directory (may have been filtered)
                _rc_figures_dir = rr.run_dir / "figures_tables" / "figures" / _cc_safe
                _rc_tables_dir = rr.run_dir / "figures_tables" / "tables" / _cc_safe

                # Gather all figure image paths (prioritizing chemical_composition directory)
                _rc_all_fig_paths: dict[str, Path] = {}
                # First, check chemical_composition directory
                if _rc_chem_figures_dir.exists():
                    for _fp in _rc_chem_figures_dir.iterdir():
                        if _fp.is_file() and _fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                            _rc_all_fig_paths[_fp.stem.lower().replace(" ", "").replace("_", "")] = _fp
                # Then, add from figures_tables (if not already present)
                if _rc_figures_dir.exists():
                    for _fp in _rc_figures_dir.iterdir():
                        if _fp.is_file() and _fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                            _norm_key = _fp.stem.lower().replace(" ", "").replace("_", "")
                            if _norm_key not in _rc_all_fig_paths:
                                _rc_all_fig_paths[_norm_key] = _fp

                # Also include figures already loaded on the DOI item
                for _fig in doi_item.figures:
                    if _fig.figure_path and _fig.figure_path.exists():
                        _norm_key = _fig.id.lower().replace(" ", "").replace("_", "")
                        if _norm_key not in _rc_all_fig_paths:
                            _rc_all_fig_paths[_norm_key] = _fig.figure_path

                # Gather all table file paths (prioritizing chemical_composition directory)
                _rc_all_table_paths: dict[str, Path] = {}
                _rc_all_table_captions: dict[str, str] = {}
                # First, check chemical_composition directory
                if _rc_chem_tables_dir.exists():
                    for _tp in _rc_chem_tables_dir.iterdir():
                        if _tp.is_file() and not _tp.name.endswith("_metadata.json"):
                            _norm_id = _tp.stem.lower().replace(" ", "").replace("_", "")
                            _rc_all_table_paths[_norm_id] = _tp
                            # Try to load caption from sidecar metadata
                            _meta_p = _tp.with_name(_tp.stem + "_metadata.json")
                            if _meta_p.exists():
                                try:
                                    _rc_all_table_captions[_norm_id] = json.loads(_meta_p.read_text()).get("caption", "")
                                except Exception:
                                    pass
                # Then, add from figures_tables (if not already present)
                if _rc_tables_dir.exists():
                    for _tp in _rc_tables_dir.iterdir():
                        if _tp.is_file() and not _tp.name.endswith("_metadata.json"):
                            _norm_id = _tp.stem.lower().replace(" ", "").replace("_", "")
                            if _norm_id not in _rc_all_table_paths:
                                _rc_all_table_paths[_norm_id] = _tp
                                # Try to load caption from sidecar metadata
                                _meta_p = _tp.with_name(_tp.stem + "_metadata.json")
                                if _meta_p.exists():
                                    try:
                                        _rc_all_table_captions[_norm_id] = json.loads(_meta_p.read_text()).get("caption", "")
                                    except Exception:
                                        pass

                # Also include tables already loaded on the DOI item
                for _tbl in doi_item.tables:
                    _norm_id = _tbl.id.lower().replace(" ", "").replace("_", "")
                    if _tbl.table_path and _tbl.table_path.exists():
                        if _norm_id not in _rc_all_table_paths:
                            _rc_all_table_paths[_norm_id] = _tbl.table_path
                    if _tbl.caption and _norm_id not in _rc_all_table_captions:
                        _rc_all_table_captions[_norm_id] = _tbl.caption

                for i, item_rc in enumerate(relevant):
                    st.markdown(
                        f"**{item_rc.get('source_type', '')}** — "
                        f"{item_rc.get('source_id', '')}"
                    )
                    # Show relevance as wrapped text
                    _rel_text = item_rc.get("relevance", "")
                    if _rel_text:
                        st.markdown(f"_{_rel_text}_")

                    _src_type = item_rc.get("source_type", "").lower()
                    _src_id = item_rc.get("source_id", "").lower().replace(" ", "")
                    _src_id_norm = _src_id.replace("_", "")

                    # Display the referenced figure/scheme image if available
                    if _src_type in ("figure/scheme", "figure", "scheme"):
                        _matched_fig = False
                        for _fkey, _fpath in _rc_all_fig_paths.items():
                            if _src_id_norm and (_src_id_norm in _fkey or _fkey in _src_id_norm):
                                _rc_img_col, _ = st.columns([1, 2])
                                with _rc_img_col:
                                    st_image_auto(str(_fpath), full_width=True)
                                _matched_fig = True
                                break
                        # Suppress "not found" message - item may have been filtered intentionally

                    # Display the referenced table if available
                    if _src_type in ("table",):
                        _matched_tbl = False
                        for _tkey, _tpath in _rc_all_table_paths.items():
                            if _src_id_norm and (_src_id_norm in _tkey or _tkey in _src_id_norm):
                                _tbl_caption = _rc_all_table_captions.get(_tkey, "")
                                if _tbl_caption:
                                    st.caption(_tbl_caption)
                                _suf = _tpath.suffix.lower()
                                if _suf in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                                    st_image_auto(str(_tpath), full_width=True)
                                elif _suf == ".csv":
                                    try:
                                        _tdf = pd.read_csv(_tpath)
                                        st.dataframe(_tdf, use_container_width=True, hide_index=True)
                                    except Exception:
                                        st.code(_tpath.read_text(errors="ignore")[:20000])
                                else:
                                    st.code(_tpath.read_text(errors="ignore")[:20000])
                                _matched_tbl = True
                                break
                        # Suppress "not found" message - item may have been filtered intentionally

                    if item_rc.get("excerpt"):
                        # Use markdown for better text wrapping (no horizontal scrolling)
                        st.markdown(f"<div style='white-space: pre-wrap; word-wrap: break-word;'>{item_rc['excerpt'][:500]}</div>", unsafe_allow_html=True)
                    if i < len(relevant) - 1:
                        st.divider()
            else:
                st.info("No relevant content identified.")

    else:
        st.info(
            "No chemical composition data found for this DOI.\n\n"
            "Enable **Extract chemical composition** in the sidebar and re-run, "
            "or use the **Run Extraction** dropdown below to extract on demand."
        )

    # --- Chemical Structure Drawing Tool ---
    with st.expander("Draw Chemical Structures", expanded=False):
        render_jsme_editor_local(height=550, key=f"jsme_{_cc_safe}")

        st.divider()
        st.markdown("#### Save Drawn Structure")

        draw_sample_name = st.text_input(
            "Sample name for this structure",
            key=f"draw_sample_{_cc_safe}",
            placeholder="e.g. QPPO-TMA-40",
        )
        draw_smiles = st.text_input(
            "SMILES (paste from editor or type manually)",
            key=f"draw_smiles_{_cc_safe}",
            placeholder="e.g. C(C(=O)O)N",
        )
        draw_notes = st.text_input(
            "Notes (optional)",
            key=f"draw_notes_{_cc_safe}",
            placeholder="e.g. backbone polymer for sample set A",
        )

        if st.button("Save structure", type="primary", key=f"save_drawn_{_cc_safe}"):
            if not draw_sample_name.strip():
                st.error("Please provide a sample name.")
            else:
                save_drawn_structure(
                    run_dir=rr.run_dir,
                    safe_doi=_cc_safe,
                    sample_name=draw_sample_name.strip(),
                    smiles=draw_smiles.strip(),
                    notes=draw_notes.strip(),
                )
                st.success(f"Saved structure for '{draw_sample_name.strip()}'")
                st.rerun()

    # --- User-Drawn Structures ---
    with st.expander("User-Drawn Structures", expanded=False):
        drawn = load_drawn_structures(rr.run_dir, _cc_safe)
        if drawn:
            for idx, ds in enumerate(drawn):
                cols = st.columns([3, 2, 1])
                with cols[0]:
                    st.markdown(f"**{ds.get('sample_name', 'Unnamed')}**")
                    if ds.get("smiles"):
                        st.code(ds["smiles"], language=None)
                with cols[1]:
                    if ds.get("notes"):
                        st.caption(ds["notes"])
                with cols[2]:
                    if st.button("Delete", key=f"del_drawn_{_cc_safe}_{idx}"):
                        delete_drawn_structure(rr.run_dir, _cc_safe, idx)
                        st.rerun()
        else:
            st.info("No user-drawn structures saved yet. Use the drawing tool below.")

    # --- Run Extraction ---
    with st.expander("Run Chamical Composition Extraction", expanded=False):
        st.caption(
            "Run the LLM-based extraction on demand for this DOI. "
            "This will analyse the parsed text, tables, and figure captions "
            "to identify chemical structures, sample names, and compositions."
        )

        if st.button("Extract chemical composition", type="primary", key=f"run_cc_{_cc_safe}"):
            try:
                with st.spinner("Extracting chemical composition..."):
                    parsed_json = doi_item.parsed_json_path
                    tables_d = rr.run_dir / "figures_tables" / "tables" / _cc_safe
                    figures_d = rr.run_dir / "figures_tables" / "figures" / _cc_safe

                    result = _cc_run(
                        run_dir=rr.run_dir,
                        safe_doi=_cc_safe,
                        parsed_json_path=parsed_json,
                        tables_dir=tables_d if tables_d.exists() else None,
                        figures_dir=figures_d if figures_d.exists() else None,
                    )

                st.success("Extraction complete!")

                # Show results inline
                samples = result.get("samples", [])
                st.markdown(f"**Found {len(samples)} sample(s)**")
                if samples:
                    for s in samples:
                        with st.expander(f"Sample: {s.get('sample_name', 'Unknown')}"):
                            st.json(s)

                relevant = result.get("relevant_content", [])
                st.markdown(f"**Identified {len(relevant)} relevant content section(s)**")

                st.info("Reload the run or switch tabs to see the full results.")

            except Exception as e:
                st.error(f"Chemical composition extraction failed: {e}")