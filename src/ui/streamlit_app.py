# SALSA/src/ui/streamlit_app.py
from __future__ import annotations
import json
import inspect
import threading
import time
import traceback
from http.cookies import SimpleCookie
from hashlib import md5
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit.elements.image as st_image
import plotly.express as px
from PIL import Image, UnidentifiedImageError
from streamlit_drawable_canvas import CanvasResult, _component_func, _data_url_to_image
from pipeline import (
    ADMIN_INPUT_ITEM_LIMIT,
    DEFAULT_INPUT_ITEM_LIMIT,
    PipelineCancelled,
    run_full_pipeline,
)
from discover import discover_run
from schemas import RunResult
from run_progress import RUN_STATUS_FILE_NAME, read_run_status, write_run_status
from intervention import latest_plextract_run_dir
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
from run_plextract_manual import run_plextract_on_crops as _run_plextract_on_crops
from custom_instructions import (
    DEFAULT_TABLE_EXTRACTION_RULES,
    default_instructions,
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
import sys
import subprocess
import sys
import base64
from typing import List, Optional
import re
from datetime import datetime
from typing import Callable, Sequence, Optional, Any
from urllib.request import urlopen
from auth import (
    AuthStore,
    build_reset_url,
    data_root_from_env,
    is_admin_user,
    run_belongs_to_user,
    safe_child_path,
    send_password_reset_email,
    user_workspaces_root,
    validate_password,
    write_run_manifest,
)
from llm_step_config import (
    GENERAL_LLM_ENV_VARS,
    LLM_STEP_CHEM_COMPOSITION,
    LLM_STEP_CHEM_RELEVANT,
    LLM_STEP_COMPLETE_DATASET,
    LLM_STEP_EXTRACT_TABLES,
    LLM_STEP_FIGURE_DATA,
    LLM_STEP_FILTER_FIGURES,
    LLM_STEP_FILTER_TABLES,
    LLM_STEP_LABELS,
    LLM_STEP_ENV_PREFIXES,
    step_env_names,
    step_env_to_general_env_map,
)

SALSA_LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "salsa_icon.png"
PUBLISHER_TDM_ENV_NAMES = {
    "elsevier_api_key": ("ELSEVIER_API_KEY", "ELS_API_KEY"),
    "wiley_tdm_token": ("WILEY_TDM_TOKEN", "WILEY_API_KEY"),
}
_INITIAL_PUBLISHER_TDM_ENV = {
    env_name: os.environ.get(env_name)
    for env_names in PUBLISHER_TDM_ENV_NAMES.values()
    for env_name in env_names
}

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

def _ignored_workspace_or_run_name(name: str) -> bool:
    return name in {"axis_presets", "instruction_presets", "settings_presets", "custom_instructions"}

def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(size, 0))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024

def _dir_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not Path(current_root, d).is_symlink()
        ]
        for filename in filenames:
            path = Path(current_root) / filename
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    return total

def _path_mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""

def _count_subdirs(path: Path) -> int:
    try:
        return sum(1 for child in path.iterdir() if child.is_dir())
    except OSError:
        return 0

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
    return "workspace_1"

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

def _normalize_ollama_host(host: str) -> str:
    value = (host or "").strip() or _default_ollama_host()
    if not value.startswith("http"):
        value = "http://" + value
    return value.rstrip("/")


@st.cache_data(show_spinner=False, ttl=5)
def _ollama_list_models_via_api(host: str) -> Optional[List[str]]:
    """
    Query the Ollama HTTP API directly so the UI reflects whether the daemon is
    actually reachable, not just whether models exist on disk.
    """
    normalized_host = _normalize_ollama_host(host)
    try:
        with urlopen(f"{normalized_host}/api/tags", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = []
        for item in payload.get("models", []):
            name = str(item.get("name", "")).strip()
            if name:
                models.append(name)
        return models
    except Exception:
        return None


def _ollama_list_models_via_cli(host: str) -> Optional[List[str]]:
    """
    Best source of truth: `ollama list` returns the models actually available to the ollama daemon.
    """
    try:
        # Note: OLLAMA_HOST can be set in env to point to a non-default daemon
        cli_env = os.environ.copy()
        cli_env["OLLAMA_HOST"] = _normalize_ollama_host(host)
        out = subprocess.check_output(
            ["ollama", "list"],
            text=True,
            stderr=subprocess.STDOUT,
            env=cli_env,
        )
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
        return None


@st.cache_data(show_spinner=False, ttl=5)
def _ollama_server_is_available(host: str) -> bool:
    normalized_host = _normalize_ollama_host(host)
    if _ollama_list_models_via_api(normalized_host) is not None:
        return True
    return _ollama_list_models_via_cli(normalized_host) is not None

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


def render_salsa_brand_header() -> None:
    if not SALSA_LOGO_PATH.exists():
        st.title("SALSA - Semi-Autonomous Literature Scraping Assistant")
        return

    _, logo_col, _ = st.columns([1, 6, 1])
    with logo_col:
        st_image_auto(str(SALSA_LOGO_PATH), full_width=True)


def _is_valid_image_path(path: Optional[Path]) -> bool:
    if path is None:
        return False
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
    except OSError:
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _open_image_safe(path: Path, *, mode: Optional[str] = None) -> Optional[Image.Image]:
    if not _is_valid_image_path(path):
        return None
    try:
        with Image.open(path) as img:
            return img.convert(mode) if mode else img.copy()
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _canvas_media_url(url: str) -> str:
    """Normalize Streamlit media URLs for the drawable-canvas iframe."""
    if not url or url.startswith(("http://", "https://", "data:")):
        return url

    base_path = str(st._config.get_option("server.baseUrlPath") or "").strip("/")
    if base_path:
        base_path = f"/{base_path}"

    if not url.startswith("/"):
        url = f"/{url}"
    if base_path and url.startswith(f"{base_path}/"):
        return url
    return f"{base_path}{url}"


def st_canvas(
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: Image.Image = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: dict = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key=None,
) -> CanvasResult:
    """Drawable canvas wrapper with robust background image URLs."""
    background_image_url = None
    if background_image:
        background_image = background_image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        raw_url = st_image.image_to_url(
            background_image,
            width,
            True,
            "RGB",
            "PNG",
            f"drawable-canvas-bg-{md5(background_image.tobytes()).hexdigest()}-{key}",
        )
        background_image_url = _canvas_media_url(raw_url)
        background_color = ""

    initial_drawing = {"version": "4.4.0"} if initial_drawing is None else dict(initial_drawing)
    initial_drawing["background"] = background_color

    component_value = _component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor=background_color,
        backgroundImageURL=background_image_url,
        realtimeUpdateStreamlit=update_streamlit and (drawing_mode != "polygon"),
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=initial_drawing,
        displayToolbar=display_toolbar,
        displayRadius=point_display_radius,
        key=key,
        default=None,
    )
    if component_value is None:
        return CanvasResult()

    return CanvasResult(
        np.asarray(_data_url_to_image(component_value["data"])),
        component_value["raw"],
    )


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
):
    """Render the full Axis Configuration dropdown, shared between figures and subfigures.

    chartdete_dir: directory containing the read-only plextract source files
                   (axis_title_texts.json, pix2val.json, label_coordinates.json).
    override_dir:  directory where user overrides are read from and written to
                   (axis_title_override.json, pix2val_override.json, axis_type_override.json).
                   Defaults to chartdete_dir when not supplied.
    """
    # Determine where overrides are stored; separate from the read-only source dir so
    # we never write back into the original plextract output.
    _override_base: Optional[Path] = override_dir if override_dir is not None else chartdete_dir

    with section_dropdown("Axis Configuration", expanded=expanded):
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
            value=int(st.session_state[width_key]), step=50, key=width_key,
        )
        disp_h = int(round(img_h * (disp_w / img_w)))

        bg = pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS).convert("RGB")
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h

        endpoint_key = f"{key_prefix}_ax_calib_endpoint"
        endpoint = st.radio("Click target", ["Xmin", "Xmax", "Ymin", "Ymax"], key=endpoint_key, horizontal=True)

        cv_ver_key = f"{key_prefix}_ax_calib_ver"
        if cv_ver_key not in st.session_state:
            st.session_state[cv_ver_key] = 0
        ensure_canvas_remount(
            boot_key=f"{key_prefix}_ax_calib_boot",
            version_key=cv_ver_key,
        )

        # Bump the canvas version when the display width changes so the canvas
        # remounts with the correct background dimensions.  The width must NOT
        # be baked into the canvas key itself — doing so creates a new key on
        # every slider move, leaving orphaned entries in st.session_state that
        # cause "dictionary changed size during iteration" in _call_callbacks.
        _ax_last_w_key = f"{key_prefix}_ax_calib_last_w"
        if st.session_state.get(_ax_last_w_key) != disp_w:
            if _ax_last_w_key in st.session_state:
                st.session_state[cv_ver_key] += 1
            st.session_state[_ax_last_w_key] = disp_w

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
        ensure_canvas_post_mount_remount(
            mount_key=(
                f"{key_prefix}_ax_calib_mounted_"
                f"{Path(source_img).name}_{disp_w}x{disp_h}"
            ),
            version_key=cv_ver_key,
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
):
    """Render the full Series Management dropdown, shared between figures and subfigures."""
    with section_dropdown("Series Management", expanded=expanded):
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
        if cv_ver_key_sm not in st.session_state:
            st.session_state[cv_ver_key_sm] = 0
        ensure_canvas_remount(
            boot_key=f"{key_prefix}_series_canvas_boot",
            version_key=cv_ver_key_sm,
        )

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
            value=int(st.session_state[width_key_sm]),
            step=50,
            key=width_key_sm,
        )

        last_w_key = f"{key_prefix}_series_last_w"
        last_w = st.session_state.get(last_w_key)

        if last_w is None:
            st.session_state[last_w_key] = disp_w
        elif last_w != disp_w:
            st.session_state[last_w_key] = disp_w
            st.session_state[cv_ver_key_sm] += 1

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
            _series_img_token = Path(source_img).name if source_img else "no_source"
            _series_canvas_mount_key = (
                f"{key_prefix}_series_canvas_mounted_{selected_idx}_{mode}_"
                f"{input_mode}_{_series_img_token}_{disp_w}x{disp_h}"
            )
            c1_sm, c2_sm = st.columns([2.2, 1.5])

            if input_mode == "Freehand drawing":
                with c1_sm:
                    st.write("Draw on the colored series image below")
                    # mask_canvas = st_canvas(
                    #     background_image=composite_disp,
                    #     drawing_mode="freedraw",
                    #     stroke_width=5, stroke_color=stroke_rgba,
                    #     fill_color=stroke_rgba, update_streamlit=True,
                    #     height=disp_h, width=disp_w,
                    #     key=f"{key_prefix}_mask_canvas_{selected_idx}_{mode}",
                    # )
                    canvas_key_freehand = (
                        f"{key_prefix}_mask_canvas_{selected_idx}_{mode}"
                        f"_v{st.session_state[cv_ver_key_sm]}"
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
                        ensure_canvas_post_mount_remount(
                            mount_key=_series_canvas_mount_key,
                            version_key=cv_ver_key_sm,
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
                        f"_v{st.session_state[cv_ver_key_sm]}"
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
                    if point_canvas is not None:
                        ensure_canvas_post_mount_remount(
                            mount_key=_series_canvas_mount_key,
                            version_key=cv_ver_key_sm,
                        )
                    if composite_disp is None:
                        st.warning("Series image could not be rendered.")
                if legend_lines:
                    st.markdown(" &nbsp; ".join(legend_lines), unsafe_allow_html=True)

                new_pts: list = []
                if point_canvas is not None and point_canvas.json_data and "objects" in point_canvas.json_data:
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
    if not _is_valid_image_path(image_path):
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
        stages = summary.get("stages") or {}
        if stages:
            stage_rows = []
            for stage_name, info in stages.items():
                stage_models = info.get("models") or {}
                model_bits = [
                    (
                        f"{model_name if model_name != 'unknown' else 'unknown model'} "
                        f"({model_info['calls']} calls, ${model_info['cost']:.4f})"
                    )
                    for model_name, model_info in stage_models.items()
                ]
                stage_rows.append(
                    {
                        "Stage": stage_name,
                        "Model(s)": "; ".join(model_bits) if model_bits else "",
                        "Calls": info["calls"],
                        "Input": info["input"],
                        "Cached": info["cached"],
                        "Output": info["output"],
                        "Cost": f"${info['cost']:.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)
        st.metric("Total", f"${summary['grand_total']:.4f}")

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
    source_names = [source_name]
    source_stem = Path(str(source_name)).stem
    if source_stem and source_stem not in source_names:
        source_names.append(source_stem)
    safe_prefixes = [
        _safe_filename_compat(f"emit_rows::{source_kind}::{doi}::{name}::")
        for name in source_names
    ]

    candidates = []
    for cache_file in cache_dir.iterdir():
        if cache_file.suffix != ".json":
            continue
        # Files are named: <safe_prefix><hash>__<sha256>.json
        # Check that the filename starts with a compatible safe_prefix.
        if any(
            cache_file.stem.startswith(prefix) or cache_file.name.startswith(prefix)
            for prefix in safe_prefixes
        ):
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

def build_doi_stats(rr: RunResult, *, doi_order: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for d in rr.dois:
        safe = doi_to_safe(d.doi)
        rows.append({
            "DOI": d.doi,
            "Tables": len(d.tables or []),
            "Figures": len(d.figures or []),
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
    host = _normalize_ollama_host(st.session_state.get("ollama_host", default_ollama_host))
    api_models = _ollama_list_models_via_api(host)
    if api_models is not None:
        return api_models
    cli_models = _ollama_list_models_via_cli(host)
    if cli_models is not None:
        return cli_models
    return []

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
    if p and _is_valid_image_path(Path(p)):
        st.image(str(p), width=preview_width)
        st.caption(Path(p).name)
    else:
        st.warning("Preview image not found for the current selection.")

    return sel

@st.cache_data(show_spinner=False)
def load_and_resize_rgba(path: str, disp_w: int):
    img = _open_image_safe(Path(path), mode="RGB")
    if img is None:
        raise ValueError(f"Could not open image: {path}")
    ow, oh = img.size
    dh = int(round(oh * (disp_w / ow)))
    bg = img.resize((disp_w, dh), Image.Resampling.LANCZOS).convert("RGB")
    return ow, oh, disp_w, dh, bg

def ensure_canvas_remount(*, boot_key: str, version_key: str):
    """
    Force a one-time rerun for canvas-heavy sections.

    Why this exists:
    `streamlit-drawable-canvas` can occasionally mount while an expander is
    collapsed, leaving the background image blank until *any* widget triggers
    a rerun. We trigger that rerun automatically once so users don't need to
    click an unrelated button just to make the image appear.
    """
    if version_key not in st.session_state:
        st.session_state[version_key] = 0
    if boot_key not in st.session_state:
        st.session_state[boot_key] = True

    if st.session_state[boot_key]:
        st.session_state[boot_key] = False
        st.session_state[version_key] += 1
        st.rerun()

def ensure_canvas_post_mount_remount(*, mount_key: str, version_key: str):
    """
    Remount a canvas once after it has been created.

    The drawable-canvas iframe sometimes paints before its background image is
    ready. A single post-mount key bump gives the component a fresh mount with
    the image already registered in Streamlit's media store.
    """
    if mount_key not in st.session_state:
        st.session_state[mount_key] = True
        st.session_state[version_key] += 1
        st.rerun()

_STEP_LLM_RENDER_ORDER = [
    LLM_STEP_EXTRACT_TABLES,
    LLM_STEP_CHEM_RELEVANT,
    LLM_STEP_CHEM_COMPOSITION,
    LLM_STEP_FILTER_FIGURES,
    LLM_STEP_FILTER_TABLES,
    LLM_STEP_FIGURE_DATA,
    LLM_STEP_COMPLETE_DATASET,
]


def _ensure_shared_llm_settings_in_state() -> None:
    if "openai_base_url" not in st.session_state:
        st.session_state["openai_base_url"] = os.environ.get(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        )
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""
    if "elsevier_api_key" not in st.session_state:
        st.session_state["elsevier_api_key"] = ""
    if "wiley_tdm_token" not in st.session_state:
        st.session_state["wiley_tdm_token"] = ""
    if "ollama_host" not in st.session_state:
        st.session_state["ollama_host"] = os.environ.get(
            "OLLAMA_HOST",
            _default_ollama_host(),
        )
    if "ollama_models_dir" not in st.session_state:
        st.session_state["ollama_models_dir"] = os.environ.get(
            "OLLAMA_MODELS",
            str(_default_ollama_models_dir()),
        )
    if "ollama_num_ctx" not in st.session_state:
        try:
            st.session_state["ollama_num_ctx"] = int(
                os.environ.get("SALSA_OLLAMA_NUM_CTX", "16384")
            )
        except ValueError:
            st.session_state["ollama_num_ctx"] = 16384


def _sync_publisher_tdm_env_from_session() -> None:
    _ensure_shared_llm_settings_in_state()
    for session_key, env_names in PUBLISHER_TDM_ENV_NAMES.items():
        value = str(st.session_state.get(session_key, "")).strip()
        for env_name in env_names:
            if value:
                os.environ[env_name] = value
            elif _INITIAL_PUBLISHER_TDM_ENV.get(env_name) is not None:
                os.environ[env_name] = _INITIAL_PUBLISHER_TDM_ENV[env_name] or ""
            else:
                os.environ.pop(env_name, None)


def _step_provider_key(step_key: str) -> str:
    return f"{step_key}_llm_provider"


def _step_model_key(step_key: str) -> str:
    return f"{step_key}_llm_model"


def _step_provider_label(step_key: str) -> str:
    return st.session_state.get(_step_provider_key(step_key), "Chat GPT")


def _step_model_value(step_key: str) -> str:
    return str(st.session_state.get(_step_model_key(step_key), "")).strip()


def _clear_general_llm_env(env: dict[str, str]) -> None:
    for key in GENERAL_LLM_ENV_VARS:
        env.pop(key, None)


def _clear_step_llm_env(step_key: str) -> None:
    for key in step_env_names(step_key):
        os.environ.pop(key, None)


def _sync_step_llm_env_from_session(step_key: str) -> None:
    _ensure_shared_llm_settings_in_state()
    _clear_step_llm_env(step_key)

    model = _step_model_value(step_key)
    if not model:
        return

    provider = _step_provider_label(step_key)
    prefix = LLM_STEP_ENV_PREFIXES[step_key]
    if provider == "Ollama":
        os.environ[f"{prefix}_LLM_PROVIDER"] = "ollama"
        os.environ[f"{prefix}_LLM_MODEL"] = model
        os.environ[f"{prefix}_OLLAMA_HOST"] = st.session_state.get("ollama_host", default_ollama_host)
        os.environ[f"{prefix}_OLLAMA_MODELS"] = st.session_state.get(
            "ollama_models_dir",
            str(default_ollama_dir),
        )
        os.environ[f"{prefix}_OLLAMA_NUM_CTX"] = str(
            int(st.session_state.get("ollama_num_ctx", 16384))
        )
    else:
        os.environ[f"{prefix}_LLM_PROVIDER"] = "openai"
        os.environ[f"{prefix}_LLM_MODEL"] = model
        os.environ[f"{prefix}_OPENAI_BASE_URL"] = st.session_state.get(
            "openai_base_url",
            "https://api.openai.com/v1",
        )
        api_key = str(st.session_state.get("openai_api_key", "")).strip()
        if api_key:
            os.environ[f"{prefix}_OPENAI_API_KEY"] = api_key


def _sync_all_step_llm_envs_from_session() -> None:
    _clear_general_llm_env(os.environ)
    for step_key in _STEP_LLM_RENDER_ORDER:
        _sync_step_llm_env_from_session(step_key)


def _general_llm_env_for_step(step_key: str) -> dict[str, str]:
    env = os.environ.copy()
    _clear_general_llm_env(env)
    for step_env, general_env in step_env_to_general_env_map(step_key).items():
        value = os.environ.get(step_env, "").strip()
        if value:
            env[general_env] = value
        else:
            env.pop(general_env, None)
    return env


def _step_has_configured_model(step_key: str) -> bool:
    return bool(_step_model_value(step_key))


def _render_step_llm_selector(
    step_key: str,
    *,
    caption: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    _ensure_shared_llm_settings_in_state()

    st.subheader(title or LLM_STEP_LABELS[step_key])
    if caption:
        st.caption(caption)

    provider_key = _step_provider_key(step_key)
    model_key = _step_model_key(step_key)
    ollama_available = _ollama_server_is_available(
        _normalize_ollama_host(st.session_state.get("ollama_host", default_ollama_host))
    )
    if not ollama_available and st.session_state.get(provider_key) == "Ollama":
        st.session_state[provider_key] = "Chat GPT"
        st.session_state.pop(model_key, None)
    provider_options = ["Chat GPT"] + (["Ollama"] if ollama_available else [])
    provider = st.radio(
        "Provider",
        options=provider_options,
        index=provider_options.index(st.session_state[provider_key])
        if st.session_state.get(provider_key) in provider_options
        else 0,
        key=provider_key,
        horizontal=True,
        on_change=_keep_create_settings_preset_open,
    )

    if provider == "Ollama":
        models = _ollama_models()
        if models:
            if st.session_state.get(model_key) not in models:
                st.session_state.pop(model_key, None)
            st.selectbox(
                "Model",
                options=models,
                index=0,
                key=model_key,
                on_change=_keep_create_settings_preset_open,
            )
        else:
            st.text_input(
                "Model (no models detected)",
                value="",
                key=model_key,
                on_change=_keep_create_settings_preset_open,
            )
            st.caption("Tip: run `ollama pull <model>` then refresh.")
    else:
        if st.session_state.get(model_key) not in _openai_models:
            st.session_state.pop(model_key, None)
        st.selectbox(
            "Model",
            options=_openai_models,
            index=0,
            key=model_key,
            on_change=_keep_create_settings_preset_open,
        )


def _active_llm_steps(
    *,
    run_extract_tables: bool,
    use_docling_pdf_uploads: bool = False,
    run_chemical_composition: bool,
    run_filter_figures: bool,
    run_filter_tables: bool,
    run_plextract: bool,
    bypass_ocr_with_llm: bool,
    run_complete_dataset: bool,
) -> list[str]:
    steps: list[str] = []
    if run_extract_tables and not use_docling_pdf_uploads:
        steps.append(LLM_STEP_EXTRACT_TABLES)
    if run_chemical_composition:
        steps.extend([LLM_STEP_CHEM_RELEVANT, LLM_STEP_CHEM_COMPOSITION])
    if run_filter_figures:
        steps.append(LLM_STEP_FILTER_FIGURES)
    if run_filter_tables:
        steps.append(LLM_STEP_FILTER_TABLES)
    if run_plextract and bypass_ocr_with_llm:
        steps.append(LLM_STEP_FIGURE_DATA)
    if run_complete_dataset:
        steps.append(LLM_STEP_COMPLETE_DATASET)
    return steps


def _missing_llm_steps(step_keys: Sequence[str]) -> list[str]:
    return [
        LLM_STEP_LABELS[step_key]
        for step_key in step_keys
        if not _step_has_configured_model(step_key)
    ]


def _any_step_uses_ollama(step_keys: Sequence[str]) -> bool:
    return any(_step_provider_label(step_key) == "Ollama" for step_key in step_keys)


def _missing_openai_api_key_steps(step_keys: Sequence[str]) -> list[str]:
    if str(st.session_state.get("openai_api_key", "")).strip():
        return []
    return [
        LLM_STEP_LABELS[step_key]
        for step_key in step_keys
        if _step_provider_label(step_key) == "Chat GPT"
    ]


def _llm_step_caption(step_key: str) -> str:
    captions = {
        LLM_STEP_EXTRACT_TABLES: "Used when parsing table content with the LLM.",
        LLM_STEP_CHEM_RELEVANT: "Finds the text, tables, and schemes relevant to chemical composition.",
        LLM_STEP_CHEM_COMPOSITION: "Uses the selected relevant evidence to extract sample compositions.",
        LLM_STEP_FILTER_FIGURES: "Decides which figures are relevant downstream.",
        LLM_STEP_FILTER_TABLES: "Decides which tables are relevant downstream.",
        LLM_STEP_FIGURE_DATA: "Used only when bypassing OCR and extracting figure data directly with the LLM.",
        LLM_STEP_COMPLETE_DATASET: "Builds the final dataset rows.",
    }
    return captions.get(step_key, "")


def _looks_like_doi_scope(value: str) -> bool:
    return bool(re.fullmatch(r"\d+\s+DOIs?", value.strip()))


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except OSError:
        return False
    return True


def _run_status_is_active(status: dict[str, Any]) -> bool:
    return status.get("state") in {"running", "cancelling"} and _pid_is_alive(status.get("pid"))


def _mark_inactive_run_status(status_path: Path, status: dict[str, Any]) -> dict[str, Any]:
    state = status.get("state")
    pid = status.get("pid")
    if state not in {"running", "cancelling"} or _pid_is_alive(pid):
        return status
    write_run_status(
        status_path,
        state="interrupted",
        stage="Interrupted",
        error=f"Last recorded pipeline process is no longer running (pid: {pid}).",
    )
    return read_run_status(status_path)


def _run_progress_status_text(status: dict[str, Any]) -> str:
    stage = status.get("stage") or "Running"
    doi = str(status.get("doi") or "").strip()
    lines = [f"Running stage: **{stage}**"]
    if doi:
        label = "Scope" if _looks_like_doi_scope(doi) else "Current DOI"
        lines.append(f"{label}: `{doi}`")
    figure = str(status.get("figure") or "").strip()
    if figure:
        lines.append(f"Current figure: `{figure}`")
    updated_at = status.get("updated_at")
    if updated_at:
        lines.append(f"Updated: `{updated_at}`")
    return "\n\n".join(lines)


def _render_run_progress_status(status: dict[str, Any]) -> None:
    st.info(_run_progress_status_text(status))


def _workspace_display_name_for_run(run_dir: Path, workspaces_root: Path) -> str:
    try:
        rel = run_dir.resolve().relative_to(workspaces_root.resolve())
        workspace_name = rel.parts[0] if rel.parts else run_dir.parent.name
    except Exception:
        workspace_name = run_dir.parent.name
    if any(ch.isupper() for ch in workspace_name):
        return workspace_name
    return workspace_name.replace("_", " ").replace("-", " ").title()


def _admin_run_records(data_root: Path, users: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for user in users:
        profile_slug = str(user["profile_slug"])
        user_workspaces = user_workspaces_root(data_root, user)
        for workspace_dir in sorted(
            [p for p in user_workspaces.iterdir() if p.is_dir() and not _ignored_workspace_or_run_name(p.name)],
            key=lambda p: p.name.lower(),
        ):
            for run_dir in sorted(
                [p for p in workspace_dir.iterdir() if p.is_dir() and not _ignored_workspace_or_run_name(p.name)],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                status = read_run_status(run_dir / RUN_STATUS_FILE_NAME)
                manifest = _read_json_file(run_dir / "run_manifest.json")
                doi_count = _count_subdirs(run_dir / "reparsed")
                records.append({
                    "user_id": int(user["id"]),
                    "username": user["username"],
                    "email": user["email"],
                    "display_name": user["display_name"],
                    "role": user["role"],
                    "profile_slug": profile_slug,
                    "workspace": workspace_dir.name,
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "owner_username": manifest.get("owner_username") or user["username"],
                    "created_at": manifest.get("created_at") or "",
                    "state": status.get("state") or "unknown",
                    "stage": status.get("stage") or "",
                    "doi": status.get("doi") or "",
                    "updated_at": status.get("updated_at") or "",
                    "last_modified": _path_mtime_iso(run_dir),
                    "doi_count": doi_count,
                    "size_bytes": _dir_size_bytes(run_dir),
                })
    return records


def _admin_user_stats(users: Sequence[Any], run_records: Sequence[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    records_by_user: dict[int, list[dict[str, Any]]] = {}
    for record in run_records:
        records_by_user.setdefault(int(record["user_id"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for user in users:
        user_id = int(user["id"])
        user_records = records_by_user.get(user_id, [])
        workspaces_root_for_user = user_workspaces_root(data_root, user)
        workspaces = [
            p for p in workspaces_root_for_user.iterdir()
            if p.is_dir() and not _ignored_workspace_or_run_name(p.name)
        ]
        state_counts: dict[str, int] = {}
        for record in user_records:
            state = str(record.get("state") or "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1
        last_activity = max(
            [str(r.get("updated_at") or r.get("last_modified") or "") for r in user_records],
            default="",
        )
        total_size = sum(int(r.get("size_bytes") or 0) for r in user_records)
        rows.append({
            "username": user["username"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "institution": user["institution"],
            "display_name": user["display_name"],
            "role": user["role"],
            "workspaces": len(workspaces),
            "runs": len(user_records),
            "done": state_counts.get("done", 0),
            "running": state_counts.get("running", 0) + state_counts.get("cancelling", 0),
            "errors": state_counts.get("error", 0),
            "interrupted": state_counts.get("interrupted", 0),
            "total_dois": sum(int(r.get("doi_count") or 0) for r in user_records),
            "disk": _format_bytes(total_size),
            "last_activity": last_activity,
        })
    return rows


def _render_admin_page(
    *,
    auth_store: AuthStore,
    data_root: Path,
    current_user: Any,
) -> bool:
    st.header("Admin")
    st.caption("Manage SALSA profiles, roles, run access, and per-user activity.")

    users = auth_store.list_users()
    run_records = _admin_run_records(data_root, users)
    stats_rows = _admin_user_stats(users, run_records, data_root)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Users", len(users))
    metric_cols[1].metric("Admins", sum(1 for user in users if user["role"] == "admin"))
    metric_cols[2].metric("Runs", len(run_records))
    metric_cols[3].metric("Storage", _format_bytes(sum(int(r.get("size_bytes") or 0) for r in run_records)))

    st.subheader("User Statistics")
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No users found.")

    st.subheader("Role Management")
    selected_user = None
    user_options = {
        f"{user['username']} ({user['email']})": user
        for user in users
    }
    if user_options:
        selected_user_label = st.selectbox(
            "Profile",
            options=list(user_options.keys()),
            key="admin_role_user",
        )
        selected_user = user_options[selected_user_label]
        role_options = ["admin", "user"]
        selected_role = st.selectbox(
            "Role",
            options=role_options,
            index=role_options.index(selected_user["role"]) if selected_user["role"] in role_options else 1,
            key=f"admin_role_value_{selected_user['id']}",
        )
        if st.button("Update role", type="primary", key="admin_update_role"):
            try:
                auth_store.set_user_role(int(selected_user["id"]), selected_role)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(f"Updated {selected_user['username']} to {selected_role}.")
                st.rerun()

    if selected_user is None:
        return False

    selected_run_records = [
        record for record in run_records
        if int(record["user_id"]) == int(selected_user["id"])
    ]
    st.subheader(f"Runs for {selected_user['username']}")
    if selected_run_records:
        runs_df = pd.DataFrame(selected_run_records)
        display_df = runs_df[[
            "workspace",
            "run_id",
            "state",
            "stage",
            "doi_count",
            "updated_at",
            "last_modified",
            "run_dir",
        ]].copy()
        display_df.insert(5, "disk", runs_df["size_bytes"].map(lambda n: _format_bytes(int(n or 0))))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        run_options = {
            f"{record['workspace']} / {record['run_id']} [{record['state']}]": record
            for record in selected_run_records
        }
        selected_run_label = st.selectbox(
            "Open run",
            options=list(run_options.keys()),
            key="admin_open_run",
        )
        if st.button("Load selected run", key="admin_load_run"):
            selected_record = run_options[selected_run_label]
            try:
                run_path = safe_child_path(
                    data_root,
                    "users",
                    str(selected_record["profile_slug"]),
                    "workspaces",
                    str(selected_record["workspace"]),
                    str(selected_record["run_id"]),
                )
                st.session_state["rr"] = discover_run(run_path)
                st.session_state["admin_loaded_run_owner"] = (
                    f"{selected_record['username']} / {selected_record['workspace']}"
                )
                st.session_state["admin_show_loaded_run"] = True
                st.success(f"Loaded {selected_record['run_id']} from {selected_record['username']}.")
            except Exception as exc:
                st.error(f"Could not load run: {exc}")
        if st.session_state.get("admin_show_loaded_run") and st.session_state.get("rr") is not None:
            owner = st.session_state.get("admin_loaded_run_owner", "")
            st.info(f"Showing loaded run below: **{owner}**")
            if st.button("Stay on admin dashboard only", key="admin_hide_loaded_run"):
                st.session_state["admin_show_loaded_run"] = False
                st.rerun()
            return True
    else:
        st.info("No runs found for this profile yet.")

    return False


def _render_guide_page() -> None:
    st.header("Guide")
    st.caption("A practical walkthrough for setting up SALSA runs, tuning settings, and correcting figure extraction results.")

    quick_start_tab, settings_tab, review_tab, interventions_tab = st.tabs(
        ["Start a run", "Settings presets", "Review results", "Figure interventions"]
    )

    with quick_start_tab:
        st.subheader("Start a new run")
        st.markdown(
            """
1. Choose or create a workspace in the sidebar.
2. In **Input Source**, choose **DOI(s)**, **PDF(s)**, or **Figure(s)**.
3. Provide the inputs:
   - **DOI(s):** paste one DOI per line, up to 50.
   - **PDF(s):** upload up to 50 PDFs. Optionally add one document ID per PDF so outputs use a readable identifier instead of the filename.
   - **Figure(s):** upload up to 50 image files. Optionally add a figure set ID and one caption per uploaded figure.
4. In **Run configuration**, enter a **Run Name** if you want a stable name. Leave it blank to use a timestamp.
5. Click **Run pipeline**.

Run names may contain letters, numbers, underscores, periods, and dashes. A named run is easiest to find later from **Load an existing run**.
            """
        )

        st.subheader("Input choice tips")
        st.markdown(
            """
- Use **DOI(s)** when SALSA should download and parse papers for you.
- Use **PDF(s)** when you already have the papers locally or want to process documents that are not easy to fetch by DOI.
- Use **Figure(s)** when you only need figure data extraction from uploaded images. In this mode, table extraction and chemical composition extraction are disabled because there is no full paper context.
            """
        )

    with settings_tab:
        st.subheader("Use bundled presets")
        st.markdown(
            f"""
SALSA creates two starter settings presets:

- **{DEFAULT_SETTINGS_PRESET_NAME}:** conservative general-purpose settings. It extracts tables, figures, OCR figure data, chemical composition, and the completed dataset. Figure and table filtering are off by default.
- **{EXAMPLE_AEM_SETTINGS_PRESET_NAME}:** an example configuration for AEM degradation workflows. It enables figure and table filtering and uses stronger LLM choices for selected steps.

Open **Load Settings Preset**, choose a preset, and click **Load**. The currently loaded preset appears at the bottom of that expander.
            """
        )

        st.subheader("Adjust cost and performance")
        st.markdown(
            """
Open **Create Settings Preset** to edit the active configuration before a run.

- Turn off stages you do not need. Each disabled LLM stage reduces cost and run time.
- Use **Chat GPT** models for managed API calls, or **Ollama** models when you want local inference and have Ollama running.
- Smaller models such as `gpt-4.1-mini` are typically cheaper and faster.
- Larger or more capable models can improve extraction quality on difficult papers, noisy tables, and nuanced filtering tasks.
- For Ollama-backed steps, increase **Ollama context window (num_ctx)** when papers are being truncated, at the cost of more memory and slower local inference.
- Keep **Use LLM to extract data from figures** off when you want the editable OCR/intervention workflow. Turn it on when you prefer direct LLM figure extraction and do not need crop, axis, or series edits.
            """
        )

        st.subheader("Save custom settings")
        st.markdown(
            """
1. Edit stage checkboxes, LLM providers/models, and instruction text in **Create Settings Preset**.
2. Add or edit **Data processing rules** and output column defaults if the completed dataset should follow a specific schema.
3. Enter a **Settings preset name**.
4. Click **Save**.

Saved presets are shared across your workspaces, so you can reload the same setup for future runs.
            """
        )

    with review_tab:
        st.subheader("Load and inspect existing runs")
        st.markdown(
            """
Use **Load an existing run** in the sidebar:

1. Pick the workspace to browse.
2. Select a **Run ID**.
3. Click **Load run**.

After loading, SALSA shows the run ID, workspace, cost summary, and a **Processed DOIs** table. Select a DOI, then switch between **Parsed text**, **Tables**, **Figures**, **Chemical Composition**, and **Dataset**.
            """
        )

        st.subheader("What to check")
        st.markdown(
            """
- **Tables:** inspect extracted table files and any LLM-generated dataset rows.
- **Figures:** inspect the original figure image, OCR-extracted data, and completed-dataset rows.
- **Dataset:** review the combined output that SALSA will export downstream.
- **Cost summary:** compare stage-level usage when changing presets or models.
            """
        )

    with interventions_tab:
        st.subheader("Open figure interventions")
        st.markdown(
            """
1. Load a completed or partially completed run.
2. Select a DOI and open **Figures**.
3. For a figure, click **Load Interventions**.

Interventions are saved under the run, so you can load the same run later and continue from the saved corrections.
            """
        )

        st.subheader("Crop figures")
        st.markdown(
            """
Use **Crop Figure** when one image contains multiple charts or panels.

1. Draw rectangles around each chart.
2. Use **Move/resize rectangles** to refine the crop boxes.
3. Click **Save crops**.
4. Run OCR data extraction on the crops so each subfigure gets its own extracted dataset.
            """
        )

        st.subheader("Modify axes and titles")
        st.markdown(
            """
Open **Axis Configuration** to correct axis titles, axis type, and calibration.

- Edit X and Y axis titles and click **Save axis titles**.
- For calibration, click Xmin, Xmax, Ymin, and Ymax on the image.
- Enter the real axis values, choose linear or log10 axes, then click **Save calibration override from clicks**.
            """
        )

        st.subheader("Modify series")
        st.markdown(
            """
Open **Series Management** to correct which plotted marks belong to each data series.

- Add, rename, or remove series as needed.
- Assign or correct detected masks/points for each series.
- Save the series mapping before rerunning extraction.
            """
        )

        st.subheader("Rerun extraction after edits")
        st.markdown(
            """
- Click **Rerun OCR data extraction** after crop, axis, or series changes. This reapplies overrides and regenerates OCR-derived figure data.
- Click **Run LLM dataset generation** after OCR data looks correct. This regenerates completed-dataset rows for the selected figure or subfigure.

If the LLM rerun is unavailable, check that **Complete dataset** has an LLM model configured in the sidebar and that an API key is present for Chat GPT-backed steps.
            """
        )


def clear_processing_llm_env_if_disabled() -> None:
    """Prevent stale processing-model env vars from affecting in-process reruns."""
    if st.session_state.get("use_separate_processing_llm"):
        return
    for key in (
        "SALSA_PROCESSING_LLM_PROVIDER",
        "SALSA_PROCESSING_LLM_MODEL",
        "SALSA_PROCESSING_OPENAI_BASE_URL",
        "SALSA_PROCESSING_OPENAI_API_KEY",
    ):
        os.environ.pop(key, None)


_CI_COLUMN_NAME = "column name"
_CI_DEFAULT_VALUE = "default value"
_CI_COLUMN_DEFAULTS_SOURCE_KEY = "ci_column_defaults_table_source"
_CI_COLUMN_DEFAULTS_VERSION_KEY = "ci_column_defaults_table_version"
_CI_PENDING_APPLY_KEY = "ci_pending_apply"
_CI_FLASH_MESSAGE_KEY = "ci_flash_message"
_CI_EDITOR_KEYS = (
    "ci_table_extraction",
    "ci_figure_filter",
    "ci_table_filter",
    "ci_chemical_composition",
    "ci_figure_data",
    "ci_data_processing",
    _CI_COLUMN_DEFAULTS_SOURCE_KEY,
    _CI_COLUMN_DEFAULTS_VERSION_KEY,
)


def _column_defaults_table_from_instructions(ci: dict[str, Any]) -> pd.DataFrame:
    output_columns = ci.get("output_columns", [])
    if isinstance(output_columns, str):
        output_columns = [c.strip() for c in output_columns.splitlines() if c.strip()]

    missing_defaults = ci.get("missing_value_defaults", {})
    if isinstance(missing_defaults, str):
        try:
            missing_defaults = json.loads(missing_defaults)
        except Exception:
            missing_defaults = {}

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_column in output_columns or []:
        column_name = str(raw_column).strip()
        if not column_name or column_name in seen:
            continue
        seen.add(column_name)
        default_value = ""
        if isinstance(missing_defaults, dict):
            raw_default = missing_defaults.get(column_name, "")
            default_value = "" if raw_default is None else str(raw_default)
        rows.append(
            {
                _CI_COLUMN_NAME: column_name,
                _CI_DEFAULT_VALUE: default_value,
            }
        )

    if isinstance(missing_defaults, dict):
        for raw_column, raw_default in missing_defaults.items():
            column_name = str(raw_column).strip()
            if not column_name or column_name in seen:
                continue
            seen.add(column_name)
            rows.append(
                {
                    _CI_COLUMN_NAME: column_name,
                    _CI_DEFAULT_VALUE: "" if raw_default is None else str(raw_default),
                }
            )

    return pd.DataFrame(rows, columns=[_CI_COLUMN_NAME, _CI_DEFAULT_VALUE])


def _normalize_column_defaults_table(
    table_value: Any,
) -> tuple[list[str], dict[str, str]]:
    if isinstance(table_value, pd.DataFrame):
        records = table_value.to_dict(orient="records")
    elif isinstance(table_value, list):
        records = table_value
    else:
        records = []

    output_columns: list[str] = []
    missing_defaults: dict[str, str] = {}
    seen: set[str] = set()

    for record in records:
        raw_column = record.get(_CI_COLUMN_NAME, "")
        raw_default = record.get(_CI_DEFAULT_VALUE, "")

        column_name = "" if pd.isna(raw_column) else str(raw_column).strip()
        if not column_name:
            continue

        if column_name not in seen:
            output_columns.append(column_name)
            seen.add(column_name)

        default_value = "" if pd.isna(raw_default) else str(raw_default).strip()
        if default_value:
            missing_defaults[column_name] = default_value
        else:
            missing_defaults.pop(column_name, None)

    return output_columns, missing_defaults


def _custom_instructions_from_editor_state(column_defaults_table: Any) -> dict[str, Any]:
    output_columns, missing_defaults = _normalize_column_defaults_table(column_defaults_table)
    return {
        "table_extraction_rules": st.session_state.get("ci_table_extraction", ""),
        "figure_filter_criteria": st.session_state.get("ci_figure_filter", ""),
        "table_filter_criteria": st.session_state.get("ci_table_filter", ""),
        "chemical_composition_rules": st.session_state.get("ci_chemical_composition", ""),
        "figure_data_extraction_rules": st.session_state.get("ci_figure_data", ""),
        "data_processing_rules": st.session_state.get("ci_data_processing", ""),
        "output_columns": output_columns,
        "missing_value_defaults": missing_defaults,
    }


def _ci_column_defaults_widget_key() -> str:
    version = int(st.session_state.get(_CI_COLUMN_DEFAULTS_VERSION_KEY, 0))
    return f"ci_column_defaults_table_widget_{version}"


def _apply_custom_instruction_state(
    ci: dict[str, Any],
    *,
    preset_name: Optional[str],
) -> None:
    st.session_state["custom_instructions"] = ci
    st.session_state["current_preset_name"] = preset_name
    st.session_state["ci_table_extraction"] = ci.get("table_extraction_rules", "")
    st.session_state["ci_figure_filter"] = ci.get("figure_filter_criteria", "")
    st.session_state["ci_table_filter"] = ci.get("table_filter_criteria", "")
    st.session_state["ci_chemical_composition"] = ci.get("chemical_composition_rules", "")
    st.session_state["ci_figure_data"] = ci.get("figure_data_extraction_rules", "")
    st.session_state["ci_data_processing"] = ci.get("data_processing_rules", "")
    if _CI_COLUMN_DEFAULTS_VERSION_KEY not in st.session_state:
        st.session_state[_CI_COLUMN_DEFAULTS_VERSION_KEY] = 0
    else:
        st.session_state[_CI_COLUMN_DEFAULTS_VERSION_KEY] += 1
    st.session_state[_CI_COLUMN_DEFAULTS_SOURCE_KEY] = _column_defaults_table_from_instructions(ci)
    st.session_state["ci_new_preset_name"] = preset_name or ""


def _queue_custom_instruction_state(
    ci: dict[str, Any],
    *,
    preset_name: Optional[str],
    flash_message: Optional[str] = None,
) -> None:
    st.session_state[_CI_PENDING_APPLY_KEY] = {
        "ci": ci,
        "preset_name": preset_name,
    }
    if flash_message:
        st.session_state[_CI_FLASH_MESSAGE_KEY] = flash_message


def _apply_pending_custom_instruction_state() -> Optional[str]:
    pending = st.session_state.pop(_CI_PENDING_APPLY_KEY, None)
    if isinstance(pending, dict):
        ci = pending.get("ci")
        if isinstance(ci, dict):
            _apply_custom_instruction_state(
                ci,
                preset_name=pending.get("preset_name"),
            )
    return st.session_state.pop(_CI_FLASH_MESSAGE_KEY, None)


SETTINGS_PRESETS_DIR_NAME = "settings_presets"
DEFAULT_SETTINGS_PRESET_NAME = "default"
EXAMPLE_AEM_SETTINGS_PRESET_NAME = "example_aem_degradation"
_SETTINGS_PENDING_APPLY_KEY = "settings_pending_apply"
_SETTINGS_FLASH_MESSAGE_KEY = "settings_flash_message"

_STAGE_STATE_KEYS = {
    "exclude_full_text": "settings_exclude_full_text",
    "run_extract_tables": "settings_run_extract_tables",
    "run_extract_figures": "settings_run_extract_figures",
    "run_filter_tables": "settings_run_filter_tables",
    "run_filter_figures": "settings_run_filter_figures",
    "run_plextract": "settings_run_plextract",
    "bypass_ocr_with_llm": "settings_bypass_ocr_with_llm",
    "run_chemical_composition": "settings_run_chemical_composition",
    "run_complete_dataset": "settings_run_complete_dataset",
}

_DEFAULT_STAGE_SETTINGS = {
    "exclude_full_text": True,
    "run_extract_tables": True,
    "run_extract_figures": True,
    "run_filter_tables": False,
    "run_filter_figures": False,
    "run_plextract": True,
    "bypass_ocr_with_llm": False,
    "run_chemical_composition": True,
    "run_complete_dataset": True,
}

_AEM_EXAMPLE_STAGE_SETTINGS = {
    **_DEFAULT_STAGE_SETTINGS,
    "run_filter_figures": True,
    "run_filter_tables": True,
}

DEFAULT_SETTINGS_LLM_MODEL = "gpt-4.1-mini"
AEM_EXAMPLE_LLM_MODELS = {
    LLM_STEP_EXTRACT_TABLES: "gpt-4.1",
    LLM_STEP_CHEM_RELEVANT: "gpt-4.1",
    LLM_STEP_CHEM_COMPOSITION: "gpt-5.5",
    LLM_STEP_FILTER_FIGURES: "gpt-4.1-mini",
    LLM_STEP_FILTER_TABLES: "gpt-4.1",
    LLM_STEP_FIGURE_DATA: "gpt-5.5",
    LLM_STEP_COMPLETE_DATASET: "gpt-4.1",
}


def _default_settings_instructions() -> dict[str, Any]:
    return {
        "table_extraction_rules": DEFAULT_TABLE_EXTRACTION_RULES,
        "figure_filter_criteria": "",
        "table_filter_criteria": "",
        "chemical_composition_rules": "",
        "figure_data_extraction_rules": "",
        "data_processing_rules": "",
        "output_columns": [],
        "missing_value_defaults": {},
    }


def _settings_presets_dir(workspaces_root: Path) -> Path:
    d = Path(workspaces_root) / SETTINGS_PRESETS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_settings_presets(workspaces_root: Path) -> list[str]:
    return sorted(
        p.stem for p in _settings_presets_dir(workspaces_root).glob("*.json") if p.is_file()
    )


def _save_settings_preset(workspaces_root: Path, name: str, settings: dict[str, Any]) -> Path:
    path = _settings_presets_dir(workspaces_root) / f"{name}.json"
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return path


def _load_settings_preset(workspaces_root: Path, name: str) -> dict[str, Any]:
    path = _settings_presets_dir(workspaces_root) / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return _normalize_settings_preset(data)


def _settings_preset_exists(workspaces_root: Path, name: str) -> bool:
    return (_settings_presets_dir(workspaces_root) / f"{name}.json").is_file()


def _settings_default_llm_steps() -> dict[str, dict[str, str]]:
    return {
        step_key: {"provider": "Chat GPT", "model": DEFAULT_SETTINGS_LLM_MODEL}
        for step_key in _STEP_LLM_RENDER_ORDER
    }


def _aem_example_llm_steps() -> dict[str, dict[str, str]]:
    return {
        step_key: {
            "provider": "Chat GPT",
            "model": AEM_EXAMPLE_LLM_MODELS.get(step_key, DEFAULT_SETTINGS_LLM_MODEL),
        }
        for step_key in _STEP_LLM_RENDER_ORDER
    }


def _make_settings_preset(
    *,
    stages: dict[str, bool],
    custom_instructions: dict[str, Any],
    llm_steps: Optional[dict[str, dict[str, str]]] = None,
    include_session_llms: bool = False,
    openai_api_key: str = "",
    elsevier_api_key: str = "",
    wiley_tdm_token: str = "",
) -> dict[str, Any]:
    llm_steps = dict(llm_steps or _settings_default_llm_steps())
    if include_session_llms:
        for step_key in _STEP_LLM_RENDER_ORDER:
            llm_steps[step_key] = {
                "provider": _step_provider_label(step_key),
                "model": _step_model_value(step_key),
            }

    return {
        "version": 1,
        "openai_api_key": openai_api_key,
        "elsevier_api_key": elsevier_api_key,
        "wiley_tdm_token": wiley_tdm_token,
        "openai_base_url": st.session_state.get("openai_base_url", "https://api.openai.com/v1")
        if include_session_llms
        else "https://api.openai.com/v1",
        "ollama_host": st.session_state.get("ollama_host", default_ollama_host)
        if include_session_llms and "default_ollama_host" in globals()
        else "",
        "ollama_models_dir": st.session_state.get("ollama_models_dir", str(default_ollama_dir))
        if include_session_llms and "default_ollama_dir" in globals()
        else "",
        "ollama_num_ctx": int(st.session_state.get("ollama_num_ctx", 16384))
        if include_session_llms
        else 16384,
        "stages": dict(stages),
        "llm_steps": llm_steps,
        "custom_instructions": dict(custom_instructions),
    }


def _normalize_settings_preset(data: dict[str, Any]) -> dict[str, Any]:
    default = _make_settings_preset(
        stages=_DEFAULT_STAGE_SETTINGS,
        custom_instructions=_default_settings_instructions(),
    )
    if not isinstance(data, dict):
        return default

    stages = data.get("stages", {})
    if isinstance(stages, dict):
        default["stages"].update(
            {
                key: bool(stages.get(key, default["stages"][key]))
                for key in _DEFAULT_STAGE_SETTINGS
            }
        )

    llm_steps = data.get("llm_steps", {})
    if isinstance(llm_steps, dict):
        for step_key in _STEP_LLM_RENDER_ORDER:
            raw = llm_steps.get(step_key, {})
            if isinstance(raw, dict):
                provider = str(raw.get("provider") or "Chat GPT")
                default["llm_steps"][step_key] = {
                    "provider": provider if provider in {"Chat GPT", "Ollama"} else "Chat GPT",
                    "model": str(raw.get("model") or "").strip(),
                }

    ci = data.get("custom_instructions", {})
    if isinstance(ci, dict):
        merged_ci = _default_settings_instructions()
        merged_ci.update(ci)
        default["custom_instructions"] = merged_ci

    for key in (
        "openai_api_key",
        "elsevier_api_key",
        "wiley_tdm_token",
        "openai_base_url",
        "ollama_host",
        "ollama_models_dir",
    ):
        if key in data:
            default[key] = str(data.get(key) or "")
    if not default["openai_base_url"]:
        default["openai_base_url"] = "https://api.openai.com/v1"
    try:
        default["ollama_num_ctx"] = int(data.get("ollama_num_ctx", default["ollama_num_ctx"]))
    except Exception:
        pass

    return default


def _current_stage_settings_from_state() -> dict[str, bool]:
    return {
        stage_name: bool(st.session_state.get(state_key, default_value))
        for stage_name, state_key in _STAGE_STATE_KEYS.items()
        for default_value in [_DEFAULT_STAGE_SETTINGS[stage_name]]
    }


def _current_settings_preset_from_state(current_ci: dict[str, Any]) -> dict[str, Any]:
    _ensure_shared_llm_settings_in_state()
    return _make_settings_preset(
        stages=_current_stage_settings_from_state(),
        custom_instructions=current_ci,
        include_session_llms=True,
        openai_api_key=str(st.session_state.get("openai_api_key", "")).strip(),
        elsevier_api_key=str(st.session_state.get("elsevier_api_key", "")).strip(),
        wiley_tdm_token=str(st.session_state.get("wiley_tdm_token", "")).strip(),
    )


def _apply_settings_preset_state(settings: dict[str, Any], *, preset_name: Optional[str]) -> None:
    settings = _normalize_settings_preset(settings)
    _ensure_shared_llm_settings_in_state()
    for stage_name, state_key in _STAGE_STATE_KEYS.items():
        st.session_state[state_key] = bool(
            settings.get("stages", {}).get(stage_name, _DEFAULT_STAGE_SETTINGS[stage_name])
        )

    st.session_state["openai_api_key"] = str(settings.get("openai_api_key") or "")
    st.session_state["elsevier_api_key"] = str(settings.get("elsevier_api_key") or "")
    st.session_state["wiley_tdm_token"] = str(settings.get("wiley_tdm_token") or "")
    st.session_state["openai_base_url"] = str(
        settings.get("openai_base_url") or "https://api.openai.com/v1"
    )
    if settings.get("ollama_host"):
        st.session_state["ollama_host"] = str(settings.get("ollama_host") or "")
    if settings.get("ollama_models_dir"):
        st.session_state["ollama_models_dir"] = str(settings.get("ollama_models_dir") or "")
    st.session_state["ollama_num_ctx"] = int(settings.get("ollama_num_ctx", 16384))

    for step_key, config in settings.get("llm_steps", {}).items():
        if step_key not in _STEP_LLM_RENDER_ORDER or not isinstance(config, dict):
            continue
        st.session_state[_step_provider_key(step_key)] = str(config.get("provider") or "Chat GPT")
        model = str(config.get("model") or "").strip()
        if model:
            st.session_state[_step_model_key(step_key)] = model
        else:
            st.session_state.pop(_step_model_key(step_key), None)

    _apply_custom_instruction_state(
        settings.get("custom_instructions", _default_settings_instructions()),
        preset_name=preset_name,
    )
    st.session_state["current_settings_preset_name"] = preset_name
    st.session_state["settings_new_preset_name"] = preset_name or ""


def _queue_settings_preset_state(
    settings: dict[str, Any],
    *,
    preset_name: Optional[str],
    flash_message: Optional[str] = None,
) -> None:
    st.session_state[_SETTINGS_PENDING_APPLY_KEY] = {
        "settings": settings,
        "preset_name": preset_name,
    }
    if flash_message:
        st.session_state[_SETTINGS_FLASH_MESSAGE_KEY] = flash_message


def _apply_pending_settings_preset_state() -> Optional[str]:
    pending = st.session_state.pop(_SETTINGS_PENDING_APPLY_KEY, None)
    if isinstance(pending, dict):
        settings = pending.get("settings")
        if isinstance(settings, dict):
            _apply_settings_preset_state(
                settings,
                preset_name=pending.get("preset_name"),
            )
    return st.session_state.pop(_SETTINGS_FLASH_MESSAGE_KEY, None)


def _bootstrap_settings_presets_for_workspaces(workspaces_root: Path) -> None:
    if not _settings_preset_exists(workspaces_root, DEFAULT_SETTINGS_PRESET_NAME):
        _save_settings_preset(
            workspaces_root,
            DEFAULT_SETTINGS_PRESET_NAME,
            _make_settings_preset(
                stages=_DEFAULT_STAGE_SETTINGS,
                custom_instructions=_default_settings_instructions(),
            ),
        )
    if not _settings_preset_exists(workspaces_root, EXAMPLE_AEM_SETTINGS_PRESET_NAME):
        _save_settings_preset(
            workspaces_root,
            EXAMPLE_AEM_SETTINGS_PRESET_NAME,
            _make_settings_preset(
                stages=_AEM_EXAMPLE_STAGE_SETTINGS,
                custom_instructions=default_instructions(),
                llm_steps=_aem_example_llm_steps(),
            ),
        )


def _bootstrap_settings_presets_for_profiles(data_root: Path, users: Sequence[Any]) -> None:
    for user in users:
        try:
            profile_workspaces_root = (
                Path(data_root) / "users" / str(user["profile_slug"]) / "workspaces"
            )
            profile_workspaces_root.mkdir(parents=True, exist_ok=True)
            _bootstrap_settings_presets_for_workspaces(profile_workspaces_root)
        except Exception:
            continue


def _keep_create_settings_preset_open() -> None:
    st.session_state["create_settings_preset_expanded"] = True


def _keep_load_settings_preset_open() -> None:
    st.session_state["load_settings_preset_expanded"] = True


def _settings_checkbox(label: str, state_key: str, default: bool, **kwargs: Any) -> bool:
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    kwargs.setdefault("on_change", _keep_create_settings_preset_open)
    return bool(st.checkbox(label, key=state_key, **kwargs))


def _render_instruction_text_area(label: str, state_key: str, height: int = 120) -> None:
    st.text_area(
        label,
        height=height,
        key=state_key,
        on_change=_keep_create_settings_preset_open,
    )


# SET OLLAMA DIR AND HOST (computed outside expander for reuse)
default_ollama_dir = _default_ollama_models_dir()
default_ollama_host = _default_ollama_host()

_openai_models = [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]

_AUTH_SESSION_COOKIE_NAME = "salsa_session"
_AUTH_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30
MAX_INPUT_ITEMS = DEFAULT_INPUT_ITEM_LIMIT


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _query_param(name: str) -> str:
    try:
        value = st.query_params.get(name, "")
    except Exception:
        params = st.experimental_get_query_params()
        value = params.get(name, [""])
        if isinstance(value, list):
            value = value[0] if value else ""
    if isinstance(value, list):
        return str(value[0] if value else "")
    return str(value or "")


def _set_query_param(name: str, value: str) -> None:
    try:
        st.query_params[name] = value
    except Exception:
        params = st.experimental_get_query_params()
        params[name] = value
        st.experimental_set_query_params(**params)


def _clear_query_param(name: str) -> None:
    try:
        if name in st.query_params:
            del st.query_params[name]
    except Exception:
        params = st.experimental_get_query_params()
        params.pop(name, None)
        st.experimental_set_query_params(**params)


def _cookie_from_header(cookie_header: str, name: str) -> str:
    if not cookie_header:
        return ""
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return ""
    morsel = cookie.get(name)
    return morsel.value if morsel is not None else ""


def _request_cookie_value(name: str) -> str:
    context = getattr(st, "context", None)
    if context is not None:
        try:
            cookies = getattr(context, "cookies", None)
            if cookies and cookies.get(name):
                return str(cookies.get(name) or "")
        except Exception:
            pass
        try:
            headers = getattr(context, "headers", None)
            if headers:
                value = _cookie_from_header(str(headers.get("cookie") or headers.get("Cookie") or ""), name)
                if value:
                    return value
        except Exception:
            pass

    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers

        headers = _get_websocket_headers() or {}
        return _cookie_from_header(str(headers.get("Cookie") or headers.get("cookie") or ""), name)
    except Exception:
        return ""


def _render_cookie_script(*, token: str = "", clear: bool = False) -> None:
    cookie_name = json.dumps(_AUTH_SESSION_COOKIE_NAME)
    cookie_writer = """
        function writeCookie(value) {
            try {
                window.parent.document.cookie = value;
            } catch (err) {}
            try {
                document.cookie = value;
            } catch (err) {}
        }
    """
    if clear:
        script = f"""
        <script>
        {cookie_writer}
        writeCookie({cookie_name} + "=; Max-Age=0; Path=/; SameSite=Lax");
        window.setTimeout(function() {{
            window.parent.location.reload();
        }}, 50);
        </script>
        """
    else:
        cookie_value = json.dumps(token)
        max_age = int(_AUTH_SESSION_MAX_AGE_SECONDS)
        script = f"""
        <script>
        {cookie_writer}
        writeCookie({cookie_name} + "=" + encodeURIComponent({cookie_value}) + "; Max-Age={max_age}; Path=/; SameSite=Lax");
        </script>
        """
    components.html(script, height=1, width=1)


def _render_pending_auth_cookie_scripts() -> None:
    token = st.session_state.pop("pending_auth_cookie_token", "")
    clear_cookie = bool(st.session_state.pop("pending_clear_auth_cookie", False))
    if clear_cookie:
        if _request_cookie_value(_AUTH_SESSION_COOKIE_NAME):
            _render_cookie_script(clear=True)
    elif token:
        _render_cookie_script(token=str(token))


def _set_auth_session(auth_store: AuthStore, user: Any) -> None:
    token = auth_store.create_auth_session_token(int(user["id"]))
    st.session_state["auth_user_id"] = int(user["id"])
    st.session_state["auth_session_token"] = token
    st.session_state["pending_auth_cookie_token"] = token


def _clear_auth_session(auth_store: Optional[AuthStore] = None) -> None:
    token = str(
        st.session_state.get("auth_session_token")
        or _request_cookie_value(_AUTH_SESSION_COOKIE_NAME)
        or ""
    )
    if auth_store is not None and token:
        auth_store.revoke_auth_session_token(token)
    st.session_state["pending_clear_auth_cookie"] = True
    for key in (
        "auth_user_id",
        "auth_session_token",
        "rr",
        "workspace_name",
        "load_workspace",
        "load_run_id",
        "pending_auth_cookie_token",
        _ACTIVE_PIPELINE_RUN_KEY if "_ACTIVE_PIPELINE_RUN_KEY" in globals() else "",
    ):
        if key:
            st.session_state.pop(key, None)


def _current_user(auth_store: AuthStore):
    user_id = st.session_state.get("auth_user_id")
    token = str(
        st.session_state.get("auth_session_token")
        or _request_cookie_value(_AUTH_SESSION_COOKIE_NAME)
        or ""
    )

    if user_id:
        user = auth_store.get_user(int(user_id))
    elif token:
        user = auth_store.user_from_auth_session_token(token)
        if user is not None:
            st.session_state["auth_user_id"] = int(user["id"])
            st.session_state["auth_session_token"] = token
    else:
        user = None

    if user is None and user_id:
        _clear_auth_session()
    elif user is None and token:
        _clear_auth_session()
        _render_cookie_script(clear=True)
    return user


def _render_auth_gate(auth_store: AuthStore):
    user = _current_user(auth_store)
    if user is not None:
        return user

    st.subheader("Sign in")
    login_tab, create_tab, forgot_tab, reset_tab = st.tabs(
        ["Login", "Create Account", "Forgot Password", "Reset Password"]
    )

    with login_tab:
        with st.form("login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            user = auth_store.authenticate(identifier, password)
            if user is None:
                st.error("Invalid username/email or password.")
            else:
                _set_auth_session(auth_store, user)
                st.rerun()

    with create_tab:
        if not _env_truthy("SALSA_AUTH_ALLOW_SIGNUP", True):
            st.info("Account creation is disabled on this deployment.")
        else:
            with st.form("create_account_form"):
                username = st.text_input("Username", key="signup_username")
                email = st.text_input("Email", key="signup_email")
                first_name = st.text_input("First name", key="signup_first_name")
                last_name = st.text_input("Last name", key="signup_last_name")
                institution = st.text_input("Institution", key="signup_institution")
                password = st.text_input("Password", type="password", key="signup_pw")
                password_again = st.text_input(
                    "Confirm password",
                    type="password",
                    key="signup_pw_again",
                )
                submitted = st.form_submit_button("Create account", type="primary")
            if submitted:
                try:
                    if password != password_again:
                        raise ValueError("Passwords do not match.")
                    user = auth_store.create_user(
                        username=username,
                        email=email,
                        first_name=first_name,
                        last_name=last_name,
                        institution=institution,
                        password=password,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    _set_auth_session(auth_store, user)
                    st.success("Account created.")
                    st.rerun()

    with forgot_tab:
        smtp_configured = bool(os.environ.get("SALSA_SMTP_HOST") and os.environ.get("SALSA_SMTP_FROM"))
        print_reset_links = _env_truthy("SALSA_AUTH_PRINT_RESET_LINKS", False)
        if smtp_configured:
            st.caption("A one-time reset token will be emailed if the account exists.")
        elif print_reset_links:
            st.caption("A one-time reset token will be written to the SALSA service logs.")
        else:
            st.warning(
                "Password reset delivery is not configured. Set SMTP environment "
                "variables, or set SALSA_AUTH_PRINT_RESET_LINKS=1 for internal-only "
                "log-based resets."
            )
        with st.form("forgot_password_form"):
            identifier = st.text_input("Username or email", key="forgot_identifier")
            submitted = st.form_submit_button("Send reset token")
        if submitted:
            if not (smtp_configured or print_reset_links):
                st.error("Password reset delivery is not configured for this deployment.")
            else:
                user = auth_store.find_user(identifier)
                delivery_failed = False
                if user is not None:
                    token = auth_store.create_password_reset_token(int(user["id"]))
                    reset_url = build_reset_url(
                        os.environ.get("SALSA_PUBLIC_URL", ""),
                        token,
                        user["email"],
                    )
                    mailed = False
                    if smtp_configured:
                        try:
                            mailed = send_password_reset_email(user["email"], reset_url)
                        except Exception as exc:
                            delivery_failed = True
                            print(f"[SALSA auth] Password reset email failed: {exc}")
                    if print_reset_links and not mailed:
                        print(
                            "[SALSA auth] Password reset token for "
                            f"{user['username']}: {reset_url}"
                        )
                if delivery_failed and not print_reset_links:
                    st.error("Reset email could not be sent. Check the SMTP settings and service logs.")
                elif smtp_configured:
                    st.success("If that account exists, a reset email has been sent.")
                else:
                    st.success("If that account exists, a reset token was written to the service logs.")
            if not smtp_configured and not print_reset_links:
                st.info(
                    "The service reads deployment settings from "
                    "`/home/wschertzer3/SALSA/.env.production`."
                )
            elif not smtp_configured and print_reset_links:
                st.info("Check the service logs with `journalctl -u salsa`.")
            elif smtp_configured and not print_reset_links:
                st.info(
                    "If email delivery still fails, enable `SALSA_AUTH_PRINT_RESET_LINKS=1` "
                    "temporarily to write fallback reset links to the service logs."
                )

    with reset_tab:
        with st.form("reset_password_form"):
            identifier = st.text_input(
                "Username or email",
                value=_query_param("reset_identifier"),
                key="reset_identifier",
            )
            token = st.text_input(
                "Reset token",
                value=_query_param("reset_token"),
                key="reset_token",
            )
            password = st.text_input("New password", type="password", key="reset_pw")
            password_again = st.text_input(
                "Confirm new password",
                type="password",
                key="reset_pw_again",
            )
            submitted = st.form_submit_button("Reset password", type="primary")
        if submitted:
            try:
                if password != password_again:
                    raise ValueError("Passwords do not match.")
                ok = auth_store.reset_password_with_token(
                    identifier=identifier,
                    token=token,
                    new_password=password,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                if ok:
                    st.success("Password reset. You can log in now.")
                    _clear_query_param("reset_token")
                    _clear_query_param("reset_identifier")
                else:
                    st.error("Invalid or expired reset token.")

    st.stop()



st.set_page_config(page_title="SALSA UI", layout="wide")
render_salsa_brand_header()

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

scripts_root = (Path(__file__).resolve().parent / "..").resolve()
salsa_data_root = data_root_from_env(scripts_root / "ui_runs")
auth_db_path = Path(os.environ.get("SALSA_AUTH_DB", str(salsa_data_root / "auth.sqlite3")))
auth_store = AuthStore(auth_db_path)
auth_store.bootstrap_default_admin()
if _query_param("salsa_session") and not st.session_state.get("_legacy_auth_query_cleared"):
    st.session_state["_legacy_auth_query_cleared"] = True
    _clear_query_param("salsa_session")
_render_pending_auth_cookie_scripts()
current_user = _render_auth_gate(auth_store)
MAX_INPUT_ITEMS = ADMIN_INPUT_ITEM_LIMIT if is_admin_user(current_user) else DEFAULT_INPUT_ITEM_LIMIT
workspaces_root = user_workspaces_root(salsa_data_root, current_user)
_bootstrap_settings_presets_for_profiles(salsa_data_root, auth_store.list_users())
show_admin_page = False

# ---------------- Sidebar ----------------
with st.sidebar:
    if "salsa_active_page" not in st.session_state:
        st.session_state["salsa_active_page"] = "tool"

    if st.session_state["salsa_active_page"] == "guide":
        if st.button("Back to SALSA tool", use_container_width=True):
            st.session_state["salsa_active_page"] = "tool"
            st.rerun()
    else:
        if st.button("Guide", use_container_width=True):
            st.session_state["salsa_active_page"] = "guide"
            st.rerun()

    st.divider()

    with st.expander("Profile", expanded=False):
        profile_label = current_user["display_name"] or current_user["username"]
        st.write(f"Signed in as **{profile_label}**")
        st.caption(current_user["email"])
        st.caption(f"Role: **{current_user['role']}**")
        if st.button("Logout"):
            _clear_auth_session(auth_store)
            st.rerun()

        with st.form("change_password_form"):
            current_password = st.text_input(
                "Current password",
                type="password",
                key="change_pw_current",
            )
            new_password = st.text_input(
                "New password",
                type="password",
                key="change_pw_new",
            )
            new_password_again = st.text_input(
                "Confirm new password",
                type="password",
                key="change_pw_new_again",
            )
            submitted = st.form_submit_button("Update password")
        if submitted:
            try:
                if new_password != new_password_again:
                    raise ValueError("Passwords do not match.")
                validate_password(new_password)
                if auth_store.authenticate(current_user["username"], current_password) is None:
                    raise ValueError("Current password is incorrect.")
                auth_store.set_password(int(current_user["id"]), new_password)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Password updated.")

    st.divider()
    if is_admin_user(current_user):
        selected_page = st.radio(
            "View",
            options=["Run workspace", "Admin"],
            horizontal=True,
            key="salsa_sidebar_view",
        )
        show_admin_page = selected_page == "Admin"
        st.divider()

    # Choose/create workspace
    st.header("Workspace")

    # List existing workspaces (filter out shared config directories)
    existing_workspaces = [
        ws for ws in _list_subdirs(workspaces_root)
        if not _ignored_workspace_or_run_name(ws)
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
    st.header("Input Source")
    input_source = st.radio(
        "Source",
        options=["DOI(s)", "PDF(s)", "Figure(s)"],
        horizontal=True,
        key="input_source",
    )
    use_docling_pdf_uploads = input_source == "PDF(s)"
    use_figure_uploads = input_source == "Figure(s)"
    use_file_uploads = use_docling_pdf_uploads or use_figure_uploads

    st.divider()
    st.header("Settings")
    _bootstrap_settings_presets_for_workspaces(workspaces_root)
    settings_flash_message = _apply_pending_settings_preset_state()
    if "custom_instructions" not in st.session_state:
        _apply_settings_preset_state(
            _load_settings_preset(workspaces_root, DEFAULT_SETTINGS_PRESET_NAME),
            preset_name=DEFAULT_SETTINGS_PRESET_NAME,
        )
    elif any(key not in st.session_state for key in _CI_EDITOR_KEYS):
        _apply_custom_instruction_state(
            st.session_state["custom_instructions"],
            preset_name=st.session_state.get("current_preset_name"),
        )
    if settings_flash_message:
        st.success(settings_flash_message)

    existing_settings_presets = _list_settings_presets(workspaces_root)

    with st.expander(
        "Create Settings Preset",
        expanded=bool(st.session_state.get("create_settings_preset_expanded", False)),
    ):
        _ensure_shared_llm_settings_in_state()
        st.text_input(
            "OpenAI API key",
            key="openai_api_key",
            type="password",
            placeholder="sk-...",
            help="Required for every selected stage that uses Chat GPT.",
            on_change=_keep_create_settings_preset_open,
        )
        publisher_key_cols = st.columns(2)
        with publisher_key_cols[0]:
            st.text_input(
                "Elsevier API key",
                key="elsevier_api_key",
                type="password",
                placeholder="Elsevier API key",
                help="Used for Elsevier TDM article and figure downloads.",
                on_change=_keep_create_settings_preset_open,
            )
        with publisher_key_cols[1]:
            st.text_input(
                "Wiley TDM token",
                key="wiley_tdm_token",
                type="password",
                placeholder="Wiley TDM token",
                help="Used for Wiley TDM article and figure downloads.",
                on_change=_keep_create_settings_preset_open,
            )

        if use_figure_uploads:
            run_extract_figures = True
            run_extract_tables = False
            run_chemical_composition = False
            run_filter_tables = False
            exclude_full_text = True
            st.checkbox("Exclude full text from context", value=True, disabled=True, key="settings_upload_exclude_full_text")
            st.checkbox("Extract tables", value=False, disabled=True, key="settings_upload_extract_tables")
            st.checkbox("Extract figures", value=True, disabled=True, key="settings_upload_extract_figures")
            st.checkbox("Filter tables", value=False, disabled=True, key="settings_upload_filter_tables")
        else:
            exclude_full_text = _settings_checkbox(
                "Exclude full text from context",
                _STAGE_STATE_KEYS["exclude_full_text"],
                _DEFAULT_STAGE_SETTINGS["exclude_full_text"],
            )
            run_extract_tables = _settings_checkbox(
                "Extract tables",
                _STAGE_STATE_KEYS["run_extract_tables"],
                _DEFAULT_STAGE_SETTINGS["run_extract_tables"],
            )
            run_extract_figures = _settings_checkbox(
                "Extract figures",
                _STAGE_STATE_KEYS["run_extract_figures"],
                _DEFAULT_STAGE_SETTINGS["run_extract_figures"],
            )
            run_filter_tables = _settings_checkbox(
                "Filter tables",
                _STAGE_STATE_KEYS["run_filter_tables"],
                _DEFAULT_STAGE_SETTINGS["run_filter_tables"],
            )

        run_filter_figures = _settings_checkbox(
            "Filter figures",
            _STAGE_STATE_KEYS["run_filter_figures"],
            _DEFAULT_STAGE_SETTINGS["run_filter_figures"],
        )
        run_plextract = _settings_checkbox(
            "Extract data from figures",
            _STAGE_STATE_KEYS["run_plextract"],
            _DEFAULT_STAGE_SETTINGS["run_plextract"],
        )
        _, figure_data_llm_col = st.columns([0.08, 0.92])
        with figure_data_llm_col:
            bypass_ocr_with_llm = _settings_checkbox(
                "Use LLM to extract data from figures",
                _STAGE_STATE_KEYS["bypass_ocr_with_llm"],
                _DEFAULT_STAGE_SETTINGS["bypass_ocr_with_llm"],
                help="Writes an LLM-digitized figure dataset. Cannot crop images or modify series/axes in this mode",
            )
        if use_figure_uploads:
            st.checkbox(
                "Extract chemical composition",
                value=False,
                disabled=True,
                key="settings_upload_chemical_composition",
            )
            run_chemical_composition = False
        else:
            run_chemical_composition = _settings_checkbox(
                "Extract chemical composition",
                _STAGE_STATE_KEYS["run_chemical_composition"],
                _DEFAULT_STAGE_SETTINGS["run_chemical_composition"],
            )
        run_complete_dataset = _settings_checkbox(
            "Complete dataset",
            _STAGE_STATE_KEYS["run_complete_dataset"],
            _DEFAULT_STAGE_SETTINGS["run_complete_dataset"],
        )

        ci = st.session_state["custom_instructions"]
        ollama_host = _normalize_ollama_host(
            st.session_state.get("ollama_host", default_ollama_host)
        )
        ollama_available = _ollama_server_is_available(ollama_host)
        if not ollama_available:
            st.info(
                f"No Ollama server is running at `{ollama_host}`. "
                "Start Ollama to enable Ollama model selection."
            )

        llm_steps_to_render = _active_llm_steps(
            run_extract_tables=run_extract_tables,
            use_docling_pdf_uploads=use_file_uploads,
            run_chemical_composition=run_chemical_composition,
            run_filter_figures=run_filter_figures,
            run_filter_tables=run_filter_tables and not use_figure_uploads,
            run_plextract=run_plextract,
            bypass_ocr_with_llm=bypass_ocr_with_llm,
            run_complete_dataset=run_complete_dataset,
        )
        if ollama_available and _any_step_uses_ollama(llm_steps_to_render):
            st.number_input(
                "Ollama context window (num_ctx)",
                min_value=4096,
                step=1024,
                value=int(st.session_state.get("ollama_num_ctx", 16384)),
                key="ollama_num_ctx",
                help="Shared across all Ollama-backed steps.",
                on_change=_keep_create_settings_preset_open,
            )

        if run_extract_tables and not use_file_uploads:
            with st.container():
                _render_step_llm_selector(
                    LLM_STEP_EXTRACT_TABLES,
                    caption=_llm_step_caption(LLM_STEP_EXTRACT_TABLES),
                    title=f"{LLM_STEP_LABELS[LLM_STEP_EXTRACT_TABLES]} settings",
                )
                _render_instruction_text_area(
                    "Table extraction rules",
                    "ci_table_extraction",
                )
            st.divider()

        if run_chemical_composition:
            with st.container():
                _render_step_llm_selector(
                    LLM_STEP_CHEM_RELEVANT,
                    caption=_llm_step_caption(LLM_STEP_CHEM_RELEVANT),
                    title=f"{LLM_STEP_LABELS[LLM_STEP_CHEM_RELEVANT]} settings",
                )
                st.divider()
                _render_step_llm_selector(
                    LLM_STEP_CHEM_COMPOSITION,
                    caption=_llm_step_caption(LLM_STEP_CHEM_COMPOSITION),
                    title=f"{LLM_STEP_LABELS[LLM_STEP_CHEM_COMPOSITION]} settings",
                )
                _render_instruction_text_area(
                    "Chemical composition instructions",
                    "ci_chemical_composition",
                    height=120,
                )
            st.divider()

        if run_filter_figures:
            with st.container():
                _render_step_llm_selector(
                    LLM_STEP_FILTER_FIGURES,
                    caption=_llm_step_caption(LLM_STEP_FILTER_FIGURES),
                    title=f"{LLM_STEP_LABELS[LLM_STEP_FILTER_FIGURES]} settings",
                )
                _render_instruction_text_area(
                    "Figure filtering criteria",
                    "ci_figure_filter",
                    height=150,
                )
            st.divider()

        if run_filter_tables and not use_figure_uploads:
            with st.container():
                _render_step_llm_selector(
                    LLM_STEP_FILTER_TABLES,
                    caption=_llm_step_caption(LLM_STEP_FILTER_TABLES),
                    title=f"{LLM_STEP_LABELS[LLM_STEP_FILTER_TABLES]} settings",
                )
                _render_instruction_text_area(
                    "Table filtering criteria",
                    "ci_table_filter",
                    height=150,
                )
            st.divider()

        if run_plextract and bypass_ocr_with_llm:
            with st.container():
                _render_step_llm_selector(
                    LLM_STEP_FIGURE_DATA,
                    caption=_llm_step_caption(LLM_STEP_FIGURE_DATA),
                    title=f"{LLM_STEP_LABELS[LLM_STEP_FIGURE_DATA]} settings",
                )
                _render_instruction_text_area(
                    "Figure data extraction instructions",
                    "ci_figure_data",
                    height=120,
                )
            st.divider()

        ci_column_defaults_table = st.session_state.get(
            _CI_COLUMN_DEFAULTS_SOURCE_KEY,
            _column_defaults_table_from_instructions(ci),
        )
        if run_complete_dataset:
            with st.container():
                _render_step_llm_selector(
                    LLM_STEP_COMPLETE_DATASET,
                    caption=_llm_step_caption(LLM_STEP_COMPLETE_DATASET),
                    title=f"{LLM_STEP_LABELS[LLM_STEP_COMPLETE_DATASET]} settings",
                )
                _render_instruction_text_area(
                    "Data processing rules",
                    "ci_data_processing",
                )
                ci_column_defaults_table = st.data_editor(
                    st.session_state.get(
                        _CI_COLUMN_DEFAULTS_SOURCE_KEY,
                        _column_defaults_table_from_instructions(ci),
                    ),
                    hide_index=True,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=_ci_column_defaults_widget_key(),
                    on_change=_keep_create_settings_preset_open,
                )

        current_ci = _custom_instructions_from_editor_state(ci_column_defaults_table)
        st.session_state["custom_instructions"] = current_ci

        st.subheader("Save Preset")
        col_name, col_save_btn = st.columns([3, 1])
        with col_name:
            preset_name = st.text_input(
                "Settings preset name",
                value=st.session_state.get("current_settings_preset_name", ""),
                key="settings_new_preset_name",
                placeholder="Enter a name for this settings preset",
                on_change=_keep_create_settings_preset_open,
            )
        with col_save_btn:
            st.write("")
            st.write("")
            if st.button("Save", type="primary", key="save_settings_preset_btn") and preset_name.strip():
                clean_preset_name = preset_name.strip()
                current_settings = _make_settings_preset(
                    stages={
                        "exclude_full_text": exclude_full_text,
                        "run_extract_tables": run_extract_tables,
                        "run_extract_figures": run_extract_figures,
                        "run_filter_tables": run_filter_tables and not use_figure_uploads,
                        "run_filter_figures": run_filter_figures,
                        "run_plextract": run_plextract,
                        "bypass_ocr_with_llm": bypass_ocr_with_llm,
                        "run_chemical_composition": run_chemical_composition,
                        "run_complete_dataset": run_complete_dataset,
                    },
                    custom_instructions=current_ci,
                    include_session_llms=True,
                    openai_api_key=str(st.session_state.get("openai_api_key", "")).strip(),
                    elsevier_api_key=str(st.session_state.get("elsevier_api_key", "")).strip(),
                    wiley_tdm_token=str(st.session_state.get("wiley_tdm_token", "")).strip(),
                )
                _save_settings_preset(workspaces_root, clean_preset_name, current_settings)
                _queue_settings_preset_state(
                    current_settings,
                    preset_name=clean_preset_name,
                    flash_message=f"Saved settings preset: {clean_preset_name}",
                )
                st.rerun()

        st.session_state["create_settings_preset_expanded"] = False

    with st.expander(
        "Load Settings Preset",
        expanded=bool(st.session_state.get("load_settings_preset_expanded", False)),
    ):
        if existing_settings_presets:
            col_load_a, col_load_b = st.columns([3, 1])
            with col_load_a:
                settings_preset_choice = st.selectbox(
                    "Saved settings preset",
                    options=[""] + existing_settings_presets,
                    key="settings_preset_choice",
                    on_change=_keep_load_settings_preset_open,
                )
            with col_load_b:
                st.write("")
                st.write("")
                if st.button(
                    "Load",
                    type="secondary",
                    key="load_settings_preset_btn",
                    on_click=_keep_load_settings_preset_open,
                ) and settings_preset_choice:
                    loaded_settings = _load_settings_preset(workspaces_root, settings_preset_choice)
                    _queue_settings_preset_state(
                        loaded_settings,
                        preset_name=settings_preset_choice,
                        flash_message=f"Loaded settings preset: {settings_preset_choice}",
                    )
                    st.rerun()
        else:
            st.info("No settings presets saved yet.")

        if st.session_state.get("current_settings_preset_name"):
            st.caption(f"Currently loaded: **{st.session_state['current_settings_preset_name']}**")
        else:
            st.caption("Currently loaded: **Unsaved settings**")

    # st.divider()
    # st.header("Axis & Units Configuration")

    # # Initialize axis config in session state
    # if "axis_config" not in st.session_state:
    #     # Try to load from active workspace if available from session state
    #     _active_rr = st.session_state.get("rr")
    #     if _active_rr and hasattr(_active_rr, 'run_dir') and _active_rr.run_dir:
    #         st.session_state["axis_config"] = load_axis_config(_active_rr.run_dir)
    #     else:
    #         st.session_state["axis_config"] = load_axis_config(workspaces_root)
    #     st.session_state["current_axis_preset_name"] = None

    # _ax_cfg = st.session_state["axis_config"]

    # # --- Load axis presets ---
    # _active_rr = st.session_state.get("rr")
    # workspace_for_axis = (_active_rr.run_dir if (_active_rr and hasattr(_active_rr, 'run_dir') and _active_rr.run_dir)
    #                       else workspaces_root)
    # existing_axis_presets = list_axis_presets(workspace_for_axis)

    # with st.expander("Load Axis Preset", expanded=False):
    #     if existing_axis_presets:
    #         col_ax_load_a, col_ax_load_b = st.columns([3, 1])
    #         with col_ax_load_a:
    #             axis_preset_choice = st.selectbox(
    #                 "Select axis preset to load",
    #                 options=[""] + existing_axis_presets,
    #                 key="axis_preset_choice",
    #             )
    #         with col_ax_load_b:
    #             if st.button("Load", type="secondary", key="load_axis_preset_btn") and axis_preset_choice:
    #                 loaded = load_axis_preset(workspace_for_axis, axis_preset_choice)
    #                 if loaded:
    #                     st.session_state["axis_config"] = loaded
    #                     st.session_state["current_axis_preset_name"] = axis_preset_choice
    #                     st.success(f"Loaded axis preset: {axis_preset_choice}")
    #                     st.rerun()
    #     else:
    #         st.info("No axis presets saved yet. Create your first preset below.")

    #     # Show currently loaded preset
    #     if st.session_state.get("current_axis_preset_name"):
    #         st.caption(f"Currently editing: **{st.session_state['current_axis_preset_name']}**")
    #     else:
    #         st.caption("Currently editing: **Unsaved changes (default)**")

    # # --- Axis config editors in expanders ---
    # with st.expander("Unit Aliases", expanded=False):
    #     st.caption("Map raw unit strings (e.g. `sec`, `hrs`) to their canonical form (e.g. `s`, `h`).")
    #     _ua = _ax_cfg.get("unit_aliases", {})
    #     _ua_rows = [{"raw": k, "canonical": v} for k, v in sorted(_ua.items())]
    #     _ua_df = pd.DataFrame(_ua_rows) if _ua_rows else pd.DataFrame(columns=["raw", "canonical"])
    #     _ua_edited = st.data_editor(
    #         _ua_df,
    #         num_rows="dynamic",
    #         use_container_width=True,
    #         key="sidebar_ax_unit_aliases_editor",
    #     )
    #     # Store edited data in session state
    #     if st.button("Apply changes", key="apply_unit_aliases"):
    #         new_ua = {}
    #         for _, row in _ua_edited.iterrows():
    #             raw = str(row.get("raw", "")).strip()
    #             can = str(row.get("canonical", "")).strip()
    #             if raw and can:
    #                 new_ua[raw] = can
    #         st.session_state["axis_config"]["unit_aliases"] = new_ua
    #         st.success("Applied unit alias changes")

    # with st.expander("Unit Dimensions", expanded=False):
    #     st.caption("Map canonical unit strings to a dimension name (e.g. `s` -> `time`, `mS/cm` -> `conductivity`).")
    #     _ud = _ax_cfg.get("unit_dimensions", {})
    #     _ud_rows = [{"unit": k, "dimension": v} for k, v in sorted(_ud.items())]
    #     _ud_df = pd.DataFrame(_ud_rows) if _ud_rows else pd.DataFrame(columns=["unit", "dimension"])
    #     _ud_edited = st.data_editor(
    #         _ud_df,
    #         num_rows="dynamic",
    #         use_container_width=True,
    #         key="sidebar_ax_unit_dims_editor",
    #     )
    #     if st.button("Apply changes", key="apply_unit_dims"):
    #         new_ud = {}
    #         for _, row in _ud_edited.iterrows():
    #             unit = str(row.get("unit", "")).strip()
    #             dim = str(row.get("dimension", "")).strip()
    #             if unit and dim:
    #                 new_ud[unit] = dim
    #         st.session_state["axis_config"]["unit_dimensions"] = new_ud
    #         st.success("Applied unit dimension changes")

    # with st.expander("Canonical Targets", expanded=False):
    #     st.caption("For each dimension, choose the canonical target unit that values are converted to.")
    #     _ct = _ax_cfg.get("canonical_targets", {})
    #     _ct_rows = [{"dimension": k, "target_unit": v} for k, v in sorted(_ct.items())]
    #     _ct_df = pd.DataFrame(_ct_rows) if _ct_rows else pd.DataFrame(columns=["dimension", "target_unit"])
    #     _ct_edited = st.data_editor(
    #         _ct_df,
    #         num_rows="dynamic",
    #         use_container_width=True,
    #         key="sidebar_ax_canon_targets_editor",
    #     )
    #     if st.button("Apply changes", key="apply_canon_targets"):
    #         new_ct = {}
    #         for _, row in _ct_edited.iterrows():
    #             dim = str(row.get("dimension", "")).strip()
    #             target = str(row.get("target_unit", "")).strip()
    #             if dim and target:
    #                 new_ct[dim] = target
    #         st.session_state["axis_config"]["canonical_targets"] = new_ct
    #         st.success("Applied canonical target changes")

    # # --- Save axis preset ---
    # with st.expander("Save Axis Preset", expanded=False):
    #     st.subheader("Save Preset")
    #     col_ax_name, col_ax_save, col_ax_del = st.columns([3, 1, 1])

    #     with col_ax_name:
    #         default_ax_name = st.session_state.get("current_axis_preset_name", "")
    #         axis_preset_name = st.text_input(
    #             "Axis preset name",
    #             value=default_ax_name,
    #             key="axis_new_preset_name",
    #             placeholder="Enter a name for this axis preset"
    #         )

    #     with col_ax_save:
    #         st.write("")  # spacing
    #         st.write("")  # spacing
    #         if st.button("Save", type="primary", key="save_axis_preset_btn") and axis_preset_name.strip():
    #             save_axis_preset(workspace_for_axis, axis_preset_name.strip(), st.session_state["axis_config"])
    #             # Also save to the workspace
    #             save_axis_config(workspace_for_axis, st.session_state["axis_config"])
    #             st.session_state["current_axis_preset_name"] = axis_preset_name.strip()
    #             st.success(f"Saved axis preset: {axis_preset_name.strip()}")
    #             st.rerun()

    #     with col_ax_del:
    #         st.write("")  # spacing
    #         st.write("")  # spacing
    #         if existing_axis_presets and st.button("Delete", type="secondary", key="delete_axis_preset_btn") and st.session_state.get("axis_preset_choice"):
    #             delete_axis_preset(workspace_for_axis, st.session_state["axis_preset_choice"])
    #             if st.session_state.get("current_axis_preset_name") == st.session_state["axis_preset_choice"]:
    #                 st.session_state["current_axis_preset_name"] = None
    #             st.success(f"Deleted axis preset: {st.session_state['axis_preset_choice']}")
    #             st.rerun()

    #     # --- Reset to defaults button ---
    #     if st.button("Reset axis config to defaults", type="secondary", key="reset_axis_btn"):
    #         st.session_state["axis_config"] = reset_axis_config(workspace_for_axis)
    #         st.session_state["current_axis_preset_name"] = None
    #         st.success("Reset axis configuration to defaults")
    #         st.rerun()

    st.divider()
    with st.expander("Load an existing run", expanded=False):

        # Choose which workspace to browse when loading runs (filter out shared config directories)
        load_ws_options = [
            ws for ws in _list_subdirs(workspaces_root)
            if not _ignored_workspace_or_run_name(ws)
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

            existing_runs = sorted(
                [
                    p.name
                    for p in load_workspace_dir.iterdir()
                    if p.is_dir() and not _ignored_workspace_or_run_name(p.name)
                ],
                reverse=True,
            )
            pick = st.selectbox("Run ID", options=[""] + existing_runs, key="load_run_id")

            if st.button("Load run") and pick:
                try:
                    run_path = safe_child_path(load_workspace_dir, pick)
                    if not run_belongs_to_user(run_path, workspaces_root, current_user):
                        raise PermissionError("This run does not belong to the signed-in user.")
                    rr = discover_run(run_path)
                except Exception as exc:
                    st.error(f"Could not load run: {exc}")
                else:
                    st.session_state["rr"] = rr
        else:
            st.info("No workspaces exist yet. Create one above, then run the pipeline.")

_ACTIVE_PIPELINE_RUN_KEY = "active_pipeline_run"


def _rerun_app() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
    rerun()


def _pipeline_status_snapshot(worker: dict[str, Any]) -> dict[str, Any]:
    with worker["lock"]:
        snapshot = dict(worker["status"])
    status_path = worker.get("status_path")
    if status_path:
        disk_status = read_run_status(status_path)
        # The on-disk status file is JSON, so object values such as RunResult are
        # stringified there. Keep the in-memory object for UI follow-up actions.
        disk_status.pop("result", None)
        snapshot.update(disk_status)
    return snapshot


def _set_pipeline_status(worker: dict[str, Any], **updates: Any) -> None:
    with worker["lock"]:
        worker["status"].update(updates)
    status_path = worker.get("status_path")
    if status_path:
        disk_updates = dict(updates)
        result = disk_updates.pop("result", None)
        if result is not None:
            disk_updates["result_run_id"] = getattr(result, "run_id", None)
            disk_updates["result_run_dir"] = str(getattr(result, "run_dir", ""))
        write_run_status(status_path, **disk_updates)


def _start_pipeline_worker(pipeline_kwargs: dict[str, Any], dois: list[str]) -> dict[str, Any]:
    lock = threading.Lock()
    cancel_event = threading.Event()
    auth_context = dict(pipeline_kwargs.pop("_auth_context", {}))
    run_dir = safe_child_path(Path(pipeline_kwargs["run_root"]), str(pipeline_kwargs["run_id"]))
    status_path = run_dir / RUN_STATUS_FILE_NAME
    status: dict[str, Any] = {
        "state": "running",
        "stage": "Starting pipeline",
        "doi": dois[0] if len(dois) == 1 else f"{len(dois)} DOIs",
        "result": None,
        "error": None,
        "traceback": None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    worker: dict[str, Any] = {
        "lock": lock,
        "cancel_event": cancel_event,
        "status": status,
        "status_path": status_path,
        "dois": list(dois),
    }

    def on_status(stage: str, doi: Optional[str] = None) -> None:
        _set_pipeline_status(
            worker,
            state="running",
            stage=stage,
            doi=doi or status.get("doi"),
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )

    def run_worker() -> None:
        try:
            rr = run_full_pipeline(
                **pipeline_kwargs,
                status_callback=on_status,
                cancel_event=cancel_event,
            )
            if auth_context:
                write_run_manifest(
                    rr.run_dir,
                    user_id=auth_context.get("user_id"),
                    username=str(auth_context.get("username") or ""),
                    workspace=str(auth_context.get("workspace") or rr.run_dir.parent.name),
                    run_id=rr.run_id,
                )
            _set_pipeline_status(worker, state="done", result=rr, stage="Done")
        except PipelineCancelled as exc:
            _set_pipeline_status(
                worker,
                state="cancelled",
                error=str(exc),
                stage="Cancelled",
            )
        except Exception as exc:
            _set_pipeline_status(
                worker,
                state="error",
                error=str(exc),
                traceback=traceback.format_exc(),
                stage="Failed",
            )

    thread = threading.Thread(target=run_worker, name="salsa-pipeline-run", daemon=True)
    worker["thread"] = thread
    thread.start()
    return worker


def _render_active_pipeline_run() -> None:
    worker = st.session_state.get(_ACTIVE_PIPELINE_RUN_KEY)
    if not worker:
        return

    snapshot = _pipeline_status_snapshot(worker)
    thread = worker["thread"]
    state = snapshot.get("state", "running")

    if thread.is_alive():
        if worker["cancel_event"].is_set():
            st.warning("Cancellation requested. Stopping the active pipeline step...")
        elif st.button("Cancel Run", type="secondary"):
            worker["cancel_event"].set()
            _set_pipeline_status(worker, state="cancelling", stage="Cancelling")
            st.warning("Cancellation requested. Stopping the active pipeline step...")

        status_slot = st.empty()
        while thread.is_alive():
            snapshot = _pipeline_status_snapshot(worker)
            status_slot.info(_run_progress_status_text(snapshot))
            time.sleep(1)

    snapshot = _pipeline_status_snapshot(worker)
    state = snapshot.get("state", state)
    st.session_state.pop(_ACTIVE_PIPELINE_RUN_KEY, None)
    if state == "done":
        rr_done = snapshot.get("result")
        if rr_done is not None:
            st.session_state["rr"] = rr_done
            st.success(f"Done. Run ID: {rr_done.run_id}")
            _render_cost_summary(rr_done.run_dir)
    elif state == "cancelled":
        st.warning("Run cancelled.")
    elif state == "error":
        st.error(f"Pipeline failed: {snapshot.get('error')}")
        with st.expander("Traceback", expanded=False):
            st.code(snapshot.get("traceback") or "", language="text")


# ---------------- Run pipeline ----------------
_render_active_pipeline_run()

if st.session_state.get("salsa_active_page") == "guide":
    _render_guide_page()
    st.stop()

if show_admin_page:
    show_loaded_run = _render_admin_page(
        auth_store=auth_store,
        data_root=salsa_data_root,
        current_user=current_user,
    )
    if not show_loaded_run:
        st.stop()
    st.divider()

if use_docling_pdf_uploads:
    st.subheader("Input PDF(s)")
    uploaded_pdf_files = st.file_uploader(
        f"Upload PDF file(s), max {MAX_INPUT_ITEMS}",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploaded_pdf_files",
    )
    if uploaded_pdf_files and len(uploaded_pdf_files) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} PDFs.")
    uploaded_pdf_ids_text = st.text_area(
        "Document IDs (optional, one per PDF)",
        height=100,
        help="Leave blank to use uploaded/<filename>. These IDs are used anywhere SALSA normally shows a DOI.",
    )
    doi_text = ""
elif use_figure_uploads:
    st.subheader("Input Figure(s)")
    uploaded_figure_files = st.file_uploader(
        f"Upload figure image file(s), max {MAX_INPUT_ITEMS}",
        type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
        key="uploaded_figure_files",
    )
    if uploaded_figure_files and len(uploaded_figure_files) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} figure images.")
    uploaded_figure_document_id = st.text_input(
        "Figure set ID (optional)",
        value="",
        help="Leave blank to use uploaded/<filename> for one figure or uploaded/figures for multiple figures.",
    )
    uploaded_figure_captions_text = st.text_area(
        "Figure captions (optional, one per uploaded figure)",
        height=120,
    )
    uploaded_pdf_files = []
    uploaded_pdf_ids_text = ""
    doi_text = ""
else:
    uploaded_pdf_files = []
    uploaded_pdf_ids_text = ""
    uploaded_figure_files = []
    uploaded_figure_document_id = ""
    uploaded_figure_captions_text = ""
    st.subheader("Input DOI(s)")
    doi_text = st.text_area(f"One DOI per line, max {MAX_INPUT_ITEMS}", height=140)

st.subheader("Run configuration")
default_run_id = st.session_state.get("pending_run_id", "")
run_id_input = st.text_input(
    "Run Name (optional)",
    value=default_run_id,
    help="Leave blank to auto-generate timestamp-based run name",
)

st.session_state["pending_run_id"] = run_id_input


if st.button("Run pipeline", type="primary"):
    uploaded_pdfs = []
    uploaded_figures = []
    figure_document_id = ""
    if use_docling_pdf_uploads:
        id_lines = [line.strip() for line in uploaded_pdf_ids_text.splitlines() if line.strip()]
        for idx, uploaded_file in enumerate(uploaded_pdf_files or []):
            document_id = id_lines[idx] if idx < len(id_lines) else None
            uploaded_pdfs.append({
                "filename": uploaded_file.name,
                "content": uploaded_file.getvalue(),
                "document_id": document_id,
            })
        dois = [
            item["document_id"]
            or f"uploaded/{re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(item['filename']).stem).strip('._-') or 'uploaded_pdf'}"
            for item in uploaded_pdfs
        ]
    elif use_figure_uploads:
        caption_lines = uploaded_figure_captions_text.splitlines()
        figure_document_id = uploaded_figure_document_id.strip()
        for idx, uploaded_file in enumerate(uploaded_figure_files or []):
            caption = caption_lines[idx].strip() if idx < len(caption_lines) else ""
            uploaded_figures.append({
                "filename": uploaded_file.name,
                "content": uploaded_file.getvalue(),
                "caption": caption,
            })
        if figure_document_id:
            dois = [figure_document_id]
        elif len(uploaded_figures) == 1:
            first_name = uploaded_figures[0]["filename"] if uploaded_figures else "uploaded_figure"
            dois = [f"uploaded/{re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(first_name).stem).strip('._-') or 'uploaded_figure'}"]
        else:
            dois = ["uploaded/figures"] if uploaded_figures else []
    else:
        dois = [d.strip() for d in doi_text.splitlines() if d.strip()]
    st.session_state["doi_order"] = dois
    if use_docling_pdf_uploads and not uploaded_pdfs:
        st.error("Please upload at least one PDF.")
    elif use_docling_pdf_uploads and len(uploaded_pdfs) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} PDFs.")
    elif use_figure_uploads and not uploaded_figures:
        st.error("Please upload at least one figure image.")
    elif use_figure_uploads and len(uploaded_figures) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} figure images.")
    elif not use_file_uploads and not dois:
        st.error("Please paste at least one DOI.")
    elif not use_file_uploads and len(dois) > MAX_INPUT_ITEMS:
        st.error(f"Enter at most {MAX_INPUT_ITEMS} DOIs.")
    else:
        active_llm_steps = _active_llm_steps(
            run_extract_tables=run_extract_tables,
            use_docling_pdf_uploads=use_file_uploads,
            run_chemical_composition=run_chemical_composition,
            run_filter_figures=run_filter_figures,
            run_filter_tables=run_filter_tables and not use_figure_uploads,
            run_plextract=run_plextract,
            bypass_ocr_with_llm=bypass_ocr_with_llm,
            run_complete_dataset=run_complete_dataset,
        )
        missing_llm_steps = _missing_llm_steps(active_llm_steps)
        missing_openai_api_key_steps = _missing_openai_api_key_steps(active_llm_steps)
        if missing_llm_steps:
            st.error(
                "Configure an LLM model for these stages before running: "
                + ", ".join(missing_llm_steps)
            )
        elif missing_openai_api_key_steps:
            st.error(
                "Enter an OpenAI API key in the sidebar before running these Chat GPT stages: "
                + ", ".join(missing_openai_api_key_steps)
            )
        else:
            custom_run_id = run_id_input.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

            _sync_all_step_llm_envs_from_session()
            _sync_publisher_tdm_env_from_session()

            try:
                target = safe_child_path(run_root, custom_run_id)
            except PermissionError as exc:
                st.error(f"Invalid run name: {exc}")
                st.stop()

            if not _WORKSPACE_RE.match(custom_run_id):
                st.error("Run name may only contain letters, numbers, underscores, dots, or dashes.")
                st.stop()

            if custom_run_id:
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

            pipeline_kwargs = dict(
                dois=dois,
                run_root=run_root,
                scripts_root=scripts_root,
                run_id=custom_run_id,
                run_extract_figures=run_extract_figures,
                run_extract_tables=run_extract_tables,
                run_filter_figures=run_filter_figures,
                run_filter_tables=run_filter_tables and not use_figure_uploads,
                run_plextract=run_plextract and not bypass_ocr_with_llm,
                run_llm_figure_extraction=run_plextract and bypass_ocr_with_llm,
                run_complete_dataset=run_complete_dataset,
                skip_llm_figures=not bypass_ocr_with_llm,
                run_chemical_composition=run_chemical_composition,
                exclude_full_text=exclude_full_text,
                custom_instructions=_run_ci,
                uploaded_pdfs=uploaded_pdfs if use_docling_pdf_uploads else None,
                uploaded_figures=uploaded_figures if use_figure_uploads else None,
                uploaded_figure_document_id=figure_document_id if use_figure_uploads else None,
                input_item_limit=MAX_INPUT_ITEMS,
                _auth_context={
                    "user_id": int(current_user["id"]),
                    "username": current_user["username"],
                    "workspace": active_workspace_name,
                },
            )
            st.session_state[_ACTIVE_PIPELINE_RUN_KEY] = _start_pipeline_worker(
                pipeline_kwargs,
                dois,
            )
            _rerun_app()

rr: RunResult | None = st.session_state.get("rr")
if not rr:
    st.stop()

st.info(f"Run ID: **{rr.run_id}**")
st.info(f"Workspace: **{_workspace_display_name_for_run(rr.run_dir, workspaces_root)}**")
_loaded_run_status_path = rr.run_dir / RUN_STATUS_FILE_NAME
_loaded_run_status = _mark_inactive_run_status(
    _loaded_run_status_path,
    read_run_status(_loaded_run_status_path),
)
if _run_status_is_active(_loaded_run_status):
    _loaded_status_slot = st.empty()
    while _run_status_is_active(_loaded_run_status):
        _loaded_status_slot.info(_run_progress_status_text(_loaded_run_status))
        time.sleep(1)
        _loaded_run_status = _mark_inactive_run_status(
            _loaded_run_status_path,
            read_run_status(_loaded_run_status_path),
        )
    _loaded_status_slot.info(_run_progress_status_text(_loaded_run_status))
_render_cost_summary(rr.run_dir)

# ---------------- DOI summary table + DOI selector ----------------
st.subheader("Processed DOIs")

df_stats = build_doi_stats(rr)

# Render as interactive table (row click -> selects DOI when supported)
df_view = df_stats.drop(columns=["_doi_safe"], errors="ignore")
_doi_table_key = "doi_summary_table"
_doi_table_applied_key = "_doi_summary_table_applied_doi"

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
            key=_doi_table_key,
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

if st.session_state.get("selected_doi") not in doi_options:
    st.session_state["selected_doi"] = doi_options[0]

if df_selected_doi in doi_options and st.session_state.get(_doi_table_applied_key) != df_selected_doi:
    # Only let the summary table drive selection when the clicked row changes.
    # Otherwise a stale selected row can overwrite the dropdown's first change
    # on the next rerun.
    st.session_state["selected_doi"] = df_selected_doi
    st.session_state[_doi_table_applied_key] = df_selected_doi
elif not df_selected_doi:
    st.session_state.pop(_doi_table_applied_key, None)

selected_doi = st.selectbox(
    "Select DOI",
    options=doi_options,
    key="selected_doi",
)

doi_item = next((d for d in rr.dois if d.doi == selected_doi), None)

if doi_item is None:
    st.error(f"Could not find DOI '{selected_doi}' in the loaded run. The data may have changed since this session was saved.")
    st.stop()

# ---------------- DOI detail view (under DOI dropdown) ----------------
_doi_view_options = ["Parsed text", "Tables", "Figures", "Chemical Composition", "Dataset"]
_doi_view_key = "selected_doi_detail_view"
if st.session_state.get(_doi_view_key) not in _doi_view_options:
    st.session_state[_doi_view_key] = _doi_view_options[0]

if hasattr(st, "segmented_control"):
    selected_doi_view = st.segmented_control(
        "View",
        options=_doi_view_options,
        key=_doi_view_key,
        label_visibility="collapsed",
    )
else:
    selected_doi_view = st.radio(
        "View",
        options=_doi_view_options,
        key=_doi_view_key,
        horizontal=True,
        label_visibility="collapsed",
    )
selected_doi_view = selected_doi_view or st.session_state[_doi_view_key]

if selected_doi_view == "Parsed text":
    if doi_item.parsed_json_path and doi_item.parsed_json_path.exists():
        st.json(json.loads(doi_item.parsed_json_path.read_text()))
    else:
        st.warning("No parsed.json found for this DOI in the run workspace.")

if selected_doi_view == "Tables":
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

if selected_doi_view == "Figures":
    # When the loaded run changes, clear any cached axis-config widget values from the
    # previous run so the new run's override files (titles, calibration) are displayed.
    _ax_run_dir_key = "_axis_cfg_last_run_dir"
    _current_run_dir_str = str(rr.run_dir)
    if st.session_state.get(_ax_run_dir_key) != _current_run_dir_str:
        _ax_cfg_patterns = (
            "_ax_x_title", "_ax_y_title",
            "_ax_xtype", "_ax_ytype",
            "_ax_val_xmin", "_ax_val_xmax", "_ax_val_ymin", "_ax_val_ymax",
            "_ax_pt_xmin", "_ax_pt_xmax", "_ax_pt_ymin", "_ax_pt_ymax",
            "_ax_calib_ver", "_ax_calib_w",
        )
        for _k in list(st.session_state.keys()):
            if any(pat in _k for pat in _ax_cfg_patterns):
                del st.session_state[_k]
        st.session_state[_ax_run_dir_key] = _current_run_dir_str

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

        # crops output directory – auto-detected subfigures are stored flat under
        # interventions/<safe_doi>/ rather than in a per-figure subdirectory, so
        # point here and glob by figure stem to find crops for this figure.
        _crops_out_dir_er = rr.run_dir / "interventions" / _safe_doi_er

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
        if _is_valid_image_path(_source_img_er):
            _fig_display_img = _source_img_er
        elif _is_valid_image_path(f.figure_path):
            _fig_display_img = f.figure_path
            _fig_display_img_caption = str(f.figure_path)
        # Collapse parent figure dropdown if subfigures exist
        _has_subfigs_er = _crops_out_dir_er.exists() and bool(list(_crops_out_dir_er.glob(f"{_fig_stem_er}_subfig_*/")))
        
        render_figure_image_dropdown(
            image=_fig_display_img,
            image_caption=_fig_display_img_caption,
            figure_caption=f.caption,
            title=f"{f.id}",
            key=f"fig_{_safe_doi_er}_{_fig_id_er}_img",
            expanded= not _has_subfigs_er,
        )
        
        # ========================
        # SECTION: OCR + LLM Data
        # ========================
        final_csv = None
        if isinstance(getattr(f, "extra", None), dict):
            final_csv = f.extra.get("final_dataset_csv")
        _final_csv_path = Path(final_csv) if final_csv else None
        _figure_dir_for_overrides = _final_csv_path.parent.parent if _final_csv_path and _final_csv_path.exists() else None
        _figure_data_title = (
            "LLM-Extracted Figure Data"
            if f.extra.get("figure_data_source") == "llm_direct"
            else "OCR-Extracted Data"
        )
        render_ocr_dropdown(
            csv_path=_final_csv_path,
            key_prefix=f"fig_{selected_doi}_{f.id}",
            axis_override_dir=_figure_dir_for_overrides,
            title=_figure_data_title,
        )

        emit_rows_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "figure", f.id)
        render_llm_dropdown(
            run_dir=rr.run_dir,
            emit_rows_path=emit_rows_path,
            key_prefix=f"fig_{selected_doi}_{f.id}",
            file_name_prefix=f.id,
        )
        safe_doi = doi_to_safe(selected_doi)

        # Pre-compute fig_stem and _per_fig_plextract so the subfigure section
        # can be rendered unconditionally (outside the interventions block).
        fig_stem = _fig_stem_er
        _int_doi_dir = rr.run_dir / "interventions" / safe_doi
        _per_fig_plextract: Path | None = None
        if _int_doi_dir.exists():
            for _int_fig_d in _int_doi_dir.iterdir():
                if _int_fig_d.is_dir() and Path(_int_fig_d.name).stem == fig_stem:
                    _candidate = _int_fig_d / "plextract_interventions"
                    if _candidate.exists():
                        _per_fig_plextract = _candidate
                    break

        # ========================
        # SECTION: Interventions
        # ========================
        _show_interventions_key = f"er_show_interventions_{_safe_doi_er}_{_fig_id_er}"
        if _show_interventions_key not in st.session_state:
            st.session_state[_show_interventions_key] = False

        intervene_button = "Load Interventions" if st.session_state[_show_interventions_key] else "Load Interventions"
        if st.button(intervene_button, type="primary", key=f"er_intervene_toggle_{_safe_doi_er}_{_fig_id_er}"):
            st.session_state[_show_interventions_key] = not st.session_state[_show_interventions_key]
            if st.session_state[_show_interventions_key]:
                _intervention_prefixes = (
                    f"er_canvas_boot_{_safe_doi_er}_{_fig_id_er}",
                    f"er_{_safe_doi_er}_{_fig_id_er}_series_canvas_boot",
                )
                for _k in list(st.session_state.keys()):
                    if _k in _intervention_prefixes or (
                        _k.startswith(f"er_{_safe_doi_er}_") and _k.endswith("_ax_calib_boot")
                    ) or _k.startswith(f"er_canvas_mounted_{_safe_doi_er}_{_fig_id_er}_") or (
                        _k.startswith(f"er_{_safe_doi_er}_")
                        and (
                            "_ax_calib_mounted_" in _k
                            or "_series_canvas_mounted_" in _k
                        )
                    ):
                        st.session_state.pop(_k, None)
            st.rerun()

        if st.session_state[_show_interventions_key]:
            if not _is_valid_image_path(_source_img_er):
                st.warning("No source image found for this figure.")
            else:
                # ========================
                # SECTION: Cropping
                # ========================
                # ---- Canvas state keys (must be outside the expander) ----
                _cv_ver_key  = f"er_canvas_ver_{_safe_doi_er}_{_fig_id_er}"
                _cv_boot_key = f"er_canvas_boot_{_safe_doi_er}_{_fig_id_er}"
                if _cv_ver_key not in st.session_state:
                    st.session_state[_cv_ver_key] = 0
                _crop_start_expanded = True
                with st.expander("Crop Figure", expanded=_crop_start_expanded):
                    if not _is_valid_image_path(_source_img_er):
                        st.warning("No source image found for this figure.")
                    else:
                        # ---- Load image ----
                        _pil_img_er = _open_image_safe(_source_img_er, mode="RGB")
                        if _pil_img_er is None:
                            st.warning(f"Could not open source image for cropping: {_source_img_er}")
                        else:
                            _orig_w_er, _orig_h_er = _pil_img_er.size
                        if _pil_img_er is not None:
                            # ---- Display width slider (Crop view) ----
                            _width_key_er = f"er_crop_view_w_{_safe_doi_er}_{_fig_id_er}"
                            if _width_key_er not in st.session_state:
                                st.session_state[_width_key_er] = min(1100, _orig_w_er)
                            _slider_min_er = min(400, max(1, _orig_w_er))
                            _slider_max_er = max(_slider_min_er, 1800)
                            st.session_state[_width_key_er] = max(
                                _slider_min_er,
                                min(_slider_max_er, int(st.session_state[_width_key_er])),
                            )

                            _disp_w_target_er = st.slider(
                                "Crop view width (px)",
                                min_value=_slider_min_er,
                                max_value=_slider_max_er,
                                value=int(st.session_state[_width_key_er]),
                                step=50,
                                key=_width_key_er,
                            )

                            _mode_key_er = f"er_canvas_mode_{_safe_doi_er}_{_fig_id_er}"

                            _orig_w_er, _orig_h_er, _disp_w_er, _disp_h_er, _bg_img_er = load_and_resize_rgba(
                                str(_source_img_er),
                                int(_disp_w_target_er),
                            )
                            ensure_canvas_remount(
                                boot_key=_cv_boot_key,
                                version_key=_cv_ver_key,
                            )

                            _draw_mode_er = st.radio(
                                "Mode",
                                ["rect", "transform"],
                                format_func=lambda x: "Draw rectangles" if x == "rect" else "Move/resize rectangles",
                                key=_mode_key_er,
                                horizontal=True,
                            )

                            # ---- Controls ----
                            _clear_canvas_clicked = st.button(
                                "Clear drawn rectangles",
                                key=f"er_clear_canvas_{_safe_doi_er}_{_fig_id_er}",
                            )

                            if _clear_canvas_clicked:
                                st.session_state[_cv_ver_key] += 1

                            # ---- Build a stable canvas key ----
                            # Width/height must NOT be in the key: changing the
                            # slider would mint a new key each run and leave orphaned
                            # entries in st.session_state, causing "dictionary changed
                            # size during iteration" in Streamlit's _call_callbacks.
                            # Instead, bump the version counter when the width changes
                            # so the canvas remounts cleanly with the right dimensions.
                            _er_last_w_key = f"er_canvas_last_w_{_safe_doi_er}_{_fig_id_er}"
                            if st.session_state.get(_er_last_w_key) != _disp_w_er:
                                if _er_last_w_key in st.session_state:
                                    st.session_state[_cv_ver_key] += 1
                                st.session_state[_er_last_w_key] = _disp_w_er
                            _canvas_key_er = (
                                f"er_canvas_{_safe_doi_er}_{_fig_id_er}"
                                f"_v{st.session_state[_cv_ver_key]}"
                            )

                            # ---- Canvas ----
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
                            ensure_canvas_post_mount_remount(
                                mount_key=(
                                    f"er_canvas_mounted_{_safe_doi_er}_{_fig_id_er}_"
                                    f"{Path(_source_img_er).name}_{_disp_w_er}x{_disp_h_er}"
                                ),
                                version_key=_cv_ver_key,
                            )

                        # ---- Extract rectangles from canvas JSON ----
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

                        # de-duplicate exact duplicates
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
                                st.rerun()

                        # ---- Existing crops ----
                        _existing_crops_er = sorted(_crops_out_dir_er.glob(f"{_fig_stem_er}_subfig_*.*")) if _crops_out_dir_er.exists() else []
                        if _existing_crops_er:
                            st.write(f"**Existing crops:** {len(_existing_crops_er)}")
                            _crop_cols_er = st.columns(min(len(_existing_crops_er), 6))
                            for _ci, _cp in enumerate(_existing_crops_er):
                                with _crop_cols_er[_ci % len(_crop_cols_er)]:
                                    st.image(str(_cp), width=300)
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
                # Overrides always go to the interventions dir so the original
                # plextract output is never modified.
                _ax_override_dir_er = rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "new_chartdete"
                render_axis_configuration_dropdown(
                    chartdete_dir=_ax_chartdete_er,
                    source_img=_calib_img_path_er,
                    key_prefix=f"er_{_safe_doi_er}_{_ax_sel_name_er}",
                    override_dir=_ax_override_dir_er,
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
                        with st.spinner("Applying mask overrides..."):
                            # Writes prepared masks + catalog + overrides into the
                            # intervention dir only; the original plextract output
                            # is never modified (item_figure_dir is read-only).
                            apply_overrides_to_plextract(_sm_fig_dir_er, rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                        # Always source lineformer from the intervention override dir
                        # so the original plextract results remain untouched.
                        _lf_dir_rerun_er = _sm_lf_override_dir_er

                        with st.spinner("Re-running data extraction..."):
                            if _is_subfig_sel_er:
                                _rerun_lf_er: dict = {}
                                if _lf_dir_rerun_er and _lf_dir_rerun_er.exists():
                                    _rerun_lf_er[_sm_source_img_er.name] = _lf_dir_rerun_er
                                _run_plextract_on_crops(
                                    scripts_root=scripts_root, run_dir=rr.run_dir,
                                    safe_doi=_safe_doi_er, image_paths=[_sm_source_img_er],
                                    override_lineformer_dirs=_rerun_lf_er,
                                    figure_id=_fig_id_er,
                                    intervention=True,
                                )
                            elif _int_crops_er:
                                # Each crop is treated as an independent new image.
                                # Only reuse a prior lineformer dir that belongs to
                                # that specific crop — never inherit the original
                                # figure's masks, which are in the wrong coordinate
                                # space for the (smaller) crop image.
                                _rerun_lf_er = {}
                                _fresh_crops_er = []
                                for _crop_er in _int_crops_er:
                                    _crop_lf_er = None
                                    # Per-crop workdir uses stem to avoid conflicting with the
                                    # crop image file stored at interventions/<safe_doi>/<name>.
                                    _per_crop_wd_er = rr.run_dir / "interventions" / _safe_doi_er / _crop_er.stem / "plextract_interventions"
                                    # Search per-crop workdir first, then figure-level workdirs
                                    for _sm_wd_r_er in [_per_crop_wd_er] + _sm_workdirs_to_search_er:
                                        if not _sm_wd_r_er.exists():
                                            continue
                                        _lat2_er = latest_plextract_run_dir(_sm_wd_r_er)
                                        if _lat2_er:
                                            _cand_lf_er = _lat2_er / "output" / _safe_doi_er / _crop_er.name / "lineformer"
                                            if _cand_lf_er.exists():
                                                _crop_lf_er = _cand_lf_er
                                                break
                                    if _crop_lf_er:
                                        _rerun_lf_er[_crop_er.name] = _crop_lf_er
                                    else:
                                        # No crop-specific lineformer exists yet;
                                        # must run full pipeline to generate fresh masks.
                                        _fresh_crops_er.append(_crop_er)
                                # Crops with prior results: intervention pipeline (skip LineFormer)
                                if _rerun_lf_er:
                                    _crops_with_lf_er = [p for p in _int_crops_er if p.name in _rerun_lf_er]
                                    for _cwlf_er in _crops_with_lf_er:
                                        _run_plextract_on_crops(
                                            scripts_root=scripts_root, run_dir=rr.run_dir,
                                            safe_doi=_safe_doi_er, image_paths=[_cwlf_er],
                                            override_lineformer_dirs={_cwlf_er.name: _rerun_lf_er[_cwlf_er.name]},
                                            figure_id=_cwlf_er.stem,
                                            intervention=True,
                                        )
                                # Fresh crops: full pipeline so LineFormer generates masks
                                # in crop-coordinate space (pixel indices reset to crop origin)
                                if _fresh_crops_er:
                                    for _fc_er in _fresh_crops_er:
                                        _run_plextract_on_crops(
                                            scripts_root=scripts_root, run_dir=rr.run_dir,
                                            safe_doi=_safe_doi_er, image_paths=[_fc_er],
                                            figure_id=_fc_er.stem,
                                            intervention=False,
                                        )
                            else:
                                if _source_img_er and _source_img_er.exists():
                                    _rerun_lf_er = {}
                                    if _lf_dir_rerun_er and _lf_dir_rerun_er.exists():
                                        _rerun_lf_er[_source_img_er.name] = _lf_dir_rerun_er
                                    _run_plextract_on_crops(
                                        scripts_root=scripts_root, run_dir=rr.run_dir,
                                        safe_doi=_safe_doi_er, image_paths=[_source_img_er],
                                        override_lineformer_dirs=_rerun_lf_er,
                                        figure_id=_fig_id_er,
                                        intervention=True,
                                    )
                        st.success("Data extraction complete!")
                        st.rerun()
                    except Exception as _e2_er:
                        st.error(f"Failed: {_e2_er}")

                # ========================
                # SECTION: Run LLM Dataset Generation
                # ========================

                # Find available figures/subfigures for LLM rerun.
                # When user-created crops exist, prefer their intervention-plextract
                # results (new OCR data) over the original figure's plextract output.
                # Each entry stores (figure_name, result_dir, extracted_figures_root).
                _llm_fig_options_er: list = []
                _llm_fig_dirs_er: list = []
                _llm_fig_efr_er: dict = {}  # figure_name → extracted_figures_root path

                if _int_crops_er:
                    # Crops exist: each crop now has its own per-crop workdir
                    # (<interventions>/<doi>/<crop_name>/plextract_interventions/).
                    # Find the latest result for each crop from its own workdir.
                    for _crop_er in _int_crops_er:
                        _crop_plextract_wd = (
                            rr.run_dir / "interventions" / _safe_doi_er
                            / _crop_er.name / "plextract_interventions"
                        )
                        if not _crop_plextract_wd.exists():
                            continue
                        _crop_lat_er = latest_plextract_run_dir(_crop_plextract_wd)
                        if not _crop_lat_er:
                            continue
                        _crop_out_er = _crop_lat_er / "output" / _safe_doi_er / _crop_er.name
                        if _crop_out_er.is_dir() and _crop_er.name not in _llm_fig_options_er:
                            _llm_fig_options_er.append(_crop_er.name)
                            _llm_fig_dirs_er.append(_crop_out_er)
                            _llm_fig_efr_er[_crop_er.name] = _crop_lat_er / "output"

                if not _llm_fig_options_er:
                    # No crop results found — fall back to original plextract output.
                    for _llm_wd_er in [
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
                                    _llm_fig_efr_er[_llm_wd_er.name] = _llm_wd_er.parent.parent

                if not _llm_fig_dirs_er:
                    st.info("No plextract output found. Run OCR data extraction first.")
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

                    if st.button("Run LLM dataset generation", type="primary", key=f"er_rerun_llm_{_safe_doi_er}_{_fig_id_er}"):
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
                                # Use the pre-computed efr for this selection so crop
                                # entries correctly point to the intervention-plextract
                                # output rather than the original plextract output.
                                _efr_er = _llm_fig_efr_er.get(_llm_sel_er, _llm_sel_dir_er.parent.parent)
                                _cmd_er = [
                                    "python", str(_complete_script_er),
                                    "--file_path", _tmp_path_er,
                                    "--output_dir", str(rr.run_dir / "complete_dataset"),
                                    "--index_csv_path", str(_index_csv_er),
                                    "--figures_tables_dir", str(rr.run_dir / "figures_tables"),
                                    "--extracted_figures_root", str(_efr_er),
                                    "--target_figure", _llm_sel_er,
                                ]

                                _sync_all_step_llm_envs_from_session()
                                _env_er = _general_llm_env_for_step(LLM_STEP_COMPLETE_DATASET)

                                if not _step_has_configured_model(LLM_STEP_COMPLETE_DATASET):
                                    st.error(
                                        "Configure an LLM for Complete dataset in the sidebar before rerunning LLM dataset generation."
                                    )
                                    Path(_tmp_path_er).unlink(missing_ok=True)
                                elif _missing_openai_api_key_steps([LLM_STEP_COMPLETE_DATASET]):
                                    st.error(
                                        "Enter an OpenAI API key in the sidebar before rerunning Chat GPT dataset generation."
                                    )
                                    Path(_tmp_path_er).unlink(missing_ok=True)
                                else:
                                    _ci_path_er = rr.run_dir.parent / "custom_instructions" / "default.json"
                                    if _ci_path_er.exists():
                                        _env_er["SALSA_CUSTOM_INSTRUCTIONS"] = str(_ci_path_er)
                                    _env_er["SALSA_RUN_ID"] = rr.run_id
                                    _env_er["SALSA_RUN_DIR"] = str(rr.run_dir)

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
                # render_metadata_dropdown(metadata_path=f.metadata_path)
        _subfig_map: dict[str, Path] = {} 
        _workdirs_priority: list[Path] = []


        if _per_fig_plextract is not None:
            _workdirs_priority.append(_per_fig_plextract)

        # 1b. Per-crop workdirs: each crop now has its own plextract_interventions dir.
        #     Add them so collect_subfig_dirs_across_runs finds their output.
        _crops_wd_er = sorted(_crops_out_dir_er.glob(f"{fig_stem}_subfig_*.png")) if _crops_out_dir_er.exists() else []
        for _c in _crops_wd_er:
            # Use stem (no extension) as the workdir name so it doesn't conflict
            # with the crop image file stored at interventions/<safe_doi>/<name>.
            _c_wd = rr.run_dir / "interventions" / safe_doi / _c.stem / "plextract_interventions"
            if _c_wd.exists():
                _workdirs_priority.append(_c_wd)

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

        #### SUBFIGURE SECTION #####
        if subfig_dirs:
                # Header for subfigures section
                st.markdown("**Cropped Subfigures**")
                for subfig_dir in subfig_dirs:
                    st.markdown(f"### {subfig_dir.name}")
                    _subfig_source_img, _subfig_image_caption = resolve_source_img_for_subfig_dir(subfig_dir, safe_doi)
                    if _subfig_source_img is None:
                        _subfig_final_dir = subfig_dir / "final"
                        if _subfig_final_dir.exists():
                            _subfig_final_img = _subfig_final_dir / "final_image.png"
                            if _is_valid_image_path(_subfig_final_img):
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
                    # Use stem (no extension) so the intervention dirs don't conflict
                    # with the crop image file stored at interventions/<safe_doi>/<name>.
                    _subfig_id_er = subfig_dir.stem
                    _subfig_override_lf = override_lineformer_dir(rr.run_dir, safe_doi, _subfig_id_er)
                    # Overrides go to the interventions dir; source data still read from subfig_dir.
                    _subfig_override_chartdete = rr.run_dir / "interventions" / safe_doi / _subfig_id_er / "new_chartdete"

                    _subfig_final_csv = subfig_dir / "final" / "final_dataset.csv"
                    render_ocr_dropdown(
                        csv_path=_subfig_final_csv if _subfig_final_csv.exists() else None,
                        key_prefix=f"subfig_{safe_doi}_{subfig_dir.name}",
                        axis_override_dir=subfig_dir,
                    )

                    # ========================
                    # SECTION: Load Interventions toggle (mirrors main-figure behaviour)
                    # ========================
                    _subfig_show_int_key = f"subfig_show_interventions_{safe_doi}_{subfig_dir.name}"
                    if _subfig_show_int_key not in st.session_state:
                        st.session_state[_subfig_show_int_key] = False
                    if st.button("Load Interventions", type="primary", key=f"subfig_intervene_toggle_{safe_doi}_{subfig_dir.name}"):
                        st.session_state[_subfig_show_int_key] = not st.session_state[_subfig_show_int_key]

                    if st.session_state[_subfig_show_int_key]:
                        render_axis_configuration_dropdown(
                            chartdete_dir=subfig_dir / "chartdete",
                            source_img=_subfig_source_img,
                            key_prefix=f"subfig_{safe_doi}_{subfig_dir.name}",
                            override_dir=_subfig_override_chartdete,
                        )
                        render_series_management_dropdown(
                            run_dir=rr.run_dir,
                            safe_doi=safe_doi,
                            fig_id=_subfig_id_er,
                            fig_dir=subfig_dir,
                            lf_dir=_subfig_lf_dir,
                            lf_override_dir=_subfig_override_lf,
                            source_img=_subfig_source_img,
                            key_prefix=f"subfig_{safe_doi}_{subfig_dir.name}",
                        )

                        # ========================
                        # SECTION: Rerun buttons (at subfigure level, matching figure structure)
                        # ========================
                        if st.button("Rerun OCR data extraction", type="primary", key=f"subfig_rerun_ocr_{safe_doi}_{subfig_dir.name}"):
                            try:
                                with st.spinner("Applying mask overrides..."):
                                    apply_overrides_to_plextract(subfig_dir, rr.run_dir, safe_doi, _subfig_id_er)

                                if not (_subfig_source_img and _subfig_source_img.exists()):
                                    st.error("Could not locate subfigure source image for rerun.")
                                else:
                                    _subfig_lf_override = override_lineformer_dir(rr.run_dir, safe_doi, _subfig_id_er)
                                    _lf_map = {}
                                    if _subfig_lf_override.exists():
                                        _lf_map[_subfig_source_img.name] = _subfig_lf_override
                                    _cd_map = {}
                                    _subfig_cd_dir = subfig_dir / "chartdete"
                                    if _subfig_cd_dir.exists():
                                        _cd_map[_subfig_source_img.name] = _subfig_cd_dir
                                    target_id = _subfig_id_er
                                    with st.spinner("Re-running data extraction..."):
                                        _run_plextract_on_crops(
                                            scripts_root=scripts_root,
                                            run_dir=rr.run_dir,
                                            safe_doi=safe_doi,
                                            image_paths=[_subfig_source_img],
                                            override_lineformer_dirs=_lf_map,
                                            figure_id=target_id,
                                            override_chartdete_dirs=_cd_map,
                                            intervention=True,
                                        )
                                    st.success(f"Data extraction complete for {subfig_dir.name}!")
                                    st.rerun()
                            except Exception as _subfig_ocr_e:
                                st.error(f"Failed: {_subfig_ocr_e}")

                        if st.button("Run LLM dataset generation", type="primary", key=f"subfig_rerun_llm_{safe_doi}_{subfig_dir.name}"):
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

                                    _sync_all_step_llm_envs_from_session()
                                    _env_sf = _general_llm_env_for_step(LLM_STEP_COMPLETE_DATASET)

                                    _env_sf["SALSA_RUN_ID"] = rr.run_id
                                    _env_sf["SALSA_RUN_DIR"] = str(rr.run_dir)

                                    if not _step_has_configured_model(LLM_STEP_COMPLETE_DATASET):
                                        st.error(
                                            "Configure an LLM for Complete dataset in the sidebar before rerunning LLM dataset generation."
                                        )
                                        Path(_tmp_path_sf).unlink(missing_ok=True)
                                    elif _missing_openai_api_key_steps([LLM_STEP_COMPLETE_DATASET]):
                                        st.error(
                                            "Enter an OpenAI API key in the sidebar before rerunning Chat GPT dataset generation."
                                        )
                                        Path(_tmp_path_sf).unlink(missing_ok=True)
                                    else:
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

                    _subfig_int_wd = rr.run_dir / "interventions" / safe_doi / _subfig_id_er / "plextract_interventions"
                    _lat_int = latest_plextract_run_dir(_subfig_int_wd) if _subfig_int_wd.exists() else None

                    if _lat_int:
                        _int_out = _lat_int / "output" / safe_doi / subfig_dir.name
                        _int_csv = _int_out / "final" / "final_dataset.csv"
                        render_ocr_dropdown(
                            csv_path=_int_csv if _int_csv.exists() else None,
                            key_prefix=f"subfig_new_ocr_{selected_doi}_{subfig_dir.name}",
                            axis_override_dir=_int_out,
                            title=f"New OCR-Extracted Data: {subfig_dir.name}",
                        )

                    # Show LLM output from the latest OCR and LLM data generation run if present.
                    if _lat_int:
                        _int_out_llm = _lat_int / "output" / safe_doi / subfig_dir.name
                        if _int_out_llm.exists():
                            _emit = find_emit_rows_for_source(rr.run_dir, selected_doi, "figure", subfig_dir.name)
                            render_llm_dropdown(
                                run_dir=rr.run_dir,
                                emit_rows_path=_emit,
                                key_prefix=f"subfig_int_llm_{safe_doi}_{subfig_dir.name}",
                                file_name_prefix=subfig_dir.name,
                                title=f"New LLM-Generated Dataset: {subfig_dir.name}",
                            )
                    
                    # render_metadata_dropdown(metadata_path=f.metadata_path)


                    # Add separator between subfigures if there are multiple
                    if subfig_dir != subfig_dirs[-1]:
                        st.markdown("---")

if selected_doi_view == "Dataset":
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
            "or use the per-figure/per-subfigure 'Run LLM dataset generation' controls."
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

if selected_doi_view == "Chemical Composition":
    st.subheader("Chemical Composition")

    # --- Add src/ to path for chemical_composition imports ---
    _src_for_chem = (Path(__file__).resolve().parent / "..").resolve()
    if str(_src_for_chem) not in sys.path:
        sys.path.insert(0, str(_src_for_chem))

    import importlib
    import llm_client as _llm_client_module
    from chemical_composition import extract_composition as _cc_module

    # Streamlit keeps imported modules alive across reruns. Reload the LLM client
    # and chemistry extractor so model fixes are picked up without restarting.
    importlib.reload(_llm_client_module)
    _cc_module = importlib.reload(_cc_module)

    _cc_identify = _cc_module.identify_relevant_content
    _cc_extract = _cc_module.extract_samples_and_compositions
    _cc_run = _cc_module.run_chemical_composition_extraction

    def _cc_build_table_fallback(samples: list[dict], *, doi: str = ""):
        def _num(value):
            text = "" if value is None else str(value).strip()
            if not text:
                return ""
            if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
                return f"{float(text):.12g}"
            assignment = re.search(r"(?:^|[\s,(;])(?:x|y|z|n|m)\s*=\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
            if assignment:
                value_f = float(assignment.group(1))
                return f"{(value_f / 100.0 if value_f > 1 else value_f):.12g}"
            exp = re.search(r"(?:experimental|experimentally|exp\.?)\D{0,60}?([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
            if exp:
                value_f = float(exp.group(1))
                return f"{(value_f / 100.0 if value_f > 1 else value_f):.12g}"
            pct = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*(?:%|mol\s*%|wt\s*%|vol\s*%)", text, flags=re.I)
            if pct:
                return f"{(float(pct[0]) / 100.0):.12g}"
            ctx = re.search(r"\b(?:dob|degree|graft(?:ing|ed)?|content|loading|ratio|fraction|functionalization)\b\D{0,120}?([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
            if ctx:
                value_f = float(ctx.group(1))
                return f"{(value_f / 100.0 if 1 < value_f <= 100 else value_f):.12g}"
            return ""

        def _smiles(value, *, polymer=False):
            smiles = "" if value is None else str(value).strip()
            if not smiles:
                return ""
            if polymer and "*" not in smiles:
                return ""
            return smiles

        max_polymer = 3
        max_additive = 3
        normalized = []
        for sample in samples:
            polymers = sample.get("polymer_components") or sample.get("co_monomers") or []
            additives = sample.get("additive_components") or []
            if not additives and isinstance(sample.get("additive"), dict):
                additive = sample["additive"]
                if additive.get("name") or additive.get("content") or additive.get("smiles"):
                    additives = [additive]
            polymers = [p for p in polymers if isinstance(p, dict)]
            additives = [a for a in additives if isinstance(a, dict)]
            normalized.append((sample.get("sample_name", ""), polymers, additives))
            max_polymer = max(max_polymer, len(polymers))
            max_additive = max(max_additive, len(additives))

        columns = (
            ["DOI", "sample_name"]
            + [f"smiles{i}" for i in range(1, max_polymer + 1)]
            + [f"c{i}" for i in range(1, max_polymer + 1)]
            + [f"additive_smiles{i}" for i in range(1, max_additive + 1)]
            + [f"additive_c{i}" for i in range(1, max_additive + 1)]
            + ["additive_name_1", "additive_name_2"]
        )
        rows = []
        for sample_name, polymers, additives in normalized:
            row = {column: "" for column in columns}
            row["DOI"] = doi
            row["sample_name"] = sample_name
            for idx, component in enumerate(polymers, start=1):
                row[f"smiles{idx}"] = _smiles(component.get("smiles", ""), polymer=True)
                row[f"c{idx}"] = _num(
                    component.get("ratio")
                    or component.get("content")
                    or component.get("amount")
                    or component.get("ratio_text")
                    or component.get("content_text")
                )
            for idx, component in enumerate(additives, start=1):
                row[f"additive_smiles{idx}"] = _smiles(component.get("smiles", ""))
                row[f"additive_c{idx}"] = _num(
                    component.get("content")
                    or component.get("ratio")
                    or component.get("amount")
                    or component.get("content_text")
                    or component.get("ratio_text")
                )
                if idx <= 2:
                    row[f"additive_name_{idx}"] = str(component.get("name", "") or "")
            rows.append(row)
        return rows, columns

    _cc_build_table = getattr(
        _cc_module,
        "build_composition_table_rows",
        _cc_build_table_fallback,
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
        with st.expander("Extracted Sample Composition Table", expanded=True):
            samples = cc_data.get("samples", [])
            if samples:
                try:
                    _rows, _cols = _cc_build_table(samples, doi=cc_data.get("doi", selected_doi))
                    df_cc = pd.DataFrame(_rows, columns=_cols)
                    st.dataframe(df_cc, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download samples CSV",
                        data=df_cc.to_csv(index=False).encode("utf-8"),
                        file_name=f"{_cc_safe}_samples.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(f"Failed to build samples table from JSON: {e}")
            elif cc_csv_path and Path(cc_csv_path).exists():
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
    with st.expander("Run Chemical Composition Extraction", expanded=False):
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
                    missing_chem_steps = _missing_llm_steps(
                        [LLM_STEP_CHEM_RELEVANT, LLM_STEP_CHEM_COMPOSITION]
                    )
                    if missing_chem_steps:
                        raise RuntimeError(
                            "Configure an LLM model for these chemistry stages first: "
                            + ", ".join(missing_chem_steps)
                        )
                    missing_openai_api_key_steps = _missing_openai_api_key_steps(
                        [LLM_STEP_CHEM_RELEVANT, LLM_STEP_CHEM_COMPOSITION]
                    )
                    if missing_openai_api_key_steps:
                        raise RuntimeError(
                            "Enter an OpenAI API key in the sidebar before running these Chat GPT chemistry stages: "
                            + ", ".join(missing_openai_api_key_steps)
                        )
                    _sync_all_step_llm_envs_from_session()

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
                    for sample_idx, sample in enumerate(samples, start=1):
                        sample_name = sample.get("sample_name", "Unknown")
                        st.markdown(f"**Sample {sample_idx}: {sample_name}**")
                        st.json(sample)

                relevant = result.get("relevant_content", [])
                st.markdown(f"**Identified {len(relevant)} relevant content section(s)**")

                st.info("Reload the run or switch tabs to see the full results.")

            except Exception as e:
                st.error(f"Chemical composition extraction failed: {e}")
