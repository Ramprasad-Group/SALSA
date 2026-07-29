"""Streamlit primitives shared between `streamlit_app.py` and `figures_view.py`.

Anything here is view-agnostic: image rendering compat shims, the drawable-canvas
wrapper, expander/section helpers, and small cached data loaders. View-specific
composition lives in the modules that import this one.
"""

from __future__ import annotations

import inspect
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from image_render import (
    DEFAULT_MAX_DISPLAY_WIDTH,
    build_canvas_background_data_url,
    build_display_data_url,
    cache_token,
    is_valid_image_path,
    open_image_safe,
)

# Vendored (and SALSA-patched) drawable-canvas. See src/ui/vendor/README.md.
from vendor.streamlit_drawable_canvas import (
    CanvasResult,
    _component_func,
    _data_url_to_image,
)


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def st_image_auto(img, *, full_width: bool = True, **kwargs):
    """Render an image, tolerating Streamlit's container-width kwarg rename.

    Newer Streamlit uses `use_container_width`, older uses `use_column_width`;
    passing both is an error, so callers pass neither and let this pick.
    """
    sig = inspect.signature(st.image)
    kwargs.pop("use_container_width", None)
    kwargs.pop("use_column_width", None)
    if "use_container_width" in sig.parameters:
        return st.image(img, use_container_width=full_width, **kwargs)
    return st.image(img, use_column_width=full_width, **kwargs)


@st.cache_data(show_spinner=False)
def cached_display_data_url(_token: tuple, path: str, max_width: int) -> Optional[str]:
    """Cached inline-display encoding of an image.

    `_token` is `cache_token(path)` — it carries mtime and size so the cache
    invalidates when interventions rewrite masks and crops in place.
    """
    return build_display_data_url(Path(path), max_width=max_width)


def st_resizable_image(image_path: Path, *, caption: Optional[str] = None, key: str = ""):
    """Render an image in a container the user can widen by dragging its corner.

    The image is downscaled and encoded once per (path, mtime, width) and cached,
    rather than base64-inlining the full-resolution original on every rerun.
    """
    if not is_valid_image_path(image_path):
        st.warning(f"Image not found: {image_path}")
        return
    try:
        data_url = cached_display_data_url(
            cache_token(image_path), str(image_path), DEFAULT_MAX_DISPLAY_WIDTH
        )
        if not data_url:
            raise ValueError(f"Could not encode image: {image_path}")
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
                <img src="{data_url}" style="width:100%;height:auto;display:block;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        if caption:
            st.caption(caption)
    except Exception as img_ex:
        st.warning(f"Failed to render resizable image: {img_ex}")
        st_image_auto(str(image_path), full_width=True)


@st.cache_data(show_spinner=False)
def load_and_resize_rgba(path: str, disp_w: int):
    """Load an image and its display-scaled RGB copy. Returns (ow, oh, dw, dh, img)."""
    img = open_image_safe(Path(path), mode="RGB")
    if img is None:
        raise ValueError(f"Could not open image: {path}")
    ow, oh = img.size
    dh = int(round(oh * (disp_w / ow)))
    bg = img.resize((disp_w, dh), Image.Resampling.LANCZOS).convert("RGB")
    return ow, oh, disp_w, dh, bg


# --------------------------------------------------------------------------
# Drawable canvas
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_canvas_background(_img_bytes: bytes, width: int, height: int) -> str:
    """Encode a canvas background as a `data:` URL, cached on its exact pixels.

    Backgrounds are passed to the component as self-contained data URLs rather
    than Streamlit media-store URLs. The media store clears every session's file
    references at the start of each script run and deletes anything not
    re-registered by the end of it, so a canvas whose `st_canvas(...)` call did
    not re-execute on some rerun (interventions collapsed, DOI switched, view
    switched) was left pointing at a deleted file — the blank-canvas bug.
    """
    img = Image.open(io.BytesIO(_img_bytes))
    return build_canvas_background_data_url(img, width=width, height=height)


def st_canvas(
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: Optional[Image.Image] = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: Optional[dict] = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key=None,
) -> CanvasResult:
    """Drawable canvas wrapper that embeds the background image directly.

    The background travels as a `data:` URL inside the component props, so it can
    never be garbage-collected out from under a mounted iframe and needs no
    reverse-proxy path fixups.
    """
    background_image_url = None
    if background_image:
        buf = io.BytesIO()
        background_image.convert("RGB").save(buf, "PNG")
        background_image_url = _cached_canvas_background(buf.getvalue(), int(width), int(height))
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


def canvas_version(version_key: str) -> int:
    """Current remount counter for a canvas, initialising it on first use.

    Bumping this counter is the only supported way to force a canvas to remount
    (and thereby discard its drawings). It exists for explicit user actions —
    "Clear markers", "Clear drawn rectangles", changing the display width — and
    is deliberately *not* driven by mount/boot heuristics: the forced-rerun
    workarounds this replaced double-rendered every section and could discard the
    first batch of shapes a user drew.
    """
    if version_key not in st.session_state:
        st.session_state[version_key] = 0
    return int(st.session_state[version_key])


def bump_canvas_version_on_width_change(*, last_width_key: str, version_key: str, width: int) -> None:
    """Remount a canvas when its display width changes.

    The width must NOT be baked into the canvas key itself — doing so mints a new
    key on every slider move and leaves orphaned entries in st.session_state,
    which surfaces as "dictionary changed size during iteration" inside
    Streamlit's `_call_callbacks`.
    """
    canvas_version(version_key)
    if st.session_state.get(last_width_key) != width:
        if last_width_key in st.session_state:
            st.session_state[version_key] += 1
        st.session_state[last_width_key] = width


# --------------------------------------------------------------------------
# Layout + data loading
# --------------------------------------------------------------------------

def section_dropdown(title: str, *, expanded: bool = False):
    """Create a section expander. Kept as a helper so figures and subfigures
    build their sections identically."""
    return st.expander(title, expanded=expanded)


@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_emit_rows(cache_path: Optional[Path]) -> Optional[pd.DataFrame]:
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
