"""Vendored `streamlit-drawable-canvas` component (upstream 0.9.3, SALSA-patched).

SALSA ships its own copy of the compiled frontend instead of depending on the
PyPI package. See ``src/ui/vendor/README.md`` for what was changed and why.

Only the pieces SALSA actually uses are exported: the component handle, the
canvas result dataclass, and the data-URL decoder.  The Python-side ``st_canvas``
wrapper lives in ``src/ui/streamlit_app.py`` because SALSA builds the background
image differently from upstream.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "build")

if not os.path.isdir(_BUILD_DIR):  # pragma: no cover - packaging guard
    raise RuntimeError(
        f"Vendored drawable-canvas frontend build is missing at {_BUILD_DIR}. "
        "See src/ui/vendor/README.md for how to restore it."
    )

_component_func = components.declare_component("st_canvas", path=_BUILD_DIR)


@dataclass
class CanvasResult:
    """Output of the canvas component.

    Attributes
    ----------
    image_data: np.ndarray
        RGBA matrix of the drawing layer.
    json_data: dict
        Fabric.js scene graph for the drawing layer.
    """

    image_data: Optional[np.ndarray] = None
    json_data: Optional[dict] = None


def _data_url_to_image(data_url: str) -> Image.Image:
    """Convert a base64 data URL into a PIL image."""
    _, payload = data_url.split(";base64,")
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def image_to_data_url(img: Image.Image, *, fmt: str = "JPEG", quality: int = 88) -> str:
    """Encode a PIL image as a self-contained ``data:`` URL.

    Used for canvas backgrounds so the image never depends on Streamlit's media
    store, which garbage-collects files between script runs.
    """
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, fmt.upper(), optimize=True)
        mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


__all__ = [
    "CanvasResult",
    "_component_func",
    "_data_url_to_image",
    "image_to_data_url",
]
