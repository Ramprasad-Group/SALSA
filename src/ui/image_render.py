"""Shared image loading and encoding helpers for the SALSA UI.

Centralises three things that used to be duplicated (and subtly wrong) across
`streamlit_app.py`:

* validating that a path is a readable image before handing it to Streamlit,
* producing a correct MIME type (the old code built `f"image/{suffix}"`, which
  yields the invalid `image/jpg` for `.jpg` and a bare `image/` for
  extensionless paths),
* downscaling large figures once and reusing the result, instead of base64
  inlining full-resolution originals on every rerun.

Everything here is pure Python so it can be unit-tested without a Streamlit
runtime; the caching decorators are applied in `streamlit_app.py`.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

# Widest we ever need to display a figure inline. Chart figures in SALSA runs are
# typically well under this, so most images pass through untouched.
DEFAULT_MAX_DISPLAY_WIDTH = 1200

# Canvas backgrounds travel to the browser inside the component's props, so they
# are bounded more tightly than plain inline images.
MAX_CANVAS_DATA_URL_BYTES = 1_500_000

_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def is_valid_image_path(path: Optional[Path]) -> bool:
    """True if `path` exists, is non-empty, and PIL can parse it."""
    if path is None:
        return False
    try:
        path = Path(path)
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


def open_image_safe(path: Path, *, mode: Optional[str] = None) -> Optional[Image.Image]:
    """Open an image, returning None instead of raising on unreadable files."""
    if not is_valid_image_path(path):
        return None
    try:
        with Image.open(path) as img:
            return img.convert(mode) if mode else img.copy()
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def guess_mime_type(path: Path) -> str:
    """Best-effort MIME type for an image file.

    Prefers the actual decoded format over the filename, because SALSA runs
    contain plenty of PNG bytes saved under a `.jpg` name (and vice versa).
    """
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
        if fmt in _PIL_FORMAT_TO_MIME:
            return _PIL_FORMAT_TO_MIME[fmt]
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    return "application/octet-stream"


def cache_token(path: Path) -> Tuple[str, float, int]:
    """Stable cache key for a file: (path, mtime, size).

    Passing this alongside the path to a `@st.cache_data` function makes the
    cache invalidate when the file is rewritten (masks and crops are rewritten
    in place during interventions).
    """
    try:
        stat = Path(path).stat()
        return (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        return (str(path), 0.0, 0)


def downscale_to_width(img: Image.Image, max_width: int) -> Image.Image:
    """Resize `img` down to `max_width` if it is wider. Never upscales."""
    width, height = img.size
    if width <= max_width or width <= 0:
        return img
    new_height = max(1, int(round(height * (max_width / width))))
    return img.resize((max_width, new_height), Image.Resampling.LANCZOS)


def encode_data_url(img: Image.Image, *, fmt: str = "JPEG", quality: int = 88) -> str:
    """Encode a PIL image as a self-contained `data:` URL."""
    buf = io.BytesIO()
    fmt = fmt.upper()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, fmt, optimize=True)
        mime = _PIL_FORMAT_TO_MIME.get(fmt, "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def build_display_data_url(
    path: Path,
    *,
    max_width: int = DEFAULT_MAX_DISPLAY_WIDTH,
) -> Optional[str]:
    """Load, downscale, and encode an image for inline display.

    Small images are passed through byte-for-byte with their true MIME type so
    line art stays crisp; only oversized images are re-encoded as JPEG.
    Returns None if the image cannot be read.
    """
    img = open_image_safe(path)
    if img is None:
        return None

    if img.width <= max_width:
        try:
            raw = Path(path).read_bytes()
        except OSError:
            return None
        mime = guess_mime_type(Path(path))
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    return encode_data_url(downscale_to_width(img, max_width))


def build_canvas_background_data_url(
    img: Image.Image,
    *,
    width: int,
    height: int,
    quality: int = 88,
    max_bytes: int = MAX_CANVAS_DATA_URL_BYTES,
) -> str:
    """Encode a canvas background at exactly `width` x `height` as a data URL.

    The canvas component draws the background stretched to the canvas size, so
    the image must be resized to match before encoding. If the result would be
    unreasonably large, quality is stepped down rather than silently shipping a
    multi-megabyte prop.
    """
    resized = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    for attempt_quality in (quality, 70, 55):
        url = encode_data_url(resized, fmt="JPEG", quality=attempt_quality)
        if len(url) <= max_bytes:
            return url
    return url
