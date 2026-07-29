"""Guard the vendored drawable-canvas build.

These patches used to live only in site-packages, so any reinstall silently
reverted them and figure canvases stopped rendering. The tests fail loudly if
the vendored build goes missing or loses a patch.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "src" / "ui" / "vendor" / "streamlit_drawable_canvas" / "frontend" / "build"


def _bundle() -> Path:
    matches = sorted((BUILD_DIR / "static" / "js").glob("main.*.salsa-patched.js"))
    assert len(matches) == 1, f"expected exactly one patched bundle, found {matches}"
    return matches[0]


def test_vendored_build_is_present():
    assert BUILD_DIR.is_dir(), f"vendored build missing at {BUILD_DIR}"
    assert (BUILD_DIR / "index.html").is_file()
    assert list((BUILD_DIR / "static" / "css").glob("*.css"))
    assert list((BUILD_DIR / "static" / "js").glob("runtime-main.*.js"))


def test_index_html_loads_the_patched_bundle():
    index = (BUILD_DIR / "index.html").read_text()
    assert _bundle().name in index, "index.html must load the SALSA-patched bundle"


def test_bundle_handles_data_urls_and_load_errors():
    source = _bundle().read_text()

    # An origin must never be prefixed onto an absolute or data: URL — SALSA
    # sends canvas backgrounds as data URIs.
    assert 'salsaAbs=0===h.indexOf("data:")' in source

    # Upstream sets img.src with no error path, so a failed background load left
    # a silently blank canvas.
    assert "e.onerror=function()" in source
    assert "Background image failed to load" in source


def test_patch_script_reports_the_bundle_as_already_patched():
    script = REPO_ROOT / "src" / "setup-scripts" / "patch_drawable_canvas.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "already patched" in result.stdout


def test_vendored_component_module_exports_the_expected_api():
    sys.path.insert(0, str(REPO_ROOT / "src" / "ui"))
    from vendor.streamlit_drawable_canvas import (  # noqa: E402
        CanvasResult,
        _component_func,
        _data_url_to_image,
    )

    assert _component_func is not None
    assert callable(_data_url_to_image)
    assert CanvasResult().image_data is None


def test_canvas_background_is_self_contained_not_a_media_store_url(tmp_path):
    """The canvas background must not depend on Streamlit's media store.

    The media store clears every session's file references at the start of each
    script run and deletes anything not re-registered by the end of it, so a
    canvas whose `st_canvas(...)` call did not re-execute on some rerun was left
    pointing at a deleted file. That was the blank-canvas bug.
    """
    sys.path.insert(0, str(REPO_ROOT / "src" / "ui"))
    import base64
    import io

    from PIL import Image

    import ui_common

    captured = {}

    def fake_component(**kwargs):
        captured.update(kwargs)
        return None

    original = ui_common._component_func
    ui_common._component_func = fake_component
    try:
        background = Image.new("RGB", (320, 240), (10, 120, 200))
        ui_common.st_canvas(
            background_image=background, width=320, height=240, key="test-canvas"
        )
    finally:
        ui_common._component_func = original

    url = captured["backgroundImageURL"]
    assert url.startswith("data:image/jpeg;base64,")
    assert "/media/" not in url

    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(";base64,", 1)[1])))
    # The component stretches the background to the canvas, so the encoded image
    # must already match the canvas dimensions exactly.
    assert decoded.size == (captured["canvasWidth"], captured["canvasHeight"]) == (320, 240)
