import base64
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ui"))

from image_render import (  # noqa: E402
    build_canvas_background_data_url,
    build_display_data_url,
    cache_token,
    downscale_to_width,
    guess_mime_type,
    is_valid_image_path,
    open_image_safe,
)


def _write_image(path: Path, size=(40, 30), fmt="PNG", color=(200, 30, 30)):
    Image.new("RGB", size, color).save(path, fmt)
    return path


def test_is_valid_image_path_rejects_missing_empty_and_non_image(tmp_path):
    assert not is_valid_image_path(None)
    assert not is_valid_image_path(tmp_path / "nope.png")

    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert not is_valid_image_path(empty)

    junk = tmp_path / "junk.png"
    junk.write_text("this is not an image")
    assert not is_valid_image_path(junk)

    assert is_valid_image_path(_write_image(tmp_path / "good.png"))


def test_open_image_safe_returns_none_instead_of_raising(tmp_path):
    junk = tmp_path / "junk.png"
    junk.write_text("nope")
    assert open_image_safe(junk) is None

    img = open_image_safe(_write_image(tmp_path / "good.png"), mode="RGB")
    assert img is not None and img.mode == "RGB"


def test_guess_mime_type_prefers_decoded_format_over_extension(tmp_path):
    # The old code built f"image/{suffix}", producing the invalid "image/jpg"
    # and a bare "image/" for extensionless paths.
    assert guess_mime_type(_write_image(tmp_path / "a.png", fmt="PNG")) == "image/png"
    assert guess_mime_type(_write_image(tmp_path / "b.jpg", fmt="JPEG")) == "image/jpeg"

    # PNG bytes stored under a .jpg name — common in SALSA run output.
    mislabelled = _write_image(tmp_path / "c.jpg", fmt="PNG")
    assert guess_mime_type(mislabelled) == "image/png"

    # No extension at all.
    assert guess_mime_type(_write_image(tmp_path / "d", fmt="PNG")) == "image/png"


def test_guess_mime_type_falls_back_for_unreadable_files(tmp_path):
    junk = tmp_path / "junk.bin"
    junk.write_text("nope")
    assert guess_mime_type(junk) == "application/octet-stream"


def test_downscale_to_width_never_upscales(tmp_path):
    img = Image.new("RGB", (100, 50))
    assert downscale_to_width(img, 200).size == (100, 50)

    shrunk = downscale_to_width(img, 50)
    assert shrunk.size == (50, 25)


def test_cache_token_changes_when_file_is_rewritten(tmp_path):
    path = _write_image(tmp_path / "mask.png", size=(20, 20))
    before = cache_token(path)

    _write_image(path, size=(40, 40), color=(0, 0, 255))
    assert cache_token(path) != before


def test_cache_token_is_stable_for_missing_files(tmp_path):
    assert cache_token(tmp_path / "gone.png") == (str(tmp_path / "gone.png"), 0.0, 0)


def test_build_display_data_url_passes_small_images_through_unchanged(tmp_path):
    path = _write_image(tmp_path / "small.png", size=(40, 30))
    url = build_display_data_url(path, max_width=1200)

    assert url.startswith("data:image/png;base64,")
    payload = base64.b64decode(url.split(";base64,", 1)[1])
    assert payload == path.read_bytes()


def test_build_display_data_url_downscales_oversized_images(tmp_path):
    path = _write_image(tmp_path / "big.png", size=(2400, 1200))
    url = build_display_data_url(path, max_width=600)

    assert url.startswith("data:image/jpeg;base64,")
    decoded = Image.open(
        __import__("io").BytesIO(base64.b64decode(url.split(";base64,", 1)[1]))
    )
    assert decoded.size == (600, 300)


def test_build_display_data_url_returns_none_for_unreadable(tmp_path):
    junk = tmp_path / "junk.png"
    junk.write_text("nope")
    assert build_display_data_url(junk) is None


def test_canvas_background_matches_requested_canvas_size(tmp_path):
    # The component stretches the background to the canvas, so the encoded image
    # must already be exactly the canvas size.
    img = Image.new("RGB", (800, 600), (10, 120, 200))
    url = build_canvas_background_data_url(img, width=300, height=225)

    assert url.startswith("data:image/jpeg;base64,")
    decoded = Image.open(
        __import__("io").BytesIO(base64.b64decode(url.split(";base64,", 1)[1]))
    )
    assert decoded.size == (300, 225)


def test_canvas_background_steps_quality_down_to_respect_budget():
    img = Image.new("RGB", (1200, 900))
    # Noise so the encoder cannot trivially compress it away.
    img.putdata([(i * 7 % 256, i * 13 % 256, i * 29 % 256) for i in range(1200 * 900)])

    generous = build_canvas_background_data_url(img, width=1200, height=900, max_bytes=10_000_000)
    constrained = build_canvas_background_data_url(img, width=1200, height=900, max_bytes=1_000)

    # The constrained call exhausts every quality step; it must still return a
    # usable data URL rather than raising or returning None.
    assert constrained.startswith("data:image/jpeg;base64,")
    assert len(constrained) < len(generous)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_supported_formats_round_trip(tmp_path, fmt):
    path = _write_image(tmp_path / f"img.{fmt.lower()}", fmt=fmt)
    assert is_valid_image_path(path)
    assert guess_mime_type(path).startswith("image/")
    assert build_display_data_url(path) is not None
