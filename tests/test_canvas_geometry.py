import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ui"))

from canvas_geometry import fabric_circle_center


def test_point_mode_circle_center_matches_click_position():
    # streamlit-drawable-canvas point mode subtracts radius + half the stroke
    # from click X and uses a centered Y origin.
    circle = {
        "left": 89,
        "top": 75,
        "radius": 1,
        "strokeWidth": 20,
        "originX": "left",
        "originY": "center",
        "scaleX": 1,
        "scaleY": 1,
    }

    assert fabric_circle_center(circle) == (100, 75)


def test_circle_center_honors_scale_and_default_origins():
    circle = {
        "left": 10,
        "top": 20,
        "radius": 4,
        "strokeWidth": 2,
        "scaleX": 2,
        "scaleY": 3,
    }

    assert fabric_circle_center(circle) == (20, 35)
