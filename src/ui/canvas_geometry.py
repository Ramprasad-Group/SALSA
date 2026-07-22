from typing import Mapping, Tuple


def fabric_circle_center(circle: Mapping[str, object]) -> Tuple[float, float]:
    """Return a Fabric.js circle's visual center in canvas coordinates."""
    left = float(circle.get("left", 0.0))
    top = float(circle.get("top", 0.0))
    radius = float(circle.get("radius", 0.0))
    stroke_width = float(circle.get("strokeWidth", 0.0))
    scale_x = float(circle.get("scaleX", 1.0))
    scale_y = float(circle.get("scaleY", 1.0))

    half_width = (radius + stroke_width / 2.0) * scale_x
    half_height = (radius + stroke_width / 2.0) * scale_y

    origin_x = str(circle.get("originX", "left"))
    origin_y = str(circle.get("originY", "top"))
    x_offset = {"left": half_width, "center": 0.0, "right": -half_width}.get(
        origin_x, half_width
    )
    y_offset = {"top": half_height, "center": 0.0, "bottom": -half_height}.get(
        origin_y, half_height
    )
    return left + x_offset, top + y_offset
