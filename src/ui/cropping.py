from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from PIL import Image

def save_crops(
    source_img: Path,
    boxes_xyxy: List[Tuple[int,int,int,int]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """
    boxes_xyxy: list of (x1,y1,x2,y2) in pixel coords of displayed image
    saves crops into out_dir as <prefix>_subfig_<k>.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(source_img).convert("RGB")

    saved = []
    for k, (x1,y1,x2,y2) in enumerate(boxes_xyxy):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(im.size[0], x2), min(im.size[1], y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        crop = im.crop((x1,y1,x2,y2))
        out_path = out_dir / f"{prefix}_subfig_{k}.png"
        crop.save(out_path)
        saved.append(out_path)
    return saved
