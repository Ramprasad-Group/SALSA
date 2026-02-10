from __future__ import annotations
import subprocess
from pathlib import Path

def run_plextract_on_manual_crops(
    scripts_root: Path,   # /home/wschertzer3/AEM_OCR/src
    run_dir: Path,        # /home/.../ui_runs/<run_id>
) -> None:
    manual_crops = run_dir / "manual_crops"
    if not manual_crops.exists():
        raise FileNotFoundError(f"No manual crops directory found: {manual_crops}")

    # only run if there is at least one DOI subdir with at least one image
    has_any = any(p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".gif"}
                  for p in manual_crops.rglob("*"))
    if not has_any:
        raise FileNotFoundError(f"manual_crops exists but contains no images: {manual_crops}")

    plextract_py = scripts_root / "extract_figure_data" / "plextract.py"

    workdir = run_dir / "plextract_work_manual"
    logs_dir = workdir / "logs"
    workdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(plextract_py),
        "--input", str(manual_crops),
        "--workdir", str(workdir),
        "--log_file", str(logs_dir),
        # optional:
        # "--log_level", "INFO",
    ]
    subprocess.run(cmd, check=True)
