from __future__ import annotations
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# Pattern that matches crop filenames produced by cropping.save_crops():
#   <prefix>_subfig_<n>.png  (e.g. Figure_1_subfig_0.png)
_SUBFIG_RE = re.compile(r"^.+_subfig_\d+\.\w+$", re.IGNORECASE)


def run_plextract_on_manual_crops(
    scripts_root: Path,   # /home/.../AEM_OCR/src
    run_dir: Path,        # /home/.../ui_runs/<run_id>
    safe_doi: str,
    figure_id: Optional[str] = None,
) -> None:
    """
    Runs plextract on intervention crops for a single DOI.

    Crops are stored at:
      <run_dir>/interventions/<safe_doi>/<figure_id>/<figure_stem>_subfig_<n>.png

    Output goes to:
      <run_dir>/interventions/interventions_plextract/results/<n>/output/<safe_doi>/...

    This function stages those files into a temporary input directory with the
    layout expected by plextract (<tmp>/figures/<safe_doi>/*.png), avoiding
    persistent duplicate crop storage.

    Args:
        scripts_root: Path to the ``src/`` directory.
        run_dir:      Root of the current pipeline run.
        safe_doi:     DOI with ``/`` replaced by ``@``.
        figure_id:    Optional figure folder name (e.g. ``Figure_1.jpg``).
                      When provided, only crops from that figure's intervention
                      directory are staged.  When ``None``, crops from *all*
                      figures under the DOI are collected.
    """
    interventions_dir = run_dir / "interventions" / safe_doi
    if not interventions_dir.exists():
        raise FileNotFoundError(
            f"No interventions directory found for DOI {safe_doi}: {interventions_dir}"
        )

    # Collect crop images only — skip lineformer mask overrides and other artifacts.
    # Crops live directly inside <figure_id>/ folders and match *_subfig_*.png.
    # Mask overrides live in <figure_id>/lineformer/ and are named mask_series_*.png.
    crop_files: list[Path] = []
    image_exts = {".png", ".jpg", ".jpeg", ".gif"}

    if figure_id:
        # Scope to a single figure's intervention directory
        search_dirs = [interventions_dir / figure_id]
    else:
        # Collect from all figure directories under this DOI
        search_dirs = [
            d for d in interventions_dir.iterdir()
            if d.is_dir() and d.name != "interventions_plextract"
        ]

    for fig_dir in search_dirs:
        if not fig_dir.exists():
            continue
        # Only look at direct children (not recursive) to avoid lineformer/ subdir.
        # Also accept the recursive _subfig_ pattern as a safety net.
        for p in fig_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in image_exts:
                continue
            # Only accept files that look like crop outputs (*_subfig_<n>.ext)
            if _SUBFIG_RE.match(p.name):
                crop_files.append(p)

    crop_files.sort()

    if not crop_files:
        raise FileNotFoundError(
            f"No intervention crop images found for DOI {safe_doi}"
            + (f" / figure {figure_id}" if figure_id else "")
            + f": searched {interventions_dir}"
        )

    plextract_py = scripts_root / "extract_figure_data" / "plextract.py"

    workdir = run_dir / "interventions" / "interventions_plextract"
    logs_dir = workdir / "logs"
    workdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="plextract_interventions_") as td:
        staged_root = Path(td) / "figures" / safe_doi
        staged_root.mkdir(parents=True, exist_ok=True)

        for cp in crop_files:
            shutil.copy2(cp, staged_root / cp.name)

        cmd = [
            "python", str(plextract_py),
            "--input", str(staged_root.parent),
            "--workdir", str(workdir),
            "--log_file", str(logs_dir),
        ]
        subprocess.run(cmd, check=True)
