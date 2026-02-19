from __future__ import annotations
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_plextract_on_manual_crops(
    scripts_root: Path,   # /home/wschertzer3/AEM_OCR/src
    run_dir: Path,        # /home/.../ui_runs/<run_id>
    safe_doi: str,
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
    """
    interventions_dir = run_dir / "interventions" / safe_doi
    if not interventions_dir.exists():
        raise FileNotFoundError(f"No interventions directory found for DOI {safe_doi}: {interventions_dir}")

    crop_files = sorted(
        p for p in interventions_dir.glob("**/*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}
    )
    if not crop_files:
        raise FileNotFoundError(f"No intervention crop images found for DOI {safe_doi}: {interventions_dir}")

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
