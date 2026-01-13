# AEM_OCR/src/ui/run_plextract_single.py
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def run_plextract_single_figure(
    *,
    scripts_root: Path,
    run_dir: Path,
    safe_doi: str,
    figure_path: Path,
    workdir_name: str = "plextract_work_interventions",
) -> None:
    """
    Runs plextract on a single figure by creating a minimal input tree:
      <run_dir>/single_input/figures/<safe_doi>/<figure_file>
    """
    input_root = run_dir / "single_input" / "figures" / safe_doi
    input_root.mkdir(parents=True, exist_ok=True)
    dst = input_root / figure_path.name
    shutil.copy2(figure_path, dst)

    workdir = run_dir / workdir_name
    workdir.mkdir(parents=True, exist_ok=True)

    p_plextract = scripts_root / "extract_figure_data" / "plextract.py"
    cmd = ["python", str(p_plextract), "--input", str(run_dir / "single_input" / "figures"), "--workdir", str(workdir)]
    subprocess.run(cmd, check=True)
