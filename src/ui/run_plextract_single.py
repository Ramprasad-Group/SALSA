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
    workdir_name: str = "plextract",
    skip_lineformer: bool = False,
    pre_populate_lineformer_dir: Optional[Path] = None,
    output_workdir: Optional[Path] = None,
) -> None:
    """
    Runs plextract on a single figure by creating a minimal input tree:
      <run_dir>/single_input/figures/<safe_doi>/<figure_file>

    If skip_lineformer is True, passes --skip-lineformer to plextract so that
    existing masks are reused instead of re-running LineFormer.

    If pre_populate_lineformer_dir is set, copies its contents into the
    plextract output lineformer directory before running extraction, so the
    pipeline uses the pre-existing masks.

    If output_workdir is set, writes results to that directory instead of
    the default workdir. Useful for intervention-specific extractions.
    """
    input_root = run_dir / "single_input" / "figures" / safe_doi
    input_root.mkdir(parents=True, exist_ok=True)
    dst = input_root / figure_path.name
    shutil.copy2(figure_path, dst)

    # Use custom output workdir if specified, otherwise use default
    if output_workdir is None:
        workdir = run_dir / workdir_name
    else:
        workdir = output_workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # If we need to pre-populate lineformer masks into the output tree,
    # figure out the run_id that plextract will use next and copy masks there.
    if pre_populate_lineformer_dir and pre_populate_lineformer_dir.exists():
        results_root = workdir / "results"
        results_root.mkdir(parents=True, exist_ok=True)
        # Determine next run_id the same way plextract does
        existing = [int(p.name) for p in results_root.iterdir() if p.is_dir() and p.name.isdigit()]
        next_id = str(max(existing) + 1 if existing else 0)
        # Build the output lineformer dir path
        figure_dir_name = figure_path.name
        out_lf_dir = results_root / next_id / "output" / safe_doi / figure_dir_name / "lineformer"
        out_lf_dir.mkdir(parents=True, exist_ok=True)
        # Copy all mask and catalog files
        for src_file in pre_populate_lineformer_dir.iterdir():
            if src_file.is_file():
                shutil.copy2(src_file, out_lf_dir / src_file.name)

    p_plextract = scripts_root / "extract_figure_data" / "plextract.py"
    cmd = ["python", str(p_plextract), "--input", str(run_dir / "single_input" / "figures"), "--workdir", str(workdir)]
    if skip_lineformer:
        cmd.append("--skip-lineformer")
    subprocess.run(cmd, check=True)
