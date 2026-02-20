from __future__ import annotations
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional


# Pattern that matches crop filenames produced by cropping.save_crops():
#   <prefix>_subfig_<n>.png  (e.g. Figure_1_subfig_0.png)
_SUBFIG_RE = re.compile(r"^.+_subfig_\d+\.\w+$", re.IGNORECASE)


def run_plextract_on_crops(
    scripts_root: Path,
    run_dir: Path,
    safe_doi: str,
    image_paths: list[Path],
    override_lineformer_dirs: Optional[Dict[str, Path]] = None,
) -> None:
    """
    Run plextract on a list of image files (crops or full figures).

    Output goes to:
      <run_dir>/interventions/interventions_plextract/results/<n>/output/<safe_doi>/...

    This function stages the images into a temporary input directory with the
    layout expected by plextract, optionally pre-populates lineformer masks
    from override directories, and runs the extraction pipeline.

    Args:
        scripts_root: Path to the ``src/`` directory.
        run_dir:      Root of the current pipeline run.
        safe_doi:     DOI with ``/`` replaced by ``@``.
        image_paths:  Image files to process.
        override_lineformer_dirs:
            Optional mapping of **image filename** (e.g. ``Figure_1_subfig_0.png``)
            to a Path containing lineformer masks to pre-populate.  When provided,
            ``--skip-lineformer`` is passed so the pipeline reuses these masks
            instead of regenerating them.
    """
    if not image_paths:
        raise FileNotFoundError("No image files provided for extraction")

    skip_lineformer = bool(override_lineformer_dirs)

    plextract_py = scripts_root / "extract_figure_data" / "plextract.py"

    workdir = run_dir / "interventions" / "interventions_plextract"
    logs_dir = workdir / "logs"
    workdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="plextract_crops_") as td:
        staged_root = Path(td) / "figures" / safe_doi
        staged_root.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            shutil.copy2(img_path, staged_root / img_path.name)

        # Pre-populate lineformer output dirs so plextract reuses the masks.
        if override_lineformer_dirs:
            results_root = workdir / "results"
            results_root.mkdir(parents=True, exist_ok=True)
            existing = [
                int(p.name)
                for p in results_root.iterdir()
                if p.is_dir() and p.name.isdigit()
            ]
            next_id = str(max(existing) + 1 if existing else 0)

            for img_name, lf_src_dir in override_lineformer_dirs.items():
                if not lf_src_dir.exists():
                    continue
                out_lf_dir = (
                    results_root / next_id / "output" / safe_doi
                    / img_name / "lineformer"
                )
                out_lf_dir.mkdir(parents=True, exist_ok=True)
                for src_file in lf_src_dir.iterdir():
                    if src_file.is_file():
                        shutil.copy2(src_file, out_lf_dir / src_file.name)

        cmd = [
            "python", str(plextract_py),
            "--input", str(staged_root.parent),
            "--workdir", str(workdir),
            "--log_file", str(logs_dir),
        ]
        if skip_lineformer:
            cmd.append("--skip-lineformer")
            # Pass the same run ID that was used to pre-populate lineformer
            # masks, so plextract writes output into the same results directory
            # instead of creating a new one.
            cmd.extend(["--run-id", next_id])
        subprocess.run(cmd, check=True)


def run_plextract_on_manual_crops(
    scripts_root: Path,
    run_dir: Path,
    safe_doi: str,
    figure_id: Optional[str] = None,
) -> None:
    """
    Convenience wrapper: collects crop images from the interventions directory
    and runs :func:`run_plextract_on_crops` on them.

    Crops are stored at:
      <run_dir>/interventions/<safe_doi>/<figure_id>/<figure_stem>_subfig_<n>.png

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

    crop_files: list[Path] = []
    image_exts = {".png", ".jpg", ".jpeg", ".gif"}

    if figure_id:
        search_dirs = [interventions_dir / figure_id]
    else:
        search_dirs = [
            d for d in interventions_dir.iterdir()
            if d.is_dir() and d.name != "interventions_plextract"
        ]

    for fig_dir in search_dirs:
        if not fig_dir.exists():
            continue
        for p in fig_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in image_exts:
                continue
            if _SUBFIG_RE.match(p.name):
                crop_files.append(p)

    crop_files.sort()

    if not crop_files:
        raise FileNotFoundError(
            f"No intervention crop images found for DOI {safe_doi}"
            + (f" / figure {figure_id}" if figure_id else "")
            + f": searched {interventions_dir}"
        )

    run_plextract_on_crops(scripts_root, run_dir, safe_doi, crop_files)
