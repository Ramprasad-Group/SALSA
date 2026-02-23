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


def _latest_result_run_dir(plextract_workdir: Path) -> Optional[Path]:
    """Return latest numeric results run dir for a plextract workdir."""
    results_root = plextract_workdir / "results"
    if not results_root.exists():
        return None
    run_dirs = [d for d in results_root.iterdir() if d.is_dir() and d.name.isdigit()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda d: int(d.name), reverse=True)
    return run_dirs[0]


def _find_chartdete_dir_for_image(
    run_dir: Path,
    safe_doi: str,
    img_name: str,
    figure_id: Optional[str],
) -> Optional[Path]:
    """
    Find an existing chartdete directory for ``img_name`` across known workdirs.

    Search priority mirrors the UI selection strategy so reruns prefer the most
    recent per-figure intervention outputs, then shared intervention outputs,
    then the original plextract output.
    """
    workdirs = []
    if figure_id:
        workdirs.append(run_dir / "interventions" / safe_doi / figure_id / "plextract_interventions")
    workdirs.extend([
        run_dir / "interventions" / "interventions_plextract",
        run_dir / "plextract",
    ])

    for wd in workdirs:
        latest = _latest_result_run_dir(wd)
        if not latest:
            continue
        cand = latest / "output" / safe_doi / img_name / "chartdete"
        if cand.exists():
            return cand

    return None


def run_plextract_on_crops(
    scripts_root: Path,
    run_dir: Path,
    safe_doi: str,
    image_paths: list[Path],
    override_lineformer_dirs: Optional[Dict[str, Path]] = None,
    figure_id: Optional[str] = None,
) -> None:
    """
    Run plextract on a list of image files (crops or full figures).

    When *figure_id* is provided output goes to a per-figure directory:
      <run_dir>/interventions/<safe_doi>/<figure_id>/plextract_interventions/results/<n>/output/<safe_doi>/...

    When *figure_id* is ``None`` the legacy shared directory is used instead:
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
        figure_id:
            When set, results are written to a per-figure subdirectory
            (``<run_dir>/interventions/<safe_doi>/<figure_id>/plextract_interventions/``)
            so that multiple figures can each track their own intervention
            extraction history without interfering with one another.
    """
    if not image_paths:
        raise FileNotFoundError("No image files provided for extraction")

    skip_lineformer = bool(override_lineformer_dirs)

    plextract_py = scripts_root / "extract_figure_data" / "plextract.py"

    if figure_id:
        workdir = run_dir / "interventions" / safe_doi / figure_id / "plextract_interventions"
    else:
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

            # Override files that plextract will never overwrite but must
            # survive into the new run (ChartDete writes axis_title_texts.json
            # and pix2val.json; it never touches the *_override variants).
            _CHARTDETE_OVERRIDES = ["axis_title_override.json", "pix2val_override.json"]

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

                # Pre-populate user override files from the sibling chartdete dir
                # so finalize_labels picks them up even after ChartDete re-runs.
                src_cd = lf_src_dir.parent / "chartdete"
                if src_cd.exists():
                    out_cd = results_root / next_id / "output" / safe_doi / img_name / "chartdete"
                    for fname in _CHARTDETE_OVERRIDES:
                        src_f = src_cd / fname
                        if src_f.exists():
                            out_cd.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(src_f), str(out_cd / fname))

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


def run_plextract_intervention_on_crops(
    scripts_root: Path,
    run_dir: Path,
    safe_doi: str,
    image_paths: list[Path],
    override_lineformer_dirs: Dict[str, Path],
    figure_id: Optional[str] = None,
) -> None:
    """
    Run the lightweight intervention pipeline (plextract_intervention.py) on a
    list of image files.

    Unlike :func:`run_plextract_on_crops`, this function:

    * Calls ``plextract_intervention.py`` instead of ``plextract.py``.
    * **Skips ChartDete and all OCR passes** — their outputs already exist on
      disk and have not changed because the user only edited series masks,
      renamed/deleted series, or updated axis-title overrides.
    * Runs only the steps that depend on the edited data:

      1. ``regenerate_coordinates_from_masks``  — rebuilds ``coordinates.json``
         from the (possibly user-edited) ``mask_series_*.png`` files.
      2. ``correct_coordinates``  — re-converts pixel coords to data-space
         values using the existing OCR / ChartDete outputs.
      3. ``match_legend``  — re-matches legend entries to series (mask colours
         may have changed).
      4. ``export_labeled_dataset_and_plot``  — produces ``final_dataset.csv``
         and ``final_image.png``, honouring ``intervention_overrides.json``
         (renamed / deleted series) and ``axis_title_override.json``.

    The chartdete outputs required by steps 2–4 are located automatically as
    the sibling ``chartdete/`` directory of each supplied lineformer dir::

        .../output/<doi>/<img>/lineformer/   ← override_lineformer_dirs value
        .../output/<doi>/<img>/chartdete/    ← derived automatically

    Args:
        scripts_root:             Path to the ``src/`` directory.
        run_dir:                  Root of the current pipeline run.
        safe_doi:                 DOI with ``/`` replaced by ``@``.
        image_paths:              Image files to process.
        override_lineformer_dirs: **Required** mapping of image filename
            (e.g. ``Figure_1_subfig_0.png``) to the lineformer directory that
            holds the pre-populated / user-edited masks and
            ``intervention_overrides.json``.  Call
            :func:`~intervention.apply_overrides_to_plextract` before this
            function to ensure the directory is up to date.
        figure_id:                When set, results are written to a per-figure
            subdirectory so multiple figures do not interfere with one another.
    """
    if not image_paths:
        raise FileNotFoundError("No image files provided for extraction")
    if not override_lineformer_dirs:
        raise ValueError(
            "override_lineformer_dirs is required for the intervention pipeline. "
            "Call apply_overrides_to_plextract() first to populate the masks."
        )

    plextract_intervention_py = (
        scripts_root / "extract_figure_data" / "plextract_intervention.py"
    )

    if figure_id:
        workdir = (
            run_dir / "interventions" / safe_doi / figure_id
            / "plextract_interventions"
        )
    else:
        workdir = run_dir / "interventions" / "interventions_plextract"
    logs_dir = workdir / "logs"
    workdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Determine the next incremental run ID up front (we always need it).
    results_root = workdir / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name)
        for p in results_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    next_id = str(max(existing) + 1 if existing else 0)

    with tempfile.TemporaryDirectory(prefix="plextract_intervention_") as td:
        staged_root = Path(td) / "figures" / safe_doi
        staged_root.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            shutil.copy2(img_path, staged_root / img_path.name)

        # Pre-populate lineformer dirs (edited masks + intervention_overrides.json).
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

        # Pre-populate chartdete dirs from the sibling of each lineformer dir.
        # The convention is: .../output/<doi>/<img>/lineformer
        #                     .../output/<doi>/<img>/chartdete
        for img_name, lf_src_dir in override_lineformer_dirs.items():
            cd_src_dir = lf_src_dir.parent / "chartdete"
            if not cd_src_dir.exists():
                # The intervention override lineformer dir may not have a sibling
                # chartdete directory. In that case, reuse the original
                # chartdete outputs from the latest available plextract run.
                cd_src_dir = _find_chartdete_dir_for_image(
                    run_dir=run_dir,
                    safe_doi=safe_doi,
                    img_name=img_name,
                    figure_id=figure_id,
                )
            if not cd_src_dir or not cd_src_dir.exists():
                continue
            out_cd_dir = (
                results_root / next_id / "output" / safe_doi
                / img_name / "chartdete"
            )
            shutil.copytree(str(cd_src_dir), str(out_cd_dir), dirs_exist_ok=True)

        cmd = [
            "python", str(plextract_intervention_py),
            "--input", str(staged_root.parent),
            "--workdir", str(workdir),
            "--log_file", str(logs_dir),
            "--run-id", next_id,
        ]
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

    run_plextract_on_crops(scripts_root, run_dir, safe_doi, crop_files, figure_id=figure_id)
