from __future__ import annotations
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

_LOG = logging.getLogger(__name__)

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




def _has_required_chartdete_output(chartdete_dir: Optional[Path]) -> bool:
    """Return True when the minimum ChartDete files needed by intervention exist."""
    print("====Chartdete dir====", chartdete_dir)
    if not chartdete_dir or not chartdete_dir.exists():
        return False
    required = [
        "bounding_boxes.json",
    ]
    return all((chartdete_dir / name).exists() for name in required)


def run_plextract_on_crops(
    scripts_root: Path,
    run_dir: Path,
    safe_doi: str,
    image_paths: list[Path],
    override_lineformer_dirs: Optional[Dict[str, Path]] = None,
    override_chartdete_dirs: Optional[Dict[str, Path]] = None,
    figure_id: Optional[str] = None,
    intervention: bool = False,
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
            to a Path containing lineformer masks to pre-populate.  When provided
            and ``intervention=False``, ``--skip-lineformer`` is passed so the
            pipeline reuses these masks instead of regenerating them.
            Required when ``intervention=True``.
        override_chartdete_dirs:
            Optional mapping of image filename to a chartdete directory.
            Only used when ``intervention=True``.  When provided for an image,
            this path is used first to supply the base ChartDete output and
            avoids relying on heuristic chartdete lookup.
        figure_id:
            When set, results are written to a per-figure subdirectory
            (``<run_dir>/interventions/<safe_doi>/<figure_id>/plextract_interventions/``)
            so that multiple figures can each track their own intervention
            extraction history without interfering with one another.
        intervention:
            When ``True``, calls ``plextract_intervention.py`` instead of
            ``plextract.py``.  The intervention pipeline skips ChartDete and
            all OCR passes — their outputs are reused from disk — and only
            re-runs the steps that depend on user-edited data:

            1. ``regenerate_coordinates_from_masks``
            2. ``correct_coordinates``
            3. ``match_legend``
            4. ``export_labeled_dataset_and_plot``

            The chartdete outputs required by steps 2–4 are located automatically
            from ``override_chartdete_dirs`` or the sibling
            ``<intervention_dir>/new_chartdete/`` of the supplied lineformer dir,
            falling back to a full plextract run if neither is available.
            ``override_lineformer_dirs`` is required in this mode.
    """
    if not image_paths:
        raise FileNotFoundError("No image files provided for extraction")
    if intervention and not override_lineformer_dirs:
        raise ValueError(
            "override_lineformer_dirs is required for the intervention pipeline. "
            "Call apply_overrides_to_plextract() first to populate the masks."
        )

    if intervention:
        script_py = scripts_root / "extract_figure_data" / "plextract_intervention.py"
    else:
        script_py = scripts_root / "extract_figure_data" / "plextract.py"

    skip_lineformer = bool(override_lineformer_dirs) and not intervention

    if figure_id:
        workdir = run_dir / "interventions" / safe_doi / figure_id / "plextract_interventions"
    else:
        workdir = run_dir / "interventions" / "interventions_plextract"
    logs_dir = workdir / "logs"
    workdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Determine the next incremental run ID.  Always needed for intervention
    # mode; only needed for full mode when override_lineformer_dirs is supplied.
    results_root = workdir / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name)
        for p in results_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    next_id = str(max(existing) + 1 if existing else 0)
    need_run_id = intervention or bool(override_lineformer_dirs)

    # Override files that plextract will never overwrite but must survive into
    # the new run (ChartDete writes axis_title_texts.json and pix2val.json;
    # it never touches the *_override variants).
    _OVERRIDE_FILES = [
        "axis_title_override.json",
        "pix2val_override.json",
        "axis_type_override.json",
    ]

    with tempfile.TemporaryDirectory(prefix="plextract_crops_") as td:
        staged_root = Path(td) / "figures" / safe_doi
        staged_root.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            shutil.copy2(img_path, staged_root / img_path.name)

        # Pre-populate lineformer output dirs so plextract reuses the masks.
        if override_lineformer_dirs:
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

        if intervention:
            # For intervention: locate the base ChartDete output for each image,
            # copy it in full, then overlay user override files on top.
            for img_name, lf_src_dir in override_lineformer_dirs.items():
                cd_src_dir = None
                if override_chartdete_dirs and img_name in override_chartdete_dirs:
                    cd_src_dir = override_chartdete_dirs[img_name]

                if not _has_required_chartdete_output(cd_src_dir):
                    sibling_cd = lf_src_dir.parent / "new_chartdete"
                    if _has_required_chartdete_output(sibling_cd):
                        cd_src_dir = sibling_cd
                    else:
                        cd_src_dir = _find_chartdete_dir_for_image(
                            run_dir=run_dir,
                            safe_doi=safe_doi,
                            img_name=img_name,
                            figure_id=figure_id,
                        )
                if not _has_required_chartdete_output(cd_src_dir):
                    _LOG.warning(
                        "[intervention] Missing required ChartDete output for %s; running full plextract fallback.",
                        img_name,
                    )
                    # Generate ChartDete outputs by running the full pipeline for this image once.
                    run_plextract_on_crops(
                        scripts_root=scripts_root,
                        run_dir=run_dir,
                        safe_doi=safe_doi,
                        image_paths=[next((p for p in image_paths if p.name == img_name), image_paths[0])],
                        override_lineformer_dirs=None,
                        figure_id=figure_id,
                    )
                    cd_src_dir = _find_chartdete_dir_for_image(
                        run_dir=run_dir,
                        safe_doi=safe_doi,
                        img_name=img_name,
                        figure_id=figure_id,
                    )

                if not _has_required_chartdete_output(cd_src_dir):
                    _LOG.warning(
                        "[intervention] Missing required ChartDete output (full fallback also unavailable): %s",
                        (cd_src_dir / "bounding_boxes.json") if cd_src_dir else f"<none for {img_name}>",
                    )
                    continue

                out_cd_dir = (
                    results_root / next_id / "output" / safe_doi
                    / img_name / "chartdete"
                )
                shutil.copytree(str(cd_src_dir), str(out_cd_dir), dirs_exist_ok=True)

                # Overlay user axis-calibration overrides from the interventions new_chartdete
                # dir on top of the base plextract chartdete.  The UI always writes overrides
                # to lf_src_dir.parent / "new_chartdete" regardless of where the full chartdete
                # output lives, so we layer them in here after the copytree.
                _user_override_cd = lf_src_dir.parent / "new_chartdete"
                if _user_override_cd.exists() and _user_override_cd != cd_src_dir:
                    for _fname in _OVERRIDE_FILES:
                        _src_f = _user_override_cd / _fname
                        if _src_f.exists():
                            shutil.copy2(str(_src_f), str(out_cd_dir / _fname))
        else:
            # For the full pipeline: only pre-populate the specific user override
            # files (ChartDete will regenerate the base chartdete outputs itself).
            if override_lineformer_dirs:
                for img_name, lf_src_dir in override_lineformer_dirs.items():
                    if not lf_src_dir.exists():
                        continue
                    src_cd = lf_src_dir.parent / "new_chartdete"
                    if src_cd.exists():
                        out_cd = results_root / next_id / "output" / safe_doi / img_name / "chartdete"
                        for fname in _OVERRIDE_FILES:
                            src_f = src_cd / fname
                            if src_f.exists():
                                out_cd.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(src_f), str(out_cd / fname))

        cmd = [
            "python", str(script_py),
            "--input", str(staged_root.parent),
            "--workdir", str(workdir),
            "--log_file", str(logs_dir),
        ]
        if skip_lineformer:
            cmd.append("--skip-lineformer")
        if need_run_id:
            # Pass the run ID so plextract writes output into the same results
            # directory that was pre-populated with lineformer masks / chartdete.
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

    run_plextract_on_crops(scripts_root, run_dir, safe_doi, crop_files, figure_id=figure_id)
