# SALSA/src/ui/pipeline_orchestrator.py
from __future__ import annotations
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from discover import discover_run
from docling_pdf import PdfUpload, extract_uploaded_pdfs_with_docling
from run_progress import RUN_STATUS_FILE_ENV, RUN_STATUS_FILE_NAME, write_run_status
from uploaded_figures import FigureUpload, stage_uploaded_figures
from llm_step_config import (
    GENERAL_LLM_ENV_VARS,
    LLM_STEP_COMPLETE_DATASET,
    LLM_STEP_EXTRACT_TABLES,
    LLM_STEP_FIGURE_DATA,
    LLM_STEP_FILTER_FIGURES,
    LLM_STEP_FILTER_TABLES,
    step_env_to_general_env_map,
)
from schemas import RunResult

StatusCallback = Callable[[str, Optional[str]], None]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DEFAULT_INPUT_ITEM_LIMIT = 50
ADMIN_INPUT_ITEM_LIMIT = 100
MAX_INPUT_ITEMS = DEFAULT_INPUT_ITEM_LIMIT


class PipelineCancelled(Exception):
    """Raised when a user cancels an in-progress UI pipeline run."""

def doi_to_safe(doi: str) -> str:
    return doi.replace("/", "@").strip()

def _run(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    cancel_event: Any = None,
):
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        start_new_session=True,
    )
    while True:
        returncode = proc.poll()
        if returncode is not None:
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd)
            return

        if cancel_event is not None and cancel_event.is_set():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait()
            raise PipelineCancelled("Pipeline run cancelled by user.")

        time.sleep(0.5)


def _raise_if_cancelled(cancel_event: Any = None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("Pipeline run cancelled by user.")


def _doi_scope(dois: list[str]) -> str:
    clean = [d.strip() for d in dois if d.strip()]
    if len(clean) == 1:
        return clean[0]
    return f"{len(clean)} DOIs"


def _validate_input_item_limit(input_count: int, input_item_limit: int | None = None) -> int:
    limit = int(input_item_limit or MAX_INPUT_ITEMS)
    if input_count > limit:
        raise ValueError(
            f"SALSA accepts at most {limit} DOIs, PDFs, or figure images per run."
        )
    return limit


def _safe_run_dir(run_root: Path, run_id: str) -> Path:
    run_id = (run_id or "").strip()
    if not _RUN_ID_RE.match(run_id):
        raise ValueError("Run name may only contain letters, numbers, underscores, dots, or dashes.")
    root = Path(run_root).resolve()
    run_dir = (root / run_id).resolve()
    if run_dir != root and root not in run_dir.parents:
        raise ValueError(f"Run directory escapes workspace root: {run_dir}")
    return run_dir


def _report_status(
    status_callback: Optional[StatusCallback],
    stage: str,
    doi: str | None,
    cancel_event: Any = None,
) -> None:
    _raise_if_cancelled(cancel_event)
    write_run_status(state="running", stage=stage, doi=doi)
    if status_callback is not None:
        status_callback(stage, doi)


def _progress_env(
    *,
    scripts_root: Path,
    stage: str,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    out = (env or os.environ.copy()).copy()
    out["SALSA_CURRENT_STAGE"] = stage
    src_path = str(scripts_root.resolve())
    existing = out.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if src_path not in parts:
        out["PYTHONPATH"] = os.pathsep.join([src_path] + parts)
    return out


def _clear_general_llm_env(env: Dict[str, str]) -> None:
    for key in GENERAL_LLM_ENV_VARS:
        env.pop(key, None)


def _subprocess_env_for_llm_step(step_key: str) -> Dict[str, str]:
    import os

    env = os.environ.copy()
    _clear_general_llm_env(env)
    for step_env, general_env in step_env_to_general_env_map(step_key).items():
        value = os.environ.get(step_env, "").strip()
        if value:
            env[general_env] = value
        else:
            env.pop(general_env, None)
    return env


def _apply_llm_step_to_process_env(step_key: str) -> None:
    import os

    _clear_general_llm_env(os.environ)
    for step_env, general_env in step_env_to_general_env_map(step_key).items():
        value = os.environ.get(step_env, "").strip()
        if value:
            os.environ[general_env] = value
        else:
            os.environ.pop(general_env, None)

def _latest_matching_csv(dir_: Path, glob_pat: str) -> Path:
    matches = sorted(dir_.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching {glob_pat} in {dir_}")
    return matches[0]

def _plextract_results_root(plextract_workdir: Path) -> Optional[Path]:
    """
    Your plextract output example:
      plextract/results/0/output/<modified_doi>/Figure_.../final/final_dataset.csv

    We want to pass --extracted_figures_root = plextract/results/<latest>/output
    so complete-dataset can find <modified_doi> subdirs under it.
    """
    results_dir = plextract_workdir / "results"
    if not results_dir.exists():
        return None

    # Choose latest numeric subdir (0,1,2,...) by mtime or numeric sort
    subdirs = [p for p in results_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None

    # Prefer numeric dirs; fall back to mtime
    numeric = []
    other = []
    for p in subdirs:
        try:
            numeric.append((int(p.name), p))
        except Exception:
            other.append(p)

    if numeric:
        numeric.sort(key=lambda x: x[0], reverse=True)
        latest = numeric[0][1]
    else:
        other.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = other[0]

    out = latest / "output"
    if out.exists() and out.is_dir():
        return out

    # fallback: search one level down for an "output" dir
    for p in latest.glob("**/output"):
        if p.is_dir():
            return p

    return None

def _copy_figure_metadata_to_plextract(figures_tables_dir: Path, plextract_output_root: Path):
    """
    Copy <run>/figures_tables/figures/<doi>/Figure_X_metadata.json
    into <run>/plextract/results/<n>/output/<doi>/Figure_X.<ext>/final/Figure_X_metadata.json
    """
    figures_root = figures_tables_dir / "figures"
    if not figures_root.exists():
        return

    for doi_dir in figures_root.iterdir():
        if not doi_dir.is_dir():
            continue
        safe_doi = doi_dir.name  # already safe doi folder name

        # where plextract wrote outputs for this doi
        plex_doi_dir = plextract_output_root / safe_doi
        if not plex_doi_dir.exists():
            continue

        for meta in doi_dir.glob("Figure_*_metadata.json"):
            # meta stem like "Figure_7_metadata"
            fig_id = meta.name.replace("_metadata.json", "")  # "Figure_7"
            # plextract figure folder is something like "Figure_7.jpg"
            candidates = list(plex_doi_dir.glob(f"{fig_id}.*"))
            for fig_folder in candidates:
                final_dir = fig_folder / "final"
                if final_dir.exists():
                    dest = final_dir / meta.name
                    try:
                        dest.write_text(meta.read_text())
                    except Exception:
                        # fallback for large files / encoding
                        import shutil
                        shutil.copy2(meta, dest)


def run_full_pipeline(
    dois: list[str],
    run_root: Path,
    *,
    scripts_root: Path,
    run_id: str | None = None,
    run_extract_tables: bool = True,
    run_extract_figures: bool = True,
    run_filter_figures: bool = True,
    run_filter_tables: bool = True,
    run_plextract: bool = True,
    run_llm_figure_extraction: bool = False,
    run_complete_dataset: bool = True,
    skip_llm_figures: bool = True,
    run_chemical_composition: bool = False,
    exclude_full_text: bool = True,
    custom_instructions: Dict[str, Any] | None = None,
    uploaded_pdfs: list[dict[str, Any]] | None = None,
    uploaded_figures: list[dict[str, Any]] | None = None,
    uploaded_figure_document_id: str | None = None,
    input_item_limit: int = MAX_INPUT_ITEMS,
    status_callback: Optional[StatusCallback] = None,
    cancel_event: Any = None,
) -> RunResult:
    """
    Orchestrates:
      1) download_records_multiple_dois_webofscience.py
      2) download_papers.py
      3) reparse_filtered_dois.py
      4) extract_figures_tables_llm.py (extraction only)
      5) chemical composition extraction (on unfiltered figures/tables)
      6) filter figures/tables
      7) plextract.py or direct LLM figure extraction (digitize)
      8) complete-dataset.py (agglomerate + GPT row completion)
    All outputs go under run_root/<run_id>/...
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = run_id.strip()
    run_dir = _safe_run_dir(run_root, run_id)
    pdf_uploads = [
        PdfUpload(
            filename=str(item.get("filename") or "uploaded.pdf"),
            content=item.get("content") or b"",
            document_id=item.get("document_id"),
        )
        for item in (uploaded_pdfs or [])
        if item.get("content")
    ]
    figure_uploads = [
        FigureUpload(
            filename=str(item.get("filename") or "uploaded_figure.png"),
            content=item.get("content") or b"",
            caption=str(item.get("caption") or ""),
        )
        for item in (uploaded_figures or [])
        if item.get("content")
    ]
    input_count = (
        len(pdf_uploads)
        or len(figure_uploads)
        or len([doi for doi in (dois or []) if str(doi).strip()])
    )
    input_item_limit = _validate_input_item_limit(input_count, input_item_limit)
    if dois:
        doi_scope = _doi_scope(dois)
    elif pdf_uploads:
        doi_scope = f"{len(pdf_uploads)} uploaded PDF" + ("" if len(pdf_uploads) == 1 else "s")
    else:
        doi_scope = f"{len(figure_uploads)} uploaded figure" + ("" if len(figure_uploads) == 1 else "s")

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Choose a different run ID or delete the existing run."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Set run dir and run ID so cost logging writes to <run_dir>/cost/usage.jsonl ---
    import os
    os.environ["SALSA_RUN_DIR"] = str(run_dir)
    os.environ["SALSA_RUN_ID"] = run_id
    os.environ[RUN_STATUS_FILE_ENV] = str(run_dir / RUN_STATUS_FILE_NAME)
    _report_status(status_callback, "Preparing run workspace", doi_scope, cancel_event)

    # --- Reference workspace-level custom instructions ---
    # Instructions are saved in workspace/custom_instructions/default.json
    # We pass them via environment variable to downstream scripts
    ci_path = None
    if custom_instructions:
        # Create a temporary reference file in the workspace (not run) directory
        workspace_ci_dir = run_root / "custom_instructions"
        workspace_ci_dir.mkdir(parents=True, exist_ok=True)
        ci_path = workspace_ci_dir / "default.json"
        with open(ci_path, "w") as f:
            json.dump(custom_instructions, f, indent=2)
        os.environ["SALSA_CUSTOM_INSTRUCTIONS"] = str(ci_path)
        print(f"[Pipeline] Saved custom instructions to: {ci_path}")
        print(f"[Pipeline] Missing value defaults: {custom_instructions.get('missing_value_defaults', {})}")
    else:
        print("[Pipeline] WARNING: No custom instructions provided, using defaults")

    # --- workspace subdirs ---
    records_dir = run_dir / "records"
    papers_dir = run_dir / "papers"
    reparsed_dir = run_dir / "reparsed"
    figures_tables_dir = run_dir / "figures_tables"
    plextract_workdir = run_dir / "plextract"
    llm_figure_workdir = run_dir / "llm_figure_extraction"
    complete_dataset_dir = run_dir / "complete_dataset"  # NEW

    for d in [records_dir, papers_dir, reparsed_dir, figures_tables_dir, plextract_workdir, llm_figure_workdir, complete_dataset_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- 0) write a minimal DOI CSV for records downloader / downstream stages ---
    # download_records script default expects column name "DOI"
    doi_csv = records_dir / "input_dois.csv"
    if pdf_uploads:
        _report_status(status_callback, "Extracting uploaded PDFs with Docling", doi_scope, cancel_event)
        docling_result = extract_uploaded_pdfs_with_docling(
            pdf_uploads,
            run_dir=run_dir,
            run_extract_tables=run_extract_tables,
            run_extract_figures=run_extract_figures,
            status_callback=status_callback,
            cancel_event=cancel_event,
        )
        dois = docling_result.dois
        doi_scope = _doi_scope(dois)
    elif figure_uploads:
        _report_status(status_callback, "Staging uploaded figures", doi_scope, cancel_event)
        figure_result = stage_uploaded_figures(
            figure_uploads,
            run_dir=run_dir,
            document_id=uploaded_figure_document_id,
        )
        dois = figure_result.dois
        doi_scope = _doi_scope(dois)

    with open(doi_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DOI"])
        for doi in dois:
            doi = doi.strip()
            if doi:
                w.writerow([doi])

    # Script paths (actual files)
    p_download_records = scripts_root / "paper-downloader" / "paper_downloader" / "download_records_multiple_dois_webofscience.py"
    p_download_papers  = scripts_root / "paper-downloader" / "paper_downloader" / "download_papers.py"
    p_reparse          = scripts_root / "paper-downloader" / "paper_downloader" / "reparse_filtered_dois.py"
    p_extract_ft       = scripts_root / "paper-downloader" / "paper_downloader" / "extract_figures_tables_llm.py"
    p_plextract        = scripts_root / "extract_figure_data" / "plextract.py"
    p_complete_dataset = scripts_root / "complete_dataset" / "complete_dataset.py"

    structured_root = papers_dir / "structured_files"
    index_csv = reparsed_dir / "index.csv"
    if not (pdf_uploads or figure_uploads):
        # --- 1) download_records_multiple_dois_webofscience.py ---
        # writes download_results_*.csv into --log_file directory.
        _report_status(status_callback, "Downloading DOI records", doi_scope, cancel_event)
        _run([
            sys.executable, str(p_download_records),
            "--file_path", str(doi_csv),
            "--doi_column", "DOI",
            "--no_db",
            "--log_file", str(records_dir),
            "--output_dir", str(records_dir),
        ], env=_progress_env(scripts_root=scripts_root, stage="Downloading DOI records"), cancel_event=cancel_event)

        results_csv = _latest_matching_csv(records_dir, "download_results_*.csv")

        # --- 2) download_papers.py (no_db) ---
        # Writes structured_files under --download_dir/structured_files
        _report_status(status_callback, "Downloading papers", doi_scope, cancel_event)
        _run([
            sys.executable, str(p_download_papers),
            "--no_db",
            "--input_csv", str(results_csv),
            "--download_dir", str(papers_dir),
            "--log_file", str(papers_dir / "logs"),
            "--sample_size", "None",
        ], env=_progress_env(scripts_root=scripts_root, stage="Downloading papers"), cancel_event=cancel_event)

        # --- 3) reparse_filtered_dois.py ---
        # Produces reparsed/index.csv and per-doi parsed.json
        _report_status(status_callback, "Re-parsing downloaded papers", doi_scope, cancel_event)
        _run([
            sys.executable, str(p_reparse),
            "--csv_path", str(results_csv),
            "--structured_root", str(structured_root),
            "--output_dir", str(reparsed_dir),
            "--doi_column", "doi",   # output from download_records is lower 'doi'
        ], env=_progress_env(scripts_root=scripts_root, stage="Re-parsing downloaded papers"), cancel_event=cancel_event)

        # --- 4) extract_figures_tables_llm.py (extraction only, no filtering yet) ---
        cmd_extract = [
            sys.executable, str(p_extract_ft),
            "--no_db",
            "--csv_path", str(index_csv),
            "--structured_files_dir", str(structured_root),
            "--output_dir", str(figures_tables_dir),
        ]
        if run_extract_figures:
            cmd_extract.append("--extract_figures")
        if run_extract_tables:
            cmd_extract.append("--extract_tables")
        _report_status(status_callback, "Extracting figures and tables", doi_scope, cancel_event)
        _run(
            cmd_extract,
            env=_progress_env(
                scripts_root=scripts_root,
                stage="Extracting figures and tables",
                env=_subprocess_env_for_llm_step(LLM_STEP_EXTRACT_TABLES)
                if run_extract_tables
                else None,
            ),
            cancel_event=cancel_event,
        )

    # --- 5) chemical composition extraction (before filtering) ---
    # Run chemical composition extraction on unfiltered figures/tables so that
    # structure-relevant figures (schemes, drawings) are available for analysis.
    if run_chemical_composition:
        _chemical_composition_dir = run_dir / "chemical_composition"
        _chemical_composition_dir.mkdir(parents=True, exist_ok=True)

        # Add the repository src/ directory to path for imports.
        # NOTE:
        #   If we accidentally add repo root (instead of repo/src), Python can resolve
        #   `chemical_composition` from an installed site-packages module.
        if scripts_root.name == "src":
            _src_path = str(scripts_root.resolve())
        elif (scripts_root / "src").exists():
            _src_path = str((scripts_root / "src").resolve())
        else:
            # Fallback, but still prefer explicit src path if discoverable.
            _src_path = str((scripts_root.parent / "src").resolve())
        if _src_path not in sys.path:
            sys.path.insert(0, _src_path)

        import importlib
        import llm_client as _llm_client_module
        from chemical_composition import extract_composition as _cc_module

        # Streamlit runs this orchestrator in-process, so a previously imported
        # LLM client/extractor can otherwise keep stale model behavior alive.
        importlib.reload(_llm_client_module)
        _cc_module = importlib.reload(_cc_module)
        run_chemical_composition_extraction = _cc_module.run_chemical_composition_extraction

        for doi in dois:
            _report_status(status_callback, "Extracting chemical composition", doi.strip(), cancel_event)
            safe = doi_to_safe(doi.strip())
            parsed_json = reparsed_dir / safe / "parsed.json"
            tables_d = figures_tables_dir / "tables" / safe
            figures_d = figures_tables_dir / "figures" / safe

            try:
                cc_kwargs = dict(
                    run_dir=run_dir,
                    safe_doi=safe,
                    parsed_json_path=parsed_json if parsed_json.exists() else None,
                    tables_dir=tables_d if tables_d.exists() else None,
                    figures_dir=figures_d if figures_d.exists() else None,
                )
                run_chemical_composition_extraction(**cc_kwargs)
                print(f"[Pipeline] Chemical composition extracted for {doi}")
            except Exception as e:
                print(f"[Pipeline] WARNING: Chemical composition failed for {doi}: {e}")

    # --- 6) filter figures/tables (after chemical composition extraction) ---
    if run_filter_figures:
        _report_status(status_callback, "Filtering figures", doi_scope, cancel_event)
        _run(
            [
                sys.executable, str(p_extract_ft),
                "--no_db",
                "--csv_path", str(index_csv),
                "--structured_files_dir", str(structured_root),
                "--output_dir", str(figures_tables_dir),
                "--filter_figures",
            ],
            env=_progress_env(
                scripts_root=scripts_root,
                stage="Filtering figures",
                env=_subprocess_env_for_llm_step(LLM_STEP_FILTER_FIGURES),
            ),
            cancel_event=cancel_event,
        )
    if run_filter_tables:
        _report_status(status_callback, "Filtering tables", doi_scope, cancel_event)
        _run(
            [
                sys.executable, str(p_extract_ft),
                "--no_db",
                "--csv_path", str(index_csv),
                "--structured_files_dir", str(structured_root),
                "--output_dir", str(figures_tables_dir),
                "--filter_tables",
            ],
            env=_progress_env(
                scripts_root=scripts_root,
                stage="Filtering tables",
                env=_subprocess_env_for_llm_step(LLM_STEP_FILTER_TABLES),
            ),
            cancel_event=cancel_event,
        )

    # --- 7) plextract.py ---
    # It expects input dir = figures_tables/figures which contains subdirs per DOI (safe doi)
    _figures_dir = figures_tables_dir / "figures"
    if run_plextract:
        _report_status(status_callback, "Extracting figure data with OCR", doi_scope, cancel_event)
        if not _figures_dir.exists() or not any(_figures_dir.iterdir()):
            print(f"[Pipeline] WARNING: No figures were extracted — skipping plextract "
                  f"(figures dir missing or empty: {_figures_dir})")
        else:
            _run([
                sys.executable, str(p_plextract),
                "--input", str(_figures_dir),
                "--workdir", str(plextract_workdir),
            ], env=_progress_env(scripts_root=scripts_root, stage="Extracting figure data with OCR"), cancel_event=cancel_event)

    if run_plextract and _figures_dir.exists() and any(_figures_dir.iterdir()):
        plextract_results_root = _plextract_results_root(plextract_workdir)
        if plextract_results_root is not None:
            _copy_figure_metadata_to_plextract(figures_tables_dir, plextract_results_root)

    # --- 7b) direct LLM figure extraction ---
    # Writes the same final_dataset.csv shape as plextract so the existing
    # complete-dataset stage can consume it through --extracted_figures_root.
    if run_llm_figure_extraction:
        _report_status(status_callback, "Extracting figure data with LLM", doi_scope, cancel_event)
        if not _figures_dir.exists() or not any(_figures_dir.iterdir()):
            print(f"[Pipeline] WARNING: No figures were extracted — skipping direct LLM figure extraction "
                  f"(figures dir missing or empty: {_figures_dir})")
        else:
            import importlib
            if str(scripts_root.resolve()) not in sys.path:
                sys.path.insert(0, str(scripts_root.resolve()))
            from extract_figure_data import llm_direct_figure_data as _llm_fig_module

            _apply_llm_step_to_process_env(LLM_STEP_FIGURE_DATA)
            _llm_fig_module = importlib.reload(_llm_fig_module)
            _llm_fig_module.run_llm_figure_data_extraction(
                figures_root=_figures_dir,
                workdir=llm_figure_workdir,
                run_dir=run_dir,
            )
            _raise_if_cancelled(cancel_event)

    # --- 8) complete-dataset.py ---
    if run_complete_dataset:
        _report_status(status_callback, "Completing dataset rows", doi_scope, cancel_event)
        cmd = [
            sys.executable, str(p_complete_dataset),
            "--file_path", str(doi_csv),
            "--doi_column", "DOI",
            "--index_csv_path", str(index_csv),
            "--output_dir", str(complete_dataset_dir),
            "--figures_tables_dir", str(figures_tables_dir),
            "--log_file", str(complete_dataset_dir),
        ]
        extracted_figures_root = None
        if run_llm_figure_extraction:
            extracted_figures_root = _plextract_results_root(llm_figure_workdir)
        elif run_plextract:
            plextract_results_root = _plextract_results_root(plextract_workdir)
            if plextract_results_root is not None:
                extracted_figures_root = plextract_results_root
        if extracted_figures_root is not None:
            cmd += ["--extracted_figures_root", str(extracted_figures_root)]
        if exclude_full_text:
            cmd.append("--exclude_full_text")
        if skip_llm_figures:
            cmd.append("--skip_llm_figures")
        if ci_path and ci_path.exists():
            cmd += ["--custom_instructions", str(ci_path)]

        _run(
            cmd,
            env=_progress_env(
                scripts_root=scripts_root,
                stage="Completing dataset rows",
                env=_subprocess_env_for_llm_step(LLM_STEP_COMPLETE_DATASET),
            ),
            cancel_event=cancel_event,
        )

    # --- discover everything for UI ---
    _report_status(status_callback, "Loading run results", doi_scope, cancel_event)
    return discover_run(run_dir)
