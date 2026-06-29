# SALSA/src/ui/discover.py
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, List

from schemas import DoiFailure, RunResult, DoiResult, FigureItem, TableItem

def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _find_metadata_sidecar(img_path: Path) -> Path | None:
    # your scripts typically use *_metadata.json
    meta = img_path.with_name(img_path.stem + "_metadata.json")
    return meta if meta.exists() else None

def _find_completed_dataset_csv(complete_dataset_dir: Path) -> dict[str, Path]:
    """
    Returns: { safe_doi : path_to_completed_dataset_csv }
    Expects files like:
      <run_dir>/complete_dataset/<safe_doi>_dataset.csv
    """
    out: dict[str, Path] = {}
    if not complete_dataset_dir.exists():
        return out

    for fp in complete_dataset_dir.glob("*_dataset.csv"):
        # Example filename: 10.1039@C6TA05090E_dataset.csv
        safe_doi = fp.name[:-len("_dataset.csv")]
        if safe_doi:
            out[safe_doi] = fp

    return out

def _find_final_dataset_csv(plextract_workdir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Returns mapping:
      { safe_doi: { fig_id: path_to_final_dataset_csv } }

    We look for:
      plextract_workdir/results/<numeric_run>/output/<safe_doi>/<Figure_X(.jpg|.png|...)>/final/final_dataset.csv
    where <Figure_X...> is a DIRECTORY named like the figure filename (often includes extension).
    """
    out: Dict[str, Dict[str, Path]] = {}
    results_root = plextract_workdir / "results"
    if not results_root.exists():
        return out

    # choose latest numeric run id (plextract uses 0,1,2,...)
    run_dirs = [p for p in results_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return out
    run_dirs.sort(key=lambda p: int(p.name), reverse=True)
    latest = run_dirs[0]

    output_root = latest / "output"
    if not output_root.exists():
        return out

    for doi_dir in output_root.iterdir():
        if not doi_dir.is_dir():
            continue
        safe_doi = doi_dir.name
        out.setdefault(safe_doi, {})

        # Each figure typically becomes a folder like:
        #   <doi_dir>/Figure_9.jpg/final/final_dataset.csv
        for fig_folder in doi_dir.iterdir():
            if not fig_folder.is_dir():
                continue

            # If folder is "Figure_9.jpg", then fig_id should match your UI id=img.stem => "Figure_9"
            fig_id = Path(fig_folder.name).stem

            csv_path = fig_folder / "final" / "final_dataset.csv"
            if csv_path.exists():
                out[safe_doi][fig_id] = csv_path

    return out

def _scan_plextract_artifacts(plextract_workdir: Path) -> Dict[str, List[Path]]:
    """
    Returns:
      { safe_doi: [all csv/json artifacts under latest plextract output] }
    Used to populate extracted_xy_paths heuristically.
    """
    out: Dict[str, List[Path]] = {}
    results_root = plextract_workdir / "results"
    if not results_root.exists():
        return out

    run_dirs = [p for p in results_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return out
    run_dirs.sort(key=lambda p: int(p.name), reverse=True)
    latest = run_dirs[0]

    output_root = latest / "output"
    if not output_root.exists():
        return out

    for doi_dir in output_root.iterdir():
        if not doi_dir.is_dir():
            continue
        safe_doi = doi_dir.name
        out.setdefault(safe_doi, [])
        for p in doi_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".csv", ".json"}:
                out[safe_doi].append(p)

    return out

def _find_chemical_composition(chem_comp_dir: Path) -> Dict[str, Path]:
    """
    Returns: { safe_doi : path_to_result.json }
    Expects: <run_dir>/chemical_composition/<safe_doi>/result.json
    """
    out: Dict[str, Path] = {}
    if not chem_comp_dir.exists():
        return out
    for d in chem_comp_dir.iterdir():
        if d.is_dir():
            rj = d / "result.json"
            if rj.exists():
                out[d.name] = rj
    return out


def _failure_message(note: dict) -> str:
    message = str(note.get("message") or "").strip()
    if message:
        return message

    reason = str(note.get("reason") or "Download failed").strip()
    status_code = note.get("status_code")
    if status_code:
        return f"{reason.replace('_', ' ')} (HTTP {status_code})"
    return reason.replace("_", " ")


def _discover_failures(run_dir: Path) -> list[DoiFailure]:
    # Keep the most actionable failure for each DOI.
    failures: dict[str, tuple[int, DoiFailure]] = {}

    def add(priority: int, failure: DoiFailure):
        doi = failure.doi.strip()
        if not doi:
            return
        current = failures.get(doi)
        if current is None or priority >= current[0]:
            failures[doi] = (priority, failure)

    records_dir = run_dir / "records"
    result_files = sorted(
        records_dir.glob("download_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if result_files:
        try:
            with result_files[-1].open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("success", "")).strip().lower() in {"true", "1", "yes"}:
                        continue
                    add(20, DoiFailure(
                        doi=str(row.get("doi") or ""),
                        publisher=str(row.get("publisher") or ""),
                        stage="Record lookup",
                        status="failed",
                        message=str(row.get("message") or "Record lookup failed"),
                    ))
        except Exception:
            pass

    status_path = run_dir / "papers" / "logs" / "download_status.jsonl"
    if status_path.exists():
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("download_status", 0) or 0) not in {5, 6, 7}:
                    continue
                note = row.get("note") if isinstance(row.get("note"), dict) else {}
                add(30, DoiFailure(
                    doi=str(row.get("doi") or ""),
                    publisher=str(row.get("publisher") or ""),
                    stage="Paper download",
                    status=str(note.get("reason") or "failed"),
                    message=_failure_message(note),
                ))
        except Exception:
            pass

    reparse_index = run_dir / "reparsed" / "index.csv"
    if reparse_index.exists():
        try:
            with reparse_index.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    status = str(row.get("status") or "").strip()
                    if status == "ok":
                        continue
                    messages = {
                        "missing_file": "No downloaded structured article file was available.",
                        "no_parser": "No parser is configured for the downloaded publisher format.",
                        "error": "The downloaded article could not be parsed.",
                    }
                    add(10, DoiFailure(
                        doi=str(row.get("doi") or ""),
                        stage="Reparse",
                        status=status or "failed",
                        message=messages.get(status, status.replace("_", " ") or "Reparse failed"),
                    ))
        except Exception:
            pass

    return [
        item[1]
        for _, item in sorted(failures.items(), key=lambda pair: pair[0].lower())
    ]


def discover_run(run_dir: Path) -> RunResult:
    reparsed = run_dir / "reparsed"
    figures_tables = run_dir / "figures_tables"
    plextract_workdir = run_dir / "plextract"
    llm_figure_workdir = run_dir / "llm_figure_extraction"
    complete_dataset_dir = run_dir / "complete_dataset"
    chem_comp_dir = run_dir / "chemical_composition"

    final_csv_map = _find_final_dataset_csv(plextract_workdir)
    llm_final_csv_map = _find_final_dataset_csv(llm_figure_workdir)
    plex_artifacts = _scan_plextract_artifacts(plextract_workdir)
    completed_map = _find_completed_dataset_csv(complete_dataset_dir)
    chem_comp_map = _find_chemical_composition(chem_comp_dir)


    dois: list[DoiResult] = []
    if reparsed.exists():
        for doi_dir in sorted([p for p in reparsed.iterdir() if p.is_dir()]):
            safe_doi = doi_dir.name
            parsed_json = doi_dir / "parsed.json"
            fulltext = None  # if you later write a fulltext.txt, add it here

            # recover original doi from metadata if present
            meta = _read_json(doi_dir / "metadata.json")
            doi = meta.get("doi", safe_doi.replace("@", "/"))

            dr = DoiResult(
                doi=doi,
                safe_doi=safe_doi,
                parsed_json_path=parsed_json if parsed_json.exists() else None,
                fulltext_path=fulltext,
                extra=meta,
            )
            
            # completed dataset
            completed_csv = completed_map.get(safe_doi)
            if completed_csv:
                dr.extra["completed_dataset_csv"] = str(completed_csv)

            # chemical composition
            chem_result = chem_comp_map.get(safe_doi)
            if chem_result:
                dr.extra["chemical_composition_json"] = str(chem_result)
                # also check for samples CSV
                samples_csv = chem_result.parent / "samples.csv"
                if samples_csv.exists():
                    dr.extra["chemical_composition_csv"] = str(samples_csv)

            # figures
            fig_dir = figures_tables / "figures" / safe_doi
            if fig_dir.exists():
                for img in sorted(list(fig_dir.glob("*.png")) + list(fig_dir.glob("*.jpg")) + list(fig_dir.glob("*.jpeg")) + list(fig_dir.glob("*.gif"))):
                    meta_path = _find_metadata_sidecar(img)
                    meta_obj = _read_json(meta_path) if meta_path else {}
                    caption = meta_obj.get("caption", "")

                    fi = FigureItem(
                        id=img.stem,
                        caption=caption,
                        figure_path=img,
                        metadata_path=meta_path,
                        filtered=True,  # “filtered” = survived your filter step (currently deletes non-matching)
                        extra=meta_obj,
                    )

                    # attach final extracted dataset if present
                    final_csv = final_csv_map.get(safe_doi, {}).get(img.stem)
                    data_source = "ocr"
                    if not final_csv:
                        final_csv = llm_final_csv_map.get(safe_doi, {}).get(img.stem)
                        data_source = "llm_direct"
                    if final_csv:
                        fi.extra["final_dataset_csv"] = str(final_csv)
                        fi.extra["figure_data_source"] = data_source

                    # attach plextract outputs if possible
                    candidates = plex_artifacts.get(safe_doi, [])
                    for c in candidates:
                        if img.stem in c.name:
                            fi.extracted_xy_paths.append(c)


                    dr.figures.append(fi)

            # tables
            tab_dir = figures_tables / "tables" / safe_doi
            if tab_dir.exists():
                for tfile in sorted([p for p in tab_dir.iterdir() if p.is_file() and not p.name.endswith("_metadata.json")]):
                    meta_path = tfile.with_name(tfile.stem + "_metadata.json")
                    if not meta_path.exists():
                        meta_path = run_dir / "docling" / safe_doi / "tables" / f"{tfile.stem}_metadata.json"
                    meta_path = meta_path if meta_path.exists() else None
                    meta_obj = _read_json(meta_path) if meta_path else {}
                    caption = meta_obj.get("caption", "")

                    dr.tables.append(TableItem(
                        id=tfile.stem,
                        caption=caption,
                        table_path=tfile,
                        metadata_path=meta_path,
                        filtered=True,
                        extra=meta_obj,
                    ))

            dois.append(dr)

    return RunResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        dois=dois,
        failures=_discover_failures(run_dir),
    )
