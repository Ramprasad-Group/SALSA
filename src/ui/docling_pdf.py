from __future__ import annotations

import csv
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from run_progress import RUN_STATUS_FILE_ENV, RUN_STATUS_FILE_NAME, write_current_doi


@dataclass(frozen=True)
class PdfUpload:
    filename: str
    content: bytes
    document_id: str | None = None


@dataclass(frozen=True)
class DoclingExtractionResult:
    dois: list[str]
    index_csv: Path
    source_pdf_dir: Path


def _doi_to_safe(doi: str) -> str:
    return doi.replace("/", "@").strip()


def _safe_filename_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "uploaded_pdf"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    stem = stem.strip("._-")
    return stem[:80] or "uploaded_pdf"


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base
    idx = 2
    while candidate in used:
        candidate = f"{base}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def _json_default(obj: Any) -> str:
    return str(obj)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _first_page_number(item: Any) -> Optional[int]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    return getattr(prov[0], "page_no", None)


def _caption_text(item: Any, doc: Any) -> str:
    caption_fn = getattr(item, "caption_text", None)
    if caption_fn is None:
        return ""
    try:
        return caption_fn(doc) or ""
    except TypeError:
        try:
            return caption_fn(doc=doc) or ""
        except Exception:
            return ""
    except Exception:
        return ""


def _save_docling_document_exports(doc: Any, artifacts_dir: Path, safe_doi: str) -> dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    text_path = artifacts_dir / "full_text.txt"
    export_text = getattr(doc, "export_to_text", None)
    if export_text is not None:
        try:
            text_path.write_text(export_text(), encoding="utf-8")
            paths["text_path"] = str(text_path)
        except Exception:
            pass

    md_path = artifacts_dir / "document.md"
    export_markdown = getattr(doc, "export_to_markdown", None)
    if export_markdown is not None:
        try:
            md_path.write_text(export_markdown(), encoding="utf-8")
            paths["markdown_path"] = str(md_path)
        except Exception:
            pass

    html_path = artifacts_dir / "document.html"
    save_as_html = getattr(doc, "save_as_html", None)
    if save_as_html is not None:
        try:
            save_as_html(html_path)
            paths["html_path"] = str(html_path)
        except Exception:
            pass

    json_path = artifacts_dir / "docling_document.json"
    save_as_json = getattr(doc, "save_as_json", None)
    if save_as_json is not None:
        try:
            save_as_json(json_path)
            paths["docling_json_path"] = str(json_path)
        except Exception:
            pass
    elif hasattr(doc, "export_to_dict"):
        try:
            _write_json(json_path, doc.export_to_dict())
            paths["docling_json_path"] = str(json_path)
        except Exception:
            pass

    page_dir = artifacts_dir / "pages"
    page_count = 0
    for page_no, page in (getattr(doc, "pages", None) or {}).items():
        try:
            pil_image = page.image.pil_image
        except Exception:
            continue
        page_dir.mkdir(parents=True, exist_ok=True)
        page_number = getattr(page, "page_no", page_no)
        page_path = page_dir / f"page_{page_number}.png"
        pil_image.save(page_path, "PNG")
        page_count += 1
    if page_count:
        paths["page_images_dir"] = str(page_dir)
        paths["page_image_count"] = str(page_count)

    paths["safe_doi"] = safe_doi
    return paths


def _extract_title(doc: Any, fallback: str) -> str:
    name = str(getattr(doc, "name", "") or "").strip()
    if name:
        return name
    texts = getattr(doc, "texts", None) or []
    for item in texts[:10]:
        label = str(getattr(item, "label", "") or "").lower()
        text = str(getattr(item, "text", "") or "").strip()
        if text and "title" in label:
            return text
    return fallback


def _section_from_doc_text(doc: Any) -> dict[str, Any]:
    export_text = getattr(doc, "export_to_text", None)
    text = ""
    if export_text is not None:
        try:
            text = export_text()
        except Exception:
            text = ""
    if not text and hasattr(doc, "export_to_markdown"):
        try:
            text = doc.export_to_markdown()
        except Exception:
            text = ""
    return {
        "type": "docling_document",
        "name": "Full text",
        "content": [text] if text else [],
    }


def _clean_csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def _write_table_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """
    Docling can return OCR/table labels with quotes, delimiters, newlines, or
    control characters. Use explicit CSV quoting/escaping so those tables remain
    serializable instead of crashing the whole PDF run.
    """
    csv_df = df.copy()
    csv_df.columns = [_clean_csv_value(col) for col in csv_df.columns]
    if hasattr(csv_df, "map"):
        csv_df = csv_df.map(_clean_csv_value)
    else:
        csv_df = csv_df.applymap(_clean_csv_value)
    csv_df.to_csv(
        csv_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        doublequote=True,
        lineterminator="\n",
    )


def _save_table(
    table: Any,
    doc: Any,
    table_dir: Path,
    doi: str,
    idx: int,
    artifacts_dir: Path | None = None,
) -> bool:
    table_dir.mkdir(parents=True, exist_ok=True)
    label = f"Table {idx}"
    stem = f"Table_{idx}"
    csv_path = table_dir / f"{stem}.csv"
    table_artifacts_dir = artifacts_dir / "tables" if artifacts_dir is not None else None
    metadata_dir = table_artifacts_dir or table_dir
    html_path = table_artifacts_dir / f"{stem}.html" if table_artifacts_dir is not None else None
    image_path = table_artifacts_dir / f"{stem}.png" if table_artifacts_dir is not None else None

    try:
        df = table.export_to_dataframe(doc=doc)
    except TypeError:
        df = table.export_to_dataframe(doc)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return False

    _write_table_csv(df, csv_path)

    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            html_path.write_text(table.export_to_html(doc=doc), encoding="utf-8")
        except TypeError:
            try:
                html_path.write_text(table.export_to_html(doc), encoding="utf-8")
            except Exception:
                pass
        except Exception:
            pass

    if image_path is not None:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = table.get_image(doc)
            if img is not None:
                img.save(image_path, "PNG")
        except Exception:
            image_path = None

    meta = {
        "label": label,
        "caption": _caption_text(table, doc),
        "notes": [],
        "metadata": {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "page": _first_page_number(table),
            "docling_self_ref": str(getattr(table, "self_ref", "")),
        },
        "doi": doi,
        "table_file_path": str(csv_path),
        "table_directory": str(table_dir),
        "table_html_path": str(html_path) if html_path is not None and html_path.exists() else "",
        "table_image_path": str(image_path) if image_path is not None and image_path.exists() else "",
        "extraction_timestamp": datetime.now().isoformat(),
        "file_type": "table",
        "publisher": "docling_pdf",
    }
    _write_json(metadata_dir / f"{stem}_metadata.json", meta)
    return True


def _save_picture(picture: Any, doc: Any, figure_dir: Path, doi: str, idx: int) -> bool:
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Figure_{idx}"
    image_path = figure_dir / f"{stem}.png"

    try:
        img = picture.get_image(doc)
    except Exception:
        img = None
    if img is None:
        return False

    img.save(image_path, "PNG")
    meta = {
        "original_label": f"Figure {idx}",
        "clean_label": stem,
        "caption": _caption_text(picture, doc),
        "url": "",
        "doi": doi,
        "figure_file_path": str(image_path),
        "figure_directory": str(figure_dir),
        "metadata": {
            "page": _first_page_number(picture),
            "docling_self_ref": str(getattr(picture, "self_ref", "")),
        },
        "extraction_timestamp": datetime.now().isoformat(),
        "file_type": "figure",
        "publisher": "docling_pdf",
    }
    _write_json(figure_dir / f"{stem}_metadata.json", meta)
    return True


def _build_docling_converter(
    *,
    run_extract_tables: bool,
    run_extract_figures: bool,
    use_ocr: bool,
) -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling is required for PDF upload extraction. The current SALSA Python "
            "environment does not provide it. Keep SALSA on Python 3.8 and set "
            "DOCLING_PYTHON to a separate Python 3.10+ environment that has Docling "
            "installed, or run SALSA itself from Python 3.10+."
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = run_extract_tables
    pipeline_options.images_scale = 1.5
    # Page images are saved as review artifacts. They are not what controls OCR;
    # Docling page OCR is controlled solely by do_ocr above.
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = run_extract_figures

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _write_uploaded_pdf_manifest(
    uploads: list[PdfUpload],
    *,
    run_dir: Path,
) -> tuple[list[dict[str, str]], list[str], Path]:
    source_pdf_dir = run_dir / "uploaded_pdfs"
    source_pdf_dir.mkdir(parents=True, exist_ok=True)
    used_ids: set[str] = set()
    dois: list[str] = []
    manifest: list[dict[str, str]] = []

    for upload in uploads:
        base_id = (upload.document_id or "").strip()
        if not base_id:
            base_id = f"uploaded/{_safe_filename_stem(upload.filename)}"
        doi = _unique_id(base_id, used_ids)
        safe_doi = _doi_to_safe(doi)
        dois.append(doi)
        pdf_path = source_pdf_dir / f"{safe_doi}.pdf"
        pdf_path.write_bytes(upload.content)
        manifest.append({
            "doi": doi,
            "safe_doi": safe_doi,
            "filename": upload.filename,
            "pdf_path": str(pdf_path),
        })

    manifest_path = run_dir / "docling_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest, dois, manifest_path


def _extract_manifest_with_docling(
    manifest: list[dict[str, str]],
    *,
    run_dir: Path,
    run_extract_tables: bool = True,
    run_extract_figures: bool = True,
    status_callback: Any = None,
    cancel_event: Any = None,
    use_ocr: bool = True,
) -> None:
    def report(stage: str, doi: str | None = None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            from pipeline import PipelineCancelled

            raise PipelineCancelled("Pipeline run cancelled by user.")
        if status_callback is not None:
            status_callback(stage, doi)

    converter = _build_docling_converter(
        run_extract_tables=run_extract_tables,
        run_extract_figures=run_extract_figures,
        use_ocr=use_ocr,
    )

    reparsed_dir = run_dir / "reparsed"
    figures_tables_dir = run_dir / "figures_tables"
    docling_artifacts_dir = run_dir / "docling"
    for path in (reparsed_dir, figures_tables_dir, docling_artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []

    total = len(manifest)
    for item_number, entry in enumerate(manifest, start=1):
        doi = entry["doi"]
        safe_doi = entry["safe_doi"]
        pdf_path = Path(entry["pdf_path"])
        filename = entry.get("filename") or pdf_path.name
        report(f"Processing PDF ({item_number}/{total})", doi)

        result = converter.convert(pdf_path)
        doc = result.document

        doi_artifacts_dir = docling_artifacts_dir / safe_doi
        export_paths = _save_docling_document_exports(doc, doi_artifacts_dir, safe_doi)

        parsed_dir = reparsed_dir / safe_doi
        parsed_dir.mkdir(parents=True, exist_ok=True)
        title = _extract_title(doc, Path(filename).stem)
        metadata = {
            "doi": doi,
            "directory": "docling_pdf",
            "doctype": "pdf",
            "source_path": str(pdf_path),
            "parsed_at": datetime.now().isoformat(),
            "title": title,
            "keywords": [],
            "docling_artifacts": export_paths,
        }
        parsed = {
            "Journal": "",
            "DOI": doi,
            "Title": title,
            "Keywords": [],
            "Abstract": "",
            "Sections": [_section_from_doc_text(doc)],
            "Docling": export_paths,
        }
        _write_json(parsed_dir / "metadata.json", metadata)
        _write_json(parsed_dir / "parsed.json", parsed)

        table_count = 0
        if run_extract_tables:
            table_dir = figures_tables_dir / "tables" / safe_doi
            for table in getattr(doc, "tables", []) or []:
                if _save_table(table, doc, table_dir, doi, table_count + 1, artifacts_dir=doi_artifacts_dir):
                    table_count += 1

        figure_count = 0
        if run_extract_figures:
            figure_dir = figures_tables_dir / "figures" / safe_doi
            for picture in getattr(doc, "pictures", []) or []:
                if _save_picture(picture, doc, figure_dir, doi, figure_count + 1):
                    figure_count += 1

        index_rows.append({
            "doi": doi,
            "status": "ok",
            "directory": "docling_pdf",
            "doctype": "pdf",
            "title": title,
            "keywords": "",
            "output_dir": str(parsed_dir),
            "source_pdf": str(pdf_path),
            "figures": figure_count,
            "tables": table_count,
        })

    index_csv = reparsed_dir / "index.csv"
    new_index = pd.DataFrame(index_rows)
    if index_csv.exists():
        try:
            existing_index = pd.read_csv(index_csv)
        except Exception:
            existing_index = pd.DataFrame()
        if not existing_index.empty and "doi" in existing_index.columns:
            replaced_dois = set(new_index.get("doi", pd.Series(dtype=str)).astype(str))
            existing_index = existing_index[
                ~existing_index["doi"].astype(str).isin(replaced_dois)
            ]
        new_index = pd.concat([existing_index, new_index], ignore_index=True, sort=False)
    new_index.to_csv(index_csv, index=False)

    structured_pdf_dir = run_dir / "papers" / "structured_files" / "docling_pdf"
    structured_pdf_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in (run_dir / "uploaded_pdfs").glob("*.pdf"):
        shutil.copy2(pdf_path, structured_pdf_dir / pdf_path.name)


def _downloaded_pdf_manifest(run_dir: Path) -> list[dict[str, str]]:
    pdf_root = run_dir / "papers" / "unstructured_files"
    manifest: list[dict[str, str]] = []
    if not pdf_root.exists():
        return manifest
    for pdf_path in sorted(pdf_root.glob("*/*.pdf")):
        safe_doi = pdf_path.stem
        manifest.append({
            "doi": safe_doi.replace("@", "/"),
            "safe_doi": safe_doi,
            "filename": pdf_path.name,
            "pdf_path": str(pdf_path),
        })
    return manifest


def _docling_available_here() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("docling") is not None
    except Exception:
        return False


def _run_docling_sidecar(
    *,
    run_dir: Path,
    manifest_path: Path,
    run_extract_tables: bool,
    run_extract_figures: bool,
    cancel_event: Any = None,
    use_ocr: bool = True,
) -> None:
    sidecar_python = os.environ.get("DOCLING_PYTHON", "").strip()
    if not sidecar_python:
        raise RuntimeError(
            "Docling is not installed in this Python 3.8 SALSA environment. "
            "Create a separate Python 3.10+ environment with Docling, then set "
            "DOCLING_PYTHON=/path/to/that/env/bin/python before starting SALSA."
        )

    cmd = [
        sidecar_python,
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir", str(run_dir),
        "--manifest", str(manifest_path),
    ]
    if run_extract_tables:
        cmd.append("--extract-tables")
    if run_extract_figures:
        cmd.append("--extract-figures")
    if not use_ocr:
        cmd.append("--no-ocr")

    env = os.environ.copy()
    env["SALSA_RUN_DIR"] = str(run_dir)
    env["SALSA_RUN_ID"] = run_dir.name
    env[RUN_STATUS_FILE_ENV] = str(run_dir / RUN_STATUS_FILE_NAME)
    src_root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if src_root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src_root] + parts)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        proc = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while proc.poll() is None:
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
                from pipeline import PipelineCancelled

                raise PipelineCancelled("Pipeline run cancelled by user.")
            import time

            time.sleep(0.5)

        if proc.returncode != 0:
            output.seek(0)
            details = output.read().strip()
            raise RuntimeError(f"Docling sidecar extraction failed: {details}")


def extract_uploaded_pdfs_with_docling(
    uploads: Iterable[PdfUpload],
    *,
    run_dir: Path,
    run_extract_tables: bool = True,
    run_extract_figures: bool = True,
    use_ocr: bool = False,
    status_callback: Any = None,
    cancel_event: Any = None,
) -> DoclingExtractionResult:
    uploads = list(uploads)
    if not uploads:
        raise ValueError("No uploaded PDFs were provided.")

    for path in (run_dir / "reparsed", run_dir / "figures_tables", run_dir / "docling"):
        path.mkdir(parents=True, exist_ok=True)

    manifest, dois, manifest_path = _write_uploaded_pdf_manifest(uploads, run_dir=run_dir)

    if _docling_available_here():
        _extract_manifest_with_docling(
            manifest,
            run_dir=run_dir,
            run_extract_tables=run_extract_tables,
            run_extract_figures=run_extract_figures,
            status_callback=status_callback,
            cancel_event=cancel_event,
            use_ocr=use_ocr,
        )
    else:
        _run_docling_sidecar(
            run_dir=run_dir,
            manifest_path=manifest_path,
            run_extract_tables=run_extract_tables,
            run_extract_figures=run_extract_figures,
            cancel_event=cancel_event,
            use_ocr=use_ocr,
        )

    return DoclingExtractionResult(
        dois=dois,
        index_csv=run_dir / "reparsed" / "index.csv",
        source_pdf_dir=run_dir / "uploaded_pdfs",
    )


def extract_downloaded_pdfs_with_docling(
    *,
    run_dir: Path,
    run_extract_tables: bool = True,
    run_extract_figures: bool = True,
    status_callback: Any = None,
    cancel_event: Any = None,
) -> DoclingExtractionResult | None:
    manifest = _downloaded_pdf_manifest(run_dir)
    if not manifest:
        return None

    manifest_path = run_dir / "downloaded_pdf_manifest.json"
    _write_json(manifest_path, manifest)
    if _docling_available_here():
        _extract_manifest_with_docling(
            manifest,
            run_dir=run_dir,
            run_extract_tables=run_extract_tables,
            run_extract_figures=run_extract_figures,
            status_callback=status_callback,
            cancel_event=cancel_event,
            use_ocr=False,
        )
    else:
        _run_docling_sidecar(
            run_dir=run_dir,
            manifest_path=manifest_path,
            run_extract_tables=run_extract_tables,
            run_extract_figures=run_extract_figures,
            cancel_event=cancel_event,
            use_ocr=False,
        )

    return DoclingExtractionResult(
        dois=[entry["doi"] for entry in manifest],
        index_csv=run_dir / "reparsed" / "index.csv",
        source_pdf_dir=run_dir / "papers" / "unstructured_files",
    )


def _worker_main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Docling PDF extraction worker")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--extract-tables", action="store_true")
    parser.add_argument("--extract-figures", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    _extract_manifest_with_docling(
        manifest,
        run_dir=run_dir,
        run_extract_tables=args.extract_tables,
        run_extract_figures=args.extract_figures,
        status_callback=lambda stage, doi: write_current_doi(doi, stage=stage),
        use_ocr=not args.no_ocr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
