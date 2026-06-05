from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class FigureUpload:
    filename: str
    content: bytes
    caption: str = ""


@dataclass(frozen=True)
class UploadedFigureResult:
    dois: list[str]
    index_csv: Path
    source_figures_dir: Path


def _doi_to_safe(doi: str) -> str:
    return doi.replace("/", "@").strip()


def _safe_stem(value: str, fallback: str) -> str:
    stem = Path(value).stem.strip() or fallback
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    stem = stem.strip("._-")
    return stem[:80] or fallback


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def stage_uploaded_figures(
    uploads: Iterable[FigureUpload],
    *,
    run_dir: Path,
    document_id: str | None = None,
) -> UploadedFigureResult:
    uploads = [u for u in uploads if u.content]
    if not uploads:
        raise ValueError("No uploaded figures were provided.")

    base_id = (document_id or "").strip()
    if not base_id:
        if len(uploads) == 1:
            base_id = f"uploaded/{_safe_stem(uploads[0].filename, 'uploaded_figure')}"
        else:
            base_id = "uploaded/figures"

    doi = base_id
    safe_doi = _doi_to_safe(doi)
    now = datetime.now().isoformat()

    source_dir = run_dir / "uploaded_figures" / safe_doi
    figures_dir = run_dir / "figures_tables" / "figures" / safe_doi
    reparsed_dir = run_dir / "reparsed" / safe_doi
    for path in (source_dir, figures_dir, reparsed_dir):
        path.mkdir(parents=True, exist_ok=True)

    for idx, upload in enumerate(uploads, start=1):
        ext = Path(upload.filename).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp"}:
            ext = ".png"
        figure_stem = f"Figure_{idx}"
        figure_path = figures_dir / f"{figure_stem}{ext}"
        source_path = source_dir / f"{figure_stem}_{_safe_stem(upload.filename, figure_stem)}{ext}"

        source_path.write_bytes(upload.content)
        figure_path.write_bytes(upload.content)

        metadata = {
            "original_label": f"Uploaded figure {idx}",
            "clean_label": figure_stem,
            "caption": upload.caption.strip(),
            "url": "",
            "doi": doi,
            "figure_file_path": str(figure_path),
            "figure_directory": str(figures_dir),
            "source_upload_path": str(source_path),
            "source_filename": upload.filename,
            "extraction_timestamp": now,
            "file_type": "figure",
            "publisher": "uploaded_figure",
        }
        _write_json(figures_dir / f"{figure_stem}_metadata.json", metadata)

    title = base_id.replace("uploaded/", "").replace("_", " ").strip() or "Uploaded figures"
    parsed = {
        "Journal": "",
        "DOI": doi,
        "Title": title,
        "Keywords": [],
        "Abstract": "",
        "Sections": [
            {
                "type": "uploaded_figure_set",
                "name": "Uploaded figures",
                "content": [
                    "User-uploaded figure image set. No source paper full text was provided."
                ],
            }
        ],
    }
    metadata = {
        "doi": doi,
        "directory": "uploaded_figures",
        "doctype": "images",
        "source_path": str(source_dir),
        "parsed_at": now,
        "title": title,
        "keywords": [],
    }
    _write_json(reparsed_dir / "metadata.json", metadata)
    _write_json(reparsed_dir / "parsed.json", parsed)

    index_csv = run_dir / "reparsed" / "index.csv"
    pd.DataFrame([
        {
            "doi": doi,
            "status": "ok",
            "directory": "uploaded_figures",
            "doctype": "images",
            "title": title,
            "keywords": "",
            "output_dir": str(reparsed_dir),
            "source_figures": str(source_dir),
            "figures": len(uploads),
            "tables": 0,
        }
    ]).to_csv(index_csv, index=False)

    return UploadedFigureResult(
        dois=[doi],
        index_csv=index_csv,
        source_figures_dir=source_dir,
    )
