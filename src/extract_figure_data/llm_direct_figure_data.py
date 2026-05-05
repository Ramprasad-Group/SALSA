from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from llm_client import get_llm_client
from ui.openai_cost import append_usage, run_id_from_env


FIGURE_DATA_COLUMNS = [
    "series_label",
    "x_name",
    "x_unit",
    "x_value",
    "y_name",
    "y_unit",
    "y_value",
]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _safe_output_name(image_path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("_")
    return name or "figure"


def _clean_json_response(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3]
    try:
        parsed = json.loads(text.strip())
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rows_schema() -> Dict[str, Any]:
    row_properties = {column: {"type": "string"} for column in FIGURE_DATA_COLUMNS}
    return {
        "name": "aem_ocr_direct_figure_data",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["rows"],
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(FIGURE_DATA_COLUMNS),
                        "properties": row_properties,
                    },
                }
            },
        },
    }


def _read_caption(image_path: Path) -> str:
    for meta_path in (
        image_path.with_name(image_path.stem + "_metadata.json"),
        image_path.with_name(image_path.stem + ".metadata.json"),
        image_path.with_name(image_path.stem + ".json"),
    ):
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        caption = meta.get("caption") or meta.get("figure_caption") or ""
        return str(caption)
    return ""


def _coerce_rows(rows: Any) -> pd.DataFrame:
    fixed: List[Dict[str, str]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            fixed.append(
                {
                    column: "" if row.get(column) is None else str(row.get(column, ""))
                    for column in FIGURE_DATA_COLUMNS
                }
            )
    return pd.DataFrame(fixed, columns=FIGURE_DATA_COLUMNS)


def _log_cost(meta: Any, *, doi: str, figure_name: str) -> None:
    try:
        append_usage(
            run_id=run_id_from_env("unknown"),
            resp=meta,
            doi=doi,
            artifact_id=figure_name,
            extra={"kind": "direct_llm_figure_data"},
        )
    except Exception as exc:
        print(f"[direct_llm_figure_data] WARNING: failed to log cost for {figure_name}: {exc}")


def extract_figure_data_llm(
    *,
    image_path: Path,
    doi: str,
    caption: str,
) -> tuple[pd.DataFrame, Any]:
    client = get_llm_client(force_new=True)
    figure_name = image_path.stem
    instructions = f"""
You are an expert at digitizing scientific plots from figure images.

Extract the plotted data directly from the attached figure image. Return one row
per visible data point. If the image is not a quantitative plot, or the axes and
values cannot be read, return an empty rows array.

Output columns:
{json.dumps(FIGURE_DATA_COLUMNS, indent=2)}

Rules:
- Fill every output column as a string.
- series_label is the legend label or plotted-series identifier. Use "" only
  when the figure has one unlabeled series.
- x_name and y_name are the axis/property names.
- x_unit and y_unit are units without parentheses. Use "" when no unit is shown.
- x_value and y_value must be numeric-looking strings when values can be read.
- Do not invent points that are not visible.
- Do not create summary rows.
- Use the caption only to interpret series labels, units, and context. The
  plotted values must come from the image.

DOI: {doi}
Figure: {figure_name}
Caption:
\"\"\"{caption}\"\"\"
""".strip()

    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Digitize this figure and return only the requested JSON rows.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path), "detail": "high"},
                },
            ],
        },
    ]

    try:
        raw, meta = client.chat(
            messages,
            response_format={"type": "json_schema", "json_schema": _rows_schema()},
            temperature=0.0,
            max_tokens=32768,
            return_meta=True,
        )
    except Exception:
        raw, meta = client.chat(
            messages,
            temperature=0.0,
            max_tokens=32768,
            return_meta=True,
        )

    parsed = _clean_json_response(raw)
    return _coerce_rows(parsed.get("rows", [])), meta


def run_llm_figure_data_extraction(
    *,
    figures_root: Path,
    workdir: Path,
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Write direct-LLM figure data under:
      <workdir>/results/<n>/output/<safe_doi>/<figure_id>/final/final_dataset.csv

    This mirrors the plextract output shape consumed by complete_dataset.py.
    """
    if run_dir is not None:
        import os

        os.environ["AEM_OCR_RUN_DIR"] = str(run_dir)

    workdir.mkdir(parents=True, exist_ok=True)
    results_root = workdir / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    existing_run_ids = []
    for path in results_root.iterdir():
        if path.is_dir() and path.name.isdigit():
            existing_run_ids.append(int(path.name))
    run_id = (max(existing_run_ids) + 1) if existing_run_ids else 0
    output_root = results_root / str(run_id) / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    for doi_dir in sorted(p for p in figures_root.iterdir() if p.is_dir()):
        safe_doi = doi_dir.name
        doi = safe_doi.replace("@", "/")
        for image_path in sorted(p for p in doi_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES):
            figure_id = _safe_output_name(image_path)
            final_dir = output_root / safe_doi / figure_id / "final"
            final_dir.mkdir(parents=True, exist_ok=True)

            caption = _read_caption(image_path)
            try:
                df, meta = extract_figure_data_llm(
                    image_path=image_path,
                    doi=doi,
                    caption=caption,
                )
                _log_cost(meta, doi=doi, figure_name=figure_id)
            except Exception as exc:
                print(f"[direct_llm_figure_data] WARNING: extraction failed for {image_path}: {exc}")
                df = pd.DataFrame(columns=FIGURE_DATA_COLUMNS)

            df = df.reindex(columns=FIGURE_DATA_COLUMNS, fill_value="").fillna("").astype(str)
            df.to_csv(final_dir / "final_dataset.csv", index=False)
            (final_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "doi": doi,
                        "safe_doi": safe_doi,
                        "figure_id": figure_id,
                        "image_path": str(image_path),
                        "caption": caption,
                        "extraction_method": "direct_llm",
                        "rows": len(df),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )

    return output_root
