from __future__ import annotations

import base64
import contextvars
import datetime
import json
import logging
import mimetypes
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from llm_client import get_llm_client
from ui.openai_cost import append_usage, run_id_from_env
from ui.run_progress import write_current_doi, write_run_status

_RUN_DIR_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_direct_figure_data_run_dir",
    default="",
)
_RUN_ID_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_direct_figure_data_run_id",
    default="",
)


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


def _run_logger(workdir: Path, run_id: int) -> tuple[logging.Logger, Path]:
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llm_figure_extraction_run_{run_id}.log"
    logger = logging.getLogger(f"salsa.llm_figure_extraction.{workdir.resolve()}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger, log_path


def _append_status_event(status_path: Path, event: Dict[str, Any]) -> None:
    record = dict(event)
    record["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(status_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


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
        "name": "salsa_direct_figure_data",
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
            run_id=_RUN_ID_CONTEXT.get() or run_id_from_env("unknown"),
            resp=meta,
            doi=doi,
            artifact_id=figure_name,
            extra={"kind": "direct_llm_figure_data"},
            run_dir=_RUN_DIR_CONTEXT.get() or None,
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
        _RUN_DIR_CONTEXT.set(str(run_dir))
        _RUN_ID_CONTEXT.set(run_dir.name)

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
    logger, log_path = _run_logger(workdir, run_id)
    status_path = workdir / "logs" / f"llm_figure_extraction_run_{run_id}.jsonl"
    successes = 0
    failures = 0
    logger.info("Starting direct LLM figure extraction")
    logger.info("figures_root=%s", figures_root)
    logger.info("output_root=%s", output_root)
    _append_status_event(
        status_path,
        {
            "event": "run_started",
            "figures_root": str(figures_root),
            "output_root": str(output_root),
            "log_path": str(log_path),
        },
    )

    for doi_dir in sorted(p for p in figures_root.iterdir() if p.is_dir()):
        safe_doi = doi_dir.name
        doi = safe_doi.replace("@", "/")
        write_current_doi(doi)
        logger.info("Processing DOI %s", doi)
        for image_path in sorted(p for p in doi_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES):
            figure_id = _safe_output_name(image_path)
            final_dir = output_root / safe_doi / figure_id / "final"
            caption = ""
            try:
                final_dir.mkdir(parents=True, exist_ok=True)
                caption = _read_caption(image_path)
                logger.info("Starting figure doi=%s figure=%s image=%s", doi, figure_id, image_path)
                write_run_status(
                    state="running",
                    stage="Extracting figure data with LLM",
                    doi=doi,
                    figure=figure_id,
                    image_path=str(image_path),
                )
                _append_status_event(
                    status_path,
                    {
                        "event": "figure_started",
                        "doi": doi,
                        "safe_doi": safe_doi,
                        "figure_id": figure_id,
                        "image_path": str(image_path),
                    },
                )
                df, meta = extract_figure_data_llm(
                    image_path=image_path,
                    doi=doi,
                    caption=caption,
                )
                _log_cost(meta, doi=doi, figure_name=figure_id)
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
                            "status": "success",
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                successes += 1
                logger.info("Finished figure doi=%s figure=%s rows=%s", doi, figure_id, len(df))
                _append_status_event(
                    status_path,
                    {
                        "event": "figure_success",
                        "doi": doi,
                        "safe_doi": safe_doi,
                        "figure_id": figure_id,
                        "image_path": str(image_path),
                        "rows": len(df),
                        "final_dataset": str(final_dir / "final_dataset.csv"),
                    },
                )
            except Exception as exc:
                failures += 1
                tb = traceback.format_exc()
                print(f"[direct_llm_figure_data] WARNING: extraction failed for {image_path}: {exc}")
                logger.error("Failed figure doi=%s figure=%s image=%s error=%s", doi, figure_id, image_path, exc)
                logger.error(tb)
                try:
                    final_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(columns=FIGURE_DATA_COLUMNS).to_csv(final_dir / "final_dataset.csv", index=False)
                    (final_dir / "metadata.json").write_text(
                        json.dumps(
                            {
                                "doi": doi,
                                "safe_doi": safe_doi,
                                "figure_id": figure_id,
                                "image_path": str(image_path),
                                "caption": caption,
                                "extraction_method": "direct_llm",
                                "rows": 0,
                                "status": "failed",
                                "error": str(exc),
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                except Exception as write_exc:
                    logger.error(
                        "Failed to write failure placeholder doi=%s figure=%s error=%s",
                        doi,
                        figure_id,
                        write_exc,
                    )
                _append_status_event(
                    status_path,
                    {
                        "event": "figure_failure",
                        "doi": doi,
                        "safe_doi": safe_doi,
                        "figure_id": figure_id,
                        "image_path": str(image_path),
                        "error": str(exc),
                        "traceback": tb,
                    },
                )

    logger.info("Finished direct LLM figure extraction successes=%s failures=%s", successes, failures)
    _append_status_event(
        status_path,
        {
            "event": "run_finished",
            "successes": successes,
            "failures": failures,
            "output_root": str(output_root),
        },
    )
    return output_root
