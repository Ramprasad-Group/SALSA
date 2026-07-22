"""Combine a SALSA run's table/figure data and chemical compositions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROVENANCE_COLUMNS = ["DOI", "source_type", "source_name", "source_path", "source_row"]


@dataclass(frozen=True)
class AggregationResult:
    data_csv: Path
    composition_csv: Path
    data_rows: int
    composition_rows: int
    warnings: tuple[str, ...] = ()


def _doi_from(path: Path, run_dir: Path) -> str:
    for part in path.relative_to(run_dir).parts:
        if part.startswith("10.") and "@" in part:
            return part.replace("@", "/", 1)
    return ""


def _read_rows(path: Path, warnings: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return [], []
            return list(reader.fieldnames), [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        warnings.append(f"Skipped {path}: {exc}")
        return [], []


def _configured_columns(run_dir: Path, key: str, warnings: list[str]) -> list[str]:
    """Read the ordered complete-dataset schema saved with the run."""
    path = run_dir / "custom_instructions.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_columns = payload.get(key, []) if isinstance(payload, dict) else []
        if not isinstance(raw_columns, list):
            raise ValueError(f"'{key}' is not a list")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"Could not read output-column order from {path}: {exc}")
        return []

    columns: list[str] = []
    for value in raw_columns:
        column = str(value).strip()
        if column and column not in columns:
            columns.append(column)
    return columns


def _write_union(
    sources: Iterable[tuple[Path, str]], run_dir: Path, destination: Path, warnings: list[str],
    configured_columns: list[str] | None = None,
) -> int:
    loaded: list[tuple[Path, str, list[dict[str, str]]]] = []
    columns: list[str] = list(configured_columns or [])
    for path, source_type in sorted(sources, key=lambda item: str(item[0])):
        fields, rows = _read_rows(path, warnings)
        if not configured_columns:
            for field in fields:
                if field not in PROVENANCE_COLUMNS and field not in columns:
                    columns.append(field)
        loaded.append((path, source_type, rows))

    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = columns if configured_columns else PROVENANCE_COLUMNS + columns
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for path, source_type, rows in loaded:
            doi = _doi_from(path, run_dir)
            for index, row in enumerate(rows, start=1):
                # Provenance is authoritative even if an input happens to use
                # one of these column names.
                row.update({
                    "DOI": doi or row.get("DOI", ""),
                    "source_type": source_type,
                    "source_name": path.parent.parent.name if source_type == "figure" else path.stem,
                    "source_path": str(path.relative_to(run_dir)),
                    "source_row": str(index),
                })
                writer.writerow(row)
                count += 1
    return count


def _write_completed_rows(run_dir: Path, destination: Path, warnings: list[str]) -> int:
    """Union current LLM-completed rows, preferring figure intervention reruns.

    SALSA's "New LLM-Generated Dataset" action writes a newer row-cache entry
    for the same DOI/figure.  Grouping figure names by filename stem and choosing
    the newest entry ensures that rerun replaces, rather than duplicates, the
    original dataset (including ``Figure_1`` versus ``Figure_1.jpg`` names).
    """
    latest_sources: dict[tuple[str, str, str], tuple[Path, list[dict[str, object]]]] = {}
    for path in sorted((run_dir / "complete_dataset" / "row_cache").glob("emit_rows_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                raise ValueError("'rows' is not a list")
            valid_rows = [row for row in rows if isinstance(row, dict)]
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"Skipped {path}: {exc}")
            continue
        first_row = valid_rows[0] if valid_rows else {}
        source_kind = str(first_row.get("source_kind", ""))
        source_name = str(first_row.get("source_name", path.stem))
        canonical_name = Path(source_name).stem if source_kind == "figure" else source_name
        source_key = (
            str(first_row.get("DOI", "")),
            source_kind,
            canonical_name,
        )
        previous = latest_sources.get(source_key)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            latest_sources[source_key] = (path, valid_rows)

    # Build the union only from the selected versions. Historical cache files
    # must not leak obsolete LLM column names into the exported schema.
    columns = _configured_columns(run_dir, "output_columns", warnings)
    if not columns:
        for _path, rows in sorted(latest_sources.values(), key=lambda item: str(item[0])):
            for row in rows:
                for field in row:
                    if field not in columns:
                        columns.append(field)
        for field in PROVENANCE_COLUMNS:
            if field not in columns:
                columns.append(field)

    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for path, rows in sorted(latest_sources.values(), key=lambda item: str(item[0])):
            for index, original in enumerate(rows, start=1):
                row = dict(original)
                row.update({
                    "DOI": str(original.get("DOI", "")),
                    "source_type": str(original.get("source_kind", "")),
                    "source_name": str(original.get("source_name", path.stem)),
                    "source_path": str(path.relative_to(run_dir)),
                    "source_row": str(index),
                })
                writer.writerow(row)
                count += 1
    return count


def aggregate_run(run_dir: str | Path, output_dir: str | Path | None = None) -> AggregationResult:
    """Create ``tables_figures.csv`` and ``chemical_compositions.csv`` for a run.

    Only the newest cached LLM result for each DOI/source is used, so reruns and
    intervention histories do not duplicate obsolete versions.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    destination = Path(output_dir).resolve() if output_dir else run_dir / "aggregated"
    warnings: list[str] = []

    data_path = destination / "tables_figures.csv"
    data_rows = _write_completed_rows(run_dir, data_path, warnings)

    compositions = ((path, "chemical_composition") for path in (run_dir / "chemical_composition").glob("*/samples.csv"))
    composition_path = destination / "chemical_compositions.csv"
    composition_columns = _configured_columns(
        run_dir, "chemical_composition_columns", warnings
    )
    composition_rows = _write_union(
        compositions, run_dir, composition_path, warnings,
        configured_columns=composition_columns or None,
    )
    return AggregationResult(data_path, composition_path, data_rows, composition_rows, tuple(warnings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = aggregate_run(args.run_dir, args.output_dir)
    print(f"Wrote {result.data_rows} table/figure rows to {result.data_csv}")
    print(f"Wrote {result.composition_rows} composition rows to {result.composition_csv}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
