#!/usr/bin/env python3
"""
Complete dataset builder with LLM-generated TRANSFORMATION PLANS (not row-emission).

Usage:
  python complete_dataset.py --file_path ... --output_dir ... --index_csv_path ... \
      --figures_tables_dir ... --extracted_figures_root ... \
      --openai_api_key ... --openai_model gpt-4o
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import sys
import os
import logging
import json
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import hashlib



_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)

from llm_client import LLMConfig, get_llm_client, load_processing_llm_config
from ui.openai_cost import append_usage, run_id_from_env
from ui.custom_instructions import load_instructions_from_env_or_path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../paper-downloader"))
sys.path.insert(0, parent_dir)
from paper_downloader import config, utils

# --- new helper module (in same directory as this script) ---
from row_emitter import (
    OUTPUT_COLUMNS as _DEFAULT_COLUMNS,
    ensure_cache_dir,
    infer_rows_for_table,
    infer_rows_for_figure_dataset,
    subsample_figure_points,
)

from figure_plan import infer_figure_series_plans, apply_figure_series_plans



parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default=os.path.join(_SRC_ROOT, 'doi-inputs', 'validate_paper_download.csv'),
                    help='Path to CSV/Excel file containing DOIs')
parser.add_argument('--output_dir', default=os.path.join(_SRC_ROOT, 'complete-dataset', 'logs', 'records'),
                    help='Directory to write combined JSONs and completed CSVs')
parser.add_argument('--index_csv_path', default=os.path.join(_SRC_ROOT, 'paper-downloader', 'paper_downloader', 'logs', 'papers', 'reparsed', 'index.csv'),
                    help="CSV where downloaded files metadata is stored")
parser.add_argument('--doi_column', default='DOI',
                    help='Name of DOI column in file')
parser.add_argument('--log_file', dest='log_file', default=os.path.join(_SRC_ROOT, 'complete-dataset', 'logs', 'records'),
                    help='Path to log file directory')

parser.add_argument('--extracted_figures_root', default='',
                    help='Optional root dir containing extracted figure-data per DOI folder (csv/json). '
                         'Expected to contain subdir named <modified_doi>.')

parser.add_argument('--figures_tables_dir', default='',
                    help="Optional override: path to run-scoped figures_tables dir (contains tables/ and figures/). "
                         "If provided, tables_dir = <figures_tables_dir>/tables/<modified_doi> and figures_dir = <figures_tables_dir>/figures/<modified_doi>.")

parser.add_argument('--exclude_full_text', default=False, action='store_true',
                    help="Exclude parsed full text from LLM prompt (reduces token usage drastically)")

# allow skipping LLM completion (still writes combined json)
parser.add_argument('--skip_llm', action='store_true', help="Skip GPT plan step (will use heuristics only)")
parser.add_argument('--skip_llm_figures', action='store_true',
                    help="Skip LLM row generation for figures (tables still processed). "
                         "Use on first pass; re-run per-figure via 'Rerun LLM dataset generation'.")

# plan caching
parser.add_argument('--plan_cache_dir', default='',
                    help="Optional override for plan cache directory. Default: <output_dir>/plans")

parser.add_argument('--custom_instructions', default='',
                    help="Path to custom_instructions.json (overrides AEM_OCR_CUSTOM_INSTRUCTIONS env var)")

parser.add_argument('--target_figure', default='',
                    help="Optional: target a specific figure/subfigure by name (e.g., 'Figure_1_subfig_0'). "
                         "If specified, only this figure/subfigure will be processed.")

args = parser.parse_args()

# --- Load custom instructions ---
_ci = load_instructions_from_env_or_path(args.custom_instructions or None)
DATASET_COLUMNS = _ci.get("output_columns", list(_DEFAULT_COLUMNS))
_MISSING_DEFAULTS = _ci.get("missing_value_defaults", {})
_EXTRA_RULES = _ci.get("data_processing_rules", "")

# Setup logging
log_file = f"{args.log_file}/complete_datasets_{datetime.now().strftime(config.DATE_TIME_FORMAT)}.log"
Path(args.log_file).mkdir(parents=True, exist_ok=True)
log = utils.setup_logger(log_file, logger_name='complete_datasets')

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

ROW_CACHE_DIR = args.plan_cache_dir.strip() or os.path.join(args.output_dir, "row_cache")
ensure_cache_dir(ROW_CACHE_DIR)

# ----------------------------
# Utilities: cost logging
# ----------------------------
def _log_cost(response, extra: dict):
    append_usage(
        run_id=run_id_from_env("unknown"),
        resp=response,
        extra=extra or {},
    )

# ----------------------------
# Utilities: load + locate
# ----------------------------
def load_file() -> pd.DataFrame:
    file_extension = args.file_path.lower().split('.')[-1]
    if file_extension in ('xls', 'xlsx'):
        df = pd.read_excel(args.file_path)
        log.info(f"Loaded Excel file: {args.file_path}")
    elif file_extension == 'csv':
        df = pd.read_csv(args.file_path)
        log.info(f"Loaded CSV file: {args.file_path}")
    else:
        log.error("Unsupported file format. Please use CSV, XLS, or XLSX.")
        sys.exit(1)

    if args.doi_column not in df.columns:
        log.error(f"DOI column '{args.doi_column}' not found. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    log.info(f"Original DataFrame size: {len(df)} rows")
    df = df.dropna(subset=[args.doi_column])
    df = df[df[args.doi_column].astype(str).str.strip() != ""]
    log.info(f"After removing empty DOI strings: {len(df)} rows")
    return df

def find_matching_files(doi: str, csv_path: str) -> Optional[str]:
    index_df = pd.read_csv(csv_path)
    matches = index_df[index_df["doi"] == doi]
    if len(matches) == 0:
        return None
    try:
        return matches.output_dir.iloc[0]
    except Exception:
        return None

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

# ----------------------------
# Agglomeration: Tables
# ----------------------------
def collect_tables(doi: str, tables_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(tables_dir):
        log.warning(f"Tables directory does not exist for DOI {doi}: {tables_dir}")
        return []

    tables: List[Dict[str, Any]] = []

    for filename in os.listdir(tables_dir):
        if not filename.lower().endswith(".csv"):
            continue

        table_base = os.path.splitext(filename)[0]
        table_csv_path = os.path.join(tables_dir, filename)
        metadata_path = os.path.join(tables_dir, f"{table_base}_metadata.json")

        if not os.path.exists(metadata_path):
            log.warning(f"No metadata file found for table '{table_csv_path}' (expected: {metadata_path})")
            continue

        try:
            table_df = pd.read_csv(table_csv_path)
        except Exception as e:
            log.error(f"Failed to read table CSV {table_csv_path}: {e}")
            continue

        try:
            with open(metadata_path, "r") as mf:
                metadata = json.load(mf)
        except Exception as e:
            log.error(f"Failed to read metadata JSON {metadata_path}: {e}")
            continue

        tables.append({
            "doi": doi,
            "table_name": table_base,
            "table_csv_path": table_csv_path,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "table": table_df.to_dict(orient="records"),
            "table_columns": list(table_df.columns),
        })

    return tables

# ----------------------------
# Agglomeration: Figures
# ----------------------------

def collect_figures(doi: str, figures_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(figures_dir):
        log.warning(f"Figures directory does not exist for DOI {doi}: {figures_dir}")
        return []

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp"}
    figures: List[Dict[str, Any]] = []

    for filename in os.listdir(figures_dir):
        fp = os.path.join(figures_dir, filename)
        if not os.path.isfile(fp):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_exts:
            continue

        base = os.path.splitext(filename)[0]
        candidate_meta = [
            os.path.join(figures_dir, f"{base}_metadata.json"),
            os.path.join(figures_dir, f"{base}.metadata.json"),
            os.path.join(figures_dir, f"{base}.json"),
        ]
        meta_path = next((m for m in candidate_meta if os.path.exists(m)), None)
        metadata = None
        if meta_path:
            try:
                with open(meta_path, "r") as mf:
                    metadata = json.load(mf)
            except Exception as e:
                log.warning(f"Failed to read figure metadata {meta_path}: {e}")

        figures.append({
            "doi": doi,
            "figure_name": base,
            "image_path": fp,
            "metadata_path": meta_path,
            "metadata": metadata,
        })

    return figures

# ----------------------------
# Extracted plextract figure data
# ----------------------------

# def load_all_extracted_figure_points(modified_doi: str) -> pd.DataFrame:
#     root = (args.extracted_figures_root or "").strip()
#     if not root:
#         return pd.DataFrame()

#     doi_dir = os.path.join(root, modified_doi)
#     if not os.path.isdir(doi_dir):
#         return pd.DataFrame()

#     dfs = []
#     for fp in Path(doi_dir).glob("**/final/final_dataset.csv"):
#         try:
#             df = pd.read_csv(fp)
#             keep = ['series_label','x_name','x_unit','x_value','y_name','y_unit','y_value']
#             if all(c in df.columns for c in keep):
#                 dfs.append(df[keep].copy())
#         except Exception:
#             pass
#     return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

@dataclass
class FigurePointDataset:
    name: str          # stable identifier for this final_dataset.csv
    path: str
    df: pd.DataFrame

def _stable_name_from_path(fp: Path) -> str:
    # short stable id, avoids insanely long cache keys
    h = hashlib.sha1(str(fp).encode("utf-8")).hexdigest()[:10]
    return f"{fp.parent.parent.name}__{fp.parent.name}__{fp.stem}__{h}"
    # Example: <some_dir>__final__final_dataset__a1b2c3d4e5


def _parse_axis_title(text: str):
    """
    Extract (name, unit) from an axis title like "Time (h)" or "Conductivity [mS/cm]".
    Returns (name_str, unit_str_or_None).
    """
    import re as _re
    if not text:
        return None, None
    text = text.strip()
    m = _re.match(r"^(.*?)[\s]*[\[\(]([\s\S]*?)[\]\)]\s*$", text)
    if m:
        name = m.group(1).strip()
        unit = m.group(2).strip()
        return name or None, unit or None
    return text or None, None


def _apply_axis_title_overrides(df: pd.DataFrame, figure_dir: Path) -> pd.DataFrame:
    """
    If an axis_title_override.json file exists in <figure_dir>/chartdete/, parse
    it and overwrite the x_name, x_unit, y_name, y_unit columns in the DataFrame.
    This propagates user-edited axis titles to the LLM without requiring a full
    re-extraction.
    """
    override_path = figure_dir / "chartdete" / "axis_title_override.json"
    if not override_path.exists():
        return df

    try:
        with open(override_path, "r") as f:
            ov = json.load(f)
    except Exception:
        return df

    x_title = (ov.get("x_title") or "").strip()
    y_title = (ov.get("y_title") or "").strip()

    if not x_title and not y_title:
        return df

    df = df.copy()

    if x_title:
        x_name, x_unit = _parse_axis_title(x_title)
        if x_name is not None and "x_name" in df.columns:
            df["x_name"] = x_name
        if x_unit is not None and "x_unit" in df.columns:
            df["x_unit"] = x_unit

    if y_title:
        y_name, y_unit = _parse_axis_title(y_title)
        if y_name is not None and "y_name" in df.columns:
            df["y_name"] = y_name
        if y_unit is not None and "y_unit" in df.columns:
            df["y_unit"] = y_unit

    return df


def load_extracted_figure_datasets(modified_doi: str, run_dir: Optional[Path] = None) -> List[FigurePointDataset]:
    """
    Load extracted figure datasets from both interventions_plextract and plextract directories.
    Checks interventions_plextract first (for intervention-based extractions), then falls back
    to plextract (for original extractions).

    For figures that have intervention results, those take priority over original plextract.

    Args:
        modified_doi: Safe DOI string for lookup
        run_dir: Optional run directory to check for interventions_plextract.
                Structure: {run_dir}/interventions/interventions_plextract/results/{n}/output/{doi}/{figure}/final/
    """
    root = (args.extracted_figures_root or "").strip()


    keep = ['series_label','x_name','x_unit','x_value','y_name','y_unit','y_value']
    out: List[FigurePointDataset] = []
    # Track which figure names have intervention results so we can skip them in plextract
    intervention_figure_names: set = set()

    # Priority 1: Check interventions_plextract if run_dir is provided
    if run_dir:
        int_workdir = Path(run_dir) / "interventions" / modified_doi
        if int_workdir.is_dir():
            # Glob all final_dataset.csv files under interventions plextract results
            pattern = f"*/plextract_interventions/results/*/output/{modified_doi}/*/final/final_dataset.csv"

            for fp in int_workdir.glob(pattern):
                try:
                    # Optional: filter by target figure
                    figure_name = fp.parent.parent.name
                    if args.target_figure and figure_name != args.target_figure:
                        continue

                    # Skip if we already have data for this figure from another workdir
                    if figure_name in intervention_figure_names:
                        continue

                    # Find the most recent run dir for THIS figure's plextract_interventions folder
                    # prefix = .../<Figure_X>/plextract_interventions
                    idx = fp.parts.index("results")
                    prefix = Path(*fp.parts[:idx])  # up to plextract_interventions
                    most_recent_run = _latest_result_run_dir(prefix)  # should be .../results/<run_id>

                    # Only process files that belong to that most recent run
                    fp_run_dir = Path(*fp.parts[:idx + 2])  # .../results/<run_id>
                    if fp_run_dir != most_recent_run:
                        continue
                
                    df = pd.read_csv(fp)
                    if not all(c in df.columns for c in keep):
                        continue

                    df = df[keep].copy()
                    if df.empty:
                        continue

                    # Apply any user-edited axis title overrides so the LLM gets corrected labels/units
                    df = _apply_axis_title_overrides(df, fp.parent.parent)

                    out.append(
                        FigurePointDataset(
                            name=_stable_name_from_path(fp),
                            path=str(fp),
                            df=df,
                        )
                    )
                    intervention_figure_names.add(figure_name)

                except Exception:
                    continue
    
    # Priority 2: Check plextract (fall back for figures without intervention results)
    if not root:
        return out

    doi_dir = Path(root) / modified_doi
    if not doi_dir.is_dir():
        return out

    for fp in doi_dir.glob("**/final/final_dataset.csv"):
        try:
            figure_name = fp.parent.parent.name
            if args.target_figure and figure_name != args.target_figure:
                continue

            # Skip figures that already have intervention results
            if figure_name in intervention_figure_names:
                continue

            df = pd.read_csv(fp)
            if not all(c in df.columns for c in keep):
                continue
            df = df[keep].copy()
            if df.empty:
                continue
            # Apply any user-edited axis title overrides.
            df = _apply_axis_title_overrides(df, fp.parent.parent)

            out.append(FigurePointDataset(
                name=_stable_name_from_path(fp),
                path=str(fp),
                df=df
            ))
        except Exception:
            continue

    return out


def complete_dataset_from_figures_llm(complete_record, client) -> pd.DataFrame:

    doi = complete_record.get("doi", "")
    full_text_raw = complete_record.get("full_text", {}) if not args.exclude_full_text else {}
    full_text = _filter_full_text_sections(full_text_raw) if full_text_raw else {}

    ef = complete_record.get("extracted_figure_data", {})
    datasets = (ef or {}).get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    # captions/context
    figure_snips = []
    for f in (complete_record.get("figures", []) or [])[:8]:
        if not isinstance(f, dict):
            continue
        md = f.get("metadata") or {}
        cap = md.get("caption") or md.get("figure_caption") or ""
        figure_snips.append({"figure_name": f.get("figure_name", ""), "caption": str(cap)})

    dfs = []
    for ds in datasets:
        ds_name = str(ds.get("dataset_name", "")).strip()
        rows = ds.get("rows", [])
        if not ds_name or not isinstance(rows, list) or not rows:
            continue

        df_points = pd.DataFrame(rows)
        if df_points.empty:
            continue

        # Keep all points for small/medium datasets so LLM row generation
        # reflects the latest OCR output (e.g., 5 OCR rows should not collapse
        # to a single sampled point). For very large datasets, downsample to
        # roughly <=200 points to control prompt size.
        _sample_step = max(1, (len(df_points) + 199) // 200)
        if _sample_step > 1:
            df_points = subsample_figure_points(
                df_points,
                sample_col="series_label",
                step=_sample_step,
            )

        if args.skip_llm or client is None:
            continue

        df_rows, meta = infer_rows_for_figure_dataset(
            client=client,
            doi=doi,
            dataset_name=ds_name,
            figure_snips=figure_snips,
            df_points=df_points,
            full_text=full_text,
            cache_dir=ROW_CACHE_DIR,
            max_points_per_call=200,
            output_columns=DATASET_COLUMNS,
            missing_value_defaults=_MISSING_DEFAULTS,
            extra_rules=_EXTRA_RULES,
        )

        try:
            _log_cost(meta, extra={"doi": doi, "kind": "emit_rows_figure", "dataset_name": ds_name})
        except Exception:
            pass

        dfs.append(df_rows)

    if not dfs:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    return pd.concat(dfs, ignore_index=True).reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)

def complete_dataset_from_tables_llm(complete_record, client) -> pd.DataFrame:
    doi = complete_record.get("doi", "")
    tables = complete_record.get("tables", [])
    full_text_raw = complete_record.get("full_text", {}) if not args.exclude_full_text else {}
    full_text = _filter_full_text_sections(full_text_raw) if full_text_raw else {}

    if not tables:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    dfs = []
    for t in tables:
        table_name = t.get("table_name", "")
        table_records = t.get("table", [])
        metadata = t.get("metadata", {}) or {}
        caption = (metadata.get("caption") or metadata.get("table_caption") or "")

        if not table_name or not isinstance(table_records, list) or not table_records:
            continue

        df_table = pd.DataFrame(table_records)
        if df_table.empty:
            continue

        if args.skip_llm or client is None:
            # If skipping LLM, you can either return empty or do your old heuristic path.
            continue

        df_rows, meta = infer_rows_for_table(
            client=client,
            doi=doi,
            table_name=table_name,
            caption=caption,
            df_table=df_table,
            full_text=full_text,
            cache_dir=ROW_CACHE_DIR,
            max_rows_per_call=60,
            output_columns=DATASET_COLUMNS,
            missing_value_defaults=_MISSING_DEFAULTS,
            extra_rules=_EXTRA_RULES,
        )
        try:
            _log_cost(meta, extra={"doi": doi, "kind": "emit_rows_table", "table_name": table_name})
        except Exception:
            pass

        dfs.append(df_rows)

    if not dfs:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    return pd.concat(dfs, ignore_index=True).reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)


def _filter_full_text_sections(
    full_text: Optional[Dict[str, Any]],
    exclude_sections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Filter out specified sections from full text before sending to LLM.

    Args:
        full_text: Parsed JSON structure with Sections array
        exclude_sections: List of section names to exclude (case-insensitive)

    Returns:
        Filtered full_text dict with excluded sections removed
    """
    if not full_text or not isinstance(full_text, dict):
        return full_text or {}

    # Default sections to exclude
    if exclude_sections is None:
        exclude_sections = [
            "Introduction", "Conclusions", "Conclusion", "References",
            "Bibliography", "Acknowledgments", "Acknowledgements",
            "Supporting Information", "Supplementary Material"
        ]

    # Normalize to lowercase for comparison
    exclude_lower = {s.lower() for s in exclude_sections}

    # Create filtered copy
    filtered = full_text.copy()

    if "Sections" in filtered and isinstance(filtered["Sections"], list):
        original_sections = filtered["Sections"]
        filtered_sections = []

        for section in original_sections:
            if not isinstance(section, dict):
                filtered_sections.append(section)
                continue

            section_name = section.get("name", "").strip()

            # Keep section if not in exclude list
            if section_name.lower() not in exclude_lower:
                filtered_sections.append(section)
            else:
                log.debug(f"Filtering out section: {section_name}")

        filtered["Sections"] = filtered_sections

    return filtered


def _load_chemical_samples(modified_doi: str) -> List[Dict[str, Any]]:
    """
    Load extracted chemical composition samples for this DOI if available.
    Returns a list of sample dicts from samples.json, or empty list.
    """
    run_dir = os.environ.get("AEM_OCR_RUN_DIR", "").strip()
    if not run_dir:
        return []
    samples_path = os.path.join(run_dir, "chemical_composition", modified_doi, "samples.json")
    if not os.path.exists(samples_path):
        return []
    try:
        with open(samples_path, "r") as f:
            samples = json.load(f)
        if isinstance(samples, list) and samples:
            return samples
    except Exception as e:
        log.warning(f"Failed to load chemical samples from {samples_path}: {e}")
    return []


def _expand_dataset_by_samples(
    df: pd.DataFrame,
    samples: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    When chemical composition samples are available, produce a copy of
    the dataset for each unique sample.  The 'Sample Name' column (if present
    in the output schema) is filled with the sample name from the extracted
    chemical composition data.
    """
    if df.empty or not samples:
        return df

    sample_names = []
    for s in samples:
        name = (s.get("sample_name") or "").strip()
        if name and name not in sample_names:
            sample_names.append(name)

    if not sample_names:
        return df

    parts = []
    for name in sample_names:
        chunk = df.copy()
        if "Sample Name" in chunk.columns:
            chunk["Sample Name"] = name
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True)


def _filter_missing_sample_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where sample name is missing or invalid.
    Returns filtered dataframe with only valid sample names.
    """
    if df.empty or "Sample Name" not in df.columns:
        return df

    original_count = len(df)

    # Filter out rows with missing/invalid sample names
    df_filtered = df[
        (df['Sample Name'].notna()) &
        (df['Sample Name'].str.strip() != '') &
        (~df['Sample Name'].str.contains('Not Found', case=False, na=False)) &
        (~df['Sample Name'].str.contains('Unknown', case=False, na=False))
    ].copy()

    filtered_count = original_count - len(df_filtered)
    if filtered_count > 0:
        log.info(
            f"Filtered out {filtered_count} rows with missing sample names "
            f"({original_count} → {len(df_filtered)} rows)"
        )

    return df_filtered.reset_index(drop=True)


def complete_dataset_llm(complete_record, client) -> pd.DataFrame:
    # When targeting a specific figure, skip table processing entirely
    # so we only re-generate rows for that figure.
    if args.target_figure:
        df_tables = pd.DataFrame(columns=DATASET_COLUMNS)
    else:
        df_tables = complete_dataset_from_tables_llm(complete_record, client)

    if args.skip_llm_figures:
        df_figs = pd.DataFrame(columns=DATASET_COLUMNS)
    else:
        df_figs = complete_dataset_from_figures_llm(complete_record, client)

    df_all = pd.concat([df_tables, df_figs], ignore_index=True)
    df_all = df_all.reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)
    df_all = df_all.drop_duplicates()

    # If chemical composition samples were extracted, compile data per unique sample
    modified_doi = complete_record.get("modified_doi", "")
    chem_samples = _load_chemical_samples(modified_doi)
    if chem_samples:
        df_all = _expand_dataset_by_samples(df_all, chem_samples)
        df_all = df_all.drop_duplicates()

    # Filter out rows without valid sample names
    df_all = _filter_missing_sample_names(df_all)

    return df_all

# ----------------------------
# Main
# ----------------------------
def main():
    # Use processing-specific LLM if configured, otherwise fall back to general
    processing_override = load_processing_llm_config()
    if processing_override is not None:
        client = get_llm_client(override=processing_override)
    else:
        client = get_llm_client()

    df = load_file()
    client = None if args.skip_llm else client

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing DOIs"):
        doi = str(row[args.doi_column]).strip()
        full_text_path = find_matching_files(doi, args.index_csv_path)

        if not isinstance(full_text_path, str) or full_text_path.strip() == "":
            log.warning(f"No full_text_path found for DOI {doi}, got: {full_text_path}")
            continue

        modified_doi = full_text_path.split('/')[-1]
        full_text_json_path = os.path.join(full_text_path, "parsed.json")

        try:
            with open(full_text_json_path, 'r') as f:
                full_text = json.load(f)
        except Exception as e:
            log.error(f"Failed to read full text JSON for DOI {doi} at {full_text_json_path}: {e}")
            continue

        if args.figures_tables_dir and isinstance(args.figures_tables_dir, str) and args.figures_tables_dir.strip():
            ft_root = args.figures_tables_dir
            tables_dir = os.path.join(ft_root, "tables", modified_doi)
            figures_dir = os.path.join(ft_root, "figures", modified_doi)
        else:
            base_dir = full_text_json_path.split('papers')[0]
            tables_dir = os.path.join(base_dir, 'figures_tables', 'tables', modified_doi)
            figures_dir = os.path.join(base_dir, 'figures_tables', 'figures', modified_doi)

        tables = collect_tables(doi, tables_dir)
        figures = collect_figures(doi, figures_dir)

        # Get run_dir from environment for checking interventions_plextract
        run_dir = os.environ.get("AEM_OCR_RUN_DIR", "").strip()
        run_dir_path = Path(run_dir) if run_dir else None

        fig_datasets = load_extracted_figure_datasets(modified_doi, run_dir=run_dir_path)

        extracted_figure_data = {
            "type": "plextract_final_dataset_bundle",
            "datasets": [
                {
                    "dataset_name": ds.name,
                    "dataset_path": ds.path,
                    "columns": list(ds.df.columns),
                    "rows": ds.df.replace([np.inf, -np.inf], np.nan).where(pd.notna(ds.df), None).to_dict(orient="records"),
                }
                for ds in fig_datasets
            ],
        }

        complete_record = {
            "doi": doi,
            "modified_doi": modified_doi,
            "full_text": full_text,
            "tables": tables,
            "figures": figures,
            "extracted_figure_data": extracted_figure_data,
        }

        # Write combined JSON per DOI
        combined_json_path = os.path.join(args.output_dir, f"{modified_doi}_complete.json")
        try:
            with open(combined_json_path, "w") as out_f:
                json.dump(complete_record, out_f, indent=2)
            log.info(f"Wrote combined record for DOI {doi} to {combined_json_path}")
        except Exception as e:
            log.error(f"Failed to write combined record for DOI {doi} to {combined_json_path}: {e}")

        # Build dataset per DOI
        try:
            # dataset_df = complete_dataset_with_plans(complete_record, client)
            dataset_df = complete_dataset_llm(complete_record, client)

        except Exception as e:
            log.exception(f"Dataset completion failed for DOI {doi}: {e}")
            # quick type debug
            log.error(f"types: tables={type(tables)}, figures={type(figures)}, extracted_figure_data={type(extracted_figure_data)}")
            continue

        dataset_csv_path = os.path.join(args.output_dir, f"{modified_doi}_dataset.csv")
        try:
            dataset_df.to_csv(dataset_csv_path, index=False)
            log.info(f"Wrote completed dataset CSV for DOI {doi} to {dataset_csv_path} ({len(dataset_df)} rows)")
        except Exception as e:
            log.error(f"Failed to write dataset CSV for DOI {doi} to {dataset_csv_path}: {e}")

    print("====Finished completing dataset====")

if __name__ == "__main__":
    main()
