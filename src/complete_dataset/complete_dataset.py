#!/usr/bin/env python3
"""
Complete dataset builder with LLM-generated TRANSFORMATION PLANS (not row-emission).

Key changes vs your current script:
- GPT is used to infer a compact, generalizable transform PLAN per series_label (figures) and per table.
- Python applies that plan deterministically to full-resolution data (no per-point GPT rewriting).
- time(h) is populated ONLY if x_semantic == "time"; otherwise time(h) is "0".
- Non-time x axes are preserved in extra columns: x_name, x_unit, x_value.


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
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.append('/home/wschertzer3/AEM_OCR/src/ui')
from ui.openai_cost import append_usage, stage_from_env, run_id_from_env

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../paper-downloader"))
sys.path.insert(0, parent_dir)
from paper_downloader import config, utils

# --- new helper module (in same directory as this script) ---
from transform_plans import (
    build_series_plan_payload,
    build_table_plan_payload,
    infer_series_transform_plan,
    infer_table_transform_plan,
    apply_series_plan_to_points,
    apply_table_plan_to_df,
    ensure_plan_cache_dir,
    load_cached_plan,
    save_cached_plan,
    heuristic_axis_semantics,
)

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='/home/wschertzer3/AEM_OCR/src/doi-inputs/validate_paper_download.csv',
                    help='Path to CSV/Excel file containing DOIs')
parser.add_argument('--output_dir', default='/home/wschertzer3/AEM_OCR/src/complete-dataset/logs/records',
                    help='Directory to write combined JSONs and completed CSVs')
parser.add_argument('--index_csv_path', default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/papers/reparsed/index.csv',
                    help="CSV where downloaded files metadata is stored")
parser.add_argument('--doi_column', default='DOI',
                    help='Name of DOI column in file')
parser.add_argument('--log_file', dest='log_file', default='/home/wschertzer3/AEM_OCR/src/complete-dataset/logs/records',
                    help='Path to log file directory')

parser.add_argument('--extracted_figures_root', default='',
                    help='Optional root dir containing extracted figure-data per DOI folder (csv/json). '
                         'Expected to contain subdir named <modified_doi>.')

parser.add_argument('--figures_tables_dir', default='',
                    help="Optional override: path to run-scoped figures_tables dir (contains tables/ and figures/). "
                         "If provided, tables_dir = <figures_tables_dir>/tables/<modified_doi> and figures_dir = <figures_tables_dir>/figures/<modified_doi>.")

parser.add_argument('--exclude_full_text', default=False, action='store_true',
                    help="Exclude parsed full text from LLM prompt (reduces token usage drastically)")

parser.add_argument('--openai_api_key', default=os.getenv("OPENAI_API_KEY", ""),
                    help="OpenAI API key (or set OPENAI_API_KEY env var)")
parser.add_argument('--openai_base_url', default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    help="OpenAI base URL (default: https://api.openai.com/v1)")
parser.add_argument('--openai_model', default='gpt-4o', help="Model name (default: gpt-4o)")

# allow skipping LLM completion (still writes combined json)
parser.add_argument('--skip_llm', action='store_true', help="Skip GPT plan step (will use heuristics only)")

# plan caching
parser.add_argument('--plan_cache_dir', default='',
                    help="Optional override for plan cache directory. Default: <output_dir>/plans")

# If you ONLY care about time-series degradation curves, you can drop non-time figure rows:
parser.add_argument('--drop_non_time_figure_rows', action='store_true',
                    help="If set, figure datapoints whose x axis is not time will be excluded.")

args = parser.parse_args()

# Setup logging
log_file = f"{args.log_file}/complete_datasets_{datetime.now().strftime(config.DATE_TIME_FORMAT)}.log"
Path(args.log_file).mkdir(parents=True, exist_ok=True)
log = utils.setup_logger(log_file, logger_name='complete_datasets')

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

PLAN_CACHE_DIR = args.plan_cache_dir.strip() or os.path.join(args.output_dir, "plans")
ensure_plan_cache_dir(PLAN_CACHE_DIR)

# ----------------------------
# Output schema
# ----------------------------
DATASET_COLUMNS = [
    "Sample Name",
    "time(h)",
    "relative humidity(%)",
    "solvent type",
    "solvent concentration (%)",
    "degradation test temp (C)",
    "prop test temp (C)",
    "property",
    "value",
    "DOI",
    "source_kind",   # "figure" or "table"
    "source_name",   # figure_name or table_name
]


# ----------------------------
# Utilities: cost logging
# ----------------------------
def _log_cost(response, extra: dict):
    append_usage(
        stage=stage_from_env("complete_dataset_plans"),
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
def subsample_series(g: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    g = g.copy()
    x_num = pd.to_numeric(g["x_value"], errors="coerce")
    if x_num.notna().any():
        g = g.assign(_x_num=x_num).sort_values("_x_num", kind="mergesort").drop(columns="_x_num")
    n = len(g)
    if n <= 2:
        return g
    keep_pos = set(range(0, n, step))
    keep_pos.add(0)
    keep_pos.add(n - 1)
    return g.iloc[sorted(keep_pos)].copy()

# def collect_extracted_figure_data(modified_doi: str) -> List[Dict[str, Any]]:
#     root = (args.extracted_figures_root or "").strip()
#     if root == "":
#         return []

#     doi_dir = os.path.join(root, modified_doi)
#     if not os.path.isdir(doi_dir):
#         log.warning(f"--extracted_figures_root provided, but no dir for DOI: {doi_dir}")
#         return []

#     extracted: List[Dict[str, Any]] = []
#     step = 10

#     for fp in Path(doi_dir).glob("**/final/final_dataset.csv"):
#         try:
#             df = pd.read_csv(fp)
#             if df.empty:
#                 continue

#             keep_cols = ['series_label', 'x_name', 'x_unit', 'x_value', 'y_name', 'y_unit', 'y_value']
#             missing = [c for c in keep_cols if c not in df.columns]
#             if missing:
#                 log.warning(f"Skipping {fp} because missing columns: {missing}")
#                 continue

#             series_payloads = []
#             for series_label, g in df.groupby("series_label", dropna=False):
#                 g_sub = subsample_series(g, step=step)
#                 for col in ["x_value", "y_value"]:
#                     g_sub[col] = pd.to_numeric(g_sub[col], errors="coerce").round(6)
#                     g_sub[col] = g_sub[col].where(g_sub[col].notna(), "")

#                 series_payloads.append({
#                     "series_label": str(series_label),
#                     "columns": list(g_sub.columns),
#                     "rows": g_sub.to_dict(orient="records"),
#                     # store path so we can reload full-res later
#                     "source_csv_path": str(fp),
#                 })

#             extracted.append({
#                 "modified_doi": modified_doi,
#                 "path": str(fp),
#                 "type": "csv",
#                 "content": {
#                     "columns": keep_cols,
#                     "series_payloads": series_payloads,
#                 },
#             })

#         except Exception as e:
#             log.warning(f"Failed to read plextract final_dataset.csv {fp}: {e}")

#     return extracted

def load_all_extracted_figure_points(modified_doi: str) -> pd.DataFrame:
    root = (args.extracted_figures_root or "").strip()
    if not root:
        return pd.DataFrame()

    doi_dir = os.path.join(root, modified_doi)
    if not os.path.isdir(doi_dir):
        return pd.DataFrame()

    dfs = []
    for fp in Path(doi_dir).glob("**/final/final_dataset.csv"):
        try:
            df = pd.read_csv(fp)
            keep = ['series_label','x_name','x_unit','x_value','y_name','y_unit','y_value']
            if all(c in df.columns for c in keep):
                dfs.append(df[keep].copy())
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ----------------------------
# OpenAI client
# ----------------------------
def make_openai_client() -> OpenAI:
    if not args.openai_api_key and not args.skip_llm:
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY or pass --openai_api_key.")
    return OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

# ----------------------------
# Completion using PLANS
# ----------------------------
def complete_dataset_from_figures_with_plans(
    complete_record: Dict[str, Any],
    client: Optional[OpenAI],
    model: str,
) -> pd.DataFrame:
    doi = complete_record.get("doi", "")
    extracted = complete_record.get("extracted_figure_data", [])

    # gather series payloads
    series_payloads = []
    for item in extracted:
        content = item.get("content", {})
        sps = content.get("series_payloads", [])
        if isinstance(sps, list) and sps:
            series_payloads.extend(sps)

    if not series_payloads:
        log.warning("No extracted_figure_data.series_payloads found for figures. Returning empty df_figs.")
        return pd.DataFrame(columns=DATASET_COLUMNS)

    # For full-resolution apply: reload each source CSV and filter to that series_label
    all_rows_out: List[Dict[str, Any]] = []

    figures = complete_record.get("figures", [])
    tables = complete_record.get("tables", [])
    full_text = complete_record.get("full_text", {}) if not args.exclude_full_text else {}

    for sp in series_payloads:
        series_label = sp.get("series_label", "")
        rows_sub = sp.get("rows", [])
        src_csv = sp.get("source_csv_path", "")

        if not series_label or not src_csv:
            continue

        # Load full-resolution for this series_label
        try:
            df_full = pd.read_csv(src_csv)
            df_full = df_full[df_full["series_label"].astype(str) == str(series_label)]
            df_full = df_full[['series_label', 'x_name', 'x_unit', 'x_value', 'y_name', 'y_unit', 'y_value']].copy()
        except Exception as e:
            log.warning(f"Could not reload full-res points for series={series_label} from {src_csv}: {e}")
            # fallback: just use subsampled rows
            df_full = pd.DataFrame(rows_sub)

        # Heuristic pre-classification
        hx = heuristic_axis_semantics(
            x_name=str(df_full["x_name"].iloc[0]) if "x_name" in df_full and len(df_full) else "",
            x_unit=str(df_full["x_unit"].iloc[0]) if "x_unit" in df_full and len(df_full) else "",
        )

        # Build plan payload
        plan_payload = build_series_plan_payload(
            doi=doi,
            series_label=str(series_label),
            points_subsample=rows_sub,
            figures=figures,
            tables=tables,
            full_text=full_text,
            heuristic=hx,
        )

        cache_key = f"series::{doi}::{series_label}::{hx.get('x_semantic','')}"
        plan = load_cached_plan(PLAN_CACHE_DIR, cache_key)

        # Infer (or heuristic-only)
        if plan is None:
            if args.skip_llm or client is None:
                # heuristic-only "plan"
                plan = {
                    "scope": {"doi": doi, "series_label": series_label, "source_kind": "figure"},
                    "semantics": {
                        "x_semantic": hx.get("x_semantic", "other"),
                        "y_semantic": "",
                    },
                    "constants": {"DOI": doi},
                    "confidence": {"x_semantic": hx.get("confidence", 0.5)},
                    "evidence": {"x_reason": hx.get("reason", "heuristic")},
                }
            else:
                resp, plan = infer_series_transform_plan(client, model, plan_payload)
                _log_cost(resp, extra={"kind": "series_plan", "series_label": series_label, "model": model})
            save_cached_plan(PLAN_CACHE_DIR, cache_key, plan)

        # Apply plan
        points = df_full.to_dict(orient="records")
        rows_out = apply_series_plan_to_points(
            points=points,
            plan=plan,
            doi=doi,
            drop_non_time=args.drop_non_time_figure_rows,
        )
        all_rows_out.extend(rows_out)

    df = pd.DataFrame(all_rows_out).reindex(columns=DATASET_COLUMNS, fill_value="")
    return df.fillna("").astype(str)

def complete_dataset_from_tables_with_plans(
    complete_record: Dict[str, Any],
    client: Optional[OpenAI],
    model: str,
) -> pd.DataFrame:
    doi = complete_record.get("doi", "")
    tables = complete_record.get("tables", [])
    figures = complete_record.get("figures", [])
    full_text = complete_record.get("full_text", {}) if not args.exclude_full_text else {}

    if not tables:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    all_rows_out: List[Dict[str, Any]] = []

    for t in tables:
        table_name = t.get("table_name", "")
        table_records = t.get("table", [])
        table_columns = t.get("table_columns", [])
        metadata = t.get("metadata", {})

        if not table_name or not isinstance(table_records, list) or not table_records:
            continue

        df_table = pd.DataFrame(table_records)

        plan_payload = build_table_plan_payload(
            doi=doi,
            table_name=table_name,
            table_columns=table_columns,
            table_head=df_table.head(8).to_dict(orient="records"),
            metadata=metadata,
            figures=figures,
            full_text=full_text,
        )

        cache_key = f"table::{doi}::{table_name}"
        plan = load_cached_plan(PLAN_CACHE_DIR, cache_key)

        if plan is None:
            if args.skip_llm or client is None:
                plan = {
                    "scope": {"doi": doi, "table_name": table_name, "source_kind": "table"},
                    "column_map": {},  # means no rows extracted
                    "constants": {"DOI": doi},
                    "confidence": 0.0,
                    "evidence": {"reason": "skip_llm or no client"},
                }
            else:
                resp, plan = infer_table_transform_plan(client, model, plan_payload)
                _log_cost(resp, extra={"kind": "table_plan", "table_name": table_name, "model": model})
            save_cached_plan(PLAN_CACHE_DIR, cache_key, plan)

        rows_out = apply_table_plan_to_df(df_table=df_table, plan=plan, doi=doi)
        all_rows_out.extend(rows_out)

    df = pd.DataFrame(all_rows_out).reindex(columns=DATASET_COLUMNS, fill_value="")
    return df.fillna("").astype(str)

def complete_dataset_with_plans(
    complete_record: Dict[str, Any],
    client: Optional[OpenAI],
    model: str,
) -> pd.DataFrame:
    df_tables = complete_dataset_from_tables_with_plans(complete_record, client, model)
    df_figs = complete_dataset_from_figures_with_plans(complete_record, client, model)

    df_all = pd.concat([df_tables, df_figs], ignore_index=True)
    df_all = df_all.reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)
    df_all = df_all.drop_duplicates()
    return df_all

# ----------------------------
# Main
# ----------------------------
def main():
    df = load_file()
    client = None if args.skip_llm else make_openai_client()

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
        extracted_figure_data = load_all_extracted_figure_points(modified_doi)

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
            dataset_df = complete_dataset_with_plans(complete_record, client, model=args.openai_model)
        except Exception as e:
            log.error(f"Plan-based completion failed for DOI {doi}: {e}")
            continue

        dataset_csv_path = os.path.join(args.output_dir, f"{modified_doi}_dataset.csv")
        try:
            dataset_df.to_csv(dataset_csv_path, index=False)
            log.info(f"Wrote completed dataset CSV for DOI {doi} to {dataset_csv_path} ({len(dataset_df)} rows)")
        except Exception as e:
            log.error(f"Failed to write dataset CSV for DOI {doi} to {dataset_csv_path}: {e}")

    print("====Finished completing dataset (plan-based)====")

if __name__ == "__main__":
    main()
