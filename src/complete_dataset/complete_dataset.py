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
from typing import Any, Dict, List, Optional
from llm_client import get_llm_client
from dataclasses import dataclass
import hashlib


sys.path.append('/home/wschertzer3/AEM_OCR/src/ui')
from ui.openai_cost import append_usage, run_id_from_env

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../paper-downloader"))
sys.path.insert(0, parent_dir)
from paper_downloader import config, utils

# --- new helper module (in same directory as this script) ---
from row_emitter import (
    OUTPUT_COLUMNS as DATASET_COLUMNS,
    ensure_cache_dir,
    infer_rows_for_table,
    infer_rows_for_figure_dataset,
    subsample_figure_points,
)

from figure_plan import infer_figure_series_plans, apply_figure_series_plans



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

def load_extracted_figure_datasets(modified_doi: str) -> List[FigurePointDataset]:
    root = (args.extracted_figures_root or "").strip()
    if not root:
        return []

    doi_dir = Path(root) / modified_doi
    if not doi_dir.is_dir():
        return []

    keep = ['series_label','x_name','x_unit','x_value','y_name','y_unit','y_value']

    out: List[FigurePointDataset] = []
    for fp in doi_dir.glob("**/final/final_dataset.csv"):
        try:
            df = pd.read_csv(fp)
            if not all(c in df.columns for c in keep):
                continue
            df = df[keep].copy()
            if df.empty:
                continue
            out.append(FigurePointDataset(
                name=_stable_name_from_path(fp),
                path=str(fp),
                df=df
            ))
        except Exception:
            continue

    return out


# ----------------------------
# Completion using PLANS
# ----------------------------
# def complete_dataset_from_figures_with_plans(
#     complete_record,
#     client,
#     ) -> pd.DataFrame:
#     doi = complete_record.get("doi", "")
#     ef = complete_record.get("extracted_figure_data", {})

#     # NEW FORMAT: ef is a dict with rows
#     if not isinstance(ef, dict) or not ef.get("rows"):
#         log.warning("No extracted_figure_data.rows found for figures. Returning empty df_figs.")
#         return pd.DataFrame(columns=DATASET_COLUMNS)

#     df_points = pd.DataFrame(ef["rows"])
#     if df_points.empty:
#         log.warning("extracted_figure_data.rows present but empty. Returning empty df_figs.")
#         return pd.DataFrame(columns=DATASET_COLUMNS)

#     # Build figure snippets (captions) for context
#     figure_snips = []
#     for f in (complete_record.get("figures", []) or [])[:8]:
#         if not isinstance(f, dict):
#             continue
#         md = f.get("metadata") or {}
#         cap = md.get("caption") or md.get("figure_caption") or ""
#         figure_snips.append({"figure_name": f.get("figure_name", ""), "caption": str(cap)[:800]})

#     # Build ONE global payload
#     plan_payload = build_figures_global_plan_payload(
#         doi=doi,
#         df_points=df_points,
#         figure_snips=figure_snips,
#         max_series=30,
#     )

#     cache_key = f"figures_global::{doi}"
#     plan = load_cached_plan(PLAN_CACHE_DIR, cache_key)

#     if plan is None:
#         if args.skip_llm or client is None:
#             # heuristic fallback based on most common x axis
#             x_name0 = str(df_points["x_name"].dropna().iloc[0]) if "x_name" in df_points and df_points["x_name"].dropna().shape[0] else ""
#             x_unit0 = str(df_points["x_unit"].dropna().iloc[0]) if "x_unit" in df_points and df_points["x_unit"].dropna().shape[0] else ""
#             hx = heuristic_axis_semantics(x_name=x_name0, x_unit=x_unit0)
#             plan = {
#                 "scope": {"doi": doi, "source_kind": "figure"},
#                 "global": {
#                     "default_x_semantic": hx.get("x_semantic", "other"),
#                     "property_ontology": {},
#                     "units": {"time_to_h": True},
#                 },
#                 "series_overrides": [],
#                 "confidence": {"x_semantic": hx.get("confidence", 0.5), "y_semantic": 0.0, "context": 0.0},
#                 "evidence": {"x_reason": hx.get("reason", "heuristic")},
#             }
#         else:
#             text, plan, meta = infer_figures_plan(client, plan_payload)
#             try:
#                 _log_cost(meta, extra={"kind": "figure_plan"})
#             except Exception as e:
#                 print(f"[error completing dataset] cannot log cost: {e}")

#         save_cached_plan(PLAN_CACHE_DIR, cache_key, plan)

#     # Apply plan deterministically to ALL points
#     df_figs = apply_global_figure_plan(df_points=df_points, plan=plan, doi=doi)

#     # Optional: drop non-time rows if requested
#     if args.drop_non_time_figure_rows:
#         # keep only rows where time(h) is non-empty
#         df_figs = df_figs[df_figs["time(h)"].astype(str).str.strip() != ""]

#     # Add provenance
#     if "source_kind" not in df_figs.columns:
#         df_figs["source_kind"] = "figure"
#     if "source_name" not in df_figs.columns:
#         df_figs["source_name"] = ""

#     df_figs = df_figs.reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)
#     return df_figs

# def complete_dataset_from_figures_with_plans(complete_record, client) -> pd.DataFrame:
#     doi = complete_record.get("doi", "")
#     full_text = complete_record.get("full_text", {}) if not args.exclude_full_text else {}

#     ef = complete_record.get("extracted_figure_data", {})
#     datasets = (ef or {}).get("datasets", [])

#     if not isinstance(datasets, list) or not datasets:
#         log.warning("No extracted_figure_data.datasets found for figures. Returning empty df_figs.")
#         return pd.DataFrame(columns=DATASET_COLUMNS)

#     # Build figure snippets (captions) for context (shared across datasets)
#     figure_snips = []
#     for f in (complete_record.get("figures", []) or [])[:8]:
#         if not isinstance(f, dict):
#             continue
#         md = f.get("metadata") or {}
#         cap = md.get("caption") or md.get("figure_caption") or ""
#         figure_snips.append({"figure_name": f.get("figure_name", ""), "caption": str(cap)[:800]})

#     all_out = []

#     for ds in datasets:
#         ds_name = str(ds.get("dataset_name", "")).strip()
#         rows = ds.get("rows", [])
#         if not ds_name or not isinstance(rows, list) or not rows:
#             continue

#         df_points = pd.DataFrame(rows)
#         if df_points.empty:
#             continue

#         # Build payload for THIS dataset only
#         plan_payload = build_figures_plan_payload(
#             doi=doi,
#             df_points=df_points,
#             figure_snips=figure_snips,
#             max_series=30,
#             full_text=full_text,
#         )

#         cache_key = f"figures_global::{doi}::{ds_name}"
#         plan = load_cached_plan(PLAN_CACHE_DIR, cache_key)

#         if plan is None:
#             if args.skip_llm or client is None:
#                 x_name0 = str(df_points["x_name"].dropna().iloc[0]) if "x_name" in df_points and df_points["x_name"].dropna().shape[0] else ""
#                 x_unit0 = str(df_points["x_unit"].dropna().iloc[0]) if "x_unit" in df_points and df_points["x_unit"].dropna().shape[0] else ""
#                 hx = heuristic_axis_semantics(x_name=x_name0, x_unit=x_unit0)
#                 plan = {
#                     "scope": {"doi": doi, "source_kind": "figure", "dataset_name": ds_name},
#                     "global": {
#                         "default_x_semantic": hx.get("x_semantic", "other"),
#                         "property_ontology": {},
#                         "units": {"time_to_h": True},
#                     },
#                     "series_overrides": [],
#                     "confidence": {"x_semantic": hx.get("confidence", 0.5), "y_semantic": 0.0, "context": 0.0},
#                     "evidence": {"x_reason": hx.get("reason", "heuristic")},
#                 }
#             else:
#                 text, plan, meta = infer_figures_plan(client, plan_payload)
#                 try:
#                     _log_cost(meta, extra={"kind": "figure_plan", "dataset_name": ds_name})
#                 except Exception as e:
#                     print(f"[error completing dataset] cannot log cost: {e}")

#             save_cached_plan(PLAN_CACHE_DIR, cache_key, plan)

#         # Apply this plan to this dataset
#         df_figs = apply_figure_plan(df_points=df_points, plan=plan, doi=doi)

#         if args.drop_non_time_figure_rows:
#             df_figs = df_figs[df_figs["time(h)"].astype(str).str.strip() != ""]

#         # Provenance: keep which extracted dataset produced these rows
#         df_figs["source_kind"] = "figure"
#         df_figs["source_name"] = ds_name

#         df_figs = df_figs.reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)
#         all_out.append(df_figs)

#     if not all_out:
#         return pd.DataFrame(columns=DATASET_COLUMNS)

#     return pd.concat(all_out, ignore_index=True)


def complete_dataset_from_figures_llm(complete_record, client) -> pd.DataFrame:
    
    # print('=====COMPLETE_RECORD=====', complete_record)

    doi = complete_record.get("doi", "")
    full_text = complete_record.get("full_text", {}) if not args.exclude_full_text else {}

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
        figure_snips.append({"figure_name": f.get("figure_name", ""), "caption": str(cap)[:800]})

    dfs = []
    for ds in datasets:
        ds_name = str(ds.get("dataset_name", "")).strip()
        rows = ds.get("rows", [])
        if not ds_name or not isinstance(rows, list) or not rows:
            continue

        df_points = pd.DataFrame(rows)
        if df_points.empty:
            continue

        # df_points = subsample_figure_points(
        #     df_points,
        #     sample_col="series_label",
        #     step=100,
        # )

        # print('=====DF_POINTS====', df_points)

        if args.skip_llm or client is None:
            continue

        # df_rows, meta = infer_rows_for_figure_dataset(
        #     client=client,
        #     doi=doi,
        #     dataset_name=ds_name,
        #     figure_snips=figure_snips,
        #     df_points=df_points,
        #     full_text=full_text,
        #     cache_dir=ROW_CACHE_DIR,
        #     max_points_per_call=200,
        # )

        plans, meta = infer_figure_series_plans(
        client=client,
        df_points=df_points,
        output_columns=DATASET_COLUMNS,
        doi=doi,
        dataset_name=ds_name,
        figure_snips=figure_snips,
        full_text=full_text,
        cache_dir=ROW_CACHE_DIRi)

        df_rows = apply_figure_series_plans(
            df_points=df_points,
            output_columns=DATASET_COLUMNS,
            doi=doi,
            dataset_name=ds_name,
            plans=plans,
            source_kind="figure")        

        try:
            _log_cost(meta, extra={"kind": "emit_rows_figure", "dataset_name": ds_name})
        except Exception:
            pass

        if args.drop_non_time_figure_rows:
            df_rows = df_rows[df_rows["time(h)"].astype(str).str.strip() != ""]

        dfs.append(df_rows)

    if not dfs:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    return pd.concat(dfs, ignore_index=True).reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)

# def complete_dataset_from_tables_with_plans(
#     complete_record,
#     client,
#     ) -> pd.DataFrame:
#     doi = complete_record.get("doi", "")
#     tables = complete_record.get("tables", [])
#     figures = complete_record.get("figures", [])
#     full_text = complete_record.get("full_text", {}) if not args.exclude_full_text else {}

#     if not tables:
#         return pd.DataFrame(columns=DATASET_COLUMNS)

#     all_rows_out: List[Dict[str, Any]] = []

#     for t in tables:
#         table_name = t.get("table_name", "")
#         table_records = t.get("table", [])
#         table_columns = t.get("table_columns", [])
#         metadata = t.get("metadata", {})

#         if not table_name or not isinstance(table_records, list) or not table_records:
#             continue

#         df_table = pd.DataFrame(table_records)

#         plan_payload = build_table_plan_payload(
#             doi=doi,
#             table_name=table_name,
#             table_columns=table_columns,
#             table_head=df_table.head(8).to_dict(orient="records"),
#             metadata=metadata,
#             figures=figures,
#             full_text=full_text,
#         )

#         cache_key = f"table::{doi}::{table_name}"
#         plan = load_cached_plan(PLAN_CACHE_DIR, cache_key)

#         if plan is None:
#             if args.skip_llm or client is None:
#                 plan = {
#                     "scope": {"doi": doi, "table_name": table_name, "source_kind": "table"},
#                     "column_map": {},  # means no rows extracted
#                     "constants": {"DOI": doi},
#                     "confidence": 0.0,
#                     "evidence": {"reason": "skip_llm or no client"},
#                 }
#             else:
#                 text, plan, meta = infer_tables_plan(client, plan_payload)
#                 try:
#                     _log_cost(meta, extra={"kind": "table_plan"})
#                 except Exception as e:
#                     print(f"[error completing dataset] cannot log cost: {e}")
#             save_cached_plan(PLAN_CACHE_DIR, cache_key, plan)

#         rows_out = apply_table_plan_to_df(df_table=df_table, plan=plan, doi=doi)
#         all_rows_out.extend(rows_out)

#     df = pd.DataFrame(all_rows_out).reindex(columns=DATASET_COLUMNS, fill_value="")
#     return df.fillna("").astype(str)

def complete_dataset_from_tables_llm(complete_record, client) -> pd.DataFrame:
    doi = complete_record.get("doi", "")
    tables = complete_record.get("tables", [])
    full_text = complete_record.get("full_text", {}) if not args.exclude_full_text else {}

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
        )
        try:
            _log_cost(meta, extra={"kind": "emit_rows_table", "table_name": table_name})
        except Exception:
            pass

        dfs.append(df_rows)

    if not dfs:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    return pd.concat(dfs, ignore_index=True).reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)

# def complete_dataset_with_plans(
#     complete_record,
#     client,
#     ) -> pd.DataFrame:
#     df_tables = complete_dataset_from_tables_with_plans(complete_record, client)
#     df_figs = complete_dataset_from_figures_with_plans(complete_record, client)

#     df_all = pd.concat([df_tables, df_figs], ignore_index=True)
#     df_all = df_all.reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)
#     df_all = df_all.drop_duplicates()
#     return df_all

def complete_dataset_llm(complete_record, client) -> pd.DataFrame:
    df_tables = complete_dataset_from_tables_llm(complete_record, client)
    df_figs = complete_dataset_from_figures_llm(complete_record, client)

    print("====DF_TABLES====", df_tables)
    print("====DF_FIGS====", df_figs)

    df_all = pd.concat([df_tables, df_figs], ignore_index=True)
    df_all = df_all.reindex(columns=DATASET_COLUMNS, fill_value="").fillna("").astype(str)
    df_all = df_all.drop_duplicates()
    return df_all

# ----------------------------
# Main
# ----------------------------
def main():
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
        fig_datasets = load_extracted_figure_datasets(modified_doi)

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
