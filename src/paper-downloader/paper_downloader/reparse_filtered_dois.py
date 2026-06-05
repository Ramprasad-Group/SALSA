#!/usr/bin/env python3
"""
Re-parse structured article files for specific DOIs (NO-DB MODE).

- Reads DOIs from a CSV (optionally filters rows, e.g., note == 1)
- Locates each structured file under --structured_root by scanning
  subdirectories (publisher directories) for files named DOI_with_atsign.*
- Chooses a LimeSoup parser based on the publisher directory
- Parses and post-processes (keywords, abstract normalization)
- Writes results to --output_dir/{doi_sanitized}/parsed.json (and metadata.json)
- Appends a row to --output_dir/index.csv for quick inspection

Requirements already in your repo:
- paper_downloader.LimeSoup (publisher parsers)
- paper_downloader.utils.parsed_output_processor (optional but used here)
- paper_downloader.utils.setup_logger (for consistent logs)
"""

import os
import sys
import argparse
import json
import glob
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd

# Local project imports (no DB modules!)
_PAPER_DL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _SRC_ROOT)
sys.path.insert(0, _PAPER_DL)
import paper_downloader.LimeSoup as LimeSoup
from paper_downloader import utils, config
from ui.run_progress import write_current_doi


# ---------------------------
# Helpers
# ---------------------------
def doi_to_filename(doi: str) -> str:
    """Convert DOI to filename used in your structured_files store."""
    return doi.replace("/", "@")

def find_structured_file(structured_root: str, doi: str) -> Optional[Tuple[str, str, str]]:
    """
    Search for the structured file under structured_root.

    Returns:
      (file_path, directory, doctype) or None if not found

    It assumes a layout like:
      structured_root/{directory}/{doi_with_@}.{doctype}
    and falls back to a recursive glob scan if direct paths fail.
    """
    base = doi_to_filename(doi)
    # Try a quick recursive glob: */<base>.*
    pattern = os.path.join(structured_root, "*", base + ".*")
    matches = glob.glob(pattern)
    if not matches:
        # As a fallback, fully recursive scan (slower, but robust)
        pattern = os.path.join(structured_root, "**", base + ".*")
        matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None

    # Prefer shortest path (closest to root) if multiple matched
    matches.sort(key=lambda p: len(Path(p).parts))
    file_path = matches[0]
    directory = Path(file_path).parent.name
    doctype = Path(file_path).suffix.lstrip(".").lower()
    return file_path, directory, doctype

def pick_parser(directory: str):
    """Map publisher directory name → LimeSoup parser class."""
    parser_map = {
        'rsc': LimeSoup.RSCSoup,
        'springer': LimeSoup.SpringerSoup,
        'elsevier': LimeSoup.ElsevierSoup,
        'nature': LimeSoup.NatureSoup,
        'acs': LimeSoup.ACSSoup,
        'wiley': LimeSoup.WileySoup,
        'aip': LimeSoup.AIPSoup,
        # historical mapping in your script:
        'hindawi': LimeSoup.SpringerSoup,
        'informa_uk': LimeSoup.InformaUKSoup,
    }
    return parser_map.get(directory.lower())

def normalize_abstract(abstract):
    """Your original logic: list → first element; otherwise str()."""
    if isinstance(abstract, list) and abstract:
        return abstract[0]
    return "" if abstract is None else str(abstract)

def write_json(path_: Path, obj: Dict):
    path_.parent.mkdir(parents=True, exist_ok=True)
    with open(path_, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Re-parse structured files locally (no DB).")
    _logs = os.path.join(_PAPER_DL, "paper_downloader", "logs")
    ap.add_argument("--csv_path", default=os.path.join(_logs, "records", "download_results.csv"),
                    help="CSV containing DOIs (and optional filter column).")
    ap.add_argument("--structured_root", default=os.path.join(_logs, "papers", "structured_files"),
                    help="Root directory where structured files are stored (publisher subfolders).")
    ap.add_argument("--output_dir", default=os.path.join(_logs, "papers", "reparsed"),
                    help="Directory to write per-DOI JSON and index.csv.")
    ap.add_argument("--filter_column", default="note",
                    help="CSV column used to filter rows (default: note).")
    ap.add_argument("--filter_value", default="1",
                    help="Value in filter_column to keep (default: '1').")
    ap.add_argument("--doi_column", default="doi", help="Name of the DOI column (default: doi).")
    args = ap.parse_args()

    # Logger
    logger_file = f'{args.output_dir}/log_reparse_no_db_{datetime.now().strftime(config.DATE_TIME_FORMAT)}'
    logger = utils.setup_logger(logger_file, "reparse_no_db")

    # Read CSV and filter
    df = pd.read_csv(args.csv_path)
    if args.filter_column in df.columns:
        filtered_df = df[df[args.filter_column].astype(str) == str(args.filter_value)]
    else:
        logger.warning(f"Filter column '{args.filter_column}' not found; processing ALL rows.")
        filtered_df = df

    if args.doi_column not in filtered_df.columns:
        raise ValueError(f"CSV must contain a '{args.doi_column}' column.")

    dois = filtered_df[args.doi_column].dropna().astype(str).tolist()
    logger.warning(f"Total rows in CSV: {len(df)}")
    logger.warning(f"Total rows after filter: {len(filtered_df)}")
    logger.warning(f"Total DOIs to process: {len(dois)}")

    # Prepare index CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for doi in tqdm(dois, desc="Re-parsing DOIs"):
        doi = doi.strip()
        write_current_doi(doi)
        logger.warning(f"Processing DOI: {doi}")

        try:
            found = find_structured_file(args.structured_root, doi)
            if not found:
                logger.warning(f"Structured file not found under {args.structured_root} for {doi}")
                index_rows.append({
                    "doi": doi,
                    "status": "missing_file",
                    "directory": "",
                    "doctype": "",
                    "title": "",
                    "keywords": "",
                    "output_dir": ""
                })
                continue

            file_path, directory, doctype = found
            parser_cls = pick_parser(directory)
            if not parser_cls:
                logger.warning(f"No parser available for directory '{directory}' (DOI: {doi}); skipping.")
                index_rows.append({
                    "doi": doi,
                    "status": "no_parser",
                    "directory": directory,
                    "doctype": doctype,
                    "title": "",
                    "keywords": "",
                    "output_dir": ""
                })
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                structured_str = f.read()

            # Parse with LimeSoup
            parsed_output = parser_cls.parse(structured_str)

            # Optional: use your existing post-processor to standardize & backfill keywords
            parsed_output_final, heuristic_keywords = utils.parsed_output_processor(parsed_output)

            # Ensure Keywords present
            if not parsed_output_final.get("Keywords"):
                parsed_output_final["Keywords"] = heuristic_keywords

            # Normalize abstract
            parsed_output_final["Abstract"] = normalize_abstract(parsed_output.get("Abstract"))

            # Title (may be empty if parser couldn't find it)
            title = (parsed_output_final.get("Title") or "").strip()

            # Write per-DOI artifacts
            doi_dir = out_dir / doi_to_filename(doi)
            metadata = {
                "doi": doi,
                "directory": directory,
                "doctype": doctype,
                "source_path": file_path,
                "parsed_at": datetime.now().isoformat(),
                "title": title,
                "keywords": parsed_output_final.get("Keywords", [])
            }
            write_json(doi_dir / "metadata.json", metadata)
            write_json(doi_dir / "parsed.json", parsed_output_final)

            # Row for index.csv
            index_rows.append({
                "doi": doi,
                "status": "ok",
                "directory": directory,
                "doctype": doctype,
                "title": title,
                "keywords": "; ".join(map(str, parsed_output_final.get("Keywords", []))),
                "output_dir": str(doi_dir)
            })

            logger.warning(f"Successfully reparsed {doi}")

        except Exception as e:
            logger.exception(f"Exception while processing {doi}: {e}")
            index_rows.append({
                "doi": doi,
                "status": "error",
                "directory": "",
                "doctype": "",
                "title": "",
                "keywords": "",
                "output_dir": ""
            })

    # Write index CSV
    idx_df = pd.DataFrame(index_rows, columns=[
        "doi", "status", "directory", "doctype", "title", "keywords", "output_dir"
    ])
    idx_df.to_csv(out_dir / "index.csv", index=False)
    logger.warning(f"Done. Wrote index to {out_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
