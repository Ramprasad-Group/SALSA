#!/usr/bin/env python3
"""
Download and insert multiple paper records from a CSV file
Handles CSV files with publisher, title, and abstract columns
Uses downloaded record data but falls back to CSV data when needed
"""

import argparse
import pandas as pd
from datetime import datetime
import sys
import logging
import json
from pathlib import Path
from tqdm import tqdm
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, parent_dir)
from paper_downloader import records_downloader, config, utils
from backend import sett, postgres
from backend.postgres.orm import PaperInfo, Papers

def make_result(doi, publisher, title, year, success, message):
    return {
        'doi': doi,
        'publisher': publisher,
        'title': title,
        'pub_year': year,
        'success': success,
        'message': message,
    }

def detect_publisher_from_doi(doi):
    """Detect publisher from DOI prefix - simple mapping"""
    if pd.isna(doi):
        return 'Unknown'
    
    doi = str(doi).strip()
    
    # Simple DOI prefix mapping
    if doi.startswith('10.1016'):
        return 'Elsevier'
    elif doi.startswith('10.1002'):
        return 'Wiley'
    elif doi.startswith('10.1007'):
        return 'Springer'
    elif doi.startswith('10.1038'):
        return 'Nature'
    elif doi.startswith('10.1021'):
        return 'ACS'
    elif doi.startswith('10.1039'):
        return 'RSC'
    elif doi.startswith('10.1088'):
        return 'IOP'
    elif doi.startswith('10.1063'):
        return 'AIP Publishing'
    elif doi.startswith('10.1149'):
        return 'ECS'
    elif doi.startswith('10.1155'):
        return 'Hindawi'
    elif doi.startswith('10.1080'):
        return 'Informa UK'
    elif doi.startswith('10.1108'):
        return 'Emerald'
    elif doi.startswith('10.1177'):
        return 'Sage'
    else:
        return 'Unknown'

def extract_title_from_record(record, csv_title=None):
    """
    Extracts title from the downloaded Crossref/record entry.
    Falls back to the CSV title if not present.
    """
    if not record:
        return csv_title or None

    # Try multiple common keys for title field
    for key in ["title", "Title", "article_title", "Article Title"]:
        if key in record and record[key]:
            return record[key]

    # Fallback if no key found
    return csv_title or None

# def _first_int(x):
#     try:
#         return int(str(x)[:4])
#     except Exception:
#         return None

# def _year_from_dateparts(obj):
#     # Crossref often uses {"date-parts": [[YYYY, M, D]]}
#     try:
#         dp = obj.get('date-parts')
#         if isinstance(dp, list) and dp and isinstance(dp[0], (list, tuple)) and dp[0]:
#             return _first_int(dp[0][0])
#     except Exception:
#         pass
#     return None

def extract_year_from_record(record, csv_year=None):
    """
    Try common Crossref-ish fields first, then fall back to CSV year.
    Returns an int year or None.
    """
    if not record:
        return 'No Year Found'

    year = record.get('year')

    # Fall back to CSV column
    return year

parser = argparse.ArgumentParser()
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PAPER_DL_LOGS = os.path.join(os.path.dirname(__file__), "logs")
parser.add_argument('--file_path', default=os.path.join(_REPO_ROOT, 'src', 'doi-inputs', 'validate_paper_download.csv'),help='Path to CSV/Excel file containing DOIs')
parser.add_argument('--doi_column', default='DOI', help='Name of DOI column in file (default: DI for Web of Science)')
parser.add_argument('--publisher_column', default='Publisher', help='Name of publisher column in file (default: PU for Web of Science)')
parser.add_argument('--title_column', default='Article Title', help='Name of title column in file (default: TI for Web of Science)')
parser.add_argument('--year_column', default='Publication Year', help='Name of year column in file (e.g., "Publication Year", "PY")')
parser.add_argument('--abstract_column', default='Abstract', help='Name of abstract column in file (default: AB for Web of Science)')
parser.add_argument('--log_file', dest='log_file', help='Path to log file directory', default=os.path.join(_PAPER_DL_LOGS, 'records'))
parser.add_argument('--journals_json', dest='journals_json', help='Path to combined_journals_issn.json', default=os.path.join(_REPO_ROOT, 'src', 'paper-downloader', 'issn_dir', 'combined_journals_issn.json'))
parser.add_argument('--sample_size', type=int, default=None, help='Number of DOIs to process (for testing)')
parser.add_argument('--start_index', type=int, default=0, help='Starting index for processing (default: 0)')
parser.add_argument('--no_db', action='store_true', default=True, help='Skip DB and save records locally instead')
parser.add_argument('--output_dir', default=os.path.join(_PAPER_DL_LOGS, 'records'), help='Directory to write JSON/CSV when --no_db')
parser.add_argument('--mailto', default='wschertzer@gatech.edu', help='Email passed to Crossref/record APIs')
args = parser.parse_args()

# Setup logging
log_file = f"{args.log_file}/log_multiple_dois_{datetime.now().strftime(config.DATE_TIME_FORMAT)}.log"
Path(args.log_file).mkdir(parents=True, exist_ok=True)
log = utils.setup_logger(log_file, logger_name='multiple-doi-downloader')

if not args.no_db:
    sett.load_settings()
    postgres.load_settings()
    db = postgres.connect('polylit')

metadata_columns = config.METADATA_COLUMNS
mailid = args.mailto
if args.no_db:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# Load data from CSV or Excel file
file_extension = args.file_path.lower().split('.')[-1]
if file_extension == 'xls' or file_extension == 'xlsx':
    df = pd.read_excel(args.file_path)
    log.info(f"Loaded Excel file: {args.file_path}")
elif file_extension == 'csv':
    df = pd.read_csv(args.file_path)
    log.info(f"Loaded CSV file: {args.file_path}")
else:
    log.error(f"Error: Unsupported file format. Please use CSV, XLS, or XLSX files.")
    sys.exit(1)

if args.doi_column not in df.columns:
    print(f"Error: DOI column '{args.doi_column}' not found in file")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)

# Remove rows with missing DOIs first (for consistency across batches)
log.info(f"Original DataFrame size: {len(df)} rows")
df = df.dropna(subset=[args.doi_column])
df = df[df[args.doi_column].str.strip() != '']
log.info(f"After removing empty DOI strings: {len(df)} rows")

# Apply start index and sample size to cleaned data
log.info(f"Start index: {args.start_index}, Sample size: {args.sample_size}")

df = df.iloc[args.start_index:]
if args.sample_size:
    df = df.head(args.sample_size)
    log.info(f"After applying sample_size: {len(df)} rows")


# Handle publisher detection - try multiple methods
log.info("Detecting publishers using multiple methods...")

# Method 1: Use existing publisher column if available
if args.publisher_column in df.columns:
    df['publisher_from_column'] = df[args.publisher_column].fillna('Unknown')
    log.info(f"Found publisher column: {args.publisher_column}")
else:
    df['publisher_from_column'] = 'Unknown'
    log.info("No publisher column found")

# Method 2: Detect from DOI
df['publisher_from_doi'] = df[args.doi_column].apply(detect_publisher_from_doi)


# Combine methods - prioritize column, then DOI, then journal
def combine_publisher_detection(row):
    if row['publisher_from_column'] != 'Unknown':
        return row['publisher_from_column']
    elif row['publisher_from_doi'] != 'Unknown':
        return row['publisher_from_doi']
    else:
        return 'Unknown'

df['final_publisher'] = df.apply(combine_publisher_detection, axis=1)

# Print detection results
detection_summary = df['final_publisher'].value_counts()
print("\nPublisher Summary:")
print(detection_summary)
print()

# Filter out rows where publisher couldn't be detected
original_count = len(df)
df = df[df['final_publisher'] != 'Unknown']
filtered_count = len(df)
log.info(f"DOIs with valid publishers: {filtered_count}/{original_count}")


downloader = records_downloader.RecordsDownloader(logger=log, logger_file=log_file)

# Load publisher->journals mapping
with open(args.journals_json, 'r') as f:
    publisher_journals = json.load(f)

# Process DOIs
successful = 0
failed = 0
already_in_db = 0
results = []

log.info(f"Starting processing of {len(df)} DOIs")

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing DOIs"):
    doi = row[args.doi_column]
    publisher = row['final_publisher']
    
    # Get CSV data for fallback
    csv_title = row.get(args.title_column, None) if args.title_column in df.columns else None
    csv_abstract = row.get(args.abstract_column, None) if args.abstract_column in df.columns else None
    
    csv_year = row.get(args.year_column, None) if args.year_column in df.columns else None
    # get year from downloaded record
    doi_record = downloader.get_record_by_doi(doi=doi, mailto=mailid)
    year_value = extract_year_from_record(doi_record or data_entry, csv_year=csv_year) # prefer downloaded record, fallback to csv_year

    # find publisher match
    publisher_value = publisher  # may be updated after matching
    log.info(f"Processing DOI {index + 1}/{len(df)}: {doi} (Publisher: {publisher})")
    found = False
    publisher_lower = publisher.lower()
    matched_publisher = None
    
    # Check if any publisher in journals mapping matches (case-insensitive)
    for journal_publisher in publisher_journals.keys():
        if journal_publisher.lower() == publisher_lower:
            matched_publisher = journal_publisher
            break
    
    if matched_publisher:
        publisher_value = matched_publisher
        log.info(f"Matched publisher '{publisher_value}' for DOI: {doi}")
    elif publisher not in publisher_journals:
        doi_publisher = row['publisher_from_doi']
        if doi_publisher != 'Unknown' and doi_publisher in publisher_journals:
            log.info(f"Publisher '{publisher}' not found, trying DOI-detected publisher '{doi_publisher}' for DOI: {doi}")
            publisher_value = doi_publisher
        else:
            log.warning(f"Publisher '{publisher}' not found in journals mapping for DOI: {doi}")
            failed += 1
            # include the CSV title if present even on failure
            results.append(make_result(doi, publisher_value, (csv_title or None), year_value, False,
                                    f"Publisher '{publisher}' not found"))
            continue

    journals = list(publisher_journals[publisher_value].keys())
    
    for journal in journals:
        try:
            log.info(f"Trying journal: {journal} for DOI: {doi}")
            doi_record = downloader.get_record_by_doi(doi=doi, mailto=mailid)
            if not doi_record:
                log.info(f"No record found for DOI: {doi} in journal: {journal}")
                continue
            data_entry = {col: doi_record.get(col, None) for col in metadata_columns}

            title_value = data_entry.get('title') or csv_title or None
            
            # Use CSV data as fallback for title and abstract if downloaded record has None
            if data_entry.get('title') is None and csv_title is not None:
                data_entry['title'] = csv_title
                log.info(f"Using CSV title for DOI: {doi}")
                
            if data_entry.get('abstract') is None and csv_abstract is not None:
                data_entry['abstract'] = csv_abstract
                log.info(f"Using CSV abstract for DOI: {doi}")
            
            if args.no_db:
                # Write one JSON file per DOI, plus append to a CSV later
                safe = doi.replace('/', '_').replace(':', '_')
                out_json = Path(args.output_dir) / f"{safe}.json"
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(data_entry, f, ensure_ascii=False, indent=2)
                log.info(f"Wrote {out_json}")
                successful += 1
                results.append(make_result(doi, publisher_value, title_value, year_value, True, f"Saved {out_json.name}"))
            else:
                paper = Papers().get_one(db, criteria={"doi": data_entry['DOI']})
                if not paper:
                    log.info(f'Adding DOI to papers table.')
                    utils._add_to_papers(db, data_entry=data_entry, directory=publisher.lower())
                    paper = Papers().get_one(db, criteria={"doi": data_entry['DOI']})
                    pid = paper.id
                else:
                    pid = paper.id
                existing_paper_info = PaperInfo().get_one(db, criteria={"pid": pid})
                if not existing_paper_info:
                    utils._add_to_papers_info(db, int(pid), data_entry)
                    log.info(f"Successfully added/updated record for DOI: {doi} in journal: {journal}")
                    successful += 1
                    results.append({'doi': doi, 'publisher': publisher, 'success': True, 'message': 'Success'})
                else:
                    log.warning(f"Paper info already exists for PID: {pid}, skipping insertion")
                    already_in_db += 1
                    results.append({'doi': doi, 'publisher': publisher, 'success': True, 'message': 'Already in database'})
            
            found = True
            break
        except Exception as e:
            log.warning(f"Error processing DOI {doi} for journal {journal}: {e}")
            failed +=1
            continue
    
    if not found:
        log.warning(f"DOI {doi} not found in any journal for publisher {publisher}")
        failed += 1
        results.append(make_result(doi, publisher_value, (csv_title or None), year_value, False, 'DOI not found in any journal'))

# Summary
log.info("=" * 60)
log.info("PROCESSING SUMMARY")
log.info("=" * 60)
log.info(f"Total DOIs processed: {len(df)}")
log.info(f"New records added: {successful}")
log.info(f"Already in database: {already_in_db}")
log.info(f"Failed downloads: {failed}")
log.info(f"Success rate: {(successful + already_in_db)/len(df)*100:.1f}%")
log.info("=" * 60)

# Print resume information for next run
end_index = args.start_index + len(df)
log.info(f"NEXT RUN: Use --start_index {end_index} to continue from where this run ended")
log.info(f"Processed DOIs from index {args.start_index} to {end_index-1}")
log.info("=" * 60)

# Save results to CSV
results_df = pd.DataFrame(results, columns=['doi', 'publisher', 'title', 'pub_year', 'success', 'message'])
results_file = f"{args.log_file.rstrip('/')}/download_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(results_file, index=False)
log.info(f"Results saved to: {results_file}")

# Print failed DOIs for review
if failed > 0:
    log.info("FAILED DOIs:")
    failed_results = results_df[results_df['success'] == False]
    for _, row in failed_results.iterrows():
        log.info(f"  {row['doi']} ({row['publisher']}): {row['message']}")

utils.code_termination(sys.argv[0])