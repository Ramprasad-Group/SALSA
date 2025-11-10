#!/usr/bin/env python3
"""
Enhanced figure and table extraction using LLM for robust table parsing
Combines rule-based figure extraction with LLM-based table extraction
Includes support for Springer, RSC, Nature, and other publishers
"""

import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import datetime
import json
from openai import OpenAI
import glob

sys.path.append('/home/wschertzer3/paper-downloader')
from paper_downloader import config
from backend import sett, postgres
from llm_table_extractor import LLMTableExtractor

# Import all available extractors
from parsers.elsevier_extractor import extract_elsevier_figures, extract_elsevier_tables
from parsers.wiley_extractor import extract_wiley_figures, extract_wiley_tables
from parsers.springer_extractor import extract_springer_figures, extract_springer_tables
from parsers.rsc_extractor import extract_rsc_figures, extract_rsc_tables
from parsers.nature_extractor import extract_nature_figures, extract_nature_tables

def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract figures and tables using LLM for robust parsing")
    parser.add_argument('--no_db', action='store_true', default=True, help='Run without Postgres; use only local CSV info')
    parser.add_argument('--csv_path', 
                       default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/papers/reparsed/index.csv',
                       help='Path to CSV file containing DOIs')
    parser.add_argument('--output_dir', 
                       default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/figures_tables',
                       help='Base output directory for figures and tables')
    parser.add_argument('--structured_files_dir',
                       default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/papers/structured_files',
                       help='Directory containing structured files')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of DOIs to process (for testing)')
    parser.add_argument('--offset', type=int, default=0,
                       help='Number of DOIs to skip from the beginning (default: 0)')
    parser.add_argument('--extract_figures', '-ef', action='store_true', default=False,
                       help='Extract figures (default: False)')
    parser.add_argument('--extract_tables', '-et', action='store_true', default=False,
                       help='Extract tables (default: False)')
    parser.add_argument('--log_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log_file', default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/figures_tables/',
                       help='Log file directory')
    parser.add_argument('--debug_publishers', action='store_true',
                       help='Debug mode: test extraction on sample files for each publisher')
    parser.add_argument('--publishers', nargs='+', 
                       choices=['elsevier', 'wiley', 'springer', 'rsc', 'nature', 'acs', 'informa_uk', 'hindawi', 'aip'],
                       help='Specific publishers to process (default: all available)')
    parser.add_argument('--filter_figures', '-ff', action='store_true', default=False)
    parser.add_argument('--filter_tables', '-ft', action='store_true', default=False)

    return parser.parse_args()

def get_papers_from_database(dois, db):
    """Get paper information from database (same as reparse code)"""
    papers = postgres.raw_sql(
        """
        SELECT *
        FROM papers 
        WHERE doi = ANY(:dois)
        ORDER BY doi
        """,
        dois=dois
    )
    papers_df = pd.DataFrame.from_records(papers, columns=papers[0]._fields)
    return papers_df

def get_papers_from_csv(df: pd.DataFrame, dois: list) -> pd.DataFrame:
    """
    Return only the required DOI rows from the CSV, assuming they have all necessary metadata
    """
    subset = df[df['doi'].isin(dois)]

    required_cols = {'doi', 'directory', 'doctype'}
    missing = required_cols - set(subset.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")

    return subset

def find_file_for_doi_from_db(doi, directory_db, doctype, structured_files_dir):
    """Find the file using database info (same logic as reparse code)"""
    # Convert DOI to filename format
    file_name = doi.replace('/', '@') + '.' + doctype
    
    # Get local directories
    local_dirs = [d for d in os.listdir(structured_files_dir) if os.path.isdir(os.path.join(structured_files_dir, d))]
    
    # Find matching directory
    directory = None
    for local_dir in local_dirs:
        if local_dir in directory_db.lower():
            directory = local_dir
            break
    
    if directory is None:
        return None, None, None
    
    # Construct file path
    file_path = os.path.join(structured_files_dir, directory, file_name)
    
    if os.path.exists(file_path):
        return file_path, directory, doctype
    
    return None, None, None

def extract_figures_tables(file_path, publisher, output_dir, doi, llm_extractor, extract_figures=True, extract_tables=True):
    """Extract figures and tables based on publisher and method preference"""
    logger = logging.getLogger('extraction')
    figures_count = 0
    tables_count = 0

    safe_doi = doi.replace('/', '@')  # consider a more robust sanitizer below
    fig_dir = Path(output_dir) / 'figures' / safe_doi
    table_dir = Path(output_dir) / 'tables' / safe_doi

    if fig_dir.exists() and table_dir.exists() :
        logger.info(f"Skipping {doi}: {fig_dir} and {table_dir} already exists")
        return figures_count, tables_count   # or return None, or a status dict


    
    if publisher.lower() == 'elsevier':
        # Extract figures if requested
        if extract_figures:
            figures_count = extract_elsevier_figures(file_path, output_dir, doi)
        else:
            logger.warning("Skipping figure extraction")
        
        # Extract tables if requested
        if extract_tables:
            if llm_extractor:
                tables_count, saved_files = extract_elsevier_tables(file_path, output_dir, doi, llm_extractor)
            else:
                logger.warning("LLM extractor not available - cannot extract tables")
                tables_count = 0
        else:
            logger.warning("Skipping table extraction")
    
    elif publisher.lower() == 'wiley':
        # Extract figures if requested
        if extract_figures:
            figures_count = extract_wiley_figures(file_path, output_dir, doi)
        else:
            logger.warning("Skipping figure extraction")
        
        # Extract tables if requested
        if extract_tables:
            if llm_extractor:
                tables_count, saved_files = extract_wiley_tables(file_path, output_dir, doi, llm_extractor)
            else:
                logger.warning("LLM extractor not available - cannot extract tables")
                tables_count = 0
        else:
            logger.warning("Skipping table extraction")
    
    elif publisher.lower() == 'springer':
        # Extract figures if requested
        if extract_figures:
            figures_count = extract_springer_figures(file_path, output_dir, doi)
        else:
            logger.warning("Skipping figure extraction")
        
        # Extract tables if requested
        if extract_tables:
            if llm_extractor:
                tables_count, saved_files = extract_springer_tables(file_path, output_dir, doi, llm_extractor)
            else:
                logger.warning("LLM extractor not available - cannot extract tables")
                tables_count = 0
        else:
            logger.warning("Skipping table extraction")
    
    elif publisher.lower() == 'royal society of chemistry' or publisher.lower() == 'rsc':
        # Extract figures if requested
        if extract_figures:
            figures_count = extract_rsc_figures(file_path, output_dir, doi)
        else:
            logger.warning("Skipping figure extraction")
        
        # Extract tables if requested
        if extract_tables:
            if llm_extractor:
                tables_count, saved_files = extract_rsc_tables(file_path, output_dir, doi, llm_extractor)
            else:
                logger.warning("LLM extractor not available - cannot extract tables")
                tables_count = 0
        else:
            logger.warning("Skipping table extraction")
    
    elif publisher == 'nature':
        # Extract figures if requested
        if extract_figures:
            figures_count = extract_nature_figures(file_path, output_dir, doi)
        else:
            logger.warning("Skipping figure extraction")
        
        # Extract tables if requested
        if extract_tables:
            if llm_extractor:
                tables_count, saved_files = extract_nature_tables(file_path, output_dir, doi, llm_extractor)
            else:
                logger.warning("LLM extractor not available - cannot extract tables")
                tables_count = 0
        else:
            logger.warning("Skipping table extraction")
    
    elif publisher == 'acs':
        # ACS extractor - figures saved as metadata, tables for LLM processing
        if extract_figures:
            try:
                from parsers.acs_extractor import extract_acs_figures
                figures_count = extract_acs_figures(file_path, output_dir, doi)
            except ImportError:
                logger.warning("ACS extractor not available")
                figures_count = 0
        else:
            logger.warning("Skipping figure extraction")
        
        if extract_tables:
            try:
                from parsers.acs_extractor import extract_acs_tables
                if llm_extractor:
                    tables_count, saved_files = extract_acs_tables(file_path, output_dir, doi, llm_extractor)
                else:
                    logger.warning("LLM extractor not available - cannot extract tables")
                    tables_count = 0
            except ImportError:
                logger.warning("ACS extractor not available")
                tables_count = 0
        else:
            logger.warning("Skipping table extraction")
    
    else:
        logger.warning(f"Publisher {publisher} not supported yet")
        return 0, 0
    
    return figures_count, tables_count

def debug_publisher_extraction(output_dir, llm_extractor, extract_figures=True, extract_tables=True):
    """Debug function to test extraction on sample files for each publisher"""
    logger = logging.getLogger('extraction')
    
    # Test files for each publisher
    test_files = {
        'elsevier': '/data/sonakshi/prod/structured_files/elsevier_xml/10.1016@j.biortech.2020.123456.xml',
        'wiley': '/data/sonakshi/prod/structured_files/wiley/10.1002@app.10031.html',
        'springer': '/data/sonakshi/prod/structured_files/springer/10.1007@s10853-020-05509-2.html',
        'rsc': '/data/sonakshi/prod/structured_files/rsc/10.1039@a001hopy.html',
        'nature': '/data/sonakshi/prod/structured_files/nature/10.1038@s41428-020-00396-5.html',
        'acs': '/data/sonakshi/prod/structured_files/acs/10.1021@acs.macromol.0c01234.xml'
    }
    
    for publisher, file_path in test_files.items():
        if os.path.exists(file_path):
            logger.info(f"Testing {publisher} extraction with file: {file_path}")
            doi = f"test_{publisher}_doi"
            
            try:
                figures_count, tables_count = extract_figures_tables(
                    file_path, publisher, output_dir, doi, llm_extractor, 
                    extract_figures, extract_tables
                )
                logger.info(f"{publisher.upper()}: Extracted {figures_count} figures and {tables_count} tables")
            except Exception as e:
                logger.error(f"Error testing {publisher}: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"Test file not found for {publisher}: {file_path}")

def figure_caption_test(meta: dict, model: str = "gpt-4o-mini") -> bool:
    """
    Use OpenAI to determine if a figure is a plot depicting anionic conductivity vs time
    based solely on its caption text.
    """
    caption = meta.get("caption", "")

    if not caption or not caption.strip():
        return False

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION")
    )

    prompt = f"""
                You are a materials science expert. Determine whether the following figure caption describes of hydroxide/ionic conductivity as a function of time or alkaline/chemical stability as a funciton of time.

                Caption:
                \"\"\"{caption}\"\"\"

                Rules:
                - Return ONLY one word: "YES" if it does desribe hydroxide/ionic conductivity as a function of time or alkaline/chemical stability as a funciton of time.
                - Return "NO" otherwise.
                """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        return result.startswith("YES")
    except Exception as e:
        print(f"[OpenAI error filtering figure]: {e}")
        return False
    
def table_caption_test(meta: dict, model: str = "gpt-4o-mini") -> bool:
    """
    Use OpenAI to determine if a figure is a plot depicting anionic conductivity vs time
    based solely on its caption text.
    """

    caption = meta.get("caption", "")
    if not caption or not caption.strip():
        return False

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION")
    )

    prompt = f"""
                You are a materials science expert. Determine whether the following table caption describes hydroxide/ionic conductivity as a function of time or alkaline/chemical stability as a funciton of time.
                

                Caption:
                \"\"\"{caption}\"\"\"

                Rules:
                - Return ONLY one word: "YES" if it does describe hydroxide/ionic conductivity as a function of time or alkaline/chemical stability as a funciton of time.
                - Return "NO" otherwise.
                """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        return result.startswith("YES")
    except Exception as e:
        print(f"[OpenAI error filtering figure]: {e}")
        return False
    
def filter_figures(output_dir: str):
    """
    Scan output_dir for figure metadata JSON files and delete figures
    that are not conductivity vs. time plots.
    """
    fig_dir = os.path.join(output_dir, "figures")
    json_files = glob.glob(os.path.join(fig_dir, "**", "*_metadata.json"), recursive=True)
    print(f"Found {len(json_files)} figure metadata files in directory: {fig_dir}")

    kept, removed = 0, 0

    for meta_path in json_files:
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            image_path = meta.get("figure_file_path")

            if figure_caption_test(meta):
                kept += 1
            else:
                removed += 1
                # Delete figure and metadata
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                os.remove(meta_path)
        except Exception as e:
            print(f"Error processing {meta_path}: {e}")

    print(f"Kept {kept} relevant figures, removed {removed} irrelevant ones.")

def filter_tables(output_dir: str):
    """
    Scan output_dir for figure metadata JSON files and delete figures
    that are not conductivity vs. time plots.
    """
    table_dir = os.path.join(output_dir, "tables")
    json_files = glob.glob(os.path.join(table_dir, "**", "*_metadata.json"), recursive=True)
    print(f"Found {len(json_files)} table metadata files in directory: {table_dir}")

    kept, removed = 0, 0

    for meta_path in json_files:
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            image_path = meta.get("table_file_path")

            if table_caption_test(meta):
                kept += 1
            else:
                removed += 1
                # Delete figure and metadata
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                os.remove(meta_path)
        except Exception as e:
            print(f"Error processing {meta_path}: {e}")

    print(f"Kept {kept} relevant tables, removed {removed} irrelevant ones.")


def main():
    args = parse_arguments()

    os.makedirs(args.log_file, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(args.log_file, f'extraction_enhanced_{timestamp}.log')
    
    # Setup logging
    logger = setup_logging(args.log_level, log_file_path)
    logger = logging.getLogger('extraction')  # Use consistent logger name
    logger.info(f"Running in {'NO-DB' if args.no_db else 'DB'} mode")
    logger.info(f"Log file: {log_file_path}")
    
    # Process extraction flags
    extract_figures = args.extract_figures
    extract_tables = args.extract_tables 
    
    # Initialize LLM extractor (only if extracting tables)
    llm_extractor = None
    if extract_tables:
        try:
            llm_extractor = LLMTableExtractor()
            logger.info("LLM extractor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM extractor: {e}")
            logger.error("Cannot proceed without LLM extractor for table extraction")
            return
    else:
        logger.info("Skipping LLM extractor initialization (tables not being extracted)")
    
    # Check for debug mode
    if args.debug_publishers:
        logger.info("Running in debug mode - testing publisher extraction")
        debug_publisher_extraction(args.output_dir, llm_extractor, extract_figures, extract_tables)
        return
    
    if not args.no_db:
        from backend import sett, postgres
        sett.load_settings("/data/sonakshi/paper-downloader/settings.yaml")
        postgres.load_settings()
        db = postgres.connect('polylit')
        logger.info("Database connection established")
    
    # Read CSV file
    try:
        df = pd.read_csv(args.csv_path)
        if 'doi' not in df.columns:
            logger.error("CSV file must contain a 'doi' column")
            return
        
        # dois = df['doi'].dropna().unique().tolist()
        dois = df[df['status'] == 'ok']['doi'].dropna().unique().tolist()
        logger.info(f"Found {len(dois)} unique DOIs")
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    
    # dois = ['10.1002/app.11736']

    # Apply offset and sample size if specified
    if args.offset > 0:
        dois = dois[args.offset:]
        logger.info(f"Skipped {args.offset} DOIs, processing {len(dois)} remaining DOIs")
    
    if args.sample_size:
        dois = dois[:args.sample_size]
        logger.info(f"Processing sample of {len(dois)} DOIs")
    
    # Get paper information from database
    logger.info("Querying database for paper information...")
    try:
        if args.no_db:
            logger.info("Running in no-DB mode")
            papers_df = get_papers_from_csv(df, dois)
        else:
            papers_df = get_papers_from_database(dois, db)
        logger.info(f"Retrieved {len(papers_df)} papers to process")
    except Exception as e:
        logger.error(f"Failed to prepare paper list: {e}")
        return
    logger.info(f"Found {len(papers_df)} papers in database")
    
        
    # target_directories = ['springer', 'rsc', 'nature']
    # papers_df = papers_df[papers_df['directory'].isin(target_directories)]
    # logger.info(f"Filtered to {len(papers_df)} papers from specified publishers: {target_directories}")
    
    # Apply sample size if specified (after filtering by publishers)
    if args.sample_size:
        papers_df = papers_df.head(args.sample_size)
        logger.info(f"Processing sample of {len(papers_df)} papers")
    
    # Process each paper
    successful = 0
    failed = 0
    total_figures = 0
    total_tables = 0
    publisher_stats = {}
    
    for _, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Processing papers"):
        print("======row======", row)
        doi = row['doi']
        directory_db = row['directory']
        doctype = row['doctype']
        
        # # Skip if directory is not springer, rsc, or nature
        # if directory_db not in ['springer', 'rsc', 'nature']:
        #     logger.debug(f"Skipping {doi} - directory '{directory_db}' not in target publishers")
        #     continue
        
        logger.info(f"Processing {doi} (Publisher: {directory_db.upper()})")
        logger.debug(f"Publisher (from DB): {directory_db}")
        logger.debug(f"File type (from DB): {doctype}")
        
        # Find the file using database info
        file_path, directory, file_type = find_file_for_doi_from_db(doi, directory_db, doctype, args.structured_files_dir)
        
        if not file_path:
            logger.warning(f"No file found for {doi}")
            failed += 1
            continue
        
        logger.info(f"Found {file_type} file: {file_path}")
        logger.debug(f"Local directory: {directory}")
        
        # Extract figures and tables
        figures_count, tables_count = extract_figures_tables(
            file_path, directory, args.output_dir, doi, 
            llm_extractor, extract_figures, extract_tables
        )
        
        # Update publisher statistics
        if directory not in publisher_stats:
            publisher_stats[directory] = {'papers': 0, 'figures': 0, 'tables': 0, 'successful': 0}
        
        publisher_stats[directory]['papers'] += 1
        publisher_stats[directory]['figures'] += figures_count
        publisher_stats[directory]['tables'] += tables_count
        
        if figures_count > 0 or tables_count > 0:
            successful += 1
            publisher_stats[directory]['successful'] += 1
            total_figures += figures_count
            total_tables += tables_count
            logger.info(f"Extracted {figures_count} figures and {tables_count} tables")
        else:
            failed += 1
            logger.warning(f"No figures or tables extracted")
    
    # Summary
    logger.info("=" * 60)
    logger.info("ENHANCED EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total papers in database: {len(papers_df)}")
    logger.info(f"DOI offset applied: {args.offset}")
    logger.info(f"Successful extractions: {successful}")
    logger.info(f"Failed extractions: {failed}")
    logger.info(f"Total figures extracted: {total_figures}")
    logger.info(f"Total tables extracted: {total_tables}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Figures extraction: {'Enabled' if extract_figures else 'Disabled'}")
    logger.info(f"Tables extraction: {'Enabled (LLM)' if extract_tables else 'Disabled'}")
    
    # Publisher breakdown
    logger.info("\nPublisher Statistics:")
    logger.info("-" * 40)
    for publisher, stats in publisher_stats.items():
        success_rate = (stats['successful'] / stats['papers'] * 100) if stats['papers'] > 0 else 0
        logger.info(f"{publisher.upper()}:")
        logger.info(f"  Papers processed: {stats['papers']}")
        logger.info(f"  Successful: {stats['successful']} ({success_rate:.1f}%)")
        logger.info(f"  Figures: {stats['figures']}")
        logger.info(f"  Tables: {stats['tables']}")
        logger.info("")
    
    logger.info("=" * 60)

    if args.filter_figures:
        print("Filtering figures for conductivity vs. time relevance...")
        filter_figures(args.output_dir)
    if args.filter_tables:
        print("Filtering tables for conductivity vs. time relevance...")
        filter_tables(args.output_dir)

if __name__ == "__main__":
    main()
