"""Download papers from a database of records into a directory"""
import argparse
import sys
import subprocess
from os import path
import os
import pandas as pd  # NEW
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, parent_dir)
from paper_downloader import downloader, utils
from paper_downloader.utils import execute
# from paper_downloader.utils import execute, _add_to_papers, _add_to_papers_info, _update_paper_info

# Try to import DB stack; it's optional now
try:
    from backend import sett, postgres
    from backend.postgres.orm import PaperInfo, Papers
    from backend.utils import checkpoint
    HAS_DB = True
except Exception:
    HAS_DB = False
    sett = None
    postgres = None
    checkpoint = None
    PaperInfo = None
    Papers = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download papers from a database of records into a directory")
    # EXISTING flags...
    parser.add_argument('--debug', type=bool, default=True, help='Store output in the dev database if True')
    parser.add_argument('--log_file', default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/papers', help='Path to log file')
    parser.add_argument('--sample_size', default=1, help='Download sample size')
    parser.add_argument('--update_dvc', type=bool, default=False, help='Update version of data if True')

    # NEW flags for No-DB mode
    parser.add_argument('--no_db', default=True, action='store_true', help='Run without Postgres')
    parser.add_argument('--input_csv', default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/records/download_results_20251110_163021.csv', help='CSV with columns: doi,title,pub_year,publisher')
    parser.add_argument('--download_dir', default='/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/logs/papers', help='Output directory (No-DB mode)')
    return parser.parse_args()

def setup_environment(args):
    if not args.no_db:
        if not HAS_DB:
            raise RuntimeError("DB mode requested but backend/Postgres not available. Use --no_db instead.")
        sett.load_settings()
        postgres.load_settings()
        db = postgres.connect('polylit')
        log_file = f'{args.log_file}/{utils.formatted_time()}.log'
        log = utils.setup_logger(log_file, logger_name='paper-downloader')
        sample_size = None if args.sample_size == 'None' else int(args.sample_size)
        download_directory = args.download_dir
        log.info(f"Using {'development' if args.debug else 'production'} environment for downloading papers")
        return log, log_file, sample_size, db, download_directory
    else:
        # No-DB setup
        os.makedirs(args.download_dir, exist_ok=True)
        os.makedirs(path.dirname(args.log_file), exist_ok=True)
        log_file = f'{args.log_file}/{utils.formatted_time()}.log'
        log = utils.setup_logger(log_file, logger_name='paper-downloader')
        sample_size = None if args.sample_size == 'None' else int(args.sample_size)
        db = None
        download_directory = args.download_dir
        log.info("Running in NO-DB mode (writing status to local JSONL/CSV logs)")
        return log, log_file, sample_size, db, download_directory

def get_query():
    return '''
    SELECT pi.id AS pi_id, pi.pid, pi.doi, p.title, pi.pub_year, pi.download_status, pi.publisher
    FROM papers_info pi
    JOIN papers p ON pi.pid = p.id
    WHERE pi.pub_year > :year_lower
    AND p.title IS NOT NULL
    AND pi.download_status IS NULL
    -- AND author_list IS NOT NULL
    AND (
        pi.publisher ILIKE '%Springer%' OR
        pi.publisher ILIKE '%Elsevier%' OR
        pi.publisher ILIKE '%Royal Society of Chemistry%' OR
        pi.publisher = 'Hindawi Limited' OR
        pi.publisher = 'Wiley' OR
        pi.publisher = 'AIP Publishing' OR
        pi.publisher = 'Informa UK Limited'
    )
    ORDER BY pi.id
    '''

def main():
    args = parse_arguments()
    log, log_file, sample_size, db, download_directory = setup_environment(args)

    # Input “cursor” source differs by mode
    records_df = None
    if args.no_db:
        if not args.input_csv:
            raise ValueError("--no_db requires --input_csv with columns: doi,title,pub_year,publisher")
        df = pd.read_csv(args.input_csv)
        required = {'doi', 'title', 'pub_year', 'publisher'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"--input_csv missing columns: {missing}")
        # Build columns expected by downloader.article_downloader:
        df = df.copy()
        df['pi_id'] = range(1, len(df) + 1)
        df['pid'] = df['pi_id']      # dummy
        df['download_status'] = None # dummy
        records_df = df[['pi_id', 'pid', 'doi', 'title', 'pub_year', 'download_status', 'publisher']]
        log.info(f"[NO-DB] Loaded input_csv with {len(records_df)} rows; columns={list(records_df.columns)}")
    else:
        # DB mode as before
        try:
            last_processed_id = checkpoint.get_last(db, name='download-papers', table=PaperInfo.__tablename__)
        except Exception:
            last_processed_id = None

    if sample_size is not None:
        input_sample_size = sample_size

    year_lower = 2015
    query = None if args.no_db else get_query()

    last_check_time = utils.formatted_time()
    count_query = '''
        SELECT COUNT(*) 
        FROM papers_info 
        WHERE note IS NOT NULL 
        AND download_time_stamp > :last_check_time
    '''

    log.info(f'Started downloading papers at {last_check_time}')
    log.info(f'Sample size {sample_size}')

    # completed_files is DB-only; in No-DB it’s just 0
    completed_files = 0 if args.no_db else execute(count_query, last_check_time=last_check_time)

    status = True
    if not args.no_db:
        try:
            last_processed_id = checkpoint.get_last(db, name='download-papers', table= PaperInfo.__tablename__)
        except Exception:
            last_processed_id = None
    else:
        last_processed_id = None

    while status:
        try:
            if sample_size is None or sample_size > 0:
                status, last_processed_id = downloader.article_downloader(
                    db,
                    query,
                    download_directory,
                    year_lower=year_lower,
                    logger=log,
                    logger_file=log_file,
                    sample_size=sample_size,
                    download_pdf=False,
                    scanned_records=completed_files,
                    last_processed_id=last_processed_id,
                    # pass a DataFrame in No-DB mode
                    records_df=records_df,
                    # where to write “DB updates” locally
                    local_status_log=args.log_file if args.no_db else None
                )
            else:
                status = False
        except Exception as e:
            log.error(f'Exception {e} occurred. Looping back')
            if sample_size is not None:
                # in No-DB we don’t have DB counts; just decrement pessimistically
                if not args.no_db:
                    completed_files = utils.execute(count_query, last_check_time=last_check_time)
                    sample_size = input_sample_size - completed_files
                else:
                    sample_size = max(0, sample_size - 1)
                log.info(f'Exception {e} occurred. Looping back. {sample_size} files left.')
            else:
                if not args.no_db:
                    files_left = utils.execute(count_query, last_check_time=last_check_time)
                    log.info(f'Exception {e} occurred. Looping back. {files_left} files left.')
                else:
                    log.info(f'Exception {e} occurred. Looping back.')

    log.info(f"Last processed doc_id: {last_processed_id}")

    # Checkpoint + counts only in DB mode
    if not args.no_db:
        checkpoint.add_new(
            db, name="download-papers", table=Papers.__tablename__, row=last_processed_id,
            comment={
                'user': sett.Run.userName,
                'filter': 'download-papers',
                'debug': sett.Run.debugCount > 0
            }
        )
        download_count = execute(count_query, last_check_time=last_check_time)
        log.info(f'Number of files downloaded: {download_count}')
        log.info(f"Checkpoint added. name = 'download-papers', table = {PaperInfo.__tablename__}, row = {last_processed_id}, "
                 f"comment = {{'user': '{sett.Run.userName}', 'filter': 'download-papers', "
                 f"'debug': {True if sett.Run.debugCount > 0 else False}}}")

    utils.count_files(path.join(download_directory, 'structured_files'), logger=log)

    if args.update_dvc and not args.no_db:
        subprocess.run(['./scripts/data_version_update.sh', download_directory])

    utils.code_termination(sys.argv[0])
if __name__ == "__main__":
    main()