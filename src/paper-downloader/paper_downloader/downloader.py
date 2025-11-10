#from numpy import isnan
import requests
from os import path, mkdir, rename, remove, stat
#import logging
from random import randint
from time import sleep
import time
import bs4 as bs
from paper_downloader import utils, config
from datetime import datetime
import json
from pathlib import Path
import requests, pickle
# from email.mime.text import MIMEText

MAX_SLEEP_TIME = 7
# FULL_DOWNLOAD_SUCCESSFUL = 1
# ABSTRACT_DOWNLOAD_SUCCESSFUL = 2
# LETTER_DOWNLOAD_SUCCESSFUL = 3
# FILE_DELETED = 4
# DOWNLOAD_FAILED = 5
# FILE_SIZE_0 = 6
# FILE_THREW_EXCEPTION = 7

def _append_jsonl(path_str: str, obj: dict):
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def update_metadata(db, doc_id, logger, download_status, note_entry, download_time_stamp, extension, local_status_log=None, doi=None, publisher=None, title=None, year=None):
    """
    If db is None -> write a JSONL row to local_status_log, else call utils._update_paper_metadata.
    """
    if db is None:
        row = {
            "doc_id": doc_id,
            "doi": doi,
            "publisher": publisher,
            "title": title,
            "year": year,
            "download_status": download_status,
            "note": note_entry or {},
            "timestamp": download_time_stamp,
            "extension": extension,
        }
        if local_status_log:
            _append_jsonl(local_status_log, row)
        if logger:
            logger.info(f"[NO-DB] status logged: {row}")
        return
    # DB mode
    update_metadata(
    db, doc_id, logger,
    download_status=config.FILE_SIZE_0, extension=extension,
    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
    local_status_log=local_status_log, doi=doi, publisher=publisher, title=title, year=year
)

def download_article(doi, publisher, title, year, download_directory, db, doc_id, extension, logger=None, timeout_sec = 90, html_url=None, download_pdf=True, debug=False, local_status_log=None):
    '''
    Downloads an article into a folder given its url and updates a database db on its status

    Parameters
    ----------
    doi: str
        DOI of article to be downloaded
        Used for setting the filename
    publisher: str

        Publisher of article with DOI=doi
    title: str
        Title of article with DOI=doi
    year: int
        Year of publishing of article with DOI=doi
    download_directory: str
        Directory in which downloaded file is saved
    db: Postgres databsase
        Postgres databsase containing download status of all files
    doc_id: str
        Unique record identifier in the Postgres databsase table paper_info
    extension: str
        file extension of the file to be downloaded
        Either xml or html or xml
    logger: logger object

    timeout_sec: int
        Duration of time after which the html request for html_url times out
    html_url: str
        URL to be downloaded
    download_pdf: bool
        Input to _validate_html_status
        If the html document turns out to not contain the full text but the full text is instead contained in a pdf, the this switch determines whether or not to download the file
    debug: bool
        A mode in which some functionality like validation and updating the database is turned off
    '''
    repl_char = '@'
    # As unix file names cannot tolerate / in the filename, / is replaced with repl_char in the DOI
    # This is done in order to recover the DOI from the file name. No existing DOI contains the @ symbol and the symbol
    # itself is intuitive in meaning
    # logger = logging.getLogger()
    header_prefix_accept = {'html':'text', 'xml':'text', 'pdf': 'application'}
    if not path.isdir(download_directory):
        mkdir(download_directory)
    file_name = doi.replace(path.sep, repl_char) + '.' + extension
    file_location = path.join(download_directory, file_name)
    
    # Check if html_url is valid
    if not html_url or not html_url.startswith("http"):
        logger.warning(f"Invalid html_url for DOI {doi}: {html_url}. Skipping.")
        return

    #Download the file if it doesn't exist
    if not path.exists(file_location):
        headers = {
            'Accept': header_prefix_accept[extension] + '/' + extension,
            'User-agent': 'Mozilla/5.6 (Windows NT 11.0; rv:68.0 Gecko/20100101 Firefox/70.0'
        }

        # Make the request to download the file
        # html_data = requests.get(html_url, stream=True, headers=headers, timeout=timeout_sec)
        # --- ACS-aware logic ---
        if 'acs' in publisher.lower():
            # use pre-authenticated session with cookies
            acs_session = get_acs_session()
            html_data = acs_session.get(html_url, stream=True, headers=headers, timeout=timeout_sec)
            if html_data.status_code == 403:
                logger.warning("[ACS] Session expired — please refresh cookies via browser login.")
        else:
            html_data = requests.get(html_url, stream=True, headers=headers, timeout=timeout_sec)

        #Process the response
        if html_data.status_code == 200:
            current_file = open(file_location, 'wb')
            current_file.write(html_data.content)
            if stat(file_location).st_size == 0: #file size is 0
                print(f'---file size is 0:  {file_location}----')
                remove(file_location)
                if extension != 'pdf' and not debug:
                    update_metadata(
                    db, doc_id, logger,
                    download_status=config.FILE_SIZE_0, extension=extension,
                    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                    local_status_log=local_status_log, doi=doi, publisher=publisher, title=title, year=year
                )
                logger.warning(extension + ' file of publisher ' + publisher + ' and DOI ' + doi + ' was of size 0 and hence it was deleted\n')
                return
            
            # # add a note when a PDF is downloaded
            # if extension == 'pdf':
            #     utils._update_paper_metadata(db, doc_id, logger = logger,
            #         note_entry= {
            #             "pdf_download_time_stamp": datetime.now().strftime(config.DATE_TIME_FORMAT),
            #             "pdf_downloaded": True
            #         })
            #     logger.info(f"PDF downloaded for DOI {doi}")
        else: #request failed
            print('---invalid html (request failed) html_data.status_code == ---:', html_data.status_code, 'html_url:', html_url)
            if extension != 'pdf' and not debug:
                update_metadata(
                    db, doc_id, logger,
                    download_status=config.FILE_SIZE_0, extension=extension,
                    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                    local_status_log=local_status_log, doi=doi, publisher=publisher, title=title, year=year
                )
            if year is not None:
                logger.warning(f'Could not download {extension} file with DOI {doi} by publisher {publisher} with title '
                               f'\'{title}\' published in {year}. Failed with HTML status code {html_data.status_code}', 'html_url:', html_url, '\n')
            else:
                logger.warning(f'Could not download {extension} file with DOI {doi} by publisher {publisher} with title '
                               f'\'{title}\'. Failed with HTML status code {html_data.status_code}', 'html_url:', html_url, '\n')
            return
        random_wait_time = randint(1, MAX_SLEEP_TIME) # Choose any value between 1 & 2 seconds randomly
        sleep(random_wait_time)
        # if extension != 'pdf' and not debug:_validate_html_status(doi, publisher, title, year, download_directory, db, doc_id, html_data, file_name, download_pdf, extension, logger=logger)


# Assume that query has been run on database outside of article_downloader
# TODO: Put doi, publisher, title and year in a dict object and pass that around to other functions.
# Change code accordingly

def article_downloader(db, query, download_directory, year_lower, logger_file=None, logger = None, logger_name='downloader',
                        sample_size=None, download_pdf=True, scanned_records=0, last_processed_id=None, records_df=None, local_status_log=None, **kwargs):
    '''
    Downloads the HTML file from the respective publisher's website given a list of DOI's. Requires authorization to access the HTML. It is adequate to access from a network that has a licensing arrangement with the publisher

    Parameters
    ----------
    db : postgres db object
        Postgres databsase of records which is updated when files are downloaded
    download_directory : str
        Directory into which files are saved
        Must be of the form ./structured_files
    download_pdf : bool
    query: dict
        query which selects records from Postgres databsase
    scanned_records: int
        Number of records already downloaded if reusing the logger file upon Cursor Error
    sample_size: int
        Number of records to download if downloading only a subset.
        If None, download all files
    logger_file: str
        File name of logger
    logger_name: str
        Name assigned to the lagger

    Returns
    -------

    '''
    if not path.isdir(path.join(download_directory, 'structured_files')):
        mkdir(path.join(download_directory, 'structured_files'))

    # pdf files are saved in the below directory
    if not path.isdir(path.join(download_directory, 'unstructured_files')):
        mkdir(path.join(download_directory, 'unstructured_files'))
    download_directory = path.join(download_directory, 'structured_files')
    download_directory_input = ''
    url_dict = config.URL_DICT
    i = 0
    repl_char = '@'
    last_download = {publisher: time.time()-config.publisher_gap_time[publisher]
                     for publisher in config.publisher_gap_time.keys()}
    download_status_check_time = datetime.now().strftime(config.DATE_TIME_FORMAT)

    if not logger:
        if not path.exists(logger_file):
            open(logger_file, 'a').close()
        logger = utils.setup_logger(logger_file, logger_name)

    headers = {
        'Accept': 'text/html',
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; rv:68.0) Gecko/20100101 Firefox/68.0'
    }

    if records_df is not None:
        # No-DB path: use DataFrame directly
        if sample_size is None:
            cursor = records_df
        else:
            cursor = records_df.iloc[:int(sample_size)]
    else:
        # DB path: use SQL query
        if sample_size is None:
            cursor = utils.execute(query, year_lower=year_lower, **kwargs)
        else:
            # Only append LIMIT when query is a string
            query = (query or "") + " LIMIT :sample_size"
            cursor = utils.execute(query, year_lower=year_lower, sample_size=sample_size, **kwargs)

    total_records = len(cursor)
    logger.info(f'Number of documents to process: {total_records}')

    # logger.warning(f'Number of documents left: {collection.count_documents(query)}\n')
    for index, doc in cursor.iterrows():
        doc_id = doc['pi_id']
        last_processed_id = doc_id
        paper_id = doc['pid']
        publisher = str(doc['publisher'])
        title = str(doc['title'])
        year = doc['pub_year'] if doc['pub_year'] is not None else None
        doi = str(doc['doi'])
        logger.info(f'Downloading DOI: {doi}')
        url = ''
        try:
            i += 1
            status_check_condition = ((i + scanned_records) % config.STATUS_CHECK == 0)
            if isinstance(status_check_condition, bool):
                status_check = status_check_condition
            else: # Assuming it's a DataFrame or Series with a 'count' column
                status_check = status_check_condition['count'].values[0] if len(status_check_condition['count'].values) > 0 else False
            
            if status_check:
                logger.warning(f'Number of records scanned: {i+scanned_records}\n')
                current_status, msg = utils.check_download_status(db, download_status_check_time)
                if current_status:
                    logger.warning('Everything as expected...Continuing download\n')
                    download_status_check_time = datetime.now().strftime(config.DATE_TIME_FORMAT)
                else:
                    logger.warning(f'Pausing download. Root must verify status. Last download_status_check_time='
                                    f'{download_status_check_time} \n {msg}')
                    # utils.send_e_mail(message=MIMEText(f'Download has been terminated, please check. \
                                # Last download_status_check_time={download_status_check_time}\n {msg}').as_string())
                    return False
            if 'elsevier' in publisher.lower():
                url = url_dict['elsevier'] + doi + '?APIKey=' + config.ELS_API_KEY
                download_directory_input = path.join(download_directory, 'elsevier')
                download_article(doi, publisher, title, year, download_directory_input, db, doc_id, 'xml', logger=logger, download_pdf=download_pdf, html_url=url)
                sleep(2)
            elif 'wiley' in publisher.lower():
                continue_status, last_download['wiley'] = utils.download_gap_time(last_download['wiley'], 'wiley')
                if continue_status:
                    # logger.info(f"Waiting before downloading Wiley paper with DOI: {doi}")
                    # sleep(config.publisher_gap_time['wiley']) 
                    continue
                url = url_dict['wiley'] + doi
                download_directory_input = path.join(download_directory, 'wiley')
                sleep(2)
            elif 'springer' in publisher.lower() and ('10.1038/' in doi or '10.1295/' in doi):
                #url = url_dict['nature'] + doi.replace('10.1038/', '').replace('10.1295/', '').replace('.', '')
                url = url_dict['doi'] + doi
                download_directory_input = path.join(download_directory, 'nature')
                sleep(2)
            elif 'springer' in publisher.lower():
                url = url_dict['doi'] + doi
                download_directory_input = path.join(download_directory, 'springer')
                sleep(2)
            elif 'acs' in publisher.lower():
                continue_status, last_download['acs'] = utils.download_gap_time(last_download['acs'], 'acs')
                if continue_status:
                    continue
                url = url_dict['acs'] + doi
                download_directory_input = path.join(download_directory, 'acs')
                sleep(2)
            elif 'informa uk' in publisher.lower():
                url = url_dict['informa uk'] + doi
                download_directory_input = path.join(download_directory, 'informa_uk')
            elif 'sage' in publisher.lower():
                url = url_dict['sage'] + doi
                download_directory_input = path.join(download_directory, 'sage')
            elif 'iop publishing' in publisher.lower():
                url = url_dict['doi'] + doi
                download_directory_input = path.join(download_directory, 'iop_publishing')
            elif 'electrochemical society' in publisher.lower():
                continue_status, last_download['ecs'] = utils.download_gap_time(last_download['ecs'], 'ecs')
                if continue_status:
                    continue
                url = url_dict['iop publishing'] + doi
                download_directory_input = path.join(download_directory, 'ecs')
            elif 'aip publishing' in publisher.lower():
                continue_status, last_download['aip'] = utils.download_gap_time(last_download['aip'], 'aip')
                if continue_status:
                    continue
                url = url_dict['aip'] + doi
                download_directory_input = path.join(download_directory, 'aip')
            elif 'emerald' in publisher.lower():
                url = url_dict['emerald'] + doi
                download_directory_input = path.join(download_directory, 'emerald')
            elif 'ieee' in publisher.lower():
                url = url_dict['doi'] + doi
                download_directory_input = path.join(download_directory, 'ieee')
            elif 'hindawi' in publisher.lower():
                url = url_dict['doi'] + doi
                download_directory_input = path.join(download_directory, 'hindawi')
            elif 'aaas' in publisher.lower():
                url = url_dict['doi'] + doi
                download_directory_input = path.join(download_directory, 'aaas')
            elif 'royal society of chemistry' in publisher.lower() or 'rsc' in publisher.lower():
                download_url = url_dict['doi'] + doi
                download_directory_input = path.join(download_directory, 'rsc')
                r = requests.get(download_url, headers=headers, timeout=config.TIME_OUT_SEC)
                url = r.url
                url = url.replace('articlelanding', 'articlehtml').replace('ArticleLanding', 'articlehtml')
                sleep(1)


            if 'elsevier' not in publisher.lower(): 
                logger.info(f"Publisher: {publisher.lower()}")
                download_article(doi, publisher, title, year, download_directory_input, db, doc_id, 'html',logger=logger, html_url=url, download_pdf=download_pdf)

        except Exception as e:
            logger.warning(doi + ' with title ' + title + ' of publisher ' + publisher + ' threw a ' + str(e) + ' exception. It will be deleted if already dowloaded.\n')
            logger.exception(e)
            logger.warning('\n')
            if path.exists(path.join(download_directory_input, doi.replace(path.sep, repl_char) + '.html')):
                remove(path.join(download_directory_input, doi.replace(path.sep, repl_char) + '.html'))
            elif path.exists(path.join(download_directory_input, doi.replace(path.sep, repl_char) + '.xml')):
                remove(path.join(download_directory_input, doi.replace(path.sep, repl_char) + '.xml'))
            elif path.exists(path.join(download_directory_input, doi.replace(path.sep, repl_char) + '.pdf')):
                remove(path.join(download_directory_input, doi.replace(path.sep, repl_char) + '.pdf'))
            # utils._update_paper_metadata(db, doc_id, download_status= config.FILE_THREW_EXCEPTION, logger=logger,
            #         note_entry= {"exception": str(e)},
            #         download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT))
            # collection.update_one({'_id': doc_id}, {'$set':{'download_status': config.FILE_THREW_EXCEPTION, 'exception': str(e),
            #                                                 'download_time_stamp': datetime.now().strftime(config.DATE_TIME_FORMAT)}})
            sleep(5)
    logger.info('Cursor exhausted')
    return False, last_processed_id


def _remove_file(download_directory, file_name, title, publisher, doi, logger):
    '''Removes a file from a particular directory and makes a log entry to that effect'''
    remove(path.join(download_directory, file_name))
    logger.warning(title + ' of publisher ' + publisher + ' with DOI ' + doi + ' did not appear to be a scientific article and hence it was deleted\n')


def _validate_html_status(doi, publisher, title, year, download_directory, db, doc_id, html_response, file_name, download_pdf, extension = None, logger = None):
    '''Validates whether the downloaded file is indeed a scientific full text document or abstract or letter or none of these'''
    # We use rules gleaned by studying the html document structure of each publisher
    #logger = logging.getLogger()
    soup = bs.BeautifulSoup(html_response.text, 'html.parser')
    valid_article_keywords = ['abstract', 'introduction', 'conclusions', 'results', 'conclusion', 'definition', 'experimental', 'synthesis', 'result']
    unstructured_download_directory = path.join(download_directory, '..', '..', 'unstructured_files')
    pdf_url_dict = {'wiley': 'https://onlinelibrary.wiley.com/doi/pdf/', 'nature': 'https://www.nature.com/articles/', 'springer': 'https://link.springer.com/content/pdf/', 'acs': 'https://pubs.acs.org/doi/pdf/'}
    # if not path.isdir(unstructured_download_directory):
    #     mkdir(unstructured_download_directory)
    if 'elsevier' in publisher.lower():
        headers = soup.find_all('ce:section-title')
        headers_text = [h.get_text().lower() for h in headers]
        #para = soup.find_all('ce:para')
        #paras_text = [h.get_text() for h in para]
        if len(soup.find_all('xocs:rawtext')) > 0: # This particular keyword appears in all pdf converted to xml in elsevier
            publisher_dir = path.join(unstructured_download_directory, 'elsevier')
            if not path.isdir(publisher_dir):
                mkdir(publisher_dir)
            file_path_original = path.join(download_directory, file_name)
            file_path_final = path.join(publisher_dir, file_name)
            rename(file_path_original, file_path_final)
            # collection.update_one({'_id': doc_id}, {'$set': {'download_status': config.ABSTRACT_DOWNLOAD_SUCCESSFUL,
            #                                                 'download_time_stamp': datetime.now().strftime(config.DATE_TIME_FORMAT)}})
            update_metadata(
    db, doc_id, logger,
    download_status=config.FILE_SIZE_0, extension=extension,
    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT), doi=doi, publisher=publisher, title=title, year=year
)
            # move file to different folder
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            # collection.update_one({'_id': doc_id}, {'$set': {'download_status': config.FULL_DOWNLOAD_SUCCESSFUL,
            #                                                 'download_time_stamp': datetime.now().strftime(config.DATE_TIME_FORMAT)}})
            update_metadata(
    db, doc_id, logger,
    download_status=config.FILE_SIZE_0, extension=extension,
    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT), doi=doi, publisher=publisher, title=title, year=year
)
            pass
        #elif len(paras_text) > 3 or len(paras_text[0]) > 1200:
        #    rename(path.join(download_directory, file_name), path.join(download_directory, 'letter_' + file_name))

        else:
            _remove_file(download_directory, file_name, title, publisher, doi, logger)
            # collection.update_one({'_id': doc_id}, {'$set': {'download_status': config.FILE_DELETED,
            #                                                 'download_time_stamp': datetime.now().strftime(config.DATE_TIME_FORMAT)}})
            update_metadata(
    db, doc_id, logger,
    download_status=config.FILE_SIZE_0, extension=extension,
    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT), doi=doi, publisher=publisher, title=title, year=year
)
            pass

    elif 'wiley' in publisher.lower():
        headers = soup.find_all('h2')
        headers_text = [h.get_text().lower() for h in headers]
        if 'abs' in html_response.url:
            publisher_dir = path.join(unstructured_download_directory, 'wiley')
            url = pdf_url_dict['wiley'] + doi
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
            if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
            update_metadata(
    db, doc_id, logger,
    download_status=config.FILE_SIZE_0, extension=extension,
    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT), doi=doi, publisher=publisher, title=title, year=year
)
        
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            update_metadata(
    db, doc_id, logger,
    download_status=config.FILE_SIZE_0, extension=extension,
    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT), doi=doi, publisher=publisher, title=title, year=year
)
            pass
        else:
            _remove_file(download_directory, file_name, title, publisher, doi,
                         logger)
            update_metadata(
    db, doc_id, logger,
    download_status=config.FILE_SIZE_0, extension=extension,
    note_entry={}, download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT), doi=doi, publisher=publisher, title=title, year=year
)

    elif 'nature' in publisher.lower() and ('10.1038/' in doi or '10.1295/' in doi):
        headers = soup.find_all('h2')
        headers_text = [h.get_text().lower() for h in headers]
        if headers_text[0] == 'abstract' and headers_text[1] == 'references':
            publisher_dir = path.join(unstructured_download_directory, 'nature')
            url = html_response.url + '.pdf'
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
            if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):    # file is ok
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.FULL_DOWNLOAD_SUCCESSFUL, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            pass
        else:
            _remove_file(download_directory, file_name, title, publisher, doi,
                         logger)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.FILE_DELETED, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            
    elif 'springer' in publisher.lower():
        headers = soup.find_all('h2')
        headers_lower = soup.find_all('h3')
        headers_text = [h.get_text().lower() for h in headers]
        headers_lower_text = [h.get_text().lower() for h in headers_lower]
        start_keywords = ['abstract', 'summary', 'definition']
        paragraphs = soup.find_all('p', {'class': 'Para'})
        if (any([kwd in headers_text[0] for kwd in start_keywords]) or any([kwd in headers_lower_text[0] for kwd in start_keywords])) and (((headers_text[1] == 'keywords' or headers_text[1] == 'key words') and ('references' in headers_text[2] or 'references' in headers_text[3] or headers_text[2] == 'copyright information' or headers_text[3] == 'copyright information')) or 'references' in headers_text[1]):
            publisher_dir = path.join(unstructured_download_directory, 'springer')
            url = pdf_url_dict['springer'] + doi
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
            if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.FILE_DELETED, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
        elif not any([start_kwd in head for start_kwd in start_keywords for head in headers_text]) and not any([start_kwd in head for start_kwd in start_keywords for head in headers_lower_text]) and len(paragraphs) > 4:
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'headerless_' + file_name))
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.FULL_DOWNLOAD_SUCCESSFUL, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            pass
        else:
            _remove_file(download_directory, file_name, title, publisher, doi, logger)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.FILE_DELETED, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            
    elif 'acs' in publisher.lower():
        paras = soup.find_all('p')
        paras_text = [h.get_text().lower() for h in paras]
        headers = soup.find_all('h2')
        headers_text = [h.get_text().lower() for h in headers]
        letter_para = soup.find_all('div', {'class' : "NLM_p"})
        if any(['in lieu of an abstract' in header for header in paras_text]):
            publisher_dir = path.join(unstructured_download_directory, 'acs')
            url = pdf_url_dict['acs'] + doi
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
            if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.FULL_DOWNLOAD_SUCCESSFUL, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            sleep(20)
            pass
        elif len(letter_para) > 2:
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'letter_' + file_name))
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.LETTER_DOWNLOAD_SUCCESSFUL, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            #sleep(20)
        else:
            _remove_file(download_directory, file_name, title, publisher, doi,
                         logger)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status= config.FILE_DELETED, 
                                        note_entry= {},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            
    elif 'informa uk' in publisher.lower():
        headers = soup.find_all('h2')
        headers_text = [h.get_text().lower() for h in headers]
        values = soup.find_all('div', {'class': "sectionHeading"})
        description = soup.find_all('meta', {'name': "dc.Description"})
        if values!=[] and 'abstract' in values[0].get_text().lower():
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            # Lack of fulltext means GT does not have access
        elif description != [] and len(description) == 1:
            publisher_dir = path.join(unstructured_download_directory, 'informa_uk')
            url = html_response.url.replace('abs', 'pdf')
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
            if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FULL_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            pass
        else:
            _remove_file(download_directory, file_name, title, publisher, doi,
                         logger)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FILE_DELETED, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)

    # elif 'royal society of chemistry' in publisher.lower() or 'rsc' in publisher.lower():
        # sub_header = soup.find_all('h3')
        # sub_header_text = [h.get_text().lower() for h in sub_header]
        # headers = soup.find_all('h2')
        # headers_text = [h.get_text().lower() for h in headers]
        # abstract = soup.find_all('p', {'class': 'abstract'})
        # if any(['the full text of this document is currently only available in the pdf' in header for header in sub_header_text]):
        #     publisher_dir = path.join(unstructured_download_directory, 'rsc')
        #     url = html_response.url.replace('articlehtml', 'articlepdf')
        #     # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
        #     if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
        #     utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
        #                                 note_entry={},
        #                                 download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
        #                                 extension=extension)

    elif 'royal society of chemistry' in publisher.lower() or 'rsc' in publisher.lower():
        sub_header = soup.find_all('h3')
        sub_header_text = [h.get_text().lower() for h in sub_header]
        headers = soup.find_all('h2')
        headers_text = [h.get_text().lower() for h in headers]
        abstract = soup.find_all('p', {'class': 'abstract'})
        if any(['the full text of this document is currently only available in the pdf' in header for header in sub_header_text]):
            publisher_dir = path.join(unstructured_download_directory, 'rsc')
            url = html_response.url.replace('articlehtml', 'articlepdf')
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
            if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FULL_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            pass
        elif len(abstract) == 1:
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.LETTER_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
        else:
            _remove_file(download_directory, file_name, title, publisher, doi,
                         logger)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FILE_DELETED, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)

    elif 'iop publishing' in publisher.lower():
        headers = soup.find_all('h2')
        headers_text = [h.get_text().lower() for h in headers]
        if 'abstract' in headers_text[0] and (len(headers_text) == 1 or 'references' in headers_text[1]):
            publisher_dir = path.join(unstructured_download_directory, 'iop_publishing')
            # rename(path.join(download_directory, file_name), path.join(download_directory, 'abstract_' + file_name))
            url = html_response.url + '/pdf'
            if download_pdf: download_article(doi, publisher, title, year, publisher_dir, db, doc_id, extension='pdf', html_url=url)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.ABSTRACT_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
        elif any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FULL_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            pass
        else:
            _remove_file(download_directory, file_name, title, publisher, doi,
                         logger)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FILE_DELETED, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)

    elif 'hindawi' in publisher.lower():
        headers = soup.find_all('h4')
        headers_text = [h.get_text().lower() for h in headers]
        if any([keyword in header for header in headers_text for keyword in valid_article_keywords]):
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FULL_DOWNLOAD_SUCCESSFUL, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)
            pass
        else:
            _remove_file(download_directory, file_name, title, publisher, doi,
                         logger)
            utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FILE_DELETED, 
                                        note_entry={},
                                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                        extension=extension)

    elif 'aip publishing' in publisher.lower():
        utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FULL_DOWNLOAD_SUCCESSFUL, 
                                    note_entry={},
                                    download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                    extension=extension)

    elif 'electrochemical society' in publisher.lower():
        utils._update_paper_metadata(db, doc_id, logger=logger, download_status=config.FULL_DOWNLOAD_SUCCESSFUL, 
                                    note_entry={},
                                    download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                                    extension=extension)


def get_acs_session(cookie_path="/home/wschertzer3/AEM_OCR/src/paper-downloader/paper_downloader/acs_cookies.pkl"):
    """Load institutional ACS cookies into a requests.Session."""
    s = requests.Session()
    try:
        with open(cookie_path, "rb") as f:
            cookies = pickle.load(f)
        for c in cookies:
            domain = c.get("domain", ".pubs.acs.org")
            s.cookies.set(c["name"], c["value"], domain=domain)
        # s.headers.update({
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        #                   "AppleWebKit/537.36 (KHTML, like Gecko) "
        #                   "Chrome/120.0.0.0 Safari/537.36",
        #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        #     "Accept-Language": "en-US,en;q=0.5",
        #     "Referer": "https://pubs.acs.org/",
        #     "Connection": "keep-alive"
        # })
        print("[ACS] Loaded cookies and created authenticated session.")
    except Exception as e:
        print(f"[ACS] Warning: Could not load cookies: {e}")
        return requests.Session()
    return s