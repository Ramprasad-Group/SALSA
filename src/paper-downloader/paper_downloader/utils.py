import logging
from paper_downloader import config
from os import walk, path
import time
import json
import pandas as pd
from datetime import datetime
from dateutil import parser
from backend import sett, postgres
from backend.postgres.orm import PaperInfo, Papers, PaperTexts, StructuredPaperTexts, StructuredPaperTexts2
from sqlalchemy import text

def setup_logger(logger_file, logger_name):
    """Boiler plate code to setup a logger"""
    if logger_file is None: return None
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logger_file, filemode='a', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.captureWarnings(True)
    logger = logging.getLogger(logger_name)
    return logger


def filename_to_doi(file_name):
    """Convert file_name to a DOI"""
    doi = file_name.replace('.xml','').replace('.html', '').replace('@', '/')
    return doi


def doi_to_filename(doi, publisher):
    """Convert DOI to a file_name"""
    extension='.html'
    if publisher.lower()=='elsevier' or publisher.lower()=='acs':
        extension='.xml'
    doi = doi.replace('/', '@') + extension
    return doi

def count_files(root_dir, logger=None):
    """Counts the number of files given a root directory and returns a dictionary"""
    total_abstract = 0
    total_fulltext = 0
    total_letter = 0
    for dirName, subdirList, fileList in walk(root_dir):
        file_count = 0
        abstract_count = 0
        letter_count = 0
        if dirName != root_dir:
            message = f'For publisher {dirName.replace(root_dir + path.sep, "")}:'
            logger.info(message) if logger else print(message)
            for file_name in fileList:
                if 'abstract_' in file_name:
                    abstract_count += 1
                elif 'letter_' in file_name:
                    letter_count += 1
                else:
                    file_count += 1
            message = f'    Number of abstract only papers: {abstract_count}'
            logger.info(message) if logger else print(message)
            message = f'    Number of letter papers: {letter_count}'
            logger.info(message) if logger else print(message)
            message = f'    Number of full text papers: {file_count}\n'
            logger.info(message) if logger else print(message)
            total_abstract += abstract_count
            total_fulltext += file_count
            total_letter += letter_count
    message = f'Total number of abstract only papers: {total_abstract}'
    logger.info(message) if logger else print(message)
    message = f'Total number of letter papers: {total_letter}'
    logger.info(message) if logger else print(message)
    message = f'Total number of full text papers: {total_fulltext}'
    logger.info(message) if logger else print(message)
    message = f'Total number of documents: {total_fulltext + total_letter + total_abstract}'
    logger.info(message) if logger else print(message)


def check_download_status(db, time):
    """Checks if the number of faulty downloads in a certain time-window exceeds a threshold and ends the program if that happens"""

    fail_count = db.execute(text('''
        SELECT COUNT(*) FROM papers_info 
        WHERE download_status = :download_failed 
        AND download_time_stamp > :time
    '''), {'download_failed': config.DOWNLOAD_FAILED, 'time': time}).scalar()

    exception_count = db.execute(text('''
        SELECT COUNT(*) FROM papers_info 
        WHERE download_status = :file_threw_exception 
        AND download_time_stamp > :time
    '''), {'file_threw_exception': config.FILE_THREW_EXCEPTION, 'time': time}).scalar()

    if exception_count > 10:
        db.execute(text('''
            UPDATE papers_info 
            SET download_status = NULL, 
                download_time_stamp = NULL, 
                exception = NULL 
            WHERE download_status = :file_threw_exception 
            AND download_time_stamp > :time
            -- AND (note IS NOT NULL AND (note::jsonb ->> 'download_time_stamp') > :time)
        '''), {'file_threw_exception': config.FILE_THREW_EXCEPTION, 'time': time})
        db.commit()
        exception_count = 0

    faulty_docs_count = fail_count + exception_count

    acs_doc_count = db.execute(text('''
        SELECT COUNT(*) FROM papers_info 
        WHERE download_status = :download_failed 
        AND publisher = 'American Chemical Society (ACS)' 
        AND download_time_stamp > :time
    '''), {'download_failed': config.DOWNLOAD_FAILED, 'time': time}).scalar()

    current_time = datetime.now().strftime(config.DATE_TIME_FORMAT)
    current_time = parser.parse(current_time)
    time = parser.parse(time)
    acs_thresh = (current_time-time)*0.05/config.THRESH
    acs_thresh = int(acs_thresh.total_seconds())

    msg = f'HTML error files: {fail_count}, Exception error files: {exception_count}, ACS failed files: {acs_doc_count}'
    if faulty_docs_count > config.THRESH or acs_doc_count > acs_thresh:
        return False, msg
    else:
        return True, msg


def code_termination(prog_name, logger=None):
    '''Boilerplate code to execute when a script terminates'''
    current_time = datetime.now().strftime(config.DATE_TIME_FORMAT)
    message = f'Execution of {prog_name} has completed at {current_time}'
    if logger: logger.warning(message)
    print(message)
    return message


def _check_membership(term, term_list):
    return any([item in term for item in term_list])


def formatted_time():
    return datetime.now().strftime(config.DATE_TIME_FORMAT)


def parsed_output_processor(output_dict, remove_end_matter=False):
    '''Takes LimeSoup's parsed json object and processes the output further'''
    # If there are dictionaries with garbage in the beginning, those might have to be deleted

    abstract_keywords = ['abstract', 'summary']
    remove_keywords = ['copyright', 'acknowledgement', 'conflicts of interest', 'data availability', 'conflict of interest',
                       'additional information', 'supplementary', 'citation', 'authors\' contributions',
                       'competing interest', 'password', 'username', 'author contributions', 'abstract', 'keywords',
                       'figures', 'disclosure statement', 'funding', 'related research data' 'supplemental data']
    abstract = ''
    delete_index = -1
    keywords = []
    if type(output_dict['Sections'])==list:
        for i, item_dict in enumerate(output_dict['Sections']):
            if type(item_dict) is dict and _check_membership(item_dict['name'].lower(), abstract_keywords):
                abstract = item_dict['content'] # abstract is a list
                delete_index = i
                break
            elif type(item_dict) is list and type(item_dict[0]) is str and _check_membership(item_dict[0], abstract_keywords):
                abstract = item_dict
                delete_index = i
                break
    elif type(output_dict['Sections']) is dict:
        if _check_membership(output_dict['Sections']['name'].lower(), abstract_keywords):
            abstract = output_dict['Sections']['content']
            output_dict['Sections']=[]

    if delete_index != -1:
        del output_dict['Sections'][delete_index]
    if abstract and type(abstract) is list:
        # Cleanup references to keywords and the word abstract in the abstract, assuming here that abstract is a list
        if type(abstract[-1])==str and abstract[-1].lower().startswith('keyword'):
            keywords = abstract[-1].lower().replace('keywords: ', '').split(',') # Assumes that keywords will always be found on the last line if at all
            abstract = abstract[:-1]
        elif len(abstract)>=2 and type(abstract[-2])==str and abstract[-2].lower().startswith('keyword'):
            keywords = abstract[-1].lower().replace('keywords: ', '').split(
                ',')  # Assumes that keywords will always be found on the last line if at all
            abstract = abstract[:-2]

        if len(abstract)>=2 and type(abstract[0])==str and type(abstract[1])==str and _check_membership(abstract[0].lower(), abstract_keywords) and _check_membership(abstract[1].lower(), abstract_keywords):
            abstract=abstract[2:]
        elif len(abstract)>=1 and type(abstract[0])==str and _check_membership(abstract[0].lower(), abstract_keywords):
            abstract=abstract[1:]

        output_dict['Abstract'] = abstract
    else:
        output_dict['Abstract'] = []

    if remove_end_matter:
        remove_indices = []
        if type(output_dict['Sections']) is list:
            for i, item_dict in enumerate(output_dict['Sections']):
                if type(item_dict) is dict and _check_membership(item_dict['name'].lower(), remove_keywords):
                    remove_indices.append(i)

        for index in sorted(remove_indices, reverse=True):
            del output_dict['Sections'][index]

    return output_dict, keywords


def create_file(output_dir, output_file):
    if not path.exists(path.join(output_dir, output_file)):
        open(path.join(output_dir, output_file), 'a').close()


def read_file(file_name, separator='\n'):
    """Reads items from a text file and returns a list"""

    with open(file_name, 'r') as f:
        item_list = f.read().split(separator)
    if item_list[-1] == '':
        item_list = item_list[:-1]
    return item_list


def download_gap_time(last_download_time, publisher):
    current_time = time.time()
    if current_time - last_download_time >= config.publisher_gap_time[publisher]:
        last_download_time = current_time
        continue_status = False
    else:
        continue_status = True
    return continue_status, last_download_time


def connect_client(db_name, user='superadmin', pwd='oNTIopSYPoCE'):
    """Connects to the client as superadmin"""
    client = MongoClient('localhost', 8161)
    client.admin.authenticate(user, pwd, source=db_name)
    return client


def connect_database(user='admin', pwd='EntyWeSTEREc'):
    """Connects to the database containing the text version of all our data"""
    db_name = 'polymer_records'
    client = MongoClient('localhost', 8161)
    client.admin.authenticate(user, pwd, source=db_name)
    db = client[db_name]
    return db

# def pick_collection(db, debug):
#     if debug:
#         collection = db['polymer_DOI_records_dev']
#         print('Using dev collection')

#     else:
#         collection = db['polymer_DOI_records_prod']
#         print('Using prod collection')

#     return collection


def execute(sql, **kwargs) -> pd.DataFrame:
    """ Query the database using raw sql.
        Return a pandas dataframe containing the results.
    """
    results = postgres.raw_sql(sql, kwargs)
    if not results:
        return None
    return pd.DataFrame.from_records(results, columns=results[0]._fields)


def _update_paper_metadata(db, doc_id, logger, note_entry, download_time_stamp, download_status=None, extension=None):
    """
    Update metadata for a paper in the database.

    Args:
        db: Database connection object
        doc_id (int): paper_info record ID
        note_entry (dict): New note entries to add/update
        download_status (int, optional): New download status
        doc_type (str, optional): New document type

    Raises:
        ValueError: If the paper or paper info is not found in the database
    """
    existing_info = PaperInfo().get_one(db, criteria={"id": int(doc_id)})
    if not existing_info:
        raise ValueError(f"Paper info not found for doc_id: {doc_id}")

    if extension:
        existing_paper = Papers().get_one(db, criteria={"id": int(existing_info.pid)})
        if not existing_paper:
            raise ValueError(f"Paper not found for pid: {existing_info.pid}")
    
        existing_paper.doctype = str(extension)
        logger.info(f"Extension for PID: {existing_info.pid} = {extension}")

    if download_status is not None:
        existing_info.download_status = int(download_status)

    if note_entry != {}:
        if existing_info.note:
            existing_note = json.loads(existing_info.note)
            existing_note.update(note_entry)
        else:
            existing_note = note_entry
        
        logger.info(f"Note entry: {existing_note}")
        existing_info.note = json.dumps(existing_note)

    existing_info.download_time_stamp = download_time_stamp
    
    try:
        db.commit()
        logger.info(f"Updated information for pi_ID: {doc_id} PID: {existing_info.pid}, Download Status: {download_status}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update paper metadata for doc_id: {doc_id}. Error: {str(e)}")
        raise


def _update_paper_info(db, pid, note_entry):
    existing_info = PaperInfo().get_one(db, criteria={"pid": int(pid)})
    if existing_info:
        if not existing_info.note:
            existing_info.note = json.dumps(note_entry)
        else:
            existing_note = json.loads(existing_info.note) if existing_info.note else {}
            existing_note.update({
                "year": note_entry["year"] if note_entry["year"] is not None else "None",
                "month": note_entry["month"] if note_entry["month"] is not None else "None",
                "day": note_entry["day"] if note_entry["day"] is not None else "None",
                "citations": note_entry["citations"] if note_entry["citations"] is not None else "None"
            })
            existing_info.note = json.dumps(existing_note)
            existing_info.pub_year = int(note_entry['year']) if note_entry['year'] is not None else None
        try: 
            db.commit()
        except Exception as e:
            db.rollback()
            logging.warning(f"Failed to update paper info (DOI: {existing_info.doi}): {e}")
    else:
        raise ValueError(f"Paper info not found for PID: {pid}")

def _add_to_papers(db, data_entry, directory):
    obj = Papers()
    obj.doi = data_entry['DOI']
    obj.publisher = data_entry['publisher']
    obj.title = data_entry['title']
    obj.doi = data_entry['DOI']
    obj.directory = directory
    obj.doctype = 'None'
    try:
        obj.insert(db)
        db.commit()
        # logging.info(f"Added DOI: {data_entry['DOI']} to papers table.")

    except Exception as e:
        db.rollback()
        logging.warning(f"Failed to insert paper (DOI: {data_entry['DOI']}): {e}")
        

def _add_to_papers_info(db, pid, data_entry):
    obj = PaperInfo()
    obj.pid = pid
    obj.doi = data_entry['DOI']
    obj.pub_year = data_entry['year']
    obj.publisher = data_entry['publisher']
    obj.journal = data_entry['journal']
    obj.issn = data_entry['issn']
    obj.note = json.dumps({
        "year": str(data_entry["year"]) if data_entry["year"] is not None else "None",  
        "month": str(data_entry["month"]) if data_entry["month"] is not None else "None",
        "day": str(data_entry["day"]) if data_entry["day"] is not None else "None",
        "citations": str(data_entry["citations"]) if data_entry["citations"] is not None else "None"
    })
        # "reference_list": data_entry["reference_list"] if data_entry["reference_list"] is not None else "None",
        # "author_list": data_entry["author_list"] if data_entry["author_list"] is not None else "None",
        # "funder_list": data_entry["funder_list"] if data_entry["funder_list"] is not None else "None"} 
        
    try:
        obj.insert(db)
        db.commit()
        # logging.info(f"Added DOI: {data_entry['DOI']} to papers_info table.")
    
    except Exception as e:
        db.rollback()
        logging.warning(f"Failed to insert paper info (DOI: {data_entry['DOI']}): {e}")


def _add_to_paper_texts(db, parsed_output, paper):
    def insert_section_content(section, paper, db, parent_name=None, section_count=None):
        """
        Recursively inserts section content into the database.
        Handles all levels of nested sections (h2, h3, h4, etc.).
        """

        # Extract the current section's name
        if isinstance(section, dict):
            current_name = f"{parent_name} - {section['name']}" if parent_name else section['name']
            section_type = section['type']
            logging.info(f"Section Type: {section['type']}, Name: {current_name}")
            content = section.get('content', [])
        elif isinstance(section, str):
            current_name = parent_name
            section_type = 'text'
            logging.info(f"Section Type: {section_type}, Name: {current_name}")
            content = section
        else:
            logging.warning(f"Unknown section type: {section}")
            return

        # if isinstance(content, list) and all(isinstance(item, dict) for item in content):
        #     # If content is a list of dictionaries, process each nested subsection
        #     for item in content:
        #         insert_section_content(item, paper, db, parent_name=current_name, section_count=section_count)

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Recurse into nested section
                    insert_section_content(item, paper, db, parent_name=current_name, section_count=section_count)
                elif isinstance(item, str):
                    # Insert individual paragraph
                    # obj = PaperTexts()
                    obj = StructuredPaperTexts2()
                    obj.pid = paper.id
                    obj.doi = paper.doi
                    obj.doctype = paper.doctype
                    obj.directory = paper.directory
                    obj.tag = section_type
                    obj.section = current_name
                    obj.text = item.strip()
                    try:
                        obj.insert(db)
                        section_count[0] += 1
                    except Exception as e:
                        db.rollback()
                        logging.warning(f"Failed to insert paragraph (PID: {paper.id}, Section: {current_name}): {e}")
                else:
                    logging.warning(f"Unknown item type in content list: {type(item)}")

        elif isinstance(content, str):
            # obj = PaperTexts()
            obj = StructuredPaperTexts2()  
            obj.pid = paper.id
            obj.doi = paper.doi
            obj.doctype = paper.doctype
            obj.directory = paper.directory
            obj.tag = section_type
            obj.section = current_name
            obj.text = content.strip()
            try:
                obj.insert(db)
                section_count[0] += 1
            except Exception as e:
                db.rollback()
                logging.warning(f"Failed to insert paper text (PID: {paper.id}, Section: {current_name}): {e}")
        else:
            logging.warning(f"Unhandled content type: {type(content)} at section: {current_name}")
            # obj = PaperTexts()
            obj = StructuredPaperTexts2()
            obj.pid = paper.id
            obj.doi = paper.doi
            obj.doctype = paper.doctype
            obj.directory = paper.directory
            obj.tag = section_type
            obj.section = current_name
            obj.text = json.dumps(content)  # Fallback
            try:
                obj.insert(db)
                section_count[0] += 1
            except Exception as e:
                db.rollback()
                logging.warning(f"Failed to insert fallback content (PID: {paper.id}, Section: {current_name}): {e}")

    # Main logic to process all top-level sections
    section_count = [0]  # Using list to allow modification in nested function
    for section in parsed_output['Sections']:
        try:
            insert_section_content(section, paper, db, section_count=section_count)
        except Exception as e:
            db.rollback()
            section_name = section.get('name') if isinstance(section, dict) else str(section)[:50]
            logging.warning(f"Error processing section (PID: {paper.id}, Section: {section_name}): {e}")

    try:
        db.commit()
        logging.info(f"Successfully added {section_count[0]} sections for PID: {paper.id} to StructuredPaperTexts2")
    except Exception as e:
        db.rollback()
        logging.warning(f"Failed to commit paper text (PID: {paper.id}): {e}")