#from numpy import isnan
import requests
from dataclasses import replace
from os import path, mkdir, rename, remove, stat
import os
import sys
#import logging
from random import randint
from time import sleep
import time
import bs4 as bs
from paper_downloader import utils, config
from paper_downloader.publisher_access import publisher_access_for
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import quote
# from email.mime.text import MIMEText

_SRC_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
from ui.run_progress import write_current_doi

MAX_SLEEP_TIME = 7
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
OPENALEX_CONTENT_ENDPOINT = "https://content.openalex.org/works"
# FULL_DOWNLOAD_SUCCESSFUL = 1
# ABSTRACT_DOWNLOAD_SUCCESSFUL = 2
# LETTER_DOWNLOAD_SUCCESSFUL = 3
# FILE_DELETED = 4
# DOWNLOAD_FAILED = 5
# FILE_SIZE_0 = 6
# FILE_THREW_EXCEPTION = 7


class PublisherAccessError(Exception):
    """A publisher API rejected or could not fulfill an access request."""


def _append_jsonl(path_str: str, obj: dict):
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _try_openalex_pdf(doi, unstructured_download_directory, logger, timeout_sec=90):
    """Download OpenAlex's cached PDF before trying credentialed publishers."""
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if not api_key:
        return False
    # OpenAlex requires the key in the query string. Prevent HTTP debug logging
    # from writing credential-bearing request URLs to pipeline logs.
    import logging
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    output_dir = Path(unstructured_download_directory) / "openalex"
    output_path = output_dir / f"{doi.replace(path.sep, '@')}.pdf"
    if _elsevier_pdf_is_complete(output_path, {}):
        logger.info("Using cached OpenAlex PDF for DOI %s", doi)
        return True
    if output_path.exists():
        output_path.unlink()

    try:
        metadata_response = requests.get(
            f"{OPENALEX_WORKS_ENDPOINT}/doi:{quote(doi, safe='/')}",
            params={
                "api_key": api_key,
                "select": "id,content_urls,has_content,best_oa_location",
            },
            headers={"Accept": "application/json"},
            timeout=timeout_sec,
        )
        if metadata_response.status_code != 200:
            metadata_response.close()
            logger.info("OpenAlex has no work metadata for DOI %s", doi)
            return False
        work = metadata_response.json()
        metadata_response.close()
        has_pdf = bool((work.get("has_content") or {}).get("pdf"))
        if not has_pdf:
            logger.info("OpenAlex has no cached PDF for DOI %s", doi)
            return False
        work_id = str(work.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
        content_urls = work.get("content_urls") or {}
        pdf_url = (
            str(content_urls.get("pdf") or "").strip()
            if isinstance(content_urls, dict)
            else ""
        )
        # Backward compatibility with the earlier OpenAlex response shape.
        if not pdf_url:
            content_url = str(work.get("content_url") or "").strip()
            pdf_url = content_url if content_url.endswith(".pdf") else ""
        if not pdf_url:
            best_oa_location = work.get("best_oa_location") or {}
            if isinstance(best_oa_location, dict):
                pdf_url = str(best_oa_location.get("pdf_url") or "").strip()
        if not pdf_url and work_id.startswith("W"):
            pdf_url = f"{OPENALEX_CONTENT_ENDPOINT}/{work_id}.pdf"
        if not pdf_url:
            return False

        pdf_response = requests.get(
            pdf_url,
            stream=True,
            params={"api_key": api_key},
            headers={"Accept": "application/pdf"},
            timeout=timeout_sec,
        )
        content_type = pdf_response.headers.get("Content-Type", "").lower()
        if pdf_response.status_code != 200:
            pdf_response.close()
            logger.info(
                "OpenAlex cached PDF download failed for DOI %s: HTTP %s",
                doi,
                pdf_response.status_code,
            )
            return False
        content = pdf_response.content
        pdf_response.close()
        if not content.startswith(b"%PDF-"):
            logger.info(
                "OpenAlex returned non-PDF content for DOI %s (%s)",
                doi,
                content_type,
            )
            return False
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_suffix(".pdf.part")
        partial_path.write_bytes(content)
        partial_path.replace(output_path)
        logger.info("Downloaded DOI %s from OpenAlex cached full text", doi)
        return True
    except (requests.RequestException, ValueError, OSError) as exc:
        logger.info("OpenAlex full-text attempt failed for DOI %s: %s", doi, exc)
        return False


def _xml_local_name(tag):
    return str(tag).rsplit("}", 1)[-1]


def _validate_response_content(content, validator):
    if not validator:
        return True, {}

    if validator == "iop_full_text_html":
        content_lower = content.lower()
        blocked_markers = (
            b"radware bot manager captcha",
            b"validate.perfdrive.com",
            b"confirm you are a human",
        )
        if any(marker in content_lower for marker in blocked_markers):
            return False, {
                "reason": "iop_bot_challenge",
                "message": (
                    "IOP returned an automated-access challenge instead of article "
                    "full text. Arrange TDM delivery with contentsupport@ioppublishing.org."
                ),
            }
        has_article_metadata = (
            b'name="citation_doi"' in content_lower
            and b'name="citation_fulltext_html_url"' in content_lower
        )
        has_article_body = (
            b"<main" in content_lower
            and len(content) >= 20_000
        )
        if not (has_article_metadata and has_article_body):
            return False, {
                "reason": "iop_incomplete_full_text",
                "message": "IOP did not return a complete full-text HTML article.",
            }
        return True, {}

    if validator == "aps_open_access_pdf":
        if content.startswith(b"%PDF-"):
            return True, {}
        return False, {
            "reason": "aps_invalid_open_access_response",
            "message": "APS Harvest did not return a PDF article.",
        }

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        return False, {
            "reason": "invalid_xml",
            "message": str(exc),
        }

    if validator == "springer_open_access_jats":
        records = next(
            (element for element in root.iter() if _xml_local_name(element.tag) == "records"),
            None,
        )
        has_article = records is not None and any(
            _xml_local_name(element.tag) == "article"
            for element in records.iter()
        )
        if not has_article:
            return False, {
                "reason": "springer_no_open_access_record",
                "message": "Springer Nature returned no open-access JATS article for this DOI.",
            }

    if validator == "aps_open_access_jats" and _xml_local_name(root.tag) != "article":
        return False, {
            "reason": "aps_invalid_open_access_response",
            "message": "APS Harvest did not return a JATS article.",
        }

    return True, {}


def _xml_has_local_name(content, local_name):
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return False
    return any(_xml_local_name(element.tag) == local_name for element in root.iter())


def _elsevier_pdf_attachment(content):
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return {}

    for element in root.iter():
        if _xml_local_name(element.tag) != "web-pdf":
            continue
        values = {}
        for child in list(element):
            name = _xml_local_name(child.tag)
            text = (child.text or "").strip()
            if name in {
                "attachment-eid",
                "filesize",
                "web-pdf-page-count",
            } and text:
                values[name] = text
        if values.get("attachment-eid"):
            try:
                filesize = int(values.get("filesize") or 0)
            except ValueError:
                filesize = 0
            try:
                page_count = int(values.get("web-pdf-page-count") or 0)
            except ValueError:
                page_count = 0
            return {
                "eid": values["attachment-eid"],
                "filesize": filesize,
                "page_count": page_count,
            }
    return {}


def _elsevier_needs_pdf(content):
    return not _xml_has_local_name(content, "body")


def _elsevier_pdf_is_complete(pdf_path, attachment):
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return False
    expected_size = int(attachment.get("filesize") or 0)
    if expected_size and pdf_path.stat().st_size < int(expected_size * 0.9):
        return False
    try:
        with pdf_path.open("rb") as pdf_file:
            return pdf_file.read(5) == b"%PDF-"
    except OSError:
        return False


def _download_elsevier_pdf(
    *,
    doi,
    xml_content,
    xml_url,
    xml_headers,
    request_params,
    download_directory,
    timeout_sec,
    logger,
):
    papers_dir = Path(download_directory).parent.parent
    pdf_dir = papers_dir / "unstructured_files" / "elsevier"
    pdf_path = pdf_dir / f"{doi.replace(path.sep, '@')}.pdf"
    attachment = _elsevier_pdf_attachment(xml_content)
    if _elsevier_pdf_is_complete(pdf_path, attachment):
        return pdf_path, {}
    if pdf_path.exists():
        logger.warning(
            "Removing incomplete cached Elsevier PDF for DOI %s (%s bytes; expected about %s bytes across %s pages)",
            doi,
            pdf_path.stat().st_size,
            attachment.get("filesize") or "unknown",
            attachment.get("page_count") or "unknown",
        )
        pdf_path.unlink()

    pdf_headers = dict(xml_headers)
    pdf_headers["Accept"] = "application/pdf"
    attachment_eid = attachment.get("eid")
    if attachment_eid:
        pdf_url = f"https://api.elsevier.com/content/object/eid/{attachment_eid}"
        pdf_params = {"view": "STANDARD"}
    else:
        pdf_url = xml_url
        pdf_params = request_params
    response = requests.get(
        pdf_url,
        stream=True,
        headers=pdf_headers,
        params=pdf_params,
        timeout=timeout_sec,
    )
    content_type = response.headers.get("Content-Type", "").lower()
    if response.status_code != 200 or "pdf" not in content_type:
        status_code = response.status_code
        response.close()
        logger.warning(
            "Elsevier PDF retrieval failed for DOI %s with HTTP %s (%s)",
            doi,
            status_code,
            content_type or "no Content-Type",
        )
        return None, {
            "reason": "elsevier_pdf_download_failed",
            "status_code": status_code,
            "content_type": content_type,
            "message": "Elsevier returned no structured body and the full PDF attachment could not be retrieved.",
        }

    pdf_dir.mkdir(parents=True, exist_ok=True)
    partial_path = pdf_path.with_suffix(pdf_path.suffix + ".part")
    with partial_path.open("wb") as current_file:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                current_file.write(chunk)
    response.close()

    if not _elsevier_pdf_is_complete(partial_path, attachment):
        actual_size = partial_path.stat().st_size if partial_path.exists() else 0
        if partial_path.exists():
            partial_path.unlink()
        return None, {
            "reason": "elsevier_pdf_incomplete",
            "expected_size": attachment.get("filesize") or None,
            "expected_pages": attachment.get("page_count") or None,
            "actual_size": actual_size,
            "message": "Elsevier returned an incomplete PDF attachment.",
        }
    partial_path.replace(pdf_path)
    logger.info("Downloaded Elsevier PDF fallback for DOI %s to %s", doi, pdf_path)
    return pdf_path, {}


def _resolve_access_plan(plan, doi, timeout_sec, logger):
    """Resolve APIs that require an identifier lookup before full-text access."""
    if plan.strategy == "direct":
        return plan

    response = requests.get(
        plan.url,
        headers=plan.headers,
        params=plan.params,
        timeout=timeout_sec,
    )
    if response.status_code >= 400:
        status_code = response.status_code
        response.close()
        if plan.strategy == "ieee_by_doi":
            raise PublisherAccessError(
                f"IEEE Xplore metadata lookup returned HTTP {status_code}. "
                "Verify that IEEE_API_KEY is current and authorized for the IEEE Xplore API."
            )
        response.raise_for_status()
    payload = response.json()
    response.close()

    if plan.strategy == "pmc_by_doi":
        records = payload.get("records") or []
        pmcid = records[0].get("pmcid") if records else ""
        if not pmcid:
            raise ValueError(f"DOI {doi} is not available in PubMed Central.")
        numeric_pmcid = str(pmcid).upper()
        if numeric_pmcid.startswith("PMC"):
            numeric_pmcid = numeric_pmcid[3:]
        return replace(
            plan,
            url="https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/",
            params={
                "verb": "GetRecord",
                "identifier": f"oai:pubmedcentral.nih.gov:{numeric_pmcid}",
                "metadataPrefix": "pmc",
            },
            headers={"Accept": "application/xml"},
            strategy="direct",
        )

    if plan.strategy == "ieee_by_doi":
        articles = payload.get("articles") or []
        article_number = articles[0].get("article_number") if articles else ""
        if not article_number:
            raise ValueError(f"IEEE did not return an article number for DOI {doi}.")
        api_key = plan.params["apikey"]
        return replace(
            plan,
            url=f"https://ieeexploreapi.ieee.org/api/v1/search/document/{article_number}/fulltext",
            params={"apikey": api_key, "format": "xml"},
            headers={"Accept": "application/xml"},
            strategy="direct",
        )

    if plan.strategy == "iop_crossref_tdm":
        message = payload.get("message") or {}
        links = message.get("link") or []
        text_mining_html = [
            link
            for link in links
            if link.get("intended-application") == "text-mining"
            and link.get("content-type") == "text/html"
        ]
        version_of_record = [
            link
            for link in text_mining_html
            if link.get("content-version") == "vor"
        ]
        candidates = version_of_record or text_mining_html
        article_url = str(candidates[0].get("URL") or "") if candidates else ""
        if not article_url:
            raise ValueError(
                f"Crossref did not publish an IOP text-mining HTML link for DOI {doi}."
            )
        if article_url.startswith("http://"):
            article_url = "https://" + article_url[len("http://"):]
        contact_email = os.environ.get(
            "SALSA_CONTACT_EMAIL",
            "wschertzer3@gatech.edu",
        )
        return replace(
            plan,
            url=article_url,
            params={},
            headers={
                "Accept": "text/html",
                "User-Agent": f"SALSA/1.0 (mailto:{contact_email})",
            },
            strategy="direct",
        )

    logger.warning("Unknown publisher access strategy %s for DOI %s", plan.strategy, doi)
    raise ValueError(f"Unknown publisher access strategy: {plan.strategy}")

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
    utils._update_paper_metadata(
        db,
        doc_id,
        logger=logger,
        download_status=download_status,
        extension=extension,
        note_entry=note_entry or {},
        download_time_stamp=download_time_stamp,
    )

def download_article(doi, publisher, title, year, download_directory, db, doc_id, extension, logger=None, timeout_sec = 90, html_url=None, download_pdf=True, debug=False, local_status_log=None, extra_headers=None, request_params=None, expected_content_type="", response_validator=""):
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

    if path.exists(file_location) and response_validator:
        with open(file_location, "rb") as existing_file:
            existing_content = existing_file.read()
        is_valid, validation_note = _validate_response_content(
            existing_content,
            response_validator,
        )
        if not is_valid:
            logger.warning(
                "Removing invalid cached %s response for DOI %s: %s",
                publisher,
                doi,
                validation_note["message"],
            )
            remove(file_location)
        elif (
            response_validator == "elsevier_full_text_or_pdf"
            and _elsevier_needs_pdf(existing_content)
        ):
            cached_headers = {
                "Accept": header_prefix_accept[extension] + "/" + extension,
                "User-agent": "Mozilla/5.6 (Windows NT 11.0; rv:68.0 Gecko/20100101 Firefox/70.0",
            }
            if extra_headers:
                cached_headers.update(extra_headers)
            pdf_path, pdf_failure = _download_elsevier_pdf(
                doi=doi,
                xml_content=existing_content,
                xml_url=html_url,
                xml_headers=cached_headers,
                request_params=request_params,
                download_directory=download_directory,
                timeout_sec=timeout_sec,
                logger=logger,
            )
            if pdf_path is None:
                update_metadata(
                    db, doc_id, logger,
                    download_status=config.DOWNLOAD_FAILED,
                    extension="pdf",
                    note_entry=pdf_failure,
                    download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                    local_status_log=local_status_log,
                    doi=doi,
                    publisher=publisher,
                    title=title,
                    year=year,
                )
                return False
    
    # Check if html_url is valid
    if not html_url or not html_url.startswith("http"):
        logger.warning(f"Invalid API URL for DOI {doi}: {html_url}. Skipping.")
        return False

    #Download the file if it doesn't exist
    if not path.exists(file_location):
        headers = {
            'Accept': header_prefix_accept[extension] + '/' + extension,
            'User-agent': 'Mozilla/5.6 (Windows NT 11.0; rv:68.0 Gecko/20100101 Firefox/70.0'
        }
        if extra_headers:
            headers.update(extra_headers)

        html_data = requests.get(
            html_url,
            stream=True,
            headers=headers,
            params=request_params,
            timeout=timeout_sec,
        )

        #Process the response
        if html_data.status_code == 200:
            content_type = html_data.headers.get("Content-Type", "").lower()
            if expected_content_type and expected_content_type not in content_type:
                logger.warning(
                    "Rejected %s response for DOI %s: expected %s content, received %s",
                    publisher,
                    doi,
                    expected_content_type,
                    content_type or "no Content-Type",
                )
                update_metadata(
                    db, doc_id, logger,
                    download_status=config.DOWNLOAD_FAILED, extension=extension,
                    note_entry={"reason": "unexpected_content_type", "content_type": content_type},
                    download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                    local_status_log=local_status_log, doi=doi, publisher=publisher, title=title, year=year
                )
                html_data.close()
                return False
            if response_validator:
                content = html_data.content
                is_valid, validation_note = _validate_response_content(
                    content,
                    response_validator,
                )
                if not is_valid:
                    logger.warning(
                        "Rejected %s response for DOI %s: %s",
                        publisher,
                        doi,
                        validation_note["message"],
                    )
                    update_metadata(
                        db, doc_id, logger,
                        download_status=config.DOWNLOAD_FAILED,
                        extension=extension,
                        note_entry=validation_note,
                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                        local_status_log=local_status_log,
                        doi=doi,
                        publisher=publisher,
                        title=title,
                        year=year,
                    )
                    html_data.close()
                    return False
                with open(file_location, "wb") as current_file:
                    current_file.write(content)
            else:
                with open(file_location, 'wb') as current_file:
                    for chunk in html_data.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            current_file.write(chunk)
            html_data.close()

            if (
                response_validator == "elsevier_full_text_or_pdf"
                and _elsevier_needs_pdf(content)
            ):
                pdf_path, pdf_failure = _download_elsevier_pdf(
                    doi=doi,
                    xml_content=content,
                    xml_url=html_url,
                    xml_headers=headers,
                    request_params=request_params,
                    download_directory=download_directory,
                    timeout_sec=timeout_sec,
                    logger=logger,
                )
                if pdf_path is None:
                    update_metadata(
                        db, doc_id, logger,
                        download_status=config.DOWNLOAD_FAILED,
                        extension="pdf",
                        note_entry=pdf_failure,
                        download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                        local_status_log=local_status_log,
                        doi=doi,
                        publisher=publisher,
                        title=title,
                        year=year,
                    )
                    return False
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
                return False
            
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
            if not debug:
                if (
                    response_validator in {
                        "aps_open_access_jats",
                        "aps_open_access_pdf",
                    }
                    and html_data.status_code in (401, 403)
                ):
                    note_entry = {
                        "reason": "aps_not_open_access",
                        "status_code": html_data.status_code,
                        "message": (
                            "APS full text is access restricted for this DOI "
                            f"(HTTP {html_data.status_code})."
                        ),
                    }
                elif html_data.status_code in (401, 403, 404):
                    note_entry = {
                        "reason": "access_restricted",
                        "status_code": html_data.status_code,
                        "message": (
                            f"{publisher} full text is access restricted or unavailable "
                            f"through its configured automated-access API "
                            f"(HTTP {html_data.status_code})."
                        ),
                    }
                else:
                    note_entry = {
                        "reason": "http_error",
                        "status_code": html_data.status_code,
                    }
                update_metadata(
                    db, doc_id, logger,
                    download_status=config.DOWNLOAD_FAILED, extension=extension,
                    note_entry=note_entry,
                    download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                    local_status_log=local_status_log, doi=doi, publisher=publisher, title=title, year=year
                )
            if year is not None:
                logger.warning(
                    "Could not download %s file with DOI %s by publisher %s with title "
                    "'%s' published in %s. Failed with HTTP status code %s; URL: %s",
                    extension, doi, publisher, title, year, html_data.status_code, html_url,
                )
            else:
                logger.warning(
                    "Could not download %s file with DOI %s by publisher %s with title "
                    "'%s'. Failed with HTTP status code %s; URL: %s",
                    extension, doi, publisher, title, html_data.status_code, html_url,
                )
            html_data.close()
            return False
        random_wait_time = randint(1, MAX_SLEEP_TIME) # Choose any value between 1 & 2 seconds randomly
        sleep(random_wait_time)
        return True
        # if extension != 'pdf' and not debug:_validate_html_status(doi, publisher, title, year, download_directory, db, doc_id, html_data, file_name, download_pdf, extension, logger=logger)
    return True


# Assume that query has been run on database outside of article_downloader
# TODO: Put doi, publisher, title and year in a dict object and pass that around to other functions.
# Change code accordingly

def article_downloader(db, query, download_directory, year_lower, logger_file=None, logger = None, logger_name='downloader',
                        sample_size=None, download_pdf=True, scanned_records=0, last_processed_id=None, records_df=None, local_status_log=None, **kwargs):
    '''
    Downloads full text only through configured publisher API/TDM endpoints.

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
    papers_root = download_directory
    if not path.isdir(path.join(papers_root, 'structured_files')):
        mkdir(path.join(papers_root, 'structured_files'))

    # pdf files are saved in the below directory
    if not path.isdir(path.join(papers_root, 'unstructured_files')):
        mkdir(path.join(papers_root, 'unstructured_files'))
    structured_download_directory = path.join(papers_root, 'structured_files')
    unstructured_download_directory = path.join(papers_root, 'unstructured_files')
    download_directory_input = ''
    i = 0
    repl_char = '@'
    last_download = {publisher: time.time()-config.publisher_gap_time[publisher]
                     for publisher in config.publisher_gap_time.keys()}
    download_status_check_time = datetime.now().strftime(config.DATE_TIME_FORMAT)

    if not logger:
        if not path.exists(logger_file):
            open(logger_file, 'a').close()
        logger = utils.setup_logger(logger_file, logger_name)

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
        write_current_doi(doi)
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
            if _try_openalex_pdf(doi, unstructured_download_directory, logger):
                update_metadata(
                    db, doc_id, logger,
                    download_status=config.FULL_DOWNLOAD_SUCCESSFUL,
                    extension="pdf",
                    note_entry={"provider": "openalex", "openalex_cached_full_text": True},
                    download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                    local_status_log=local_status_log,
                    doi=doi,
                    publisher=publisher,
                    title=title,
                    year=year,
                )
                continue

            access = publisher_access_for(doi, publisher)
            if access.plan is None:
                logger.warning("Skipping DOI %s: %s", doi, access.reason)
                update_metadata(
                    db, doc_id, logger,
                    download_status=config.DOWNLOAD_FAILED,
                    extension="",
                    note_entry={"reason": access.reason},
                    download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                    local_status_log=local_status_log,
                    doi=doi,
                    publisher=publisher,
                    title=title,
                    year=year,
                )
                continue

            plan = access.plan
            plan = _resolve_access_plan(plan, doi, config.TIME_OUT_SEC, logger)
            if plan.directory in last_download:
                continue_status, last_download['wiley'] = utils.download_gap_time(last_download['wiley'], 'wiley')
                if continue_status:
                    continue

            download_root = (
                unstructured_download_directory
                if plan.extension == "pdf"
                else structured_download_directory
            )
            download_directory_input = path.join(download_root, plan.directory)
            logger.info("Using %s for DOI %s", plan.provider, doi)
            download_article(
                doi,
                publisher,
                title,
                year,
                download_directory_input,
                db,
                doc_id,
                plan.extension,
                logger=logger,
                download_pdf=download_pdf,
                html_url=plan.url,
                extra_headers=plan.headers,
                request_params=plan.params,
                expected_content_type=plan.expected_content_type,
                response_validator=plan.response_validator,
                local_status_log=local_status_log,
            )

        except PublisherAccessError as e:
            logger.warning("%s: %s", doi, e)
            update_metadata(
                db, doc_id, logger,
                download_status=config.DOWNLOAD_FAILED,
                extension="",
                note_entry={"reason": "publisher_access_error", "message": str(e)},
                download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                local_status_log=local_status_log,
                doi=doi,
                publisher=publisher,
                title=title,
                year=year,
            )
        except Exception as e:
            logger.warning(doi + ' with title ' + title + ' of publisher ' + publisher + ' threw a ' + str(e) + ' exception. It will be deleted if already dowloaded.\n')
            logger.exception(e)
            logger.warning('\n')
            update_metadata(
                db, doc_id, logger,
                download_status=config.FILE_THREW_EXCEPTION,
                extension="",
                note_entry={"reason": "exception", "message": str(e)},
                download_time_stamp=datetime.now().strftime(config.DATE_TIME_FORMAT),
                local_status_log=local_status_log,
                doi=doi,
                publisher=publisher,
                title=title,
                year=year,
            )
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
            # The response did not include full text.
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
