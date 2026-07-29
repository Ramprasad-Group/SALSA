import os
from requests.utils import quote
from csv import reader
from time import sleep
from random import randint
#from os import path, remove
from collections import defaultdict
from paper_downloader import utils, config
from paper_downloader.http_session import get_session

class RecordsDownloader:
    '''
    Provides functions for downloading DOI records and DOI metadata from CrossRef
    '''

    def __init__(self, logger=None, logger_file=None, logger_name='records_downloader'):
        '''

        Parameters
        ----------
        els_api_key: str
            API key for Elsevier
        sleep_sec: int
            Sleep time between API calls (default = 1s)
        timeout_sec: int
            Max time before timeout (default = 90s)
        '''
        self.els_api_key = config.ELS_API_KEY
        self.sleep_sec = config.SLEEP_SEC
        self.timeout_sec = config.TIME_OUT_SEC
        if logger:
            self.logger = logger
        else:
            if logger_file is not None: self.logger = utils.setup_logger(logger_file, logger_name)
            
        self.blocked_metadata_type = ['erratum', 'news', 'events', 'acknowledgement', 'announcement', 'bookreview',
                                      'editorialnotes']
        self.flagged_keywords = ['call for papers', 'special issue', 'ieee', 'comments on', 'comment on',
                                'table of contents', 'classified contents', 'conference', 'retract', 'correction to',
                                'erratum', 'corrigendum', 'reply to', 'cheminform abstract', 'foreword', 'commentary',
                                'prelude', 'workshop', 'retract', 'memorial', 'symposium', 'editorial', 'memoriam', 'errata',
                                'author correction', 'publisher correction', 'author index']
        self.metadata_columns = config.METADATA_COLUMNS

    def get_dois_from_search(self, query, rows=500, mailto="null@null.com"):
        '''

        Parameters
        ----------
        query: str
            Search string for which DOI records are retrieved
        rows: int
            the maximum number of DOIs to download
        mailto: str
            mailto address for API

        Returns
        -------
        List of DOI's corresponding to the query string
        '''

        dois = []
        base_url = 'https://api.crossref.org/works?query='
        max_rows = 1000 #Defined by CrossRef API

        headers = {
          'Accept': 'application/json',
          'User-agent': 'mailto:' + mailto
        }

        if rows <= max_rows: #No multi-query needed
            search_url = f'{base_url}{query}&rows={rows}'
            response = get_session().get(search_url, headers=headers, timeout=self.timeout_sec).json()

            for item in response["message"]["items"]:
                dois.append(item["DOI"])

        else: #Need to split queries
            cursor = "*"
            max_paging = rows//max_rows+1
            page_count = 0
            while page_count<max_paging:
                sleep(self.sleep_sec)
                print(f'Downloading page: {page_count+1}')
                page_count+=1
                r = get_session().get(base_url + query + "&rows=" + str(max_rows) + "&cursor=" + cursor,
                                 headers=headers, timeout=self.timeout_sec)
                cursor = quote(r.json()['message']['next-cursor'], safe='')
                for item in r.json()['message']['items']:
                    if len(dois)<rows: dois.append(item['DOI'])
                    else: break

        return list(set(dois))

    def get_records_from_search(self, rows, pub_after, get_all,
                                exit_downloader=None, query=None, issn=None, max_paging=None,
                                page_count=1, cursor=None, mailto="null@null.com"):
        '''
        Retrieves DOI and corresponding metadata from CrossRef in batches given either a query string or a journal issn

        DOI's with publisher belonging to a certain list, published after a certain year and

        Parameters
        ----------
        query: str
            search string fed into CrossRed API
        rows: int
            the maximum number of DOIs to find
        pub_after: int
            DOI's published on or after this year are downloaded
        logger_file
        get_all: bool
            True in order to download all records associated with a query string or journal
        exit_downloader: int
            After exit_downloader number of pages of records have been downloaded, the function returns the DOI's downloaded
            along with the cursor and the current page_count
        issn: str
            ISSN corresponding to journal for which records are to be downloaded
        max_paging: int
            Number of pages that have to be downloaded for rows number of records to be downloaded
            Set equal to the total number of pages returned by CrossRef if get_all is True
        page_count: int
            Current page being downloaded
        cursor: str
            Cursor of last downloaded page in the batch
            None if the first page is being downloaded
        mailto: str
            mailto address for API

        Returns
        -------

        Raises
        -------
        ValueError if both issn and query are None or both are provided

        '''


        dois = defaultdict(list) # can also be a MongoDB record
        # Behaviour of function changes depending on whether query is None or issn is None
        if query is not None and issn is None: base_url = 'https://api.crossref.org/works?query='
        elif issn is not None and query is None: base_url = 'https://api.crossref.org/journals/' + issn + '/works?'
        else: raise ValueError

        max_rows = 1000 #Defined by CrossRef API
        if cursor is None: cursor = "*"
        headers = {
          'Accept': 'application/json',
          'User-agent': 'mailto:' + mailto
        }

        if rows <= max_rows and not get_all: #No multi-query needed
            # query url format different depending on whether issn mode or query search is used
            if query is not None: search_url = base_url + query + '&rows=' + str(rows) + "&cursor=" + cursor + "&filter=from-pub-date:" + str(pub_after)
            elif issn is not None: search_url = base_url + 'rows=' + str(rows) + "&cursor=" + cursor  + "&filter=from-pub-date:" + str(pub_after)
            response = get_session().get(search_url, headers=headers, timeout=self.timeout_sec).json()

            for item in response["message"]["items"]:
                try:
                    # Checks that records is published after pub_after, belongs to the list of publishers specified,
                    # title meets some conditions and number of entries appended so far is less than the maximum number of dois to find
                    if len(dois[self.metadata_columns[0]]) < rows and  \
                            ("published-print" not in item or int(
                                item['published-print']['date-parts'][0][0]) >= pub_after) and (
                            "title" not in item or self._check_doi_title(str(item["title"][0]))):
                        metadata_record = self._item_checker(item, issn)
                        if not metadata_record:
                            continue
                        else:
                            for column in self.metadata_columns: # Can use zip here?
                                dois[column].append(metadata_record[column])
                except Exception as e:
                    self.logger.warning(item['DOI'] + ' threw a ' + str(e) + ' exception')
                    self.logger.exception(e)
                    self.logger.warning('\n')
            return dois

        else: #Need to split queries
            if max_paging is None: max_paging = rows//max_rows+1
            while page_count <= max_paging:
                print(f'Downloading page: {page_count}')
                try:
                    if query is not None: search_url = base_url + query + "&rows=" + str(max_rows) + "&cursor=" + cursor + "&filter=from-pub-date:" + str(pub_after)
                    elif issn is not None: search_url = base_url + "rows=" + str(max_rows) + "&cursor=" + cursor + "&filter=from-pub-date:" + str(pub_after)
                    r = get_session().get(search_url, headers=headers, timeout=self.timeout_sec)
                except Exception as e:
                    self.logger.warning(f'Exception {e} occurred')
                    sleep(30)
                    continue

                if page_count == 1 and get_all:
                    rows_count = int(r.json()['message']['total-results'])
                    self.logger.info(f"Total results: {rows_count}")
                    max_paging = rows_count//max_rows + 1

                try:
                    cursor = quote(r.json()['message']['next-cursor'], safe='')
                except Exception as e:
                    self.logger.warning(str(r.json()['message']['next-cursor'])+' threw a '+str(e)+' error\n') # This block might not be necessary
                    self.logger.exception(e)
                    self.logger.warning('\n')
                    break
                page_count += 1
                # To implement robustness to connection failure in the records downloader,
                # the results are returned after exit_downloader number of pages is completed
                # The outer loop must call the function again using the cursor
                # returned in order to download further records
                if exit_downloader is None: raise ValueError

                if page_count % (exit_downloader+1) == 0:
                    if page_count==max_paging+1:
                        cursor=False
                        return dois, max_paging, page_count, cursor
                    else:
                        return dois, max_paging, page_count, cursor

                for item in r.json()['message']['items']:
                    # if "title" in item and item["title"] == 'Poly(dimethyltin glutarate) as a Prospective Material for High Dielectric Applications':
                    #     print('Found!')
                    #     print(item["DOI"])
                    try:
                        if (len(dois[self.metadata_columns[0]]) < rows or get_all) and 'publisher' in item and  \
                                ("published-print" not in item or int(item['published-print']['date-parts'][0][0]) >= pub_after) \
                                and ("title" not in item or self._check_doi_title(str(item["title"][0]))):
                            # if "title" in item and item["title"] == 'Poly(dimethyltin glutarate) as a Prospective Material for High Dielectric Applications': print('Inside Loop!')
                            metadata_record = self._item_checker(item, issn)
                            if not metadata_record:
                                continue
                            else:
                                for column in self.metadata_columns:
                                    dois[column].append(metadata_record[column])
                    except Exception as e:
                        self.logger.warning(str(item['DOI']) + ' threw a ' + str(e) + ' exception\n')
                        self.logger.exception(e)
                        self.logger.warning('\n')

        sleep_sec_rand = randint(self.sleep_sec, 10 * self.sleep_sec)
        print(f'Waiting for {sleep_sec_rand}  seconds...')
        sleep(sleep_sec_rand)
        cursor = False
        return dois, max_paging, page_count, cursor
    

    def get_record_by_doi(self, doi, mailto="null@null.com"):
        """
        Fetches metadata for a specific DOI from CrossRef.
        """
        url = f"https://api.crossref.org/works/{doi}"
        headers = {
            'Accept': 'application/json',
            'User-agent': f'mailto:{mailto}'
        }
        try:
            response = get_session().get(url, headers=headers, timeout=self.timeout_sec)
            response.raise_for_status()
            item = response.json()["message"]
            # Use your existing _item_checker to process the item if needed
            metadata_record = self._item_checker(item, issn=None)
            return metadata_record
        except Exception as e:
            self.logger.warning(f"Exception fetching DOI {doi}: {e}")
            return None

    def _check_doi_title(self, title):
        no_of_words = len(title.split(' '))
        if no_of_words > 3 and not any([phrase in title.lower() for phrase in self.flagged_keywords]):
            return True
        else:
            return False

    def _item_checker(self, item, issn):
        year_list = []
        if "published-print" in item and item['published-print']['date-parts'][0][0] is not None:
            year_list.append(int(item['published-print']['date-parts'][0][0]))
        if "published-online" in item and item['published-online']['date-parts'][0][0] is not None:
            year_list.append(int(item['published-online']['date-parts'][0][0]))
        if "published" in item and item['published']['date-parts'][0][0] is not None:
            year_list.append(int(item['published']['date-parts'][0][0]))
        if "created" in item and item['created']['date-parts'][0][0] is not None:
            year_list.append(int(item['created']['date-parts'][0][0]))
        if "issued" in item and item['issued']['date-parts'][0][0] is not None:
            year_list.append(int(item['issued']['date-parts'][0][0]))

        if year_list:
            year = min(year_list)
        else:
            year = None

        # For older papers, published-print and published-online might be in different years and so using published-online as the primary criteria may give
        # incorrect months while published-print may not even have a month
        if "published-online" in item and len(item['published-online']['date-parts'][0])>=2 and item['published-online']['date-parts'][0][1] is not None:
            month = int(item['published-online']['date-parts'][0][1])
        elif "issued" in item and len(item['issued']['date-parts'][0])>=2 and item['issued']['date-parts'][0][1] is not None:
            month = int(item['issued']['date-parts'][0][1])
        elif "created" in item and len(item['created']['date-parts'][0])>=2 and item['created']['date-parts'][0][1] is not None:
            month = int(item['created']['date-parts'][0][1])
        elif "published-print" in item and len(item['published-print']['date-parts'][0])>=2 and item['published-print']['date-parts'][0][1] is not None:
            month = int(item['published-print']['date-parts'][0][1])
        else:
            month=None

        if "published-online" in item and len(item['published-online']['date-parts'][0])>=3 and item['published-online']['date-parts'][0][2] is not None:
            day = int(item['published-online']['date-parts'][0][2])
        elif "issued" in item and len(item['issued']['date-parts'][0])>=3 and item['issued']['date-parts'][0][2] is not None:
            day = int(item['issued']['date-parts'][0][2])
        elif "created" in item and len(item['created']['date-parts'][0])>=3 and item['created']['date-parts'][0][2] is not None:
            day = int(item['created']['date-parts'][0][2])
        elif "published-print" in item and len(item['published-print']['date-parts'][0])>=3 and item['published-print']['date-parts'][0][2] is not None:
            day = int(item['published-print']['date-parts'][0][2])
        else:
            day=None


        if "title" in item:
            title = item["title"][0]
        else:
            title = None

        if "container-title" in item:
            journal = item["container-title"][0]
        else:
            journal = None

        if "is-referenced-by-count" in item:
            citations = item["is-referenced-by-count"]
        else:
            citations = None

        if "ISSN" in item and issn is None:
            issn = item["ISSN"][0]

        if "type" in item:
            metadata_type = item["type"][0]
            if metadata_type is not None and any([kwd in metadata_type.lower() for kwd in self.blocked_metadata_type]):
                return False
        if "language" in item:
            if item["language"] != 'en':
                return False

        if "reference" in item:
            reference_list = []
            for i in range(len(item["reference"])):
                if 'DOI' in item["reference"][i].keys():
                    reference_list.append(item["reference"][i]['DOI'])
        else:
            reference_list = None

        if "author" in item:
            author_list = []
            for author in item["author"]:
                author_item = {}
                if 'ORCID' in author.keys():
                    author_item['ORCID'] = author['ORCID'].replace('http://orcid.org/', '')
                if 'given' in author.keys() and 'family' in author.keys():
                    # author_item.extend([author["given"], author["family"]])
                    author_item['given'], author_item['family'] = author["given"], author["family"]
                if 'affiliation' in author.keys() and author['affiliation'] \
                        and 'name' in author['affiliation'][0].keys():
                    author_item['affiliation'] = author['affiliation'][0]['name']

                if author_item:
                    author_list.append(author_item)
        else:
            author_list = None

        if 'funder' in item:
            funder_list = []
            for funder in item['funder']:
                if 'name' in funder:
                    funder_list.append(funder['name'])
            if not funder_list: funder_list = None
        else:
            funder_list = None

        metadata_record = {
          "DOI": item["DOI"],
          "title": title,
          # "prefix": item["prefix"],
          "journal": journal,
          "publisher": item["publisher"],
          # "volume": volume,
          # "issue": issue,
          # "page": page,
          "year": year,
          "month": month,
          "day": day,
          # "num_references": item['references-count'],
          "citations": citations,
          "reference_list": reference_list,
          "author_list": author_list,
          "funder_list": funder_list,
          "issn": issn
      }
        return metadata_record

    def get_metadata_from_doi(self, doi, mailto="null@null.com"):
        base_url = 'https://api.crossref.org/works/' + str(doi)

        headers = {
          'Accept': 'application/json',
          'User-agent': 'mailto:' + mailto
        }

        search_url = str(base_url)
        response = get_session().get(search_url, headers=headers, timeout=self.timeout_sec).json()

        item = response["message"]
        metadata_record = None
        try:
            metadata_record = self._item_checker(item, issn=None)
        except Exception as e:
            print(e)

        return metadata_record # Will return False if the DOI type belongs to one of the blocked types

    def get_title_from_doi(self, doi, mode):
        '''

        Parameters
        ----------
        doi: str
            DOI string for the article we want to download title for
        mode: str
            Only supports 'crossref' for now
        Returns
        -------
        A title (or None on failure)
        '''

        if mode == 'crossref':
            try:
                url='http://api.crossref.org/works/' + doi
                headers = {
                    'X-ELS-APIKEY': self.els_api_key,
                    'Accept': 'application/json'
                }

                r = get_session().get(url, headers=headers, timeout=self.timeout_sec)
                if r.status_code == 200:
                    title = r.json()['message']['title'][0]
                    return title
            except:
                # API download limit exceeded or no title exists
                return None

        return None

    def load_queries_from_csv(self, csvf):
        '''

        Parameters
        ----------
        csvf: file object
            CSV file with one query per line

        Returns
        -------
        A list of queries, processed to be insertable into REST API (GET) calls
        '''

        csvf.seek(0)
        csv_reader = reader(csvf, delimiter=',')
        queries = []
        for line in csv_reader:
            #Build search query (assume 1st column is queries)
            query = quote(line[0])
            query = '+'.join(query.split())
            queries.append(query)
        return queries
