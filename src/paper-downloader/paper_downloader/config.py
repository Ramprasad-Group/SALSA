import os

FULL_DOWNLOAD_SUCCESSFUL = 1
ABSTRACT_DOWNLOAD_SUCCESSFUL = 2
LETTER_DOWNLOAD_SUCCESSFUL = 3
FILE_DELETED = 4
DOWNLOAD_FAILED = 5
FILE_SIZE_0 = 6
FILE_THREW_EXCEPTION = 7

TIME_OUT_SEC = 90
SLEEP_SEC = 1


def _first_env(*names):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""



ELS_API_KEY = _first_env('ELSEVIER_API_KEY', 'ELS_API_KEY')
WILEY_TDM_TOKEN = _first_env('WILEY_TDM_TOKEN', 'WILEY_API_KEY')
SPRINGER_NATURE_API_KEY = _first_env('SPRINGER_NATURE_API_KEY', 'SPRINGER_API_KEY')
RSC_TDM_TOKEN = _first_env('RSC_TDM_TOKEN', 'RSC_API_KEY')
NCBI_API_KEY = _first_env('NCBI_API_KEY', 'PUBMED_API_KEY')
IEEE_API_KEY = _first_env('IEEE_API_KEY', 'IEEE_XPLORE_API_KEY')
DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
publisher_gap_time = {'wiley': 6}

METADATA_COLUMNS = ['DOI', 'publisher', 'journal', 'year', 'month', 'day', 'title', 'citations', 'reference_list', 'author_list',
                    'funder_list', 'issn']

STATUS_CHECK = 5000
THRESH = STATUS_CHECK*0.05 # Number of faulty downloads
