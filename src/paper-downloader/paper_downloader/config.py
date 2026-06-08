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



ELS_API_KEY = _first_env('ELSEVIER_API_KEY', 'ELS_API_KEY') or _DEFAULT_ELS_API_KEY
WILEY_TDM_TOKEN = _first_env('WILEY_TDM_TOKEN', 'WILEY_API_KEY') or _DEFAULT_WILEY_TDM_TOKEN
DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
publisher_gap_time = {'acs': 481, 'aip': 3, 'wiley': 6, 'ecs': 120}

METADATA_COLUMNS = ['DOI', 'publisher', 'journal', 'year', 'month', 'day', 'title', 'citations', 'reference_list', 'author_list',
                    'funder_list', 'issn']

STATUS_CHECK = 5000
THRESH = STATUS_CHECK*0.05 # Number of faulty downloads

USER_ID = 'melda.notifications@gmail.com'
PASSWORD = 'bree-wonk1FICT-gul'
ROOT_EMAIL = 'pranav.shetty@gatech.edu'

URL_DICT = {'acs': 'http://pubs.acs.org/doi/full/', 'nature': 'https://www.nature.com/articles/',
                'springer': 'http://link.springer.com/article/', 'wiley': 'https://api.wiley.com/onlinelibrary/tdm/v1/articles/', 'elsevier':'https://api.elsevier.com/content/article/doi/',
                'emerald':'http://www.emeraldinsight.com/doi/full/', 'informa uk':'https://www.tandfonline.com/doi/full/', 'iop publishing': 'https://iopscience.iop.org/article/',
                'sage': 'https://journals.sagepub.com/doi/', 'aip': 'https://aip.scitation.org/doi/', 'doi': 'https://dx.doi.org/'}
