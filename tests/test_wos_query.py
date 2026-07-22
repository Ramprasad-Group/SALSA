import json

from wos_query.expanded import WebOfScienceClient, extract_dois


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


def test_extract_dois_handles_expanded_identifier_shape():
    payload = {"REC": {"records": {"REC": [
        {"dynamic_data": {"cluster_related": {"identifiers": {"identifier": [
            {"type": "doi", "value": "10.1000/ABC"},
            {"type": "issn", "value": "1234-5678"},
        ]}}}},
        {"DOI": "https://doi.org/10.2000/example"},
    ]}}}
    assert extract_dois(payload) == ["10.1000/ABC", "10.2000/example"]


def test_client_paginates_and_deduplicates():
    pages = iter([
        {"QueryResult": {"RecordsFound": 2, "RecordsSearched": 1}, "DOI": "10.1000/a"},
        {"QueryResult": {"RecordsFound": 2, "RecordsSearched": 1}, "DOI": "10.1000/b"},
    ])
    client = WebOfScienceClient("secret", opener=lambda *_args, **_kwargs: _Response(next(pages)))
    assert client.search_dois("TS=test", page_size=1, request_delay=0) == ["10.1000/a", "10.1000/b"]
