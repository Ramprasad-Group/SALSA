import io
import json
from urllib.parse import parse_qs, urlparse

from literature_query import OpenAlexClient, ScopusClient


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _response(payload):
    return Response(json.dumps(payload).encode("utf-8"))


def test_openalex_search_uses_cursor_and_normalizes_dois():
    requests = []
    pages = [
        {"meta": {"next_cursor": "next"}, "results": [{"doi": "https://doi.org/10.1234/ABC"}]},
        {"meta": {"next_cursor": None}, "results": [{"doi": "10.5678/def"}]},
    ]

    def opener(request, **_kwargs):
        requests.append(request)
        return _response(pages.pop(0))

    result = OpenAlexClient(api_key="key", opener=opener).search_dois(
        "membrane", page_size=1, request_delay=0
    )

    assert result == ["10.1234/ABC", "10.5678/def"]
    assert parse_qs(urlparse(requests[0].full_url).query)["cursor"] == ["*"]
    assert parse_qs(urlparse(requests[1].full_url).query)["cursor"] == ["next"]


def test_openalex_applies_publisher_oa_sort_and_top_doi_limit():
    requests = []
    responses = [
        {"results": [{
            "id": "https://openalex.org/P1234567890",
            "display_name": "Example Publisher",
        }]},
        {"meta": {"next_cursor": "next"}, "results": [
            {"doi": None}, {"doi": "https://doi.org/10.1234/one"},
        ]},
        {"meta": {"next_cursor": None}, "results": [
            {"doi": "10.1234/two"}, {"doi": "10.1234/three"},
        ]},
    ]

    def opener(request, **_kwargs):
        requests.append(request)
        return _response(responses.pop(0))

    result = OpenAlexClient(api_key="key", opener=opener).search_dois(
        "membrane",
        page_size=2,
        max_dois=2,
        publisher="Example Publisher",
        open_access=True,
        sort="relevance_score:desc",
        request_delay=0,
    )

    assert result == ["10.1234/one", "10.1234/two"]
    publisher_params = parse_qs(urlparse(requests[0].full_url).query)
    assert publisher_params["search"] == ["Example Publisher"]
    works_params = parse_qs(urlparse(requests[1].full_url).query)
    assert works_params["filter"] == [
        "locations.source.publisher_lineage:P1234567890,open_access.is_oa:true"
    ]
    assert works_params["sort"] == ["relevance_score:desc"]


def test_openalex_semantic_search_uses_semantic_parameter():
    requests = []

    def opener(request, **_kwargs):
        requests.append(request)
        return _response({"meta": {"next_cursor": None}, "results": []})

    OpenAlexClient(api_key="key", opener=opener).search_dois(
        "materials that conduct ions without swelling",
        search_mode="semantic",
        request_delay=0,
    )

    params = parse_qs(urlparse(requests[0].full_url).query)
    assert params["search.semantic"] == [
        "materials that conduct ions without swelling"
    ]
    assert "search" not in params
    assert params["page"] == ["1"]
    assert "cursor" not in params


def test_openalex_semantic_search_uses_pages_and_stops_at_fifty_records():
    requests = []
    pages = [
        {"results": [{"doi": f"10.1234/{index}"} for index in range(30)]},
        {"results": [{"doi": f"10.1234/{index}"} for index in range(30, 50)]},
    ]

    def opener(request, **_kwargs):
        requests.append(request)
        return _response(pages.pop(0))

    result = OpenAlexClient(api_key="key", opener=opener).search_dois(
        "a detailed research description",
        search_mode="semantic",
        page_size=30,
        request_delay=0,
    )

    assert len(result) == 50
    first = parse_qs(urlparse(requests[0].full_url).query)
    second = parse_qs(urlparse(requests[1].full_url).query)
    assert first["page"] == ["1"]
    assert second["page"] == ["2"]
    assert second["per_page"] == ["20"]
    assert "cursor" not in first
    assert "cursor" not in second


def test_scopus_search_paginates_and_sends_elsevier_headers():
    requests = []
    pages = [
        {"search-results": {"opensearch:totalResults": "2", "entry": [{"prism:doi": "10.1234/a"}]}},
        {"search-results": {"opensearch:totalResults": "2", "entry": [{"prism:doi": "10.5678/b"}]}},
    ]

    def opener(request, **_kwargs):
        requests.append(request)
        return _response(pages.pop(0))

    result = ScopusClient(api_key="key", inst_token="token", opener=opener).search_dois(
        "TITLE(test)", page_size=1, request_delay=0
    )

    assert result == ["10.1234/a", "10.5678/b"]
    assert requests[0].get_header("X-els-apikey") == "key"
    assert requests[0].get_header("X-els-insttoken") == "token"
    assert parse_qs(urlparse(requests[1].full_url).query)["start"] == ["1"]
