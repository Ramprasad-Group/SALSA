import importlib
import inspect
import logging
import os
from pathlib import Path
import sys

import pytest


PAPER_DOWNLOADER_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "paper-downloader"
)
sys.path.insert(0, str(PAPER_DOWNLOADER_SRC))

from paper_downloader.publisher_access import publisher_access_for


@pytest.fixture(autouse=True)
def clear_publisher_credentials(monkeypatch):
    for name in (
        "ELSEVIER_API_KEY",
        "ELS_API_KEY",
        "WILEY_TDM_TOKEN",
        "WILEY_API_KEY",
        "SPRINGER_NATURE_API_KEY",
        "SPRINGER_API_KEY",
        "RSC_TDM_TOKEN",
        "RSC_API_KEY",
        "NCBI_API_KEY",
        "PUBMED_API_KEY",
        "IEEE_API_KEY",
        "IEEE_XPLORE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("publisher", "doi", "env_name"),
    [
        ("Elsevier", "10.1016/j.polymer.2024.1", "ELSEVIER_API_KEY"),
        ("Wiley", "10.1002/app.12345", "WILEY_API_KEY"),
        ("Springer", "10.1007/s10853-020-05509-2", "SPRINGER_NATURE_API_KEY"),
        ("Nature", "10.1038/s41428-020-00396-5", "SPRINGER_NATURE_API_KEY"),
        ("RSC", "10.1039/C8TA01114A", "RSC_API_KEY"),
        ("IEEE", "10.1109/example", "IEEE_API_KEY"),
    ],
)
def test_credentialed_publishers_fail_closed(publisher, doi, env_name):
    decision = publisher_access_for(doi, publisher)

    assert decision.plan is None
    assert env_name in decision.reason


def test_documented_api_routes(monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "elsevier-test")
    monkeypatch.setenv("WILEY_TDM_TOKEN", "wiley-test")
    monkeypatch.setenv("SPRINGER_NATURE_API_KEY", "springer-test")

    elsevier = publisher_access_for("10.1016/j.polymer.2024.1", "Elsevier").plan
    wiley = publisher_access_for("10.1002/app.12345", "Wiley").plan
    springer = publisher_access_for("10.1007/s10853-020-05509-2", "Springer").plan

    assert elsevier.url.startswith("https://api.elsevier.com/")
    assert elsevier.headers["X-ELS-APIKey"] == "elsevier-test"
    assert elsevier.response_validator == "elsevier_full_text_or_pdf"
    assert wiley.url.startswith("https://api.wiley.com/")
    assert wiley.headers["Wiley-TDM-Client-Token"] == "wiley-test"
    assert springer.url == "https://api.springernature.com/openaccess/jats"
    assert springer.params["q"] == "doi:10.1007/s10853-020-05509-2"
    assert springer.response_validator == "springer_open_access_jats"


def test_aps_uses_harvest_and_allows_open_access_without_token():
    plan = publisher_access_for(
        "10.1103/PhysRevX.5.021001",
        "APS",
    ).plan

    assert plan.url.startswith("https://harvest.aps.org/")
    assert plan.headers == {"Accept": "application/pdf"}
    assert plan.extension == "pdf"
    assert plan.response_validator == "aps_open_access_pdf"


def test_open_access_routes_require_no_user_credential():
    pubmed = publisher_access_for("10.1371/journal.pone.0303005", "PubMed").plan
    plos = publisher_access_for("10.1371/journal.pone.0303005", "PLOS").plan

    assert pubmed.strategy == "pmc_by_doi"
    assert pubmed.url.startswith("https://pmc.ncbi.nlm.nih.gov/")
    assert plos.strategy == "direct"
    assert plos.params == {
        "id": "10.1371/journal.pone.0303005",
        "type": "manuscript",
    }


def test_rsc_uses_fixed_publications_endpoint(monkeypatch):
    monkeypatch.setenv("RSC_API_KEY", "rsc-test")

    plan = publisher_access_for("10.1039/C8TA01114A", "RSC").plan

    assert plan.url == "https://api.rsc.org/articles/10.1039/C8TA01114A/pdf"
    assert plan.headers["apikey"] == "rsc-test"
    assert plan.extension == "pdf"


def test_iop_uses_crossref_tdm_lookup():
    plan = publisher_access_for(
        "10.1088/1361-6528/aa9ab0",
        "IOP Publishing",
    ).plan

    assert plan.strategy == "iop_crossref_tdm"
    assert plan.url.endswith("10.1088%2F1361-6528%2Faa9ab0")
    assert plan.response_validator == "iop_full_text_html"
    assert plan.directory == "iop"


def test_iop_crossref_lookup_prefers_version_of_record(monkeypatch):
    downloader = importlib.import_module("paper_downloader.downloader")
    plan = publisher_access_for(
        "10.1088/1361-6528/aa9ab0",
        "IOP Publishing",
    ).plan

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "message": {
                    "link": [
                        {
                            "URL": "http://example.test/accepted",
                            "content-type": "text/html",
                            "content-version": "am",
                            "intended-application": "text-mining",
                        },
                        {
                            "URL": "http://example.test/version-of-record",
                            "content-type": "text/html",
                            "content-version": "vor",
                            "intended-application": "text-mining",
                        },
                    ]
                }
            }

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: Response())

    resolved = downloader._resolve_access_plan(
        plan,
        "10.1088/1361-6528/aa9ab0",
        30,
        logging.getLogger(__name__),
    )

    assert resolved.url == "https://example.test/version-of-record"
    assert resolved.strategy == "direct"
    assert resolved.headers["Accept"] == "text/html"


def test_ieee_uses_doi_lookup_before_full_text(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "ieee-test")

    plan = publisher_access_for("10.1109/example", "IEEE").plan

    assert plan.strategy == "ieee_by_doi"
    assert plan.url == "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    assert plan.params == {
        "apikey": "ieee-test",
        "doi": "10.1109/example",
    }


@pytest.mark.parametrize(
    "publisher",
    [
        "ACS",
        "ECS",
        "AIP Publishing",
        "Hindawi",
        "Informa UK",
        "Sage",
        "Emerald",
    ],
)
def test_publishers_without_configured_tdm_api_require_upload(publisher):
    decision = publisher_access_for("10.0000/example", publisher)

    assert decision.plan is None
    assert "Upload an authorized PDF" in decision.reason


def test_active_downloader_does_not_resolve_dois_or_scrape_landing_pages():
    downloader = importlib.import_module("paper_downloader.downloader")
    source = inspect.getsource(downloader.article_downloader)

    assert "doi.org" not in source
    assert "dx.doi.org" not in source
    assert "pubs.acs.org" not in source
    assert "tandfonline.com" not in source
