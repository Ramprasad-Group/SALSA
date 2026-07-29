#!/usr/bin/env python3
"""Non-destructive publisher full-text access diagnostics."""

import json
import logging
from urllib.parse import urlparse

from paper_downloader.downloader import (
    _elsevier_needs_pdf,
    _resolve_access_plan,
    _validate_response_content,
)
from paper_downloader.http_session import get_session
from paper_downloader.publisher_access import publisher_access_for


REPRESENTATIVE_DOIS = {
    "Elsevier": "10.1016/j.polymer.2020.122355",
    "Wiley": "10.1002/app.10031",
    "ACS": "10.1021/acs.macromol.6c01122",
    "Nature": "10.1038/s41428-020-00396-5",
    "RSC": "10.1039/C8TA01114A",
    "PubMed": "10.1371/journal.pone.0303005",
    "PLOS": "10.1371/journal.pone.0303005",
    "IEEE": "10.1109/5.771073",
    "Springer": "10.1007/s10853-020-05509-2",
    "Informa UK": "10.1080/21663831.2026.2684786",
    "Hindawi": "10.1155/ijod/9930736",
    "ECS": "10.1149/1945-7111/ae7476",
    "APS": "10.1103/PhysRevX.5.021001",
    "AIP Publishing": "10.1063/5.0332580",
    "IOP Publishing": "10.1088/1361-6528/aa9ab0",
}

PROBE_USER_AGENT = "SALSA publisher access probe (wschertzer3@gatech.edu)"


def _full_text_probe(plan, doi, timeout=30):
    request_headers = {"User-Agent": PROBE_USER_AGENT}
    request_headers.update(plan.headers)
    try:
        resolved_plan = _resolve_access_plan(
            plan,
            doi,
            timeout,
            logging.getLogger("publisher_access_probe"),
        )
        request_headers.update(resolved_plan.headers)
        response = get_session().get(
            resolved_plan.url,
            headers=request_headers,
            params=resolved_plan.params,
            allow_redirects=True,
            timeout=timeout,
        )
        content = response.content
        content_type = response.headers.get("Content-Type", "")
        result = {
            "status": response.status_code,
            "final_host": urlparse(response.url).netloc,
            "content_type": content_type,
            "redirects": len(response.history),
            "bytes": len(content),
            "full_text": False,
        }
        if response.status_code == 200:
            if resolved_plan.extension == "pdf":
                result["full_text"] = content.startswith(b"%PDF") and len(content) > 1000
            elif resolved_plan.response_validator:
                valid, note = _validate_response_content(
                    content,
                    resolved_plan.response_validator,
                )
                result["full_text"] = valid
                if (
                    valid
                    and resolved_plan.response_validator == "elsevier_full_text_or_pdf"
                    and _elsevier_needs_pdf(content)
                ):
                    result["full_text"] = False
                    note = {"reason": "elsevier_pdf_fallback_required"}
                if note:
                    result["note"] = note
            elif resolved_plan.extension == "xml":
                result["full_text"] = (
                    b"<article" in content.lower()
                    or b"<body" in content.lower()
                )
            elif resolved_plan.extension == "html":
                result["full_text"] = len(content) > 20_000
        response.close()
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    # Publisher API keys are commonly carried in query strings. Keep HTTP client
    # debug logs disabled so diagnostics never echo credentials.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    results = []
    for publisher, doi in REPRESENTATIVE_DOIS.items():
        row = {
            "publisher": publisher,
            "doi": doi,
        }
        decision = publisher_access_for(doi, publisher)
        if decision.plan is None:
            row["salsa_access"] = {"available": False, "reason": decision.reason}
        else:
            plan = decision.plan
            row["salsa_access"] = {
                "available": True,
                "provider": plan.provider,
                "full_text_probe": _full_text_probe(plan, doi),
            }
        results.append(row)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
