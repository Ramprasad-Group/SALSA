"""Query Web of Science API Expanded and export the result DOIs.

The API key is read from ``WOS_API_KEY`` unless explicitly supplied.  Keeping
credentials out of command arguments prevents them from appearing in shell
history and process listings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_ENDPOINT = "https://api.clarivate.com/api/wos"
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


class WebOfScienceError(RuntimeError):
    """A usable error returned by Web of Science or the network."""


def _candidate_dois(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        kind = str(value.get("type", value.get("name", ""))).lower()
        candidate = value.get("value", value.get("identifier"))
        if "doi" in kind and isinstance(candidate, str):
            yield candidate
        for key, child in value.items():
            if str(key).lower() == "doi" and isinstance(child, str):
                yield child
            yield from _candidate_dois(child)
    elif isinstance(value, list):
        for child in value:
            yield from _candidate_dois(child)


def extract_dois(payload: Any) -> list[str]:
    """Extract unique, normalized DOIs from Expanded API JSON."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in _candidate_dois(payload):
        doi = raw.strip()
        for prefix in ("https://doi.org/", "http://doi.org/"):
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
        doi = doi.rstrip(".,;)")
        key = doi.casefold()
        if DOI_RE.match(doi) and key not in seen:
            seen.add(key)
            result.append(doi)
    return result


class WebOfScienceClient:
    """Small dependency-free client for Web of Science API Expanded."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.api_key = (api_key or os.environ.get("WOS_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Set WOS_API_KEY to your Web of Science API Expanded key.")
        self.endpoint = (endpoint or os.environ.get("WOS_API_URL") or DEFAULT_ENDPOINT).rstrip("/")
        self.timeout = timeout
        self._opener = opener

    def _page(self, query: str, database: str, count: int, first_record: int) -> dict[str, Any]:
        params = urlencode({
            "databaseId": database,
            "usrQuery": query,
            "count": count,
            "firstRecord": first_record,
        })
        request = Request(
            f"{self.endpoint}?{params}",
            headers={"X-ApiKey": self.api_key, "Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise WebOfScienceError(f"Web of Science returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise WebOfScienceError(f"Could not reach Web of Science: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise WebOfScienceError("Web of Science returned invalid JSON.") from exc

    def search_dois(
        self,
        query: str,
        *,
        database: str = "WOS",
        page_size: int = 100,
        max_records: int | None = None,
        request_delay: float = 0.5,
    ) -> list[str]:
        """Return all unique DOIs matching a Web of Science advanced query."""
        if not query.strip():
            raise ValueError("The Web of Science query cannot be empty.")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100.")

        first, dois, seen = 1, [], set()
        while True:
            wanted = page_size if max_records is None else min(page_size, max_records - (first - 1))
            if wanted <= 0:
                break
            payload = self._page(query.strip(), database, wanted, first)
            page_dois = extract_dois(payload)
            for doi in page_dois:
                if doi.casefold() not in seen:
                    seen.add(doi.casefold())
                    dois.append(doi)

            query_result = payload.get("QueryResult", {})
            total = int(query_result.get("RecordsFound", first - 1 + wanted))
            # RecordsSearched is the database search scope, not the number of
            # records in this page. An empty DOI page is also not a reliable
            # stopping condition because records may simply lack a DOI.
            first += wanted
            if first > total or (max_records is not None and first > max_records):
                break
            if request_delay:
                time.sleep(request_delay)
        return dois


def write_dois(dois: Iterable[str], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["DOI"])
        writer.writerows((doi,) for doi in dois)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help='Web of Science advanced query, e.g. TS=("anion exchange membrane")')
    parser.add_argument("--output", type=Path, default=Path("wos_dois.csv"))
    parser.add_argument("--database", default="WOS")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    dois = WebOfScienceClient().search_dois(args.query, database=args.database, max_records=args.max_records)
    write_dois(dois, args.output)
    print(f"Wrote {len(dois)} unique DOIs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
