"""Dependency-free Elsevier Scopus Search API client."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from wos_query.expanded import extract_dois

DEFAULT_ENDPOINT = "https://api.elsevier.com/content/search/scopus"


class ScopusError(RuntimeError):
    """A usable error returned by Scopus or the network."""


class ScopusClient:
    def __init__(
        self,
        api_key: str | None = None,
        inst_token: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.api_key = (api_key or os.environ.get("ELSEVIER_API_KEY", "") or os.environ.get("ELS_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Set ELSEVIER_API_KEY to your Elsevier API key.")
        self.inst_token = (inst_token or os.environ.get("SCOPUS_INST_TOKEN", "")).strip()
        self.endpoint = (endpoint or os.environ.get("SCOPUS_API_URL") or DEFAULT_ENDPOINT).rstrip("/")
        self.timeout = timeout
        self._opener = opener

    def _page(self, query: str, count: int, start: int) -> dict[str, Any]:
        params = urlencode({"query": query, "count": count, "start": start, "view": "STANDARD"})
        headers = {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}
        if self.inst_token:
            headers["X-ELS-Insttoken"] = self.inst_token
        request = Request(f"{self.endpoint}?{params}", headers=headers)
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ScopusError(f"Scopus returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise ScopusError(f"Could not reach Scopus: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ScopusError("Scopus returned invalid JSON.") from exc

    def search_dois(
        self,
        query: str,
        *,
        page_size: int = 200,
        max_records: int | None = None,
        request_delay: float = 0.1,
    ) -> list[str]:
        if not query.strip():
            raise ValueError("The Scopus query cannot be empty.")
        if not 1 <= page_size <= 200:
            raise ValueError("page_size must be between 1 and 200.")

        start, dois, seen = 0, [], set()
        while True:
            wanted = page_size if max_records is None else min(page_size, max_records - start)
            if wanted <= 0:
                break
            payload = self._page(query.strip(), wanted, start)
            search_results = payload.get("search-results", {})
            entries = search_results.get("entry", [])
            if not isinstance(entries, list):
                raise ScopusError("Scopus returned an unexpected response shape.")
            # Scopus names the DOI field ``prism:doi`` rather than ``doi``.
            page_dois = extract_dois(
                [{"doi": entry.get("prism:doi")} for entry in entries if isinstance(entry, dict)]
            )
            for doi in page_dois:
                if doi.casefold() not in seen:
                    seen.add(doi.casefold())
                    dois.append(doi)
            try:
                total = int(search_results.get("opensearch:totalResults", len(entries)))
            except (TypeError, ValueError):
                total = start + len(entries)
            start += len(entries)
            if not entries or start >= total or (max_records is not None and start >= max_records):
                break
            if request_delay:
                time.sleep(request_delay)
        return dois
