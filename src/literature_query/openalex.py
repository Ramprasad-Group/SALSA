"""Dependency-free OpenAlex Works search client."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from wos_query.expanded import extract_dois

DEFAULT_ENDPOINT = "https://api.openalex.org/works"
DEFAULT_PUBLISHERS_ENDPOINT = "https://api.openalex.org/publishers"
PUBLISHER_ID_RE = re.compile(r"^(?:https?://openalex\.org/)?(P\d+)$", re.IGNORECASE)


class OpenAlexError(RuntimeError):
    """A usable error returned by OpenAlex or the network."""


class OpenAlexClient:
    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        publishers_endpoint: str | None = None,
        timeout: float = 30.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.api_key = (api_key or os.environ.get("OPENALEX_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Set OPENALEX_API_KEY to your OpenAlex API key.")
        self.endpoint = (endpoint or os.environ.get("OPENALEX_API_URL") or DEFAULT_ENDPOINT).rstrip("/")
        self.publishers_endpoint = (
            publishers_endpoint
            or os.environ.get("OPENALEX_PUBLISHERS_API_URL")
            or DEFAULT_PUBLISHERS_ENDPOINT
        ).rstrip("/")
        self.timeout = timeout
        self._opener = opener

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{endpoint}?{urlencode(params)}",
            headers={"Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OpenAlexError(f"OpenAlex returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise OpenAlexError(f"Could not reach OpenAlex: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OpenAlexError("OpenAlex returned invalid JSON.") from exc

    def resolve_publisher_id(self, publisher: str) -> str:
        """Resolve an OpenAlex publisher name, URL, or ID to a bare publisher ID."""
        publisher = publisher.strip()
        if not publisher:
            raise ValueError("The OpenAlex publisher cannot be empty.")
        match = PUBLISHER_ID_RE.match(publisher)
        if match:
            return match.group(1).upper()
        payload = self._get(self.publishers_endpoint, {
            "search": publisher,
            "per_page": 10,
            "select": "id,display_name",
            "api_key": self.api_key,
        })
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            raise OpenAlexError(f"No OpenAlex publisher matched {publisher!r}.")
        exact = next(
            (
                item for item in results
                if isinstance(item, dict)
                and str(item.get("display_name", "")).strip().casefold() == publisher.casefold()
            ),
            None,
        )
        selected = exact or results[0]
        raw_id = str(selected.get("id", "")) if isinstance(selected, dict) else ""
        id_match = PUBLISHER_ID_RE.match(raw_id)
        if not id_match:
            raise OpenAlexError("OpenAlex returned an invalid publisher ID.")
        return id_match.group(1).upper()

    def _page(
        self,
        query: str,
        count: int,
        *,
        filters: list[str],
        sort: str,
        search_mode: str,
        cursor: str = "",
        page: int = 1,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "search.semantic" if search_mode == "semantic" else "search": query,
            "per_page": count,
            "select": "doi",
            "api_key": self.api_key,
        }
        if search_mode == "semantic":
            params["page"] = page
        else:
            params["cursor"] = cursor
        if filters:
            params["filter"] = ",".join(filters)
        if sort:
            params["sort"] = sort
        return self._get(self.endpoint, params)

    def search_dois(
        self,
        query: str,
        *,
        page_size: int = 100,
        max_records: int | None = None,
        max_dois: int | None = None,
        publisher: str = "",
        open_access: bool | None = None,
        sort: str = "relevance_score:desc",
        search_mode: str = "keyword",
        request_delay: float = 0.1,
    ) -> list[str]:
        if not query.strip():
            raise ValueError("The OpenAlex query cannot be empty.")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100.")
        if max_dois is not None and max_dois < 1:
            raise ValueError("max_dois must be at least 1.")
        if search_mode not in {"keyword", "semantic"}:
            raise ValueError("search_mode must be 'keyword' or 'semantic'.")

        filters: list[str] = []
        if publisher.strip():
            publisher_id = self.resolve_publisher_id(publisher)
            filters.append(f"locations.source.publisher_lineage:{publisher_id}")
        if open_access is not None:
            filters.append(f"open_access.is_oa:{str(open_access).lower()}")

        semantic_limit = 50
        record_limit = max_records
        if search_mode == "semantic":
            record_limit = min(
                semantic_limit,
                max_records if max_records is not None else semantic_limit,
            )

        cursor, page, processed, dois, seen = "*", 1, 0, [], set()
        while cursor if search_mode == "keyword" else processed < record_limit:
            wanted = page_size if record_limit is None else min(page_size, record_limit - processed)
            if wanted <= 0:
                break
            payload = self._page(
                query.strip(),
                wanted,
                filters=filters,
                sort=sort,
                search_mode=search_mode,
                cursor=cursor,
                page=page,
            )
            results = payload.get("results", [])
            if not isinstance(results, list):
                raise OpenAlexError("OpenAlex returned an unexpected response shape.")
            for doi in extract_dois(results):
                if doi.casefold() not in seen:
                    seen.add(doi.casefold())
                    dois.append(doi)
                    if max_dois is not None and len(dois) >= max_dois:
                        return dois[:max_dois]
            processed += len(results)
            if search_mode == "semantic":
                page += 1
            else:
                cursor = str(payload.get("meta", {}).get("next_cursor") or "")
            if not results or (record_limit is not None and processed >= record_limit):
                break
            if request_delay:
                time.sleep(request_delay)
        return dois
