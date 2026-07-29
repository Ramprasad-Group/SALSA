"""Shared requests.Session for outbound HTTP calls.

Reusing one Session gives every caller connection pooling (keep-alive,
no repeated TCP+TLS handshakes) and a consistent retry policy, instead of
each call site building a fresh connection via bare `requests.get(...)`.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session: requests.Session | None = None


def get_session() -> requests.Session:
    global _session
    if _session is None:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _session = session
    return _session
