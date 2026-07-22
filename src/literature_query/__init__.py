"""DOI search clients for literature-index APIs."""

from .openalex import OpenAlexClient, OpenAlexError
from .scopus import ScopusClient, ScopusError

__all__ = ["OpenAlexClient", "OpenAlexError", "ScopusClient", "ScopusError"]
