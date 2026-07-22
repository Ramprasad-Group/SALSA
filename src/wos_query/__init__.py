"""Web of Science query support for SALSA."""

from .expanded import WebOfScienceClient, WebOfScienceError, extract_dois

__all__ = ["WebOfScienceClient", "WebOfScienceError", "extract_dois"]
