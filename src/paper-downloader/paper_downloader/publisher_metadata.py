"""Normalize publisher metadata discovered from DOI records."""

from typing import Any, Mapping, Optional


def canonical_publisher_name(value: Any, doi: str = "") -> str:
    """Return a SALSA publisher label while preserving unsupported names."""
    publisher = str(value or "").strip()
    if not publisher or publisher.lower() == "unknown":
        return "Unknown"

    lowered = publisher.lower()
    aliases = (
        (("royal society of chemistry", "rsc"), "RSC"),
        (("elsevier",), "Elsevier"),
        (("john wiley", "wiley"), "Wiley"),
        (("springer",), "Nature" if str(doi).lower().startswith("10.1038/") else "Springer"),
        (("nature portfolio",), "Nature"),
        (("public library of science", "plos"), "PLOS"),
        (("institute of electrical and electronics engineers", "ieee"), "IEEE"),
        (("american physical society",), "APS"),
        (("iop publishing", "institute of physics"), "IOP"),
        (("american chemical society",), "ACS"),
    )
    for names, canonical in aliases:
        if any(name in lowered for name in names):
            return canonical
    return publisher


def publisher_from_metadata(
    record: Optional[Mapping[str, Any]], doi: str = ""
) -> str:
    """Extract and normalize a publisher returned by Crossref-style metadata."""
    if not record:
        return "Unknown"
    for key in ("publisher", "Publisher"):
        publisher = canonical_publisher_name(record.get(key), doi)
        if publisher != "Unknown":
            return publisher
    return "Unknown"
