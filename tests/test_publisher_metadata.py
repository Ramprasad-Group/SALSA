from pathlib import Path
import sys


PAPER_DOWNLOADER_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "paper-downloader"
)
sys.path.insert(0, str(PAPER_DOWNLOADER_SRC))

from paper_downloader.publisher_metadata import (  # noqa: E402
    canonical_publisher_name,
    publisher_from_metadata,
)


def test_crossref_rsc_name_is_normalized():
    assert publisher_from_metadata(
        {"publisher": "Royal Society of Chemistry (RSC)"},
        "10.9999/example",
    ) == "RSC"


def test_crossref_can_resolve_an_unknown_prefix_to_supported_publisher():
    assert publisher_from_metadata(
        {"publisher": "John Wiley & Sons, Ltd"},
        "10.5555/example",
    ) == "Wiley"


def test_unsupported_crossref_publisher_is_preserved_for_reporting():
    assert canonical_publisher_name("Example University Press") == "Example University Press"


def test_missing_crossref_publisher_remains_unknown():
    assert publisher_from_metadata({"title": "Example"}) == "Unknown"
