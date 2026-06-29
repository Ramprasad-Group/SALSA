"""Publisher full-text access policy.

DOI resolution is deliberately not used here. A DOI redirect only locates a
publisher landing page; it does not grant permission for automated full-text
retrieval. SALSA downloads only through documented machine-access endpoints.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, Optional
from urllib.parse import quote


@dataclass(frozen=True)
class PublisherAccessPlan:
    provider: str
    directory: str
    extension: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    expected_content_type: str = ""
    response_validator: str = ""
    strategy: str = "direct"


@dataclass(frozen=True)
class PublisherAccessDecision:
    plan: Optional[PublisherAccessPlan]
    reason: str = ""


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _missing_credential(provider: str, env_name: str) -> PublisherAccessDecision:
    return PublisherAccessDecision(
        plan=None,
        reason=f"{provider} full-text access requires {env_name}.",
    )


def publisher_access_for(doi: str, publisher: str) -> PublisherAccessDecision:
    """Return the documented machine-access route for a publisher."""
    doi = str(doi).strip()
    publisher_lower = str(publisher).strip().lower()

    if "elsevier" in publisher_lower:
        api_key = _first_env("ELSEVIER_API_KEY", "ELS_API_KEY")
        if not api_key:
            return _missing_credential("Elsevier", "ELSEVIER_API_KEY")
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="Elsevier Article Retrieval API",
                directory="elsevier",
                extension="xml",
                url=f"https://api.elsevier.com/content/article/doi/{doi}",
                headers={"X-ELS-APIKey": api_key, "Accept": "text/xml"},
                expected_content_type="xml",
                response_validator="elsevier_full_text_or_pdf",
            )
        )

    if "wiley" in publisher_lower:
        token = _first_env("WILEY_TDM_TOKEN", "WILEY_API_KEY")
        if not token:
            return _missing_credential("Wiley", "WILEY_API_KEY")
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="Wiley TDM API",
                directory="wiley",
                extension="pdf",
                url=f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}",
                headers={
                    "Wiley-TDM-Client-Token": token,
                    "Accept": "application/pdf",
                },
                expected_content_type="pdf",
            )
        )

    if "springer" in publisher_lower or "nature" in publisher_lower:
        api_key = _first_env("SPRINGER_NATURE_API_KEY", "SPRINGER_API_KEY")
        if not api_key:
            return _missing_credential(
                "Springer Nature Open Access API",
                "SPRINGER_NATURE_API_KEY",
            )
        directory = "nature" if "nature" in publisher_lower or doi.startswith("10.1038/") else "springer"
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="Springer Nature Open Access API",
                directory=directory,
                extension="xml",
                url="https://api.springernature.com/openaccess/jats",
                params={"api_key": api_key, "q": f"doi:{doi}"},
                headers={"Accept": "application/xml"},
                expected_content_type="xml",
                response_validator="springer_open_access_jats",
            )
        )

    if "royal society of chemistry" in publisher_lower or publisher_lower == "rsc":
        token = _first_env("RSC_TDM_TOKEN", "RSC_API_KEY")
        if not token:
            return _missing_credential("RSC", "RSC_API_KEY")
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="RSC Publications API",
                directory="rsc",
                extension="pdf",
                url=f"https://api.rsc.org/articles/{doi}/pdf",
                headers={"apikey": token, "Accept": "application/pdf"},
                expected_content_type="pdf",
            )
        )

    if "iop publishing" in publisher_lower or publisher_lower == "iop":
        contact_email = os.environ.get(
            "SALSA_CONTACT_EMAIL",
            "wschertzer3@gatech.edu",
        )
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="IOP Publishing via Crossref TDM",
                directory="iop",
                extension="html",
                url=f"https://api.crossref.org/works/{quote(doi, safe='')}",
                params={"mailto": contact_email},
                headers={
                    "Accept": "application/json",
                    "User-Agent": f"SALSA/1.0 (mailto:{contact_email})",
                },
                expected_content_type="html",
                response_validator="iop_full_text_html",
                strategy="iop_crossref_tdm",
            )
        )

    if (
        "pubmed central" in publisher_lower
        or publisher_lower == "pubmed"
        or publisher_lower == "pmc"
    ):
        params = {
            "ids": doi,
            "format": "json",
            "tool": "SALSA",
            "email": os.environ.get(
                "NCBI_EMAIL",
                os.environ.get("SALSA_CONTACT_EMAIL", "wschertzer3@gatech.edu"),
            ),
        }
        ncbi_api_key = _first_env("NCBI_API_KEY", "PUBMED_API_KEY")
        if ncbi_api_key:
            params["api_key"] = ncbi_api_key
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="PubMed Central OAI-PMH API",
                directory="pubmed",
                extension="xml",
                url="https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
                params=params,
                headers={"Accept": "application/json"},
                expected_content_type="xml",
                strategy="pmc_by_doi",
            )
        )

    if "plos" in publisher_lower or doi.lower().startswith("10.1371/"):
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="PLOS manuscript XML",
                directory="plos",
                extension="xml",
                url="https://journals.plos.org/plosone/article/file",
                params={"id": doi, "type": "manuscript"},
                headers={"Accept": "application/xml"},
                expected_content_type="xml",
            )
        )

    if "ieee" in publisher_lower:
        api_key = _first_env("IEEE_API_KEY", "IEEE_XPLORE_API_KEY")
        if not api_key:
            return _missing_credential("IEEE Xplore", "IEEE_API_KEY")
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="IEEE Xplore Full-Text API",
                directory="ieee",
                extension="xml",
                url="https://ieeexploreapi.ieee.org/api/v1/search/articles",
                params={"apikey": api_key, "doi": doi},
                headers={"Accept": "application/json"},
                expected_content_type="xml",
                strategy="ieee_by_doi",
            )
        )

    if publisher_lower == "aps" or "american physical society" in publisher_lower:
        return PublisherAccessDecision(
            PublisherAccessPlan(
                provider="APS Harvest Open Access API",
                directory="aps",
                extension="pdf",
                url=f"https://harvest.aps.org/v2/journals/articles/{doi}",
                headers={"Accept": "application/pdf"},
                expected_content_type="pdf",
                response_validator="aps_open_access_pdf",
            )
        )

    return PublisherAccessDecision(
        plan=None,
        reason=(
            f"SALSA has no configured TDM API for {publisher}. "
            "Upload an authorized PDF instead."
        ),
    )
