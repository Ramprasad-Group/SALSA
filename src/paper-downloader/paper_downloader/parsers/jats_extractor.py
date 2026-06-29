"""Figure and table extraction for JATS XML publishers."""

import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from paper_downloader.jats import (
    descendants,
    first_descendant,
    local_name,
    parse_xml,
    text_content,
)


XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
PLOS_JOURNAL_SLUGS = {
    "pcsy": "complexsystems",
    "pone": "plosone",
    "pbio": "plosbiology",
    "pgen": "plosgenetics",
    "pmed": "plosmedicine",
    "ppat": "plospathogens",
    "pcbi": "ploscompbiol",
    "pntd": "plosntds",
    "pctr": "plosclinicaltrials",
    "pclm": "plosclimate",
}


def _safe_label(value, fallback):
    clean = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    ).strip("_")
    return clean or fallback


def _load_article(file_path):
    content = Path(file_path).read_text(encoding="utf-8")
    return parse_xml(content)[1]


def _figure_url(publisher, doi, href, figure):
    if href.startswith(("http://", "https://")):
        return href
    if publisher in {"springer", "nature"} and href.startswith("MediaObjects/"):
        return (
            "https://media.springernature.com/full/springer-static/image/"
            f"art%3A{quote(doi, safe='')}/{href}"
        )
    if doi.lower().startswith("10.1371/") and (
        href.startswith("info:doi/") or ".g" in href
    ):
        parts = doi.lower().split(".")
        journal_code = parts[2] if len(parts) > 3 else ""
        slug = PLOS_JOURNAL_SLUGS.get(journal_code)
        if slug:
            figure_id = href
            if not figure_id.startswith("info:doi/"):
                match = re.search(r"\.(g\d+)(?:\.|$)", href, re.IGNORECASE)
                if not match:
                    return ""
                figure_id = f"info:doi/{doi}.{match.group(1)}"
            return (
                f"https://journals.plos.org/{slug}/article/figure/image"
                f"?size=large&id={figure_id}"
            )
    if publisher in {"pubmed", "pmc"}:
        pmcid = text_content(
            next(
                (
                    element
                    for element in descendants(figure, "article-id")
                    if element.get("pub-id-type") in {"pmc", "pmcid"}
                ),
                None,
            )
        )
        if pmcid:
            numeric_pmcid = pmcid.upper()
            if numeric_pmcid.startswith("PMC"):
                numeric_pmcid = numeric_pmcid[3:]
            return (
                "https://pmc.ncbi.nlm.nih.gov/articles/instance/"
                f"{numeric_pmcid}/bin/{href}"
            )
    return ""


def extract_jats_figures(file_path, output_dir, doi, publisher):
    logger = logging.getLogger("extraction")
    article = _load_article(file_path)
    figures = descendants(article, "fig")
    if not figures:
        return 0

    figure_dir = Path(output_dir) / "figures" / doi.replace("/", "@")
    saved = 0
    for index, figure in enumerate(figures, 1):
        label = text_content(first_descendant(figure, "label")) or f"Figure {index}"
        caption = text_content(first_descendant(figure, "caption"))
        graphics = descendants(figure, "graphic")
        graphic = next(
            (
                candidate
                for candidate in graphics
                if ".g" in candidate.get(XLINK_HREF, "")
                or "MediaObjects/" in candidate.get(XLINK_HREF, "")
            ),
            graphics[0] if graphics else None,
        )
        href = graphic.get(XLINK_HREF, "") if graphic is not None else ""
        url = _figure_url(publisher.lower(), doi, href, article)
        if not url:
            logger.warning("No resolvable JATS figure URL for %s (%s)", label, href)
            continue

        response = requests.get(
            url,
            headers={"Accept": "image/*", "User-Agent": "SALSA/1.0"},
            timeout=30,
        )
        content_type = response.headers.get("Content-Type", "").lower()
        if response.status_code != 200 or "image" not in content_type:
            logger.warning(
                "Could not retrieve %s for %s: HTTP %s (%s)",
                label,
                doi,
                response.status_code,
                content_type,
            )
            response.close()
            continue

        extension = content_type.split("/", 1)[-1].split(";", 1)[0]
        extension = {"jpeg": "jpg", "svg+xml": "svg"}.get(extension, extension)
        clean_label = _safe_label(label, f"Figure_{index}")
        figure_dir.mkdir(parents=True, exist_ok=True)
        image_path = figure_dir / f"{clean_label}.{extension}"
        image_path.write_bytes(response.content)
        response.close()
        metadata = {
            "original_label": label,
            "clean_label": clean_label,
            "caption": caption,
            "url": url,
            "doi": doi,
            "figure_file_path": str(image_path),
            "figure_directory": str(figure_dir),
            "file_type": "figure",
            "publisher": publisher,
        }
        (figure_dir / f"{clean_label}_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved += 1
    return saved


def _table_rows(table):
    rows = []
    for row in descendants(table, "tr") + descendants(table, "row"):
        values = [
            text_content(cell)
            for cell in list(row)
            if local_name(cell.tag) in {"th", "td", "entry"}
        ]
        if values:
            rows.append(values)
    return rows


def extract_jats_tables(file_path, output_dir, doi):
    article = _load_article(file_path)
    wrappers = descendants(article, "table-wrap")
    table_dir = Path(output_dir) / "tables" / doi.replace("/", "@")
    saved = 0
    saved_files = []
    for index, wrapper in enumerate(wrappers, 1):
        table = first_descendant(wrapper, "table")
        if table is None:
            continue
        rows = _table_rows(table)
        if not rows:
            continue
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        label = text_content(first_descendant(wrapper, "label")) or f"Table {index}"
        caption = text_content(first_descendant(wrapper, "caption"))
        clean_label = _safe_label(label, f"Table_{index}")
        table_dir.mkdir(parents=True, exist_ok=True)
        csv_path = table_dir / f"{clean_label}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, header=False)
        metadata = {
            "label": label,
            "caption": caption,
            "doi": doi,
            "table_file_path": str(csv_path),
            "file_type": "table",
        }
        (table_dir / f"{clean_label}_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved += 1
        saved_files.append(str(csv_path))
    return saved, saved_files
