"""Shared parsing helpers for JATS XML publisher responses."""

import re
import xml.etree.ElementTree as ET


def local_name(tag):
    return str(tag).rsplit("}", 1)[-1]


def text_content(element):
    if element is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def first_descendant(element, name, **attrs):
    if element is None:
        return None
    for candidate in element.iter():
        if local_name(candidate.tag) != name:
            continue
        if all(candidate.get(key) == value for key, value in attrs.items()):
            return candidate
    return None


def descendants(element, name):
    if element is None:
        return []
    return [
        candidate
        for candidate in element.iter()
        if local_name(candidate.tag) == name
    ]


def article_element(root):
    if local_name(root.tag) == "article":
        return root
    return first_descendant(root, "article")


def parse_xml(xml_content):
    root = ET.fromstring(xml_content)
    article = article_element(root)
    if article is None:
        raise ValueError("No JATS article element was found.")
    return root, article


def _section_paragraphs(section):
    paragraphs = []
    for child in list(section):
        child_name = local_name(child.tag)
        if child_name in {"p", "list", "disp-quote", "boxed-text"}:
            value = text_content(child)
            if value:
                paragraphs.append(value)
    return paragraphs


def _collect_sections(section, parent_title=""):
    title_element = next(
        (child for child in list(section) if local_name(child.tag) == "title"),
        None,
    )
    title = text_content(title_element) or "Untitled section"
    section_title = f"{parent_title} > {title}" if parent_title else title
    result = []
    paragraphs = _section_paragraphs(section)
    if paragraphs:
        result.append(
            {
                "name": section_title,
                "type": "section",
                "content": paragraphs,
            }
        )
    for child in list(section):
        if local_name(child.tag) == "sec":
            result.extend(_collect_sections(child, section_title))
    return result


def parse_jats(xml_content):
    root = ET.fromstring(xml_content)
    article = article_element(root)
    if article is None:
        return parse_generic_xml(root)
    front = first_descendant(article, "front")
    body = first_descendant(article, "body")

    doi_element = None
    for candidate in descendants(front, "article-id"):
        if candidate.get("pub-id-type") == "doi":
            doi_element = candidate
            break

    abstract_element = first_descendant(front, "abstract")
    abstract_parts = [
        text_content(element)
        for element in descendants(abstract_element, "p")
        if text_content(element)
    ]

    sections = []
    loose_paragraphs = []
    if body is not None:
        for child in list(body):
            child_name = local_name(child.tag)
            if child_name == "sec":
                sections.extend(_collect_sections(child))
            elif child_name == "p":
                value = text_content(child)
                if value:
                    loose_paragraphs.append(value)
    if loose_paragraphs:
        sections.insert(
            0,
            {
                "name": "Full text",
                "type": "section",
                "content": loose_paragraphs,
            },
        )

    return {
        "DOI": text_content(doi_element),
        "Keywords": [
            text_content(element)
            for element in descendants(front, "kwd")
            if text_content(element)
        ],
        "Title": text_content(first_descendant(front, "article-title")),
        "Journal": text_content(first_descendant(front, "journal-title")),
        "Abstract": " ".join(abstract_parts),
        "Sections": sections,
    }


def parse_generic_xml(root):
    """Best-effort fallback for structured publisher XML that is not JATS."""
    title = first_descendant(root, "article-title")
    if title is None:
        title = first_descendant(root, "title")
    doi = next(
        (
            element
            for element in root.iter()
            if local_name(element.tag) in {"doi", "article-id"}
            and (
                element.get("pub-id-type") in {None, "doi"}
                or local_name(element.tag) == "doi"
            )
        ),
        None,
    )
    body = first_descendant(root, "body")
    if body is None:
        body = first_descendant(root, "fulltext")
    paragraphs = [
        text_content(element)
        for element in descendants(body, "p")
        if text_content(element)
    ]
    return {
        "DOI": text_content(doi),
        "Keywords": [],
        "Title": text_content(title),
        "Journal": text_content(first_descendant(root, "journal-title")),
        "Abstract": text_content(first_descendant(root, "abstract")),
        "Sections": [
            {
                "name": "Full text",
                "type": "section",
                "content": paragraphs,
            }
        ]
        if paragraphs
        else [],
    }
