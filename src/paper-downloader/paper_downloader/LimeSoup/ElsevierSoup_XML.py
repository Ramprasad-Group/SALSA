import re

import bs4

from LimeSoup.lime_soup import Soup, RuleIngredient
from LimeSoup.parser.elsevier_xml import (
    resolve_elsevier_entities, extract_ce_text, find_non_empty_children,
    node_named, extract_ce_section, extract_ce_abstract,
    extract_ce_title, remove_consecutive_whitespaces, extract_ce_para    ) #extract_ce_para

__author__ = 'Haoyan Huo, Nicolas Mingione'
__maintainer__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__version__ = '0.3.2-xml'
__all__ = ['ElsevierXMLSoup']

class ElsevierParseXML(RuleIngredient):
    @staticmethod
    def _parse(xml_str):
        xml_str = resolve_elsevier_entities(xml_str)
        return bs4.BeautifulSoup(xml_str, 'lxml-xml')


class ElsevierReadMetaData(RuleIngredient):
    @staticmethod
    def get_text_or_none(soup, name, handler=None):
        if soup is None:
            return None

        node = soup.find(name=name)
        if node is None:
            return None
        elif handler is not None:
            return handler(node)
        else:
            return node.get_text().strip()

    @staticmethod
    def _parse(soup):
        # journal
        journal_name = ElsevierReadMetaData.get_text_or_none(soup, 'xocs:srctitle') or \
                       ElsevierReadMetaData.get_text_or_none(soup, 'prism:publicationName')
        doi = ElsevierReadMetaData.get_text_or_none(soup, 'xocs:doi')

        # https://www.elsevier.com/__data/assets/pdf_file/0003/58872/ja5_tagbytag5_v1.9.5.pdf
        # Elsevier XML definition pp. 46
        head_node = soup.find('head')

        title = ElsevierReadMetaData.get_text_or_none(head_node, 'ce:title', extract_ce_title) or \
                ElsevierReadMetaData.get_text_or_none(soup, 'dc:title')

        keywords = []
        if head_node is not None:
            # Elsevier XML definition pp. 366
            for node in head_node.find_all('ce:keyword'):
                text_node = node.find('ce:text')
                if text_node is not None:
                    keyword = remove_consecutive_whitespaces(
                        extract_ce_text(text_node),
                        keep_newline=False
                    ).strip()
                    keywords.append(keyword)

        if len(keywords) == 0:
            for subject in soup.find_all('dcterms:subject'):
                keywords.append(subject.get_text().strip())

        return soup, {
            'Journal': journal_name,
            'DOI': doi,
            'Title': title,
            'Keywords': keywords
        }


class ElsevierCollect(RuleIngredient):

    @staticmethod
    def _parse(args):
        soup, obj = args
        paragraphs = []

        # find all sections
        for node in soup.find_all('ce:abstract'):
            abstract_paragraph = extract_ce_abstract(node)
            normalized_name = re.sub(r'[^\w]', '', abstract_paragraph['name'])
            if re.match(r'abstracts?', normalized_name, re.IGNORECASE):
                paragraphs.append(abstract_paragraph)

        sections = soup.find('ce:sections')
        if sections is not None:
            for node in find_non_empty_children(sections):
                if node_named(node, 'ce:para'):
                    paragraphs.extend(extract_ce_para(node).split('\n'))
                elif node_named(node, 'ce:section'):
                    # Preserve citation numbers by replacing ce:cross-ref tags before processing
                    citation_refs = node.find_all('ce:cross-ref')
                    for ref in citation_refs:
                        ref.replace_with(f"{ref.get_text()}")
                    paragraphs.append(extract_ce_section(node))

        # Find and add reference list/bibliography
        reference_sections = []
        
        # Look for bibliography in different possible locations
        bibliography = soup.find('ce:bibliography')
        if bibliography is not None:
            # Extract bibliography content manually since extract_ce_section only works with ce:section
            bib_content = []
            for ref in bibliography.find_all('ce:bib-reference'):
                ref_text = ref.get_text().strip()
                if ref_text:
                    # Clean up excessive whitespace and normalize text
                    cleaned_text = re.sub(r'\s+', ' ', ref_text).strip()
                    bib_content.append(cleaned_text)
            if bib_content:
                reference_sections.append({
                    'type': 'ce_bibliography',
                    'name': 'References',
                    'content': bib_content
                })
        
        # Also check for reference-list
        reference_list = soup.find('ce:reference-list')
        if reference_list is not None:
            # Extract reference list content manually
            ref_content = []
            for ref in reference_list.find_all('ce:reference'):
                ref_text = ref.get_text().strip()
                if ref_text:
                    # Clean up excessive whitespace and normalize text
                    cleaned_text = re.sub(r'\s+', ' ', ref_text).strip()
                    ref_content.append(cleaned_text)
            if ref_content:
                reference_sections.append({
                    'type': 'ce_reference_list',
                    'name': 'References',
                    'content': ref_content
                })
            
        # Check for references within sections
        for ref_section in soup.find_all('ce:section'):
            section_title = ref_section.find('ce:title')
            if section_title:
                title_text = section_title.get_text().lower()
                if any(keyword in title_text for keyword in ['reference', 'bibliography', 'cited']):
                    # Preserve citation numbers in references too
                    citation_refs = ref_section.find_all('ce:cross-ref')
                    for ref in citation_refs:
                        ref.replace_with(f"{ref.get_text()}")
                    reference_sections.append(extract_ce_section(ref_section))
        
        # Add reference sections to paragraphs
        paragraphs.extend(reference_sections)

        obj['Sections'] = paragraphs
        return obj


ElsevierXMLSoup = Soup(parser_version=__version__)
ElsevierXMLSoup.add_ingredient(ElsevierParseXML())
ElsevierXMLSoup.add_ingredient(ElsevierReadMetaData())
ElsevierXMLSoup.add_ingredient(ElsevierCollect())
