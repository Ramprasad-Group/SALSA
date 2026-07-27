import logging
import sys
from pathlib import Path


PAPER_DOWNLOADER = Path(__file__).resolve().parents[1] / "src" / "paper-downloader" / "paper_downloader"
sys.path.insert(0, str(PAPER_DOWNLOADER))

from llm_table_extractor import LLMTableExtractor


ELSEVIER_XML = """\
<root xmlns:ce="http://www.elsevier.com/xml/common/dtd">
  <ce:table><row><entry>first</entry></row></ce:table>
  <ce:table><row><entry>second</entry></row></ce:table>
</root>
"""


def _extractor_with_results(results):
    extractor = LLMTableExtractor.__new__(LLMTableExtractor)
    extractor.logger = logging.getLogger("test-table-extraction")
    extractor.failed_tables = []
    remaining = iter(results)
    extractor.extract_single_table_with_llm_xml = lambda *_args: next(remaining)
    return extractor


def test_failed_xml_table_does_not_stop_subsequent_tables():
    extractor = _extractor_with_results([None, {"table": "second"}])

    tables = extractor.extract_tables_hybrid_xml(ELSEVIER_XML)

    assert tables == [{"table": "second"}]
    assert extractor.failed_tables == [{"source_type": "xml", "table_number": 1}]


def test_all_failed_xml_tables_are_recorded_without_exiting():
    extractor = _extractor_with_results([None, None])

    tables = extractor.extract_tables_hybrid_xml(ELSEVIER_XML)

    assert tables == []
    assert extractor.failed_tables == [
        {"source_type": "xml", "table_number": 1},
        {"source_type": "xml", "table_number": 2},
    ]


def test_failed_html_table_does_not_stop_subsequent_tables():
    extractor = LLMTableExtractor.__new__(LLMTableExtractor)
    extractor.logger = logging.getLogger("test-table-extraction")
    extractor.failed_tables = []
    remaining = iter([None, {"table": "second"}])
    extractor.extract_single_table_with_llm_html = lambda *_args: next(remaining)
    html = "<html><table><tr><td>first</td></tr></table><table><tr><td>second</td></tr></table></html>"

    tables = extractor.extract_tables_hybrid_html(html)

    assert tables == [{"table": "second"}]
    assert extractor.failed_tables == [{"source_type": "html", "table_number": 1}]
