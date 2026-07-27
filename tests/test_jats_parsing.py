import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests


PAPER_DOWNLOADER_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "paper-downloader"
)
sys.path.insert(0, str(PAPER_DOWNLOADER_SRC))

from paper_downloader.jats import parse_jats
from paper_downloader.parsers.jats_extractor import (
    extract_jats_figures,
    extract_jats_tables,
)


JATS_RESPONSE = """<?xml version="1.0"?>
<response xmlns:xlink="http://www.w3.org/1999/xlink">
  <records>
    <article>
      <front>
        <journal-meta><journal-title>Test Journal</journal-title></journal-meta>
        <article-meta>
          <article-id pub-id-type="doi">10.1371/journal.pone.test</article-id>
          <title-group><article-title>Test Article</article-title></title-group>
          <abstract><p>Test abstract.</p></abstract>
          <kwd-group><kwd>Polymers</kwd></kwd-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Results</title>
          <p>Result paragraph.</p>
          <sec><title>Details</title><p>Detail paragraph.</p></sec>
          <fig>
            <label>Fig 1</label>
            <caption><p>Figure caption.</p></caption>
            <graphic xlink:href="info:doi/10.1371/journal.pone.test.g001"/>
          </fig>
          <table-wrap>
            <label>Table 1</label>
            <caption><p>Table caption.</p></caption>
            <table><tbody><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></tbody></table>
          </table-wrap>
        </sec>
      </body>
    </article>
  </records>
</response>
"""


class FakeImageResponse:
    status_code = 200
    content = b"\x89PNG\r\n\x1a\nexample"
    headers = {"Content-Type": "image/png"}

    def close(self):
        pass


class JATSParsingTests(unittest.TestCase):
    def test_springer_response_envelope_is_unwrapped(self):
        parsed = parse_jats(JATS_RESPONSE)

        self.assertEqual(parsed["DOI"], "10.1371/journal.pone.test")
        self.assertEqual(parsed["Title"], "Test Article")
        self.assertEqual(parsed["Abstract"], "Test abstract.")
        self.assertEqual(
            [section["name"] for section in parsed["Sections"]],
            ["Results", "Results > Details"],
        )

    def test_jats_tables_are_saved_without_an_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "article.xml"
            xml_path.write_text(JATS_RESPONSE, encoding="utf-8")

            count, paths = extract_jats_tables(
                str(xml_path),
                temp_dir,
                "10.1371/journal.pone.test",
            )

            self.assertEqual(count, 1)
            self.assertEqual(len(paths), 1)
            self.assertIn("A,B", Path(paths[0]).read_text())

    def test_plos_figure_url_is_resolved_and_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "article.xml"
            xml_path.write_text(JATS_RESPONSE, encoding="utf-8")

            with patch(
                "paper_downloader.parsers.jats_extractor.requests.get",
                return_value=FakeImageResponse(),
            ) as get:
                count = extract_jats_figures(
                    str(xml_path),
                    temp_dir,
                    "10.1371/journal.pone.test",
                    "plos",
                )

            self.assertEqual(count, 1)
            self.assertIn("plosone/article/figure/image", get.call_args.args[0])
            metadata_path = next(Path(temp_dir).rglob("*_metadata.json"))
            self.assertEqual(
                json.loads(metadata_path.read_text())["caption"],
                "Figure caption.",
            )

    def test_figure_timeout_is_retried_then_skipped_without_aborting_batch(self):
        two_figures = JATS_RESPONSE.replace(
            "</fig>",
            "</fig><fig><label>Fig 2</label><caption><p>Second.</p></caption>"
            '<graphic xlink:href="info:doi/10.1371/journal.pone.test.g002"/></fig>',
            1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "article.xml"
            xml_path.write_text(two_figures, encoding="utf-8")

            with patch(
                "paper_downloader.parsers.jats_extractor.requests.get",
                side_effect=[
                    requests.exceptions.ReadTimeout("slow"),
                    requests.exceptions.ReadTimeout("slow"),
                    requests.exceptions.ReadTimeout("slow"),
                    FakeImageResponse(),
                ],
            ) as get, patch(
                "paper_downloader.parsers.jats_extractor.time.sleep"
            ):
                count = extract_jats_figures(
                    str(xml_path),
                    temp_dir,
                    "10.1371/journal.pone.test",
                    "plos",
                )

            self.assertEqual(count, 1)
            self.assertEqual(get.call_count, 4)
            self.assertEqual(get.call_args.kwargs["timeout"], (10, 60))

    def test_non_jats_xml_has_a_full_text_fallback(self):
        parsed = parse_jats(
            "<document><title>IEEE title</title><doi>10.1109/test</doi>"
            "<body><p>First paragraph.</p><p>Second paragraph.</p></body></document>"
        )

        self.assertEqual(parsed["Title"], "IEEE title")
        self.assertEqual(parsed["DOI"], "10.1109/test")
        self.assertEqual(len(parsed["Sections"][0]["content"]), 2)


if __name__ == "__main__":
    unittest.main()
