import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd


PAPER_DOWNLOADER_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "paper-downloader"
)
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PAPER_DOWNLOADER_SRC))
sys.path.insert(0, str(SRC_ROOT))

from paper_downloader.downloader import article_downloader, download_article
from paper_downloader.publisher_access import (
    PublisherAccessDecision,
    PublisherAccessPlan,
)


SPRINGER_EMPTY_RESPONSE = b"""<?xml version="1.0"?>
<response>
  <result><total>0</total><recordsDisplayed>0</recordsDisplayed></result>
  <records/>
</response>
"""


class FakeResponse:
    def __init__(self, status_code, content=b"", content_type="application/xml"):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type}

    def close(self):
        pass

    def iter_content(self, chunk_size=65536):
        del chunk_size
        yield self.content


class OpenAccessValidationTests(unittest.TestCase):
    def test_pdf_access_plans_are_saved_under_unstructured_files(self):
        plan = PublisherAccessPlan(
            provider="Test PDF API",
            directory="rsc",
            extension="pdf",
            url="https://example.test/article.pdf",
        )
        records = pd.DataFrame(
            [
                {
                    "pi_id": 1,
                    "pid": 1,
                    "doi": "10.1039/example",
                    "publisher": "RSC",
                    "title": "Example",
                    "pub_year": 2025,
                }
            ]
        )
        captured = {}

        def fake_download(*args, **kwargs):
            captured["directory"] = args[4]
            return True

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "paper_downloader.downloader.publisher_access_for",
            return_value=PublisherAccessDecision(plan),
        ), patch(
            "paper_downloader.downloader.download_article",
            side_effect=fake_download,
        ):
            article_downloader(
                None,
                None,
                temp_dir,
                2015,
                logger=Mock(),
                records_df=records,
            )

        self.assertEqual(
            Path(captured["directory"]),
            Path(temp_dir) / "unstructured_files" / "rsc",
        )

    def test_iop_bot_challenge_is_not_saved_as_full_text(self):
        challenge = b"""<html><head><title>Radware Bot Manager Captcha</title></head>
<body>Confirm you are a human at validate.perfdrive.com.</body></html>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "iop"
            status_path = Path(temp_dir) / "download_status.jsonl"

            with patch(
                "paper_downloader.downloader.requests.get",
                return_value=FakeResponse(200, challenge, "text/html"),
            ):
                result = download_article(
                    "10.1088/example",
                    "IOP Publishing",
                    "Example",
                    2025,
                    str(output_dir),
                    None,
                    1,
                    "html",
                    logger=Mock(),
                    html_url="https://iopscience.iop.org/article/10.1088/example",
                    local_status_log=str(status_path),
                    expected_content_type="html",
                    response_validator="iop_full_text_html",
                )

            self.assertFalse(result)
            self.assertFalse((output_dir / "10.1088@example.html").exists())
            status = json.loads(status_path.read_text().strip())
            self.assertEqual(status["note"]["reason"], "iop_bot_challenge")

    def test_elsevier_rawtext_response_downloads_pdf_fallback(self):
        xml = b"""<?xml version="1.0"?>
<full-text-retrieval-response xmlns:xocs="http://www.elsevier.com/xml/xocs/dtd">
  <originalText><xocs:doc><xocs:rawtext>PDF only</xocs:rawtext></xocs:doc></originalText>
</full-text-retrieval-response>
"""
        pdf = b"%PDF-1.4\nexample"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "papers" / "structured_files" / "elsevier"
            output_dir.mkdir(parents=True)

            with patch(
                "paper_downloader.downloader.requests.get",
                side_effect=[
                    FakeResponse(200, xml),
                    FakeResponse(200, pdf, "application/pdf"),
                ],
            ), patch("paper_downloader.downloader.sleep"):
                result = download_article(
                    "10.1016/example",
                    "Elsevier",
                    "Example",
                    2020,
                    str(output_dir),
                    None,
                    1,
                    "xml",
                    logger=Mock(),
                    html_url="https://api.elsevier.com/content/article/doi/10.1016/example",
                    extra_headers={"X-ELS-APIKey": "test-key", "Accept": "text/xml"},
                    expected_content_type="xml",
                    response_validator="elsevier_full_text_or_pdf",
                )

            self.assertTrue(result)
            self.assertTrue((output_dir / "10.1016@example.xml").exists())
            self.assertEqual(
                (
                    Path(temp_dir)
                    / "papers"
                    / "unstructured_files"
                    / "elsevier"
                    / "10.1016@example.pdf"
                ).read_bytes(),
                pdf,
            )

    def test_elsevier_abstract_only_response_downloads_declared_pdf_attachment(self):
        xml = b"""<?xml version="1.0"?>
<full-text-retrieval-response xmlns:xocs="http://www.elsevier.com/xml/xocs/dtd">
  <coredata><dc:description xmlns:dc="http://purl.org/dc/elements/1.1/">Abstract only</dc:description></coredata>
  <originalText>
    <xocs:doc>
      <xocs:attachment-metadata-doc>
        <xocs:attachments>
          <xocs:web-pdf>
            <xocs:attachment-eid>1-s2.0-example-main.pdf</xocs:attachment-eid>
            <xocs:filesize>20</xocs:filesize>
            <xocs:web-pdf-page-count>7</xocs:web-pdf-page-count>
          </xocs:web-pdf>
        </xocs:attachments>
      </xocs:attachment-metadata-doc>
    </xocs:doc>
  </originalText>
</full-text-retrieval-response>
"""
        pdf = b"%PDF-1.4\nfull-pdf-data"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "papers" / "structured_files" / "elsevier"
            output_dir.mkdir(parents=True)

            with patch(
                "paper_downloader.downloader.requests.get",
                side_effect=[
                    FakeResponse(200, xml),
                    FakeResponse(200, pdf, "application/pdf"),
                ],
            ) as request_get, patch("paper_downloader.downloader.sleep"):
                result = download_article(
                    "10.1006/example",
                    "Elsevier",
                    "Example",
                    2001,
                    str(output_dir),
                    None,
                    1,
                    "xml",
                    logger=Mock(),
                    html_url="https://api.elsevier.com/content/article/doi/10.1006/example",
                    extra_headers={"X-ELS-APIKey": "test-key", "Accept": "text/xml"},
                    expected_content_type="xml",
                    response_validator="elsevier_full_text_or_pdf",
                )

            self.assertTrue(result)
            pdf_request = request_get.call_args_list[1]
            self.assertEqual(
                pdf_request.args[0],
                "https://api.elsevier.com/content/object/eid/1-s2.0-example-main.pdf",
            )
            self.assertEqual(pdf_request.kwargs["params"], {"view": "STANDARD"})

    def test_elsevier_incomplete_cached_preview_is_replaced(self):
        xml = b"""<?xml version="1.0"?>
<full-text-retrieval-response xmlns:xocs="http://www.elsevier.com/xml/xocs/dtd">
  <originalText>
    <xocs:doc>
      <xocs:rawtext>Converted text</xocs:rawtext>
      <xocs:attachment-metadata-doc>
        <xocs:attachments>
          <xocs:web-pdf>
            <xocs:attachment-eid>1-s2.0-example-main.pdf</xocs:attachment-eid>
            <xocs:filesize>100</xocs:filesize>
            <xocs:web-pdf-page-count>7</xocs:web-pdf-page-count>
          </xocs:web-pdf>
        </xocs:attachments>
      </xocs:attachment-metadata-doc>
    </xocs:doc>
  </originalText>
</full-text-retrieval-response>
"""
        full_pdf = b"%PDF-" + (b"x" * 95)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "papers" / "structured_files" / "elsevier"
            output_dir.mkdir(parents=True)
            xml_path = output_dir / "10.1016@example.xml"
            xml_path.write_bytes(xml)
            pdf_path = (
                Path(temp_dir)
                / "papers"
                / "unstructured_files"
                / "elsevier"
                / "10.1016@example.pdf"
            )
            pdf_path.parent.mkdir(parents=True)
            pdf_path.write_bytes(b"%PDF-preview")

            with patch(
                "paper_downloader.downloader.requests.get",
                return_value=FakeResponse(200, full_pdf, "application/pdf"),
            ), patch("paper_downloader.downloader.sleep"):
                result = download_article(
                    "10.1016/example",
                    "Elsevier",
                    "Example",
                    1976,
                    str(output_dir),
                    None,
                    1,
                    "xml",
                    logger=Mock(),
                    html_url="https://api.elsevier.com/content/article/doi/10.1016/example",
                    extra_headers={"X-ELS-APIKey": "test-key", "Accept": "text/xml"},
                    expected_content_type="xml",
                    response_validator="elsevier_full_text_or_pdf",
                )

            self.assertTrue(result)
            self.assertEqual(pdf_path.read_bytes(), full_pdf)

    def test_springer_zero_record_response_is_not_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nature"
            output_dir.mkdir()
            article_path = output_dir / "10.1038@227680a0.xml"
            article_path.write_bytes(SPRINGER_EMPTY_RESPONSE)
            status_path = Path(temp_dir) / "download_status.jsonl"

            with patch(
                "paper_downloader.downloader.requests.get",
                return_value=FakeResponse(200, SPRINGER_EMPTY_RESPONSE),
            ):
                result = download_article(
                    "10.1038/227680a0",
                    "Nature",
                    "Example",
                    1970,
                    str(output_dir),
                    None,
                    1,
                    "xml",
                    logger=Mock(),
                    html_url="https://api.springernature.com/openaccess/jats",
                    local_status_log=str(status_path),
                    expected_content_type="xml",
                    response_validator="springer_open_access_jats",
                )

            self.assertFalse(result)
            self.assertFalse(article_path.exists())
            status = json.loads(status_path.read_text().strip())
            self.assertEqual(
                status["note"]["reason"],
                "springer_no_open_access_record",
            )

    def test_aps_unauthorized_response_is_recorded_as_not_open_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "aps"
            status_path = Path(temp_dir) / "download_status.jsonl"

            with patch(
                "paper_downloader.downloader.requests.get",
                return_value=FakeResponse(401, b'{"errors": []}', "application/json"),
            ):
                result = download_article(
                    "10.1103/PhysRevB.54.11169",
                    "APS",
                    "Example",
                    1996,
                    str(output_dir),
                    None,
                    1,
                    "pdf",
                    logger=Mock(),
                    html_url="https://harvest.aps.org/v2/journals/articles/example",
                    local_status_log=str(status_path),
                    expected_content_type="pdf",
                    response_validator="aps_open_access_pdf",
                )

            self.assertFalse(result)
            self.assertFalse(
                (output_dir / "10.1103@PhysRevB.54.11169.pdf").exists()
            )
            status = json.loads(status_path.read_text().strip())
            self.assertEqual(status["note"]["reason"], "aps_not_open_access")
            self.assertEqual(status["note"]["status_code"], 401)


if __name__ == "__main__":
    unittest.main()
