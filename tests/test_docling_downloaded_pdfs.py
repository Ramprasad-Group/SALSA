import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


UI_SRC = Path(__file__).resolve().parents[1] / "src" / "ui"
sys.path.insert(0, str(UI_SRC))

from docling_pdf import PdfUpload, _downloaded_pdf_manifest, _run_docling_sidecar, extract_uploaded_pdfs_with_docling


class DownloadedPdfManifestTests(unittest.TestCase):
    def test_discovers_elsevier_pdf_and_recovers_doi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            pdf_dir = run_dir / "papers" / "unstructured_files" / "elsevier"
            pdf_dir.mkdir(parents=True)
            pdf_path = pdf_dir / "10.1016@example.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            manifest = _downloaded_pdf_manifest(run_dir)

            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["doi"], "10.1016/example")
            self.assertEqual(manifest[0]["safe_doi"], "10.1016@example")
            self.assertEqual(manifest[0]["pdf_path"], str(pdf_path))
            self.assertEqual(manifest[0]["publisher"], "Elsevier")

    def test_downloaded_pdf_sidecar_disables_ocr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text("[]", encoding="utf-8")

            old_python = os.environ.get("DOCLING_PYTHON")
            os.environ["DOCLING_PYTHON"] = sys.executable
            try:
                with mock.patch("docling_pdf.subprocess.Popen") as popen:
                    process = popen.return_value
                    process.poll.return_value = 0
                    process.returncode = 0

                    _run_docling_sidecar(
                        run_dir=run_dir,
                        manifest_path=manifest_path,
                        run_extract_tables=True,
                        run_extract_figures=True,
                        use_ocr=False,
                    )

                command = popen.call_args.args[0]
                self.assertIn("--no-ocr", command)
            finally:
                if old_python is None:
                    os.environ.pop("DOCLING_PYTHON", None)
                else:
                    os.environ["DOCLING_PYTHON"] = old_python

    def test_uploaded_pdf_sidecar_disables_ocr_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)

            old_python = os.environ.get("DOCLING_PYTHON")
            os.environ["DOCLING_PYTHON"] = sys.executable
            try:
                with mock.patch("docling_pdf._docling_available_here", return_value=False):
                    with mock.patch("docling_pdf.subprocess.Popen") as popen:
                        process = popen.return_value
                        process.poll.return_value = 0
                        process.returncode = 0

                        extract_uploaded_pdfs_with_docling(
                            [PdfUpload(filename="paper.pdf", content=b"%PDF-1.4")],
                            run_dir=run_dir,
                            run_extract_tables=True,
                            run_extract_figures=True,
                        )

                command = popen.call_args.args[0]
                self.assertIn("--no-ocr", command)
            finally:
                if old_python is None:
                    os.environ.pop("DOCLING_PYTHON", None)
                else:
                    os.environ["DOCLING_PYTHON"] = old_python

    def test_uploaded_pdf_sidecar_can_enable_ocr_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)

            old_python = os.environ.get("DOCLING_PYTHON")
            os.environ["DOCLING_PYTHON"] = sys.executable
            try:
                with mock.patch("docling_pdf._docling_available_here", return_value=False):
                    with mock.patch("docling_pdf.subprocess.Popen") as popen:
                        process = popen.return_value
                        process.poll.return_value = 0
                        process.returncode = 0

                        extract_uploaded_pdfs_with_docling(
                            [PdfUpload(filename="paper.pdf", content=b"%PDF-1.4")],
                            run_dir=run_dir,
                            run_extract_tables=True,
                            run_extract_figures=True,
                            use_ocr=True,
                        )

                command = popen.call_args.args[0]
                self.assertNotIn("--no-ocr", command)
            finally:
                if old_python is None:
                    os.environ.pop("DOCLING_PYTHON", None)
                else:
                    os.environ["DOCLING_PYTHON"] = old_python


if __name__ == "__main__":
    unittest.main()
