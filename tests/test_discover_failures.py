import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


UI_SRC = Path(__file__).resolve().parents[1] / "src" / "ui"
sys.path.insert(0, str(UI_SRC))

from discover import discover_run


class DiscoverFailuresTests(unittest.TestCase):
    def test_unknown_prefix_record_failure_is_shown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            records_dir = run_dir / "records"
            records_dir.mkdir(parents=True)

            with (records_dir / "download_results_1.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "doi", "publisher", "title", "pub_year",
                        "success", "message",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "doi": "10.9999/example",
                    "publisher": "Unknown",
                    "success": "False",
                    "message": (
                        "Unknown DOI prefix '10.9999'. "
                        "No publisher mapping is configured for this prefix."
                    ),
                })

            result = discover_run(run_dir)

            self.assertEqual(len(result.failures), 1)
            self.assertEqual(result.failures[0].stage, "Record lookup")
            self.assertIn("10.9999", result.failures[0].message)

    def test_download_failure_overrides_reparse_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            records_dir = run_dir / "records"
            logs_dir = run_dir / "papers" / "logs"
            reparsed_dir = run_dir / "reparsed"
            records_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)
            reparsed_dir.mkdir(parents=True)

            with (records_dir / "download_results_1.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "doi", "publisher", "title", "pub_year",
                        "success", "message",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "doi": "10.1109/example",
                    "publisher": "IEEE",
                    "success": "True",
                    "message": "Metadata saved",
                })

            status = {
                "doi": "10.1109/example",
                "publisher": "IEEE",
                "download_status": 5,
                "note": {
                    "reason": "publisher_access_error",
                    "message": "IEEE Xplore returned HTTP 403.",
                },
            }
            (logs_dir / "download_status.jsonl").write_text(
                json.dumps(status) + "\n",
                encoding="utf-8",
            )

            with (reparsed_dir / "index.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["doi", "status"])
                writer.writeheader()
                writer.writerow({
                    "doi": "10.1109/example",
                    "status": "missing_file",
                })

            result = discover_run(run_dir)

            self.assertEqual(len(result.failures), 1)
            self.assertEqual(result.failures[0].stage, "Paper download")
            self.assertEqual(
                result.failures[0].message,
                "IEEE Xplore returned HTTP 403.",
            )


if __name__ == "__main__":
    unittest.main()
