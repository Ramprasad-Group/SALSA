import os
from pathlib import Path
import sys
import tempfile
import unittest


UI_SRC = Path(__file__).resolve().parents[1] / "src" / "ui"
sys.path.insert(0, str(UI_SRC))

from pipeline import PipelineCancelled, _raise_if_cancelled
from run_progress import (
    RUN_STATUS_FILE_ENV,
    cancel_request_path,
    clear_run_cancellation,
    read_run_status,
    request_run_cancellation,
    write_run_status,
)


class RunCancellationTests(unittest.TestCase):
    def test_disk_cancel_request_is_seen_by_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            old_status_file = os.environ.get(RUN_STATUS_FILE_ENV)
            os.environ[RUN_STATUS_FILE_ENV] = str(run_dir / "run_status.json")
            try:
                request_run_cancellation(run_dir)
                self.assertTrue(cancel_request_path(run_dir).exists())
                with self.assertRaises(PipelineCancelled):
                    _raise_if_cancelled()

                clear_run_cancellation(run_dir)
                self.assertFalse(cancel_request_path(run_dir).exists())
                _raise_if_cancelled()
            finally:
                if old_status_file is None:
                    os.environ.pop(RUN_STATUS_FILE_ENV, None)
                else:
                    os.environ[RUN_STATUS_FILE_ENV] = old_status_file

    def test_status_updates_preserve_existing_owner_pid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "run_status.json"

            write_run_status(status_path, state="running", stage="Starting", pid=12345)
            write_run_status(status_path, state="running", stage="Downloading papers")

            self.assertEqual(read_run_status(status_path)["pid"], 12345)


if __name__ == "__main__":
    unittest.main()
