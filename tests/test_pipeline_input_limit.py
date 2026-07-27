import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ui"))
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

from pipeline import (  # noqa: E402
    ADMIN_INPUT_ITEM_LIMIT,
    DEFAULT_INPUT_ITEM_LIMIT,
    _unexpected_preexisting_run_outputs,
    _validate_input_item_limit,
)


class TestPipelineInputLimit(unittest.TestCase):
    def test_default_limit_is_ten(self):
        self.assertEqual(_validate_input_item_limit(10), DEFAULT_INPUT_ITEM_LIMIT)
        with self.assertRaisesRegex(ValueError, "at most 10"):
            _validate_input_item_limit(11)

    def test_admin_limit_allows_one_hundred(self):
        self.assertEqual(
            _validate_input_item_limit(100, ADMIN_INPUT_ITEM_LIMIT),
            ADMIN_INPUT_ITEM_LIMIT,
        )
        with self.assertRaisesRegex(ValueError, "at most 100"):
            _validate_input_item_limit(101, ADMIN_INPUT_ITEM_LIMIT)

    def test_run_manifest_is_allowed_before_pipeline_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_manifest.json").write_text("{}")
            (run_dir / "run_status.json").write_text("{}")
            (run_dir / "pipeline_worker.log").write_text("")
            self.assertEqual(_unexpected_preexisting_run_outputs(run_dir), [])

            output = run_dir / "records"
            output.mkdir()
            self.assertEqual(_unexpected_preexisting_run_outputs(run_dir), [output])


if __name__ == "__main__":
    unittest.main()
