import sys
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
    _validate_input_item_limit,
)


class TestPipelineInputLimit(unittest.TestCase):
    def test_default_limit_remains_fifty(self):
        self.assertEqual(_validate_input_item_limit(50), DEFAULT_INPUT_ITEM_LIMIT)
        with self.assertRaisesRegex(ValueError, "at most 50"):
            _validate_input_item_limit(51)

    def test_admin_limit_allows_one_hundred(self):
        self.assertEqual(
            _validate_input_item_limit(100, ADMIN_INPUT_ITEM_LIMIT),
            ADMIN_INPUT_ITEM_LIMIT,
        )
        with self.assertRaisesRegex(ValueError, "at most 100"):
            _validate_input_item_limit(101, ADMIN_INPUT_ITEM_LIMIT)


if __name__ == "__main__":
    unittest.main()
