import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cost.sum_cost import summarise_usage  # noqa: E402


class TestCostStageSummary(unittest.TestCase):
    def test_summarise_usage_groups_cost_by_stage(self):
        records = [
            {
                "model": "gpt-4.1-mini",
                "input_tokens": 1000,
                "cached_tokens": 100,
                "output_tokens": 200,
                "extra": {"kind": "table_xml"},
            },
            {
                "model": "gpt-4.1-mini",
                "input_tokens": 500,
                "cached_tokens": 0,
                "output_tokens": 50,
                "extra": {"kind": "table_html"},
            },
            {
                "model": "gpt-5.1",
                "input_tokens": 100,
                "cached_tokens": 0,
                "output_tokens": 10,
                "extra": {"kind": "filter_figure"},
            },
            {
                "model": "gpt-5.1",
                "input_tokens": 120,
                "cached_tokens": 0,
                "output_tokens": 12,
                "extra": {"kind": "filter_table"},
            },
        ]

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()

            summary = summarise_usage(handle.name)

        self.assertEqual(summary["stages"]["Table recreation"]["calls"], 2)
        self.assertEqual(
            summary["stages"]["Table recreation"]["models"]["gpt-4.1-mini"]["calls"],
            2,
        )
        self.assertEqual(summary["stages"]["Figure filtering"]["calls"], 1)
        self.assertEqual(
            summary["stages"]["Figure filtering"]["models"]["gpt-5.1"]["calls"],
            1,
        )
        self.assertEqual(summary["stages"]["Table filtering"]["calls"], 1)
        self.assertAlmostEqual(
            summary["grand_total"],
            sum(stage["cost"] for stage in summary["stages"].values()),
        )


if __name__ == "__main__":
    unittest.main()
