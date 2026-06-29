from pathlib import Path
import sys
import unittest
from unittest import mock

import pandas as pd


COMPLETE_DATASET_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "complete_dataset"
)
sys.path.insert(0, str(COMPLETE_DATASET_SRC))

from row_emitter import OUTPUT_COLUMNS, _coerce_rows_schema, infer_rows_for_table


class CompleteDatasetDefaultSchemaTests(unittest.TestCase):
    def test_default_schema_is_general_purpose(self):
        self.assertIn("independent variable", OUTPUT_COLUMNS)
        self.assertIn("dependant variable", OUTPUT_COLUMNS)
        self.assertIn("dependent value", OUTPUT_COLUMNS)
        self.assertIn("dependent unit", OUTPUT_COLUMNS)
        self.assertNotIn("property", OUTPUT_COLUMNS)
        self.assertNotIn("value", OUTPUT_COLUMNS)
        self.assertNotIn("unit", OUTPUT_COLUMNS)
        self.assertNotIn("degradation test temp (C)", OUTPUT_COLUMNS)
        self.assertNotIn("solvent molarity (M)", OUTPUT_COLUMNS)

    def test_empty_custom_columns_use_general_schema(self):
        frame = _coerce_rows_schema(
            [{
                "dependant variable": "conductivity",
                "dependent value": "5",
                "dependent unit": "mS/cm",
            }],
            doi="10.1/example",
            source_kind="table",
            source_name="Table 1",
            columns=[],
        )

        self.assertEqual(list(frame.columns), OUTPUT_COLUMNS)

    def test_cached_rows_preserve_requested_custom_schema(self):
        custom_columns = ["material", "measurement", "DOI"]
        with mock.patch(
            "row_emitter.load_cached_rows",
            return_value=[
                {
                    "material": "A",
                    "measurement": "5",
                    "DOI": "10.1/example",
                }
            ],
        ):
            frame, meta = infer_rows_for_table(
                client=None,
                doi="10.1/example",
                table_name="Table 1",
                caption="",
                df_table=pd.DataFrame([{"source": "value"}]),
                full_text=None,
                cache_dir="/tmp",
                output_columns=custom_columns,
            )

        self.assertIsNone(meta)
        self.assertEqual(list(frame.columns), custom_columns)

    def test_cache_key_changes_with_requested_schema(self):
        cache_keys = []

        def capture_cache_key(_cache_dir, cache_key):
            cache_keys.append(cache_key)
            return []

        with mock.patch(
            "row_emitter.load_cached_rows",
            side_effect=capture_cache_key,
        ):
            for columns in (["material", "value"], ["sample", "value"]):
                infer_rows_for_table(
                    client=None,
                    doi="10.1/example",
                    table_name="Table 1",
                    caption="",
                    df_table=pd.DataFrame([{"source": "value"}]),
                    full_text=None,
                    cache_dir="/tmp",
                    output_columns=columns,
                )

        self.assertEqual(len(cache_keys), 2)
        self.assertNotEqual(cache_keys[0], cache_keys[1])


if __name__ == "__main__":
    unittest.main()
