import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemical_composition.extract_composition import build_composition_table_rows  # noqa: E402


class TestCompositionTableSchema(unittest.TestCase):
    def test_empty_samples_keep_required_columns(self):
        rows, columns = build_composition_table_rows([], doi="10.1000/example")

        self.assertEqual(rows, [])
        self.assertEqual(columns[0:2], ["DOI", "sample_name"])
        for column in (
            "smiles1",
            "smiles2",
            "smiles3",
            "c1",
            "c2",
            "c3",
            "additive_smiles1",
            "additive_smiles2",
            "additive_smiles3",
            "additive_c1",
            "additive_c2",
            "additive_c3",
            "additive_name_1",
            "additive_name_2",
        ):
            self.assertIn(column, columns)

    def test_sample_rows_include_doi_and_additive_names(self):
        samples = [
            {
                "sample_name": "AEM-1",
                "smiles1": "[*]CC[*]",
                "c1": 0.7,
                "additive_smiles1": "O",
                "additive_c1": "5 wt%",
                "additive_name_1": "water",
                "notes": "fixture",
            }
        ]

        rows, columns = build_composition_table_rows(samples, doi="10.1000/example")

        self.assertEqual(columns[0:2], ["DOI", "sample_name"])
        self.assertEqual(rows[0]["DOI"], "10.1000/example")
        self.assertEqual(rows[0]["sample_name"], "AEM-1")
        self.assertEqual(rows[0]["additive_name_1"], "water")
        self.assertEqual(rows[0]["additive_c1"], "0.05")
        self.assertIn("smiles3", rows[0])
        self.assertIn("additive_smiles3", rows[0])


if __name__ == "__main__":
    unittest.main()
