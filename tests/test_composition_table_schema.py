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

    def test_custom_sample_fields_are_preserved(self):
        samples = [
            {
                "sample_name": "PFT60-x60",
                "smiles1": "[*]CC[*]",
                "c1": 1,
                "contains_add": 0,
                "contains_crosslink": 1,
                "notes": "fixture",
            }
        ]

        rows, columns = build_composition_table_rows(samples, doi="10.1000/example")

        self.assertIn("contains_add", columns)
        self.assertIn("contains_crosslink", columns)
        self.assertEqual(rows[0]["contains_add"], "0")
        self.assertEqual(rows[0]["contains_crosslink"], "1")
        self.assertEqual(
            columns.index("contains_add") < columns.index("notes"),
            True,
        )

    def test_standard_aliases_do_not_create_duplicate_columns(self):
        samples = [
            {
                "sample_name": "AEM-1",
                "Sample Name": "AEM-1",
                "DOI": "10.9999/from-model",
                "smiles1": "[*]CC[*]",
                "c1": 1,
                "additive_smiles1": "O",
                "additivec1": 0.05,
                "additive_name_1": "water",
                "cont_add": 1,
                "contains_add": 1,
                "cont_cl": 0,
                "contains_crosslink": 0,
                "llm notes": "alias notes",
                "confidence": 0.5,
            }
        ]

        rows, columns = build_composition_table_rows(samples, doi="10.1000/example")

        self.assertEqual(len(columns), len(set(columns)))
        self.assertNotIn("Sample Name", columns)
        self.assertNotIn("additivec1", columns)
        self.assertNotIn("cont_add", columns)
        self.assertNotIn("cont_cl", columns)
        self.assertNotIn("llm notes", columns)
        self.assertEqual(columns.count("DOI"), 1)
        self.assertEqual(rows[0]["DOI"], "10.1000/example")
        self.assertEqual(rows[0]["sample_name"], "AEM-1")
        self.assertEqual(rows[0]["additive_c1"], "0.05")
        self.assertEqual(rows[0]["contains_add"], "1")
        self.assertEqual(rows[0]["contains_crosslink"], "0")
        self.assertEqual(rows[0]["notes"], "alias notes")
        self.assertEqual(rows[0]["confidence"], "0.5")


if __name__ == "__main__":
    unittest.main()
