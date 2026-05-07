import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemical_composition.extract_composition import _parsed_json_to_text  # noqa: E402


class TestChemicalPromptText(unittest.TestCase):
    def test_parsed_json_to_text_flattens_sections_and_skips_references(self):
        parsed = {
            "Title": "Example Paper",
            "Abstract": "Short abstract.",
            "Sections": [
                {
                    "name": "Experimental",
                    "content": [
                        {
                            "name": "Synthesis",
                            "content": ["Step one.", "Step two."],
                        }
                    ],
                },
                {
                    "name": "References",
                    "content": ["[1] Should not appear."],
                },
            ],
        }

        text = _parsed_json_to_text(parsed)

        self.assertIn("# Example Paper", text)
        self.assertIn("## Abstract", text)
        self.assertIn("Experimental", text)
        self.assertIn("Synthesis", text)
        self.assertIn("Step one.", text)
        self.assertNotIn("Should not appear.", text)


if __name__ == "__main__":
    unittest.main()
