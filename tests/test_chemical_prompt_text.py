import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemical_composition.extract_composition import (  # noqa: E402
    _build_composition_prompt,
    _parsed_json_to_text,
)


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

    def test_composition_prompt_includes_custom_composition_rules(self):
        prompt = _build_composition_prompt(
            [
                {
                    "source_type": "text",
                    "source_id": "body_text",
                    "content": "Sample A contains additive X.",
                }
            ],
            composition_rules="include contains_add as 1 or 0",
        )

        self.assertIn("User-provided extraction instructions:", prompt)
        self.assertIn("include contains_add as 1 or 0", prompt)
        self.assertIn("Sample A contains additive X.", prompt)


if __name__ == "__main__":
    unittest.main()
