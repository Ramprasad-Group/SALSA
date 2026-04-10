import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemical_composition.chemistry_text_pipeline import (  # noqa: E402
    canonicalize_chemistry_schema,
    chemistry_schema_to_text,
    collect_sample_description_text,
    extract_chemistry_schema_rule_based,
    filter_target_leakage_sentences,
    get_text_embedding,
    save_chemistry_text_artifacts,
)


FIXTURE_TEXT = (
    "QAPEK-OH membrane was prepared by chloromethylation of PEK block copolymer, "
    "followed by quaternization with triethylamine and conversion to hydroxide form. "
    "The membrane has a fully aromatic backbone and phase-separated hydrophilic/hydrophobic morphology."
)


class TestChemistryTextPipeline(unittest.TestCase):
    def test_collect_sample_description_text_multi_source(self):
        sample = {
            "sample_name": "QAPEK-OH",
            "components": [{"name": "PEK", "role": "polymer_backbone", "amount": "80 wt%"}],
            "co_monomers": [{"name": "11-bromo-1-undecene", "ratio": "20 mol%", "smiles": ""}],
            "additive": {"name": "SiO2", "content": "5 wt%"},
        }
        bundle = collect_sample_description_text(
            sample=sample,
            parsed_text=FIXTURE_TEXT,
            tables_text="QAPEK-OH | composition 80/20",
            figures_text="Figure 1 caption: synthesis route of QAPEK-OH",
            relevant_content=[{"source_type": "text", "source_id": "body", "excerpt": "QAPEK-OH was quaternized."}],
        )
        self.assertEqual(bundle.sample_id, "QAPEK-OH")
        self.assertIn("chloromethylation", bundle.raw_text.lower())
        self.assertGreater(len(bundle.snippets), 3)

    def test_filter_target_leakage_sentences(self):
        text = "Sample showed excellent conductivity. It was prepared by chloromethylation."
        out = filter_target_leakage_sentences(text)
        self.assertNotIn("excellent conductivity", out.lower())
        self.assertIn("chloromethylation", out.lower())

    def test_rule_based_schema_extraction(self):
        schema = extract_chemistry_schema_rule_based(FIXTURE_TEXT, sample_name="QAPEK-OH")
        self.assertEqual(schema.backbone_family, "polyetherketone")
        self.assertEqual(schema.polymer_architecture, "block copolymer")
        self.assertIn("chloromethylation", schema.functionalization_route)
        self.assertIn("quaternization", schema.functionalization_route)
        self.assertEqual(schema.anion_or_counterion, "hydroxide")
        self.assertEqual(schema.aromaticity, "aromatic")
        self.assertIn("phase-separated", schema.morphology_cues)

    def test_canonicalization_aliases(self):
        schema = extract_chemistry_schema_rule_based("QA groups in OH-form, PPO membrane.", sample_name="S1")
        schema = canonicalize_chemistry_schema(schema)
        self.assertIn(schema.cation_type, ("quaternary ammonium", None))
        self.assertIn(schema.anion_or_counterion, ("hydroxide", None))
        if schema.backbone_family:
            self.assertEqual(schema.backbone_family, "poly(2,6-dimethyl-1,4-phenylene oxide)")

    def test_canonical_text_is_deterministic(self):
        schema = canonicalize_chemistry_schema(
            extract_chemistry_schema_rule_based(FIXTURE_TEXT, sample_name="QAPEK-OH")
        )
        t1 = chemistry_schema_to_text(schema).canonical_text
        t2 = chemistry_schema_to_text(schema).canonical_text
        self.assertEqual(t1, t2)

    def test_embedding_cache_behavior(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "cache"
            emb1 = get_text_embedding(
                "sample chemistry text",
                sample_id="S1",
                model_name="hash-embedding-v1",
                backend="local",
                cache_dir=cache_dir,
            )
            emb2 = get_text_embedding(
                "sample chemistry text",
                sample_id="S1",
                model_name="hash-embedding-v1",
                backend="local",
                cache_dir=cache_dir,
            )
            self.assertEqual(emb1.cache_key, emb2.cache_key)
            self.assertEqual(emb1.embedding_dim, emb2.embedding_dim)
            self.assertTrue((cache_dir / f"{emb1.cache_key}.npy").exists())

    def test_end_to_end_sample_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "sample"
            bundle = collect_sample_description_text(
                sample={"sample_name": "QAPEK-OH"},
                parsed_text=FIXTURE_TEXT,
            )
            schema = canonicalize_chemistry_schema(
                extract_chemistry_schema_rule_based(bundle.raw_text, sample_name="QAPEK-OH")
            )
            canonical = chemistry_schema_to_text(schema)
            emb = get_text_embedding(
                canonical.canonical_text,
                sample_id="QAPEK-OH",
                model_name="hash-embedding-v1",
                backend="local",
                cache_dir=Path(td) / "cache",
            )
            save_chemistry_text_artifacts(base, bundle, schema, canonical, emb)

            self.assertTrue((base / "raw_description.json").exists())
            self.assertTrue((base / "chemistry_schema.json").exists())
            self.assertTrue((base / "canonical_chemistry_text.json").exists())
            self.assertTrue((base / "chemistry_embedding.npy").exists())
            meta = json.loads((base / "chemistry_embedding_metadata.json").read_text())
            vec = np.load(base / "chemistry_embedding.npy")
            self.assertEqual(meta["embedding_dim"], int(vec.shape[0]))


if __name__ == "__main__":
    unittest.main()
