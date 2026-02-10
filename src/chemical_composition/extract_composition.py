"""
Chemical composition extraction module.

Uses the LLM to:
  1. Identify text, tables, and figures relevant to chemical structures
     — including figures/schemas depicting chemical structures or reactions.
  2. Extract unique sample names with their components and compositions,
     filtered to only those samples that have associated materials-property
     data (e.g. conductivity, stability, IEC) reported in the manuscript.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports from sibling packages when invoked from src/ui/
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from llm_client import get_llm_client
from ui.openai_cost import append_usage, run_id_from_env


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_IDENTIFY_RELEVANT_PROMPT = """\
You are an expert materials-science chemist.

Given the following parsed content from a scientific paper, identify ALL
sections of text, tables, and figures that are relevant to the **chemical
structures, chemical reactions, and chemical composition** of the materials
discussed.

IMPORTANT — you MUST include:
- Figures or schemes that **depict chemical structures**, molecular
  diagrams, polymerisation reactions, synthesis routes, or reaction
  mechanisms (e.g. "Scheme 1", "Figure 1 – synthesis of …").
  These are extremely important even if they contain no numerical data.
- Figures that show chemical drawings, repeat-unit structures, monomer
  structures, cross-linking diagrams, or any graphical representation of
  molecules.
- Tables listing polymer compositions, monomer feed ratios, IEC values,
  water uptake, degree of functionalization, etc.
- Body-text paragraphs describing synthesis procedures, reagent amounts,
  grafting degrees, functionalization steps, or naming conventions for
  samples.

For each relevant item, return a JSON array of objects with these keys:
- "source_type": one of "text", "table", "figure/scheme"
- "source_id": the ID as used in the paper (e.g. "Table 1", "Figure 3",
  "Scheme 1") or "body_text" for inline passages
- "excerpt": the relevant passage (≤500 chars) or caption/description
- "relevance": brief explanation of why this is relevant to chemical
  composition or structure

Return ONLY a JSON array.  No markdown fences.

Paper content:
{content}
"""

_EXTRACT_SAMPLES_PROMPT = """\
You are an expert materials-science chemist.

From the paper content below, identify every **unique sample name** that
has associated **materials-property data** reported in this manuscript
(e.g. conductivity, stability/retention, IEC, water uptake, swelling
ratio, mechanical properties, or any other measured property).

Do NOT include samples that are only mentioned as literature references,
reagents consumed during synthesis, or comparison points from other
papers without new measurements in this manuscript.

For each qualifying sample extract:
- Its chemical components (monomers, polymers, fillers, dopants, solvents…)
- The composition of each component (mole %, weight %, volume %, ratio,
  or however it is expressed in the paper)
- Any SMILES or molecular formula you can determine from the paper text
- The chemical structure class (e.g. "quaternary ammonium", "imidazolium",
  "polysulfone", "PEEK", …)

Return a JSON array of objects.  Each object MUST have these keys:
- "sample_name": str – the exact sample label used in the paper
- "components": list of objects, each with:
    - "name": str – component name
    - "role": str – one of "polymer_backbone", "functional_group",
      "crosslinker", "filler", "dopant", "solvent", "other"
    - "formula": str or null – molecular formula if available
    - "smiles": str or null – SMILES string if determinable
    - "amount": str or null – composition amount as stated
- "structure_class": str or null – high-level classification
- "notes": str or null – any additional context

Return ONLY a JSON array.  No markdown fences.

Paper content:
{content}
"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 60000) -> str:
    """Truncate text to fit within LLM context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]"


def _clean_json_response(text: str) -> str:
    """Strip markdown fences if the LLM wraps its JSON in them."""
    text = text.strip()
    if text.startswith("```"):
        # remove opening fence (possibly with language tag)
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _log_cost(meta: Any, *, doi: str = "", extra: Optional[Dict] = None):
    """Log LLM usage to <run_dir>/cost/usage.jsonl."""
    try:
        append_usage(
            run_id=run_id_from_env("unknown"),
            resp=meta,
            doi=doi,
            extra=extra or {},
        )
    except Exception as e:
        print(f"[chemical_composition] WARNING: failed to log cost: {e}")


def identify_relevant_content(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """
    Ask the LLM to identify which parts of the paper are relevant to
    chemical composition — including figures/schemes depicting structures
    or reactions.

    Parameters
    ----------
    parsed_text : str
        The full parsed text of the paper.
    tables_text : str
        Concatenated table data (captions + content).
    figures_text : str
        Concatenated figure captions / descriptions.
    doi : str
        DOI for cost logging.

    Returns
    -------
    list of dicts
        Each dict has source_type, source_id, excerpt, relevance.
    """
    combined = ""
    if parsed_text:
        combined += "=== BODY TEXT ===\n" + _truncate(parsed_text) + "\n\n"
    if tables_text:
        combined += "=== TABLES ===\n" + _truncate(tables_text, 20000) + "\n\n"
    if figures_text:
        combined += "=== FIGURE CAPTIONS ===\n" + _truncate(figures_text, 10000) + "\n\n"

    if not combined.strip():
        return []

    prompt = _IDENTIFY_RELEVANT_PROMPT.format(content=combined)
    client = get_llm_client()
    raw, meta = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        return_meta=True,
    )

    _log_cost(meta, doi=doi, extra={"kind": "chemical_composition_identify"})

    try:
        items = json.loads(_clean_json_response(raw))
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    return []


def extract_samples_and_compositions(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """
    Ask the LLM to extract every unique sample that has associated
    materials-property data, along with its chemical composition.

    Returns
    -------
    list of dicts
        Each dict has sample_name, components, structure_class, notes.
    """
    combined = ""
    if parsed_text:
        combined += "=== BODY TEXT ===\n" + _truncate(parsed_text) + "\n\n"
    if tables_text:
        combined += "=== TABLES ===\n" + _truncate(tables_text, 20000) + "\n\n"
    if figures_text:
        combined += "=== FIGURE CAPTIONS ===\n" + _truncate(figures_text, 10000) + "\n\n"

    if not combined.strip():
        return []

    prompt = _EXTRACT_SAMPLES_PROMPT.format(content=combined)
    client = get_llm_client()
    raw, meta = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        return_meta=True,
    )

    _log_cost(meta, doi=doi, extra={"kind": "chemical_composition_extract"})

    try:
        items = json.loads(_clean_json_response(raw))
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    return []


# ---------------------------------------------------------------------------
# Pipeline-level runner
# ---------------------------------------------------------------------------

def run_chemical_composition_extraction(
    run_dir: Path,
    safe_doi: str,
    parsed_json_path: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end extraction for one DOI.

    Reads parsed text + tables + figures, calls LLM twice
    (identify relevant content, then extract compositions),
    writes results to ``<run_dir>/chemical_composition/<safe_doi>/``.

    Returns the combined result dict.
    """
    import os
    import pandas as pd

    # Set AEM_OCR_RUN_DIR so cost logging writes to the correct location
    os.environ["AEM_OCR_RUN_DIR"] = str(run_dir)

    doi_str = safe_doi.replace("@", "/")

    # --- gather text ---
    parsed_text = ""
    if parsed_json_path and parsed_json_path.exists():
        try:
            pj = json.loads(parsed_json_path.read_text())
            # parsed.json typically has sections or a flat "text" key
            if isinstance(pj, dict):
                parsed_text = pj.get("text", "") or json.dumps(pj, indent=1)
            elif isinstance(pj, str):
                parsed_text = pj
            else:
                parsed_text = json.dumps(pj, indent=1)
        except Exception:
            pass

    # --- gather table data ---
    tables_text = ""
    if tables_dir and tables_dir.exists():
        parts = []
        for tf in sorted(tables_dir.iterdir()):
            if tf.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(tf)
                    parts.append(f"### {tf.stem}\n{df.to_string(index=False)}\n")
                except Exception:
                    pass
            elif tf.suffix.lower() == ".json" and "metadata" in tf.name:
                try:
                    meta = json.loads(tf.read_text())
                    cap = meta.get("caption", "")
                    if cap:
                        parts.append(f"### {tf.stem} caption\n{cap}\n")
                except Exception:
                    pass
        tables_text = "\n".join(parts)

    # --- gather figure captions (including schemes) ---
    figures_text = ""
    if figures_dir and figures_dir.exists():
        parts = []
        # Collect captions from metadata JSON files
        for mf in sorted(figures_dir.glob("*_metadata.json")):
            try:
                meta = json.loads(mf.read_text())
                cap = meta.get("caption", "")
                fig_id = meta.get("id", mf.stem.replace("_metadata", ""))
                if cap:
                    parts.append(f"### {fig_id}\n{cap}\n")
            except Exception:
                pass
        # Also list figure filenames to help identify scheme/structure figures
        # even when metadata is sparse
        for img in sorted(
            list(figures_dir.glob("*.png"))
            + list(figures_dir.glob("*.jpg"))
            + list(figures_dir.glob("*.jpeg"))
        ):
            meta_path = img.with_name(img.stem + "_metadata.json")
            if not meta_path.exists():
                # No metadata; include filename so LLM knows the figure exists
                parts.append(f"### {img.stem}\n(figure image: {img.name})\n")
        figures_text = "\n".join(parts)

    # --- LLM calls ---
    relevant = identify_relevant_content(
        parsed_text, tables_text, figures_text, doi=doi_str,
    )
    samples = extract_samples_and_compositions(
        parsed_text, tables_text, figures_text, doi=doi_str,
    )

    result = {
        "doi": doi_str,
        "safe_doi": safe_doi,
        "relevant_content": relevant,
        "samples": samples,
    }

    # --- persist ---
    out_dir = run_dir / "chemical_composition" / safe_doi
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "relevant_content.json").write_text(
        json.dumps(relevant, indent=2, ensure_ascii=False)
    )
    (out_dir / "samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False)
    )
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    # Also produce a flat CSV of samples for convenience
    rows = []
    for s in samples:
        base = {
            "sample_name": s.get("sample_name", ""),
            "structure_class": s.get("structure_class", ""),
            "notes": s.get("notes", ""),
        }
        for comp in s.get("components", []):
            row = dict(base)
            row["component_name"] = comp.get("name", "")
            row["component_role"] = comp.get("role", "")
            row["formula"] = comp.get("formula", "")
            row["smiles"] = comp.get("smiles", "")
            row["amount"] = comp.get("amount", "")
            rows.append(row)
        if not s.get("components"):
            rows.append(base)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "samples.csv", index=False)

    return result
