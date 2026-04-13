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
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports from sibling packages when invoked from src/ui/
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from llm_client import get_llm_client, load_processing_llm_config
from ui.openai_cost import append_usage, run_id_from_env
from chemical_composition.chemistry_text_pipeline import (
    ChemistryTextPipelineConfig,
    canonicalize_chemistry_schema,
    chemistry_schema_to_text,
    collect_sample_description_text,
    extract_chemistry_schema_llm,
    extract_chemistry_schema_rule_based,
    get_text_embedding,
    save_chemistry_text_artifacts,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_COMBINED_EXTRACTION_PROMPT = """\
You are an expert materials-science chemist.

Given the following parsed content from a scientific paper, perform TWO tasks:

TASK 1: Identify ALL sections of text, tables, and figures that are relevant to the **chemical structures, chemical reactions, and chemical composition** of the materials discussed.

IMPORTANT — you MUST include:
- Figures or schemes that **depict chemical structures**, molecular diagrams, polymerisation reactions, synthesis routes, or reaction mechanisms (e.g. "Scheme 1", "Figure 1 – synthesis of …"). These are extremely important even if they contain no numerical data.
- Figures that show chemical drawings, repeat-unit structures, monomer structures, cross-linking diagrams, or any graphical representation of molecules.
- Tables listing polymer compositions, monomer feed ratios, IEC values, water uptake, degree of functionalization, etc.
- Body-text paragraphs describing synthesis procedures, reagent amounts, grafting degrees, functionalization steps, or naming conventions for samples.

IMPORTANT EXCLUSION:
- Do NOT include figures that are primarily property-performance plots/data (e.g. conductivity vs temperature, tensile strength, IEC trends, water uptake/swelling, durability/retention curves, polarization/power-density curves, Nyquist/EIS plots, TGA/DSC/DTG, SAXS/WAXS/XRD property analyses), unless that same figure explicitly contains a chemical structure/reaction diagram as a central element.

TASK 2: From the paper content, identify every **unique sample name** that has associated **materials-property data** reported in this manuscript (e.g. conductivity, stability/retention, IEC, water uptake, swelling ratio, mechanical properties, or any other measured property).

Do NOT include samples that are only mentioned as literature references, reagents consumed during synthesis, or comparison points from other papers without new measurements in this manuscript.

For each qualifying sample extract:
- One SMILES string for each co-monomer (if determinable from text/table/caption context)
- The composition ratio of each co-monomer in the monomer feed or polymer (mol%, wt%, feed ratio, etc.; keep original unit text)
- Additive information: additive name + additive content/amount for the sample.
  - If no additive is reported for the sample, leave additive fields blank ("").
- Its broader chemical components (monomers, polymers, fillers, dopants, solvents…)
- Any SMILES or molecular formula you can determine from the paper text
- The chemical structure class (e.g. "quaternary ammonium", "imidazolium", "polysulfone", "PEEK", …)

Return a JSON object with TWO keys:

1. "relevant_content": array of objects, each with:
   - "source_type": one of "text", "table", "figure/scheme"
   - "source_id": the ID as used in the paper (e.g. "Table 1", "Figure 3", "Scheme 1") or "body_text" for inline passages
   - "excerpt": the relevant passage (≤500 chars) or caption/description
   - "relevance": brief explanation of why this is relevant to chemical composition or structure

2. "samples": array of objects, each with:
   - "sample_name": str – the exact sample label used in the paper
   - "co_monomers": list of objects, each with:
      - "name": str – co-monomer name
      - "smiles": str – SMILES for this co-monomer, or "" if unavailable
      - "ratio": str – ratio/content for this co-monomer with units/context, or "" if unavailable
   - "additive": object with:
      - "name": str – additive name, or ""
      - "content": str – additive amount/content, or ""
   - "components": list of objects, each with:
       - "name": str – component name
       - "role": str – one of "polymer_backbone", "functional_group", "crosslinker", "filler", "dopant", "solvent", "other"
       - "formula": str or null – molecular formula if available
       - "smiles": str or null – SMILES string if determinable
       - "amount": str or null – composition amount as stated
   - "structure_class": str or null – high-level classification
   - "notes": str or null – any additional context

Return ONLY a JSON object with these two keys.  No markdown fences.

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


_PROPERTY_FIGURE_KEYWORDS = (
    "conductivity",
    "water uptake",
    "swelling",
    "iec",
    "ion exchange capacity",
    "mechanical",
    "tensile",
    "elongation",
    "thermal stability",
    "durability",
    "retention",
    "polarization",
    "power density",
    "eis",
    "nyquist",
    "tga",
    "dsc",
    "xrd",
    "saxs",
    "waxs",
)


def _is_property_figure(item: Dict[str, Any]) -> bool:
    """Heuristic filter for figures that are primarily property-data visuals."""
    src_type = (item.get("source_type") or "").strip().lower()
    if src_type not in ("figure/scheme", "figure"):
        return False

    haystack = " ".join(
        [
            str(item.get("source_id", "")),
            str(item.get("excerpt", "")),
            str(item.get("relevance", "")),
        ]
    ).lower()
    return any(keyword in haystack for keyword in _PROPERTY_FIGURE_KEYWORDS)


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model output into a stable schema for downstream consumers."""
    relevant = result.get("relevant_content", [])
    if not isinstance(relevant, list):
        relevant = []
    # Hard guard: never keep property-data-only figures in relevant content.
    relevant = [item for item in relevant if not _is_property_figure(item)]

    normalized_samples = []
    raw_samples = result.get("samples", [])
    if isinstance(raw_samples, list):
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            co_monomers = sample.get("co_monomers", [])
            if not isinstance(co_monomers, list):
                co_monomers = []
            additive = sample.get("additive", {})
            if not isinstance(additive, dict):
                additive = {}

            normalized_samples.append(
                {
                    **sample,
                    "co_monomers": [
                        {
                            "name": str(cm.get("name", "")),
                            "smiles": str(cm.get("smiles", "")),
                            "ratio": str(cm.get("ratio", "")),
                        }
                        for cm in co_monomers
                        if isinstance(cm, dict)
                    ],
                    "additive": {
                        "name": str(additive.get("name", "")),
                        "content": str(additive.get("content", "")),
                    },
                }
            )

    return {"relevant_content": relevant, "samples": normalized_samples}


def extract_chemical_composition_combined(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> Dict[str, Any]:
    """
    Combined function that identifies relevant content AND extracts samples
    in a single LLM call.

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
    dict with keys:
        - "relevant_content": list of dicts (source_type, source_id, excerpt, relevance)
        - "samples": list of dicts (sample_name, components, structure_class, notes)
    """
    combined = ""
    if parsed_text:
        combined += "=== BODY TEXT ===\n" + _truncate(parsed_text) + "\n\n"
    if tables_text:
        combined += "=== TABLES ===\n" + _truncate(tables_text, 20000) + "\n\n"
    if figures_text:
        combined += "=== FIGURE CAPTIONS ===\n" + _truncate(figures_text, 10000) + "\n\n"

    if not combined.strip():
        return {"relevant_content": [], "samples": []}

    prompt = _COMBINED_EXTRACTION_PROMPT.format(content=combined)
    # Use processing-specific LLM if configured, otherwise fall back to general
    processing_override = load_processing_llm_config()
    if processing_override is not None:
        client = get_llm_client(override=processing_override)
    else:
        client = get_llm_client()
    raw, meta = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        return_meta=True,
    )

    # Log cost for combined call
    _log_cost(meta, doi=doi, extra={"kind": "chemical_composition_combined"})

    try:
        result = json.loads(_clean_json_response(raw))
        if isinstance(result, dict):
            # Validate structure
            if "relevant_content" in result and "samples" in result:
                return _normalize_result(result)
    except json.JSONDecodeError:
        pass

    # Return empty structure if parsing failed
    return {"relevant_content": [], "samples": []}


# Legacy functions kept for backward compatibility
def identify_relevant_content(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use extract_chemical_composition_combined instead.

    This function is kept for backward compatibility but now calls
    the combined extraction function and returns only relevant_content.
    """
    result = extract_chemical_composition_combined(
        parsed_text, tables_text, figures_text, doi=doi
    )
    return result.get("relevant_content", [])


def extract_samples_and_compositions(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use extract_chemical_composition_combined instead.

    This function is kept for backward compatibility but now calls
    the combined extraction function and returns only samples.
    """
    result = extract_chemical_composition_combined(
        parsed_text, tables_text, figures_text, doi=doi
    )
    return result.get("samples", [])


# ---------------------------------------------------------------------------
# Pipeline-level runner
# ---------------------------------------------------------------------------

def run_chemical_composition_extraction(
    run_dir: Path,
    safe_doi: str,
    parsed_json_path: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    chemistry_text_config: Optional[ChemistryTextPipelineConfig] = None,
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

    # --- Combined LLM call ---
    extraction_result = extract_chemical_composition_combined(
        parsed_text, tables_text, figures_text, doi=doi_str,
    )

    result = {
        "doi": doi_str,
        "safe_doi": safe_doi,
        "relevant_content": extraction_result.get("relevant_content", []),
        "samples": extraction_result.get("samples", []),
    }

    # Extract for convenience
    relevant = result["relevant_content"]
    samples = result["samples"]

    # --- persist ---
    out_dir = run_dir / "chemical_composition" / safe_doi
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy relevant figures and tables to output directory ---
    # This ensures they're preserved even if later filtering removes them
    figures_out_dir = out_dir / "figures"
    tables_out_dir = out_dir / "tables"
    figures_out_dir.mkdir(parents=True, exist_ok=True)
    tables_out_dir.mkdir(parents=True, exist_ok=True)

    for item in relevant:
        src_type = item.get("source_type", "").lower()
        src_id = item.get("source_id", "").lower()

        # Copy relevant figures/schemes
        if src_type in ("figure/scheme", "figure", "scheme") and figures_dir and figures_dir.exists():
            # Normalize source_id for matching
            src_id_norm = src_id.replace(" ", "").replace("_", "")

            # Find matching figure files
            for fig_file in figures_dir.iterdir():
                if fig_file.is_file() and fig_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                    fig_stem_norm = fig_file.stem.lower().replace(" ", "").replace("_", "")
                    if src_id_norm and (src_id_norm in fig_stem_norm or fig_stem_norm in src_id_norm):
                        # Copy the image file
                        try:
                            shutil.copy2(fig_file, figures_out_dir / fig_file.name)
                            # Copy metadata if it exists
                            meta_file = fig_file.with_name(fig_file.stem + "_metadata.json")
                            if meta_file.exists():
                                shutil.copy2(meta_file, figures_out_dir / meta_file.name)
                        except Exception as e:
                            print(f"[chemical_composition] WARNING: failed to copy {fig_file}: {e}")
                        break

        # Copy relevant tables
        elif src_type == "table" and tables_dir and tables_dir.exists():
            # Normalize source_id for matching
            src_id_norm = src_id.replace(" ", "").replace("_", "")

            # Find matching table files
            for tbl_file in tables_dir.iterdir():
                if tbl_file.is_file() and not tbl_file.name.endswith("_metadata.json"):
                    tbl_stem_norm = tbl_file.stem.lower().replace(" ", "").replace("_", "")
                    if src_id_norm and (src_id_norm in tbl_stem_norm or tbl_stem_norm in src_id_norm):
                        # Copy the table file
                        try:
                            shutil.copy2(tbl_file, tables_out_dir / tbl_file.name)
                            # Copy metadata if it exists
                            meta_file = tbl_file.with_name(tbl_file.stem + "_metadata.json")
                            if meta_file.exists():
                                shutil.copy2(meta_file, tables_out_dir / meta_file.name)
                        except Exception as e:
                            print(f"[chemical_composition] WARNING: failed to copy {tbl_file}: {e}")
                        break

    (out_dir / "relevant_content.json").write_text(
        json.dumps(relevant, indent=2, ensure_ascii=False)
    )
    (out_dir / "samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False)
    )
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    # --- Optional chemistry text -> schema -> canonical text -> embedding pipeline ---
    chemistry_text_cfg = chemistry_text_config or ChemistryTextPipelineConfig()
    if chemistry_text_cfg.enable_chemistry_text_pipeline:
        chem_root = run_dir / "chemistry_text" / safe_doi
        cache_dir = run_dir / "chemistry_text" / "_cache"
        chem_root.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        schema_all_path = chem_root / "chemistry_schema_all.jsonl"
        canonical_all_path = chem_root / "canonical_text_all.jsonl"
        emb_index_path = chem_root / "embedding_index.jsonl"
        # reset aggregate files for this DOI run
        schema_all_path.write_text("")
        canonical_all_path.write_text("")
        emb_index_path.write_text("")

        for idx, sample in enumerate(samples):
            sample_name = str(sample.get("sample_name", "")).strip() or f"sample_{idx+1}"
            sample_dir_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_name).strip("_") or f"sample_{idx+1}"
            sample_out_dir = chem_root / sample_dir_name

            try:
                bundle = collect_sample_description_text(
                    sample=sample,
                    parsed_text=parsed_text,
                    tables_text=tables_text,
                    figures_text=figures_text,
                    relevant_content=relevant,
                    include_caption_text=chemistry_text_cfg.include_caption_text,
                    include_table_context=chemistry_text_cfg.include_table_context,
                    include_experimental_section=chemistry_text_cfg.include_experimental_section,
                    exclude_performance_sentences=chemistry_text_cfg.exclude_performance_sentences,
                )

                if chemistry_text_cfg.chemistry_schema_extractor == "llm":
                    schema = extract_chemistry_schema_llm(bundle.raw_text, sample_name=sample_name)
                elif chemistry_text_cfg.chemistry_schema_extractor == "hybrid":
                    schema = extract_chemistry_schema_llm(bundle.raw_text, sample_name=sample_name)
                    if not schema.backbone_family and not schema.cation_type:
                        schema = extract_chemistry_schema_rule_based(
                            bundle.raw_text, sample_name=sample_name, extraction_method="hybrid_fallback"
                        )
                else:
                    schema = extract_chemistry_schema_rule_based(
                        bundle.raw_text, sample_name=sample_name, extraction_method="rule_based"
                    )

                schema = canonicalize_chemistry_schema(schema)
                canonical = chemistry_schema_to_text(schema)

                embedding = None
                try:
                    embedding = get_text_embedding(
                        canonical.canonical_text,
                        sample_id=sample_name,
                        model_name=chemistry_text_cfg.chemistry_embedding_model,
                        backend=chemistry_text_cfg.chemistry_embedding_backend,
                        cache_dir=cache_dir,
                    )
                    print(
                        f"[text chemical_composition]: embedding generated for {sample_name}"
                    )
                    print("====embedding====", embedding)
                except Exception as e:
                    print(
                        f"[text chemical_composition] WARNING: embedding failed for {sample_name}: {e}"
                    )

                print()
                save_chemistry_text_artifacts(sample_out_dir, bundle, schema, canonical, embedding)

                with schema_all_path.open("a", encoding="utf-8") as sf:
                    sf.write(json.dumps({"sample_id": sample_name, **schema.__dict__}, ensure_ascii=False) + "\n")
                with canonical_all_path.open("a", encoding="utf-8") as cf:
                    cf.write(json.dumps(canonical.__dict__, ensure_ascii=False) + "\n")
                if embedding is not None:
                    emb_meta = dict(embedding.__dict__)
                    emb_meta.pop("embedding_vector", None)
                    with emb_index_path.open("a", encoding="utf-8") as ef:
                        ef.write(json.dumps(emb_meta, ensure_ascii=False) + "\n")
            except Exception as e:
                print(
                    f"[chemical_composition] WARNING: chemistry text pipeline failed for {sample_name}: {e}"
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
            row["co_monomers"] = json.dumps(
                s.get("co_monomers", []), ensure_ascii=False
            )
            additive = s.get("additive", {}) or {}
            row["additive_name"] = additive.get("name", "")
            row["additive_content"] = additive.get("content", "")
            rows.append(row)
        if not s.get("components"):
            row = dict(base)
            row["co_monomers"] = json.dumps(
                s.get("co_monomers", []), ensure_ascii=False
            )
            additive = s.get("additive", {}) or {}
            row["additive_name"] = additive.get("name", "")
            row["additive_content"] = additive.get("content", "")
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "samples.csv", index=False)

    return result
