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
import base64
import mimetypes
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from sibling packages when invoked from src/ui/
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from llm_client import get_llm_client, load_optional_llm_config_from_env
from ui.llm_step_config import (
    LLM_STEP_CHEM_COMPOSITION,
    LLM_STEP_CHEM_RELEVANT,
    step_env_prefix,
)
from ui.openai_cost import append_usage, run_id_from_env


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RELEVANT_CONTENT_PROMPT = """\
You are an expert materials-science chemist.

Given the following parsed content from a scientific paper, identify ALL
sections of text, tables, and figures that are relevant to the **chemical
structures, chemical reactions, sample naming, sample composition, and measured
sample/property association** of the materials discussed.

IMPORTANT — you MUST include:
- Figures or schemes that **depict chemical structures**, molecular diagrams, polymerisation reactions, synthesis routes, or reaction mechanisms (e.g. "Scheme 1", "Figure 1 – synthesis of …"). These are extremely important even if they contain no numerical data.
- Figures that show chemical drawings, repeat-unit structures, monomer structures, cross-linking diagrams, or any graphical representation of molecules.
- Tables listing polymer compositions, monomer feed ratios, IEC values, water uptake, degree of functionalization, sample names, or other measured properties.
- Body-text paragraphs describing synthesis procedures, reagent amounts, grafting degrees, functionalization steps, or naming conventions for samples.
- Body-text paragraphs that connect sample names to measured property data.

IMPORTANT EXCLUSION:
- Do NOT include figures that are primarily property-performance plots/data (e.g. conductivity vs temperature, tensile strength, IEC trends, water uptake/swelling, durability/retention curves, polarization/power-density curves, Nyquist/EIS plots, TGA/DSC/DTG, SAXS/WAXS/XRD property analyses), unless that same figure explicitly contains a chemical structure/reaction diagram as a central element.

Return a JSON object with one key:

1. "relevant_content": array of objects, each with:
   - "source_type": one of "text", "table", "figure/scheme"
   - "source_id": the ID as used in the paper (e.g. "Table 1", "Figure 3", "Scheme 1") or "body_text" for inline passages
   - "excerpt": a short relevant passage or caption (<=500 chars)
   - "content": the relevant passage, table rows, caption, or description with enough detail to support downstream chemical composition extraction (<=2500 chars)
   - "relevance": brief explanation of why this is relevant to chemical composition, structure, sample names, or measured-sample identification

Return ONLY a JSON object with this key. No markdown fences.

Paper content:
{content}
"""


_COMPOSITION_EXTRACTION_PROMPT = """\
You are an expert materials-science chemist.

You will receive relevant evidence extracted from one scientific paper. Using
ONLY that evidence, extract a sample-level chemical composition table for every
unique sample in this paper that has associated materials-property data reported
in this manuscript.

Do NOT include samples that are only mentioned as literature references, reagents
consumed during synthesis, solvents/catalysts, or comparison points from other
papers without new measurements in this manuscript.

Return a JSON object with one key:

1. "samples": array of row objects. Each row object must contain:
   - "sample_name": exact sample label used in the paper
   - "smiles1", "smiles2", ..., "smilesN": polymer/repeat-unit SMILES strings for polymer components in the final measured sample
   - "c1", "c2", ..., "cN": numeric relative ratios for smiles1..smilesN
   - "additive_smiles1", "additive_smiles2", ..., "additive_smilesN": free additive/filler/dopant/crosslinker SMILES strings
   - "additive_c1", "additive_c2", ..., "additive_cN": numeric relative contents for additive_smiles1..additive_smilesN
   - "additive_name_1", "additive_name_2", ..., "additive_name_N": additive/filler/dopant/crosslinker names corresponding to additive_smilesN when available
   - "notes": concise reasoning and source wording for the row, including uncertainty, units, and why the sample was included

Strict requirements:
- c and additive_c fields MUST be JSON numbers or null. Never put text in these
  fields. Use fractions on a 0-1 scale: 40 mol% -> 0.4, 5 wt% -> 0.05, 1:3 ->
  0.25 and 0.75 for the two components. If both theoretical and experimental
  values are given, prefer the experimental value. If a numeric value cannot be
  determined, use null.
- Put all source wording, units, formulas, assumptions, and ambiguity in "notes",
  not in c or additive_c.
- Every smiles1..smilesN value must be a polymer/repeat-unit SMILES for a final
  polymer component. It must include dummy attachment atoms such as "[*]" to
  mark repeat-unit connection points. Do not put detached small molecules,
  monomers, free side chains, free cations, reagents, names, formulas, "R",
  "Me", "Ph", or invalid condensed groups in smiles columns. If a valid
  polymer/repeat-unit SMILES cannot be determined, use "".
- If a side chain or cation is covalently attached to a polymer, encode the
  whole functionalized repeat unit in the corresponding smiles column; do not
  list the detached side-chain/cation molecule as a polymer smiles column.
- additive_smiles columns may be either small-molecule SMILES or
  polymer/repeat-unit SMILES, but only for free additives/fillers/dopants/
  crosslinkers that remain in the measured sample.
- Use as many numbered columns as needed for the paper. Omit unused numbered
  fields from a row or set them to ""/null.

Return ONLY a JSON object with the "samples" key. No markdown fences.

Relevant evidence:
{relevant_content}
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


def _candidate_json_payloads(text: str) -> List[str]:
    """Yield likely JSON payload substrings from an LLM response."""
    cleaned = _clean_json_response(text)
    if cleaned:
        yield cleaned

    # Some local models prepend prose such as "Here is the JSON:" before the
    # fenced payload, so look for fenced blocks anywhere in the response.
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if candidate:
            yield candidate

    # Last-resort fallback: take the outermost JSON object-looking slice.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        if candidate:
            yield candidate


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


def _safe_doi(doi: str) -> str:
    return (doi or "unknown").replace("/", "@")


def _debug_output_dir(doi: str) -> Optional[Path]:
    run_dir = os.environ.get("SALSA_RUN_DIR", "").strip()
    if not run_dir:
        return None
    out_dir = Path(run_dir) / "chemical_composition" / _safe_doi(doi)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_debug_text(doi: str, filename: str, content: str) -> None:
    out_dir = _debug_output_dir(doi)
    if out_dir is None:
        return
    try:
        (out_dir / filename).write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"[chemical_composition] WARNING: failed to write debug artifact {filename}: {e}")


def _write_debug_json(doi: str, filename: str, payload: Any) -> None:
    _write_debug_text(
        doi,
        filename,
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


_SKIP_PARSED_SECTION_NAMES = {
    "introduction",
    "conclusions",
    "conclusion",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "supporting information",
    "supplementary material",
}


def _normalize_section_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _should_skip_parsed_section(name: str) -> bool:
    return _normalize_section_name(name) in _SKIP_PARSED_SECTION_NAMES


def _section_content_to_text(content: Any, depth: int = 0) -> str:
    """Flatten nested parsed-paper sections into readable plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            text = _section_content_to_text(item, depth=depth)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    if isinstance(content, dict):
        name = _to_text(content.get("name")).strip()
        if name and _should_skip_parsed_section(name):
            return ""
        body = _section_content_to_text(content.get("content"), depth=depth + 1)
        if not body:
            return ""
        if not name:
            return body
        header = "#" * min(depth + 2, 6)
        return f"{header} {name}\n{body}"

    return _to_text(content).strip()


def _parsed_json_to_text(parsed: Any) -> str:
    """Convert structured parsed.json output into prompt-friendly paper text."""
    if isinstance(parsed, str):
        return parsed

    if not isinstance(parsed, dict):
        return json.dumps(parsed, indent=1, ensure_ascii=False)

    direct_text = _to_text(parsed.get("text")).strip()
    if direct_text:
        return direct_text

    parts: List[str] = []
    title = _to_text(parsed.get("Title")).strip()
    if title:
        parts.append(f"# {title}")

    abstract_text = _section_content_to_text(parsed.get("Abstract"))
    if abstract_text:
        parts.append(f"## Abstract\n{abstract_text}")

    sections = parsed.get("Sections")
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                section_name = _to_text(section.get("name")).strip()
                if section_name and _should_skip_parsed_section(section_name):
                    continue
            section_text = _section_content_to_text(section, depth=0)
            if section_text:
                parts.append(section_text)
    elif sections is not None:
        section_text = _section_content_to_text(sections, depth=0)
        if section_text:
            parts.append(section_text)

    if parts:
        return "\n\n".join(parts)

    return json.dumps(parsed, indent=1, ensure_ascii=False)


def _estimate_tokens(text: str) -> int:
    """Crude token estimate for prompt-size warning logs."""
    return max(1, len(text) // 4) if text else 0


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


def _normalize_relevant_content_items(value: Any) -> List[Dict[str, str]]:
    """Normalize relevant-content LLM output for persistence and prompt chaining."""
    if not isinstance(value, list):
        return []

    relevant: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized = {
            "source_type": _to_text(item.get("source_type")).strip(),
            "source_id": _to_text(item.get("source_id")).strip(),
            "excerpt": _to_text(item.get("excerpt")).strip(),
            "content": _to_text(item.get("content") or item.get("excerpt")).strip(),
            "relevance": _to_text(item.get("relevance")).strip(),
        }
        for image_key in ("image_filename", "image_path"):
            if item.get(image_key):
                normalized[image_key] = _to_text(item.get(image_key)).strip()
        if not _is_property_figure(normalized):
            relevant.append(normalized)
    return relevant


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model output into a stable schema for downstream consumers."""
    relevant = _normalize_relevant_content_items(result.get("relevant_content", []))
    samples = _normalize_composition_samples(result.get("samples", []))
    return {"relevant_content": relevant, "samples": samples}


def _to_text(value: Any) -> str:
    """Return a stable string for values that may be null, numeric, or nested."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    """Coerce a model field into a list of dictionaries."""
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _pick_first(item: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return _to_text(value)
    return ""


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def _normalize_numeric_fraction(value: Any) -> str:
    """Coerce model ratio/content output into a numeric fraction string."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return _format_number(float(value))

    text = _to_text(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "n/a", "not available"}:
        return ""

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        return _format_number(float(text))

    assignment = re.search(
        r"(?:^|[\s,(;])(?:x|y|z|n|m)\s*=\s*([+-]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if assignment:
        value_f = float(assignment.group(1))
        return _format_number(value_f / 100.0 if value_f > 1 else value_f)

    variable_example = re.search(
        r"(?:^|[\s,(;])(?:x|y|z|n|m)\b\D{0,30}?(?:e\.g\.|eg|example)\D{0,20}?([+-]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if variable_example:
        value_f = float(variable_example.group(1))
        return _format_number(value_f / 100.0 if value_f > 1 else value_f)

    experimental = re.search(
        r"(?:experimental|experimentally|exp\.?)\D{0,60}?([+-]?\d+(?:\.\d+)?)\s*(?:%|mol\s*%|wt\s*%|vol\s*%)?",
        text,
        flags=re.IGNORECASE,
    )
    if experimental:
        value_f = float(experimental.group(1))
        return _format_number(value_f / 100.0 if value_f > 1 else value_f)

    percent_matches = re.findall(
        r"([+-]?\d+(?:\.\d+)?)\s*(?:%|mol\s*%|wt\s*%|vol\s*%)",
        text,
        flags=re.IGNORECASE,
    )
    if percent_matches:
        return _format_number(float(percent_matches[0]) / 100.0)

    ratio_match = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*[:/]\s*([+-]?\d+(?:\.\d+)?)\s*",
        text,
    )
    if ratio_match:
        first = float(ratio_match.group(1))
        second = float(ratio_match.group(2))
        total = first + second
        if total:
            return _format_number(first / total)

    contextual_number = re.search(
        r"\b(?:dob|degree|graft(?:ing|ed)?|content|loading|ratio|fraction|functionalization)\b\D{0,120}?([+-]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if contextual_number:
        value_f = float(contextual_number.group(1))
        return _format_number(value_f / 100.0 if 1 < value_f <= 100 else value_f)

    return ""


def _table_numeric_value(value: Any) -> str:
    """Last-mile guard for c/additive_c columns: numeric strings only."""
    normalized = _normalize_numeric_fraction(value)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", normalized):
        return normalized
    return ""


def _normalize_smiles(value: Any, *, require_polymer: bool = False) -> str:
    """
    Keep only plain SMILES-like strings.

    For polymer components, require dummy attachment atoms so a small-molecule
    precursor does not get exported as a polymer/repeat-unit SMILES.
    """
    smiles = _to_text(value).strip()
    if not smiles:
        return ""

    invalid_patterns = [
        r"\.\.\.",
        r"\bR\d*\b",
        r"\b(Me|Et|Ph)\b",
        r"(?<!\[)\bCH[23]\b",
        r"(?<!\[)[A-Z][a-z]?\+",
        r"[{}]",
    ]
    if any(re.search(pattern, smiles) for pattern in invalid_patterns):
        return ""
    if require_polymer and "*" not in smiles:
        return ""
    return smiles


def _numeric_or_none(value: Any) -> Optional[float]:
    """Return a JSON-number-safe float, or None when no numeric value exists."""
    normalized = _normalize_numeric_fraction(value)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", normalized):
        return float(normalized)
    return None


def _numbered_indices(row: Dict[str, Any], *prefixes: str) -> List[int]:
    indices = set()
    for key in row:
        for prefix in prefixes:
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(key))
            if match:
                indices.add(int(match.group(1)))
    return sorted(indices)


def _numbered_additive_name_indices(row: Dict[str, Any]) -> List[int]:
    indices = set()
    for key in row:
        match = re.fullmatch(r"additive_name_?(\d+)", str(key))
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _normalize_composition_sample_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a sample into the requested flat composition-table schema."""
    row: Dict[str, Any] = {
        "sample_name": _to_text(sample.get("sample_name")).strip(),
        "notes": _to_text(sample.get("notes")).strip(),
    }

    polymer_indices = _numbered_indices(sample, "smiles", "c")
    additive_indices = _numbered_indices(sample, "additive_smiles", "additive_c")
    additive_name_indices = _numbered_additive_name_indices(sample)

    if polymer_indices or additive_indices or additive_name_indices:
        for idx in polymer_indices:
            row[f"smiles{idx}"] = _normalize_smiles(
                sample.get(f"smiles{idx}", ""),
                require_polymer=True,
            )
            row[f"c{idx}"] = _numeric_or_none(sample.get(f"c{idx}"))

        for idx in additive_indices:
            row[f"additive_smiles{idx}"] = _normalize_smiles(
                sample.get(f"additive_smiles{idx}", "")
            )
            row[f"additive_c{idx}"] = _numeric_or_none(sample.get(f"additive_c{idx}"))
        for idx in additive_name_indices:
            row[f"additive_name_{idx}"] = _to_text(
                sample.get(f"additive_name_{idx}", sample.get(f"additive_name{idx}", ""))
            ).strip()
        return row

    # Backward-compatible conversion for older LLM outputs that used nested
    # polymer_components/additive_components.
    for idx, component in enumerate(_normalize_polymer_components(sample), start=1):
        row[f"smiles{idx}"] = _normalize_smiles(
            component.get("smiles", ""),
            require_polymer=True,
        )
        row[f"c{idx}"] = _numeric_or_none(
            component.get("ratio", "") or component.get("ratio_text", "")
        )

    for idx, component in enumerate(_normalize_additive_components(sample), start=1):
        row[f"additive_smiles{idx}"] = _normalize_smiles(component.get("smiles", ""))
        row[f"additive_c{idx}"] = _numeric_or_none(
            component.get("content", "") or component.get("content_text", "")
        )
        row[f"additive_name_{idx}"] = _to_text(component.get("name", "")).strip()

    return row


def _normalize_composition_samples(value: Any) -> List[Dict[str, Any]]:
    """Normalize model output samples into flat composition rows."""
    if not isinstance(value, list):
        return []

    rows: List[Dict[str, Any]] = []
    for sample in value:
        if not isinstance(sample, dict):
            continue
        row = _normalize_composition_sample_row(sample)
        if row.get("sample_name"):
            rows.append(row)
    return rows


def _normalize_polymer_components(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize polymer composition entries from new or legacy model fields."""
    raw_items = _as_dict_list(sample.get("polymer_components"))
    if not raw_items:
        raw_items = _as_dict_list(sample.get("co_monomers"))
    if not raw_items:
        # Last-resort compatibility for older component-only samples.
        raw_items = [
            c for c in _as_dict_list(sample.get("components"))
            if _to_text(c.get("role")).lower()
            in {"polymer_backbone", "functional_group", "other", ""}
        ]

    normalized: List[Dict[str, str]] = []
    for item in raw_items:
        ratio_value = _pick_first(item, ["ratio", "relative_ratio", "content", "amount", "composition"])
        ratio_text = _pick_first(item, ["ratio_text", "ratio_source", "ratio_units", "composition_text"])
        normalized.append(
            {
                "name": _pick_first(item, ["name", "component_name", "polymer_name", "monomer_name"]),
                "smiles": _normalize_smiles(
                    _pick_first(item, ["smiles", "polymer_smiles", "repeat_unit_smiles", "monomer_smiles"]),
                    require_polymer=True,
                ),
                "ratio": _normalize_numeric_fraction(ratio_value),
                "ratio_text": ratio_text or _to_text(ratio_value),
            }
        )
    return normalized


def _normalize_additive_components(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize additive/crosslinker/filler entries from new or legacy fields."""
    raw_items: List[Dict[str, Any]] = []
    for key in ("additive_components", "additives", "crosslinkers"):
        raw_items.extend(_as_dict_list(sample.get(key)))

    legacy_additive = sample.get("additive")
    if not raw_items and isinstance(legacy_additive, dict) and any(legacy_additive.values()):
        raw_items.append(legacy_additive)

    if not raw_items:
        raw_items = [
            c for c in _as_dict_list(sample.get("components"))
            if _to_text(c.get("role")).lower()
            in {"additive", "crosslinker", "filler", "dopant"}
        ]

    normalized: List[Dict[str, str]] = []
    for item in raw_items:
        content_value = _pick_first(item, ["content", "relative_content", "ratio", "amount", "composition"])
        content_text = _pick_first(item, ["content_text", "content_source", "content_units", "composition_text"])
        normalized.append(
            {
                "name": _pick_first(item, ["name", "component_name", "additive_name", "crosslinker_name"]),
                "role": _pick_first(item, ["role", "component_role"]) or "additive",
                "smiles": _normalize_smiles(
                    _pick_first(item, ["smiles", "additive_smiles", "crosslinker_smiles"]),
                ),
                "content": _normalize_numeric_fraction(content_value),
                "content_text": content_text or _to_text(content_value),
            }
        )
    return normalized


def _normalize_components(
    sample: Dict[str, Any],
    polymer_components: List[Dict[str, str]],
    additive_components: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Keep legacy component rows available for older downstream code."""
    raw_components = _as_dict_list(sample.get("components"))
    if raw_components:
        return [
            {
                "name": _pick_first(c, ["name", "component_name"]),
                "role": _pick_first(c, ["role", "component_role"]),
                "formula": _to_text(c.get("formula")),
                "smiles": _to_text(c.get("smiles")),
                "amount": _pick_first(c, ["amount", "content", "ratio", "composition"]),
            }
            for c in raw_components
        ]

    components: List[Dict[str, str]] = []
    for pc in polymer_components:
        components.append(
            {
                "name": pc.get("name", ""),
                "role": "polymer_backbone",
                "formula": "",
                "smiles": pc.get("smiles", ""),
                "amount": pc.get("ratio", ""),
            }
        )
    for ac in additive_components:
        components.append(
            {
                "name": ac.get("name", ""),
                "role": ac.get("role", "") or "additive",
                "formula": "",
                "smiles": ac.get("smiles", ""),
                "amount": ac.get("content", ""),
            }
        )
    return components


def _first_additive(additive_components: List[Dict[str, str]], legacy_additive: Any) -> Dict[str, str]:
    """Return the legacy single-additive summary for downstream compatibility."""
    if additive_components:
        first = additive_components[0]
        return {
            "name": first.get("name", ""),
            "content": first.get("content", ""),
            "smiles": first.get("smiles", ""),
        }
    if isinstance(legacy_additive, dict):
        return {
            "name": _to_text(legacy_additive.get("name")),
            "content": _to_text(legacy_additive.get("content")),
            "smiles": _to_text(legacy_additive.get("smiles")),
        }
    return {"name": "", "content": "", "smiles": ""}


def build_composition_table_rows(
    samples: List[Dict[str, Any]],
    *,
    doi: str = "",
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Build the sample-level chemical composition table requested by the UI.

    Columns are:
      DOI, sample_name, smiles1..smilesN, c1..cN,
      additive_smiles1..additive_smilesM, additive_c1..additive_cM,
      additive_name_1..additive_name_K, notes.
    """
    normalized_samples = _normalize_composition_samples(samples)
    max_polymer = 3
    max_additive = 3
    max_additive_names = 2
    for sample in normalized_samples:
        polymer_indices = _numbered_indices(sample, "smiles", "c")
        additive_indices = _numbered_indices(sample, "additive_smiles", "additive_c")
        additive_name_indices = _numbered_additive_name_indices(sample)
        if polymer_indices:
            max_polymer = max(max_polymer, max(polymer_indices))
        if additive_indices:
            max_additive = max(max_additive, max(additive_indices))
        if additive_name_indices:
            max_additive_names = max(max_additive_names, max(additive_name_indices))
    max_additive_names = max(max_additive_names, min(max_additive, 2))

    columns = (
        ["DOI", "sample_name"]
        + [f"smiles{i}" for i in range(1, max_polymer + 1)]
        + [f"c{i}" for i in range(1, max_polymer + 1)]
        + [f"additive_smiles{i}" for i in range(1, max_additive + 1)]
        + [f"additive_c{i}" for i in range(1, max_additive + 1)]
        + [f"additive_name_{i}" for i in range(1, max_additive_names + 1)]
        + ["notes"]
    )

    rows: List[Dict[str, str]] = []
    for sample in normalized_samples:
        row = {column: "" for column in columns}
        row["DOI"] = doi
        row["sample_name"] = _to_text(sample.get("sample_name"))
        row["notes"] = _to_text(sample.get("notes"))
        for idx in range(1, max_polymer + 1):
            row[f"smiles{idx}"] = _normalize_smiles(
                sample.get(f"smiles{idx}", ""),
                require_polymer=True,
            )
            row[f"c{idx}"] = _table_numeric_value(sample.get(f"c{idx}"))
        for idx in range(1, max_additive + 1):
            row[f"additive_smiles{idx}"] = _normalize_smiles(
                sample.get(f"additive_smiles{idx}", "")
            )
            row[f"additive_c{idx}"] = _table_numeric_value(sample.get(f"additive_c{idx}"))
        for idx in range(1, max_additive_names + 1):
            row[f"additive_name_{idx}"] = _to_text(
                sample.get(f"additive_name_{idx}", sample.get(f"additive_name{idx}", ""))
            )
        rows.append(row)

    return rows, columns


def _build_paper_content(parsed_text: str, tables_text: str = "", figures_text: str = "") -> str:
    """Build the paper-context block shared by the relevance prompt."""
    combined = ""
    if parsed_text:
        combined += "=== BODY TEXT ===\n" + _truncate(parsed_text) + "\n\n"
    if tables_text:
        combined += "=== TABLES ===\n" + _truncate(tables_text, 20000) + "\n\n"
    if figures_text:
        combined += "=== FIGURE CAPTIONS ===\n" + _truncate(figures_text, 10000) + "\n\n"
    return combined


_FIGURE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _normalize_source_id_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _has_digit(value: str) -> bool:
    return any(char.isdigit() for char in value)


def _find_matching_figure_image(figures_dir: Path, source_id: str) -> Optional[Path]:
    """Find the local figure image that corresponds to a relevant-content ID."""
    src_norm = _normalize_source_id_for_match(source_id)
    if not src_norm or not figures_dir.exists():
        return None

    image_files = sorted(
        path
        for path in figures_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _FIGURE_IMAGE_SUFFIXES
    )

    for image_path in image_files:
        stem_norm = _normalize_source_id_for_match(image_path.stem)
        if stem_norm == src_norm:
            return image_path

    for image_path in image_files:
        stem_norm = _normalize_source_id_for_match(image_path.stem)
        if _has_digit(src_norm) and not _has_digit(stem_norm):
            continue
        if src_norm in stem_norm or stem_norm in src_norm:
            return image_path

    for meta_path in sorted(figures_dir.glob("*_metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        candidates = [
            _to_text(meta.get("id")),
            meta_path.stem.replace("_metadata", ""),
        ]
        if not any(
            src_norm in _normalize_source_id_for_match(candidate)
            or _normalize_source_id_for_match(candidate) in src_norm
            for candidate in candidates
            if candidate
        ):
            continue

        base_stem = meta_path.stem.replace("_metadata", "")
        for suffix in _FIGURE_IMAGE_SUFFIXES:
            image_path = meta_path.with_name(base_stem + suffix)
            if image_path.exists():
                return image_path

    return None


def _is_structure_figure_for_vision(item: Dict[str, Any]) -> bool:
    """Select chemical drawings/schemes for multimodal composition extraction."""
    src_type = _to_text(item.get("source_type")).strip().lower()
    if src_type not in ("figure/scheme", "figure", "scheme"):
        return False

    haystack = " ".join(
        _to_text(item.get(key))
        for key in ("source_id", "excerpt", "content", "relevance")
    ).lower()

    if "scheme" in haystack:
        return True

    spectroscopy_terms = (
        "ftir",
        "xps",
        "nmr",
        "spectra",
        "spectrum",
        "survey scan",
        "f 1s",
        "n 1s",
    )
    if any(term in haystack for term in spectroscopy_terms):
        return False

    structure_terms = (
        "synthesis route",
        "chemical structure",
        "chemical drawing",
        "molecular diagram",
        "reaction",
        "repeat-unit",
        "repeat unit",
        "microstructure",
    )
    return any(term in haystack for term in structure_terms)


def _attach_relevant_figure_images(
    relevant_content: List[Dict[str, Any]],
    figures_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """Attach local image paths to relevant chemical drawing/scheme evidence."""
    if not figures_dir or not figures_dir.exists():
        return relevant_content

    attached: List[Dict[str, Any]] = []
    for item in relevant_content:
        item_out = dict(item)
        if _is_structure_figure_for_vision(item_out):
            image_path = _find_matching_figure_image(figures_dir, _to_text(item_out.get("source_id")))
            if image_path is not None:
                item_out["image_filename"] = image_path.name
                item_out["image_path"] = str(image_path)
        attached.append(item_out)
    return attached


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_composition_messages(
    prompt: str,
    relevant_content: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    image_items = [
        item
        for item in relevant_content
        if item.get("image_path") and _is_structure_figure_for_vision(item)
    ]
    if not image_items:
        return [{"role": "user", "content": prompt}]

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                prompt
                + "\n\nAttached chemical drawing/scheme images correspond to the "
                "evidence items that include image_filename/image_path. Inspect the "
                "images directly when deriving repeat-unit SMILES and composition "
                "fractions; do not rely on the local file path text itself."
            ),
        }
    ]

    for item in image_items[:3]:
        image_path = Path(_to_text(item.get("image_path")))
        if not image_path.exists():
            continue
        content.append(
            {
                "type": "text",
                "text": (
                    f"Attached image for {item.get('source_id', 'figure')}: "
                    f"{image_path.name}"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(image_path),
                    "detail": "high",
                },
            }
        )

    return [{"role": "user", "content": content}]


def _get_chemical_composition_client(step_key: str) -> tuple[Any, bool]:
    """Use the chemistry-step LLM override when configured, else fall back."""
    prefix = step_env_prefix(step_key)
    override = load_optional_llm_config_from_env(
        provider_env=f"{prefix}_LLM_PROVIDER",
        model_env=f"{prefix}_LLM_MODEL",
        openai_base_url_envs=(f"{prefix}_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
        openai_api_key_envs=(f"{prefix}_OPENAI_API_KEY", "OPENAI_API_KEY"),
        ollama_host_envs=(f"{prefix}_OLLAMA_HOST", "OLLAMA_HOST"),
        ollama_num_ctx_envs=(
            f"{prefix}_OLLAMA_NUM_CTX",
            "SALSA_OLLAMA_NUM_CTX",
            "OLLAMA_NUM_CTX",
        ),
        default_provider="openai",
    )
    # Streamlit mutates env vars between runs, while get_llm_client() caches a
    # global client. Force a fresh client so chemistry extraction follows the
    # currently selected per-step model instead of a stale cached client.
    return get_llm_client(force_new=True, override=override), override is not None


def _parse_json_object(raw: str) -> Dict[str, Any]:
    for candidate in _candidate_json_payloads(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _relevant_content_to_prompt_text(relevant_content: List[Dict[str, Any]]) -> str:
    """Serialize relevant content compactly for the composition prompt."""
    normalized = _normalize_relevant_content_items(relevant_content)
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def extract_relevant_content_llm(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """Identify relevant chemistry/sample evidence in a dedicated LLM call."""
    combined = _build_paper_content(parsed_text, tables_text, figures_text)
    if not combined.strip():
        return []

    estimated_tokens = _estimate_tokens(combined)
    # if estimated_tokens > 3500:
    #     print(
    #         "[chemical_composition] WARNING: estimated relevant-content prompt "
    #         f"is ~{estimated_tokens} tokens before model-side tokenization. "
    #         "If Ollama is using a 4096-token context, it may truncate the paper. "
    #         "Set SALSA_OLLAMA_NUM_CTX to a larger value such as 16384 or 32768."
    #     )

    prompt = _RELEVANT_CONTENT_PROMPT.format(content=combined)
    _write_debug_text(doi, "relevant_content_prompt.txt", prompt)
    client, used_step_llm_override = _get_chemical_composition_client(LLM_STEP_CHEM_RELEVANT)
    raw, meta = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        return_meta=True,
    )
    _write_debug_text(doi, "relevant_content_raw.txt", raw)
    _log_cost(
        meta,
        doi=doi,
        extra={
            "kind": "chemical_composition_relevant_content",
            "configured_model": client.model,
            "used_step_llm_override": used_step_llm_override,
        },
    )

    result = _parse_json_object(raw)
    _write_debug_json(doi, "relevant_content_parsed.json", result)
    return _normalize_relevant_content_items(result.get("relevant_content", []))


def extract_composition_from_relevant_content_llm(
    relevant_content: List[Dict[str, Any]],
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """Extract flat sample composition rows from previously selected evidence."""
    relevant_prompt_text = _relevant_content_to_prompt_text(relevant_content)
    if not relevant_prompt_text.strip() or relevant_prompt_text.strip() == "[]":
        return []

    prompt = _COMPOSITION_EXTRACTION_PROMPT.format(
        relevant_content=relevant_prompt_text
    )
    _write_debug_text(doi, "samples_prompt.txt", prompt)
    client, used_step_llm_override = _get_chemical_composition_client(LLM_STEP_CHEM_COMPOSITION)
    messages = _build_composition_messages(prompt, relevant_content)
    raw, meta = client.chat(
        messages,
        temperature=0.0,
        return_meta=True,
    )
    _write_debug_text(doi, "samples_raw.txt", raw)
    _log_cost(
        meta,
        doi=doi,
        extra={
            "kind": "chemical_composition_table",
            "configured_model": client.model,
            "used_step_llm_override": used_step_llm_override,
        },
    )

    result = _parse_json_object(raw)
    _write_debug_json(doi, "samples_parsed.json", result)
    return _normalize_composition_samples(result.get("samples", []))


def extract_chemical_composition_combined(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
    figures_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compatibility wrapper that identifies relevant content, then extracts
    sample-level chemical composition rows from that relevant content.

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
        - "samples": list of flat rows with sample_name, smiles/c,
          additive_smiles/additive_c, and notes
    """
    relevant = extract_relevant_content_llm(
        parsed_text, tables_text, figures_text, doi=doi
    )
    relevant = _attach_relevant_figure_images(relevant, figures_dir)
    samples = extract_composition_from_relevant_content_llm(relevant, doi=doi)
    return {"relevant_content": relevant, "samples": samples}


# Legacy functions kept for backward compatibility
def identify_relevant_content(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """
    Identify relevant content for chemical composition extraction.
    """
    return extract_relevant_content_llm(
        parsed_text, tables_text, figures_text, doi=doi
    )

def extract_samples_and_compositions(
    parsed_text: str,
    tables_text: str = "",
    figures_text: str = "",
    *,
    doi: str = "",
) -> List[Dict[str, Any]]:
    """
    Extract flat sample composition rows after relevant-content selection.
    """
    relevant = extract_relevant_content_llm(
        parsed_text, tables_text, figures_text, doi=doi
    )
    return extract_composition_from_relevant_content_llm(relevant, doi=doi)

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

    Reads parsed text + tables + figures, calls relevance selection and then
    composition-table extraction,
    writes results to ``<run_dir>/chemical_composition/<safe_doi>/``.

    Returns the combined result dict.
    """
    import os
    import pandas as pd

    # Set SALSA_RUN_DIR so cost logging writes to the correct location
    os.environ["SALSA_RUN_DIR"] = str(run_dir)

    doi_str = safe_doi.replace("@", "/")

    # --- gather text ---
    parsed_text = ""
    if parsed_json_path and parsed_json_path.exists():
        try:
            pj = json.loads(parsed_json_path.read_text())
            parsed_text = _parsed_json_to_text(pj)
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

    # --- Two-stage LLM extraction: relevant evidence, then composition table ---
    extraction_result = extract_chemical_composition_combined(
        parsed_text,
        tables_text,
        figures_text,
        doi=doi_str,
        figures_dir=figures_dir,
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

    # Produce the UI-facing sample-level composition table:
    # sample_name, smiles1..smilesN, c1..cN, additive_smiles1.., additive_c1..
    composition_rows, composition_columns = build_composition_table_rows(samples, doi=doi_str)
    df = pd.DataFrame(composition_rows, columns=composition_columns)
    df.to_csv(out_dir / "samples.csv", index=False)

    return result
