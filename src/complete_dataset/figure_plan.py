from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd


# -----------------------------
# Defaults (match your pipeline)
# -----------------------------
DEFAULTS = {
    "prop test temp (C)": "25 Not Found",
    "degradation test temp (C)": "None Not Found",
    "time(h)": "0 Not Found",
    "relative humidity(%)": "100 Not Found",
    "solvent type": "None Not Found",
    "solvent molarity (M)": "None Not Found",
}

# Provenance fields you should always force from code, not the model
FORCE_FIELDS = {"DOI", "source_kind", "source_name"}


def _df_to_records_clean(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df2 = df.replace([pd.NA, float("inf"), float("-inf")], None)
    df2 = df2.where(pd.notna(df2), None)
    return df2.to_dict(orient="records")


def _hash_payload(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -----------------------------
# Plan schema for the LLM
# -----------------------------
JSONType = Literal["string", "number", "integer", "boolean", "null", "object", "array"]


def build_figure_plan_json_schema(
    output_columns: List[str],
    *,
    schema_name: str = "aem_ocr_figure_plan",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    The model returns:
    {
      "series_plans": [
        {
          "series_label": "...",
          "x_maps_to": "<one of output_columns or ''>",
          "y_property": "conductivity",
          "constants": { "<col>": "<string>", ... },
          "confidence": 0.0-1.0,
          "rationale": "..."
        },
        ...
      ]
    }

    - x_maps_to is an enum of output_columns plus "" (meaning "no matching x column")
    - constants must include ALL output columns (strings), but you can ignore/override FORCE_FIELDS in code
    """
    if not output_columns:
        raise ValueError("output_columns must be non-empty")

    col_enum = list(output_columns) + [""]  # allow "no mapping"

    return {
        "name": schema_name,
        "strict": strict,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["series_plans"],
            "properties": {
                "series_plans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["series_label", "x_maps_to", "y_property", "constants", "confidence", "rationale"],
                        "properties": {
                            "series_label": {"type": "string"},
                            "x_maps_to": {"type": "string", "enum": col_enum},
                            "y_property": {"type": "string"},
                            "constants": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": list(output_columns),
                                "properties": {c: {"type": "string"} for c in output_columns},
                            },
                            "confidence": {"type": "number"},
                            "rationale": {"type": "string"},
                        },
                    },
                }
            },
        },
    }


# -----------------------------
# Prompt builder
# -----------------------------
def create_figure_plan_prompt(
    df_points: pd.DataFrame,
    output_columns: List[str],
    *,
    doi: str,
    dataset_name: str,
    figure_snips: List[Dict[str, Any]],
    full_text: Any,
    head_n: int = 8,
) -> str:
    """
    Groups by series_label and provides:
    - axis names/units (first non-null)
    - small head of points
    - column list
    - caption snippets + (optional) full text
    """
    if df_points is None or df_points.empty:
        payload = {
            "doi": doi,
            "dataset_name": dataset_name,
            "output_columns": output_columns,
            "series": [],
            "figure_snippets": figure_snips or [],
            "full_text": full_text or "",
        }
        return "INPUT_PAYLOAD_JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    df = df_points.copy()
    if "series_label" not in df.columns:
        df["series_label"] = "UNKNOWN_SERIES"

    series_payloads: List[Dict[str, Any]] = []

    for series_label, g in df.groupby("series_label", dropna=False):
        g = g.copy()

        def first_nonnull(col: str) -> str:
            if col not in g.columns:
                return ""
            s = g[col].dropna()
            return str(s.iloc[0]) if len(s) else ""

        series_payloads.append({
            "series_label": str(series_label),
            "data_columns": list(g.columns),
            "x_name": first_nonnull("x_name"),
            "x_unit": first_nonnull("x_unit"),
            "y_name": first_nonnull("y_name"),
            "y_unit": first_nonnull("y_unit"),
            "head_points": _df_to_records_clean(g.head(head_n)),
            "n_points_total": int(len(g)),
        })

    payload = {
        "doi": doi,
        "dataset_name": dataset_name,
        "output_columns": output_columns,
        "figure_snippets": figure_snips or [],
        "full_text": full_text or "",
        "series": series_payloads,
        "notes": "The head_points are just for context; mapping must apply to ALL points in each series.",
    }
    return "INPUT_PAYLOAD_JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def figure_plan_instructions(output_columns: List[str], doi: str, dataset_name: str) -> str:
    """
    System message for plan inference.
    """
    return f"""
You are an expert at interpreting extracted scientific figure data (already digitized into points).

Infer a per-series mapping PLAN.

For EACH series_label you must output:
- x_maps_to: which output column should be filled from x_value.
  This is NOT necessarily time; choose the BEST matching column from output_columns.
  If none match, set x_maps_to = "".
- y_property: the canonical property name represented by y_value (e.g., "conductivity", "IEC", "water uptake", etc.).
- constants: fill EVERY output column as strings:
  * Put the appropriate constant value if stated in captions/full text.
  * Otherwise use the required Not Found defaults.
  * You MUST include ALL output columns as keys.

Hard rules:
- Use ONLY the provided payload (series axis names/units, figure snippets, full text).
- Do not invent experimental conditions.
- If uncertain, use Not Found defaults and lower confidence.

Missing-value defaults (exact strings):
- prop test temp (C): "25 Not Found"
- degradation test temp (C): "None Not Found"
- time(h): "0 Not Found"
- relative humidity(%): "100 Not Found"
- solvent type: "None Not Found"
- solvent molarity (M): "None Not Found"

Output must be valid JSON matching the provided JSON schema.
""".strip()


# -----------------------------
# Plan data class
# -----------------------------
@dataclass
class SeriesPlan:
    series_label: str
    x_maps_to: str               # "" means none
    y_property: str
    constants: Dict[str, str]
    confidence: float
    rationale: str


# -----------------------------
# LLM inference
# -----------------------------
def infer_figure_series_plans(
    client,
    *,
    df_points: pd.DataFrame,
    output_columns: List[str],
    doi: str,
    dataset_name: str,
    figure_snips: List[Dict[str, Any]],
    full_text: Any,
    cache_dir: Optional[str] = None) -> Tuple[List[SeriesPlan], Any]:
    """
    Calls the LLM once per dataset to infer series plans.
    Cache optional; if you already have a cache utility you can plug it in.
    """
    prompt = create_figure_plan_prompt(
        df_points=df_points,
        output_columns=output_columns,
        doi=doi,
        dataset_name=dataset_name,
        figure_snips=figure_snips,
        full_text=full_text,
    )
    instructions = figure_plan_instructions(output_columns, doi=doi, dataset_name=dataset_name)
    schema = build_figure_plan_json_schema(output_columns)

    # Optional caching hook (simple file cache)
    cache_key = None
    if cache_dir:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = f"figure_plan::{doi}::{dataset_name}::{_hash_payload({'prompt': prompt})}"
        fp = os.path.join(cache_dir, cache_key.replace("/", "_") + ".json")
        if os.path.exists(fp):
            with open(fp, "r") as f:
                obj = json.load(f)
            return _parse_series_plans(obj), None

    text, meta = client.chat(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.0,
        max_tokens=4000,   # plans are small
        return_meta=True,
    )

    print("====FIGURE TEXT====", text)

    obj = json.loads(text) if isinstance(text, str) else (text or {})
    if cache_dir and cache_key:
        with open(fp, "w") as f:
            json.dump(obj, f, indent=2)

    return _parse_series_plans(obj), meta

def _parse_series_plans(obj: Dict[str, Any]) -> List[SeriesPlan]:
    out: List[SeriesPlan] = []
    for sp in (obj or {}).get("series_plans", []) or []:
        out.append(SeriesPlan(
            series_label=str(sp.get("series_label", "")),
            x_maps_to=str(sp.get("x_maps_to", "")),
            y_property=str(sp.get("y_property", "")),
            constants=dict(sp.get("constants", {}) or {}),
            confidence=float(sp.get("confidence", 0.0) or 0.0),
            rationale=str(sp.get("rationale", "")),
        ))
    return out

# -----------------------------
# Apply plans deterministically
# -----------------------------
def apply_figure_series_plans(
    df_points: pd.DataFrame,
    output_columns: List[str],
    *,
    doi: str,
    dataset_name: str,
    plans: List[SeriesPlan],
    source_kind: str = "figure") -> pd.DataFrame:
    """
    Deterministic conversion:
    - one output row per input point
    - y_value -> "value"
    - property -> plan.y_property
    - x_value -> plan.x_maps_to (if non-empty)
    - constants fill the rest (with code-forced provenance)
    """
    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=output_columns)

    df = df_points.copy()
    if "series_label" not in df.columns:
        df["series_label"] = "UNKNOWN_SERIES"

    plan_by_label = {p.series_label: p for p in plans}

    all_out: List[pd.DataFrame] = []

    for series_label, g in df.groupby("series_label", dropna=False):
        s = str(series_label)
        p = plan_by_label.get(s)

        # Fallback plan if model omitted this series
        if p is None:
            # minimal safe plan
            constants = {c: DEFAULTS.get(c, "") for c in output_columns}
            p = SeriesPlan(
                series_label=s,
                x_maps_to="",
                y_property="None Not Found",
                constants=constants,
                confidence=0.0,
                rationale="missing plan; used fallback",
            )

        # Start output frame
        out = pd.DataFrame({c: [""] * len(g) for c in output_columns})

        # Fill constants from plan (but don't trust provenance fields)
        for c in output_columns:
            if c in FORCE_FIELDS:
                continue
            v = p.constants.get(c, "")
            out[c] = "" if v is None else str(v)

        # Fill defaults for any still-empty critical fields
        for k, dv in DEFAULTS.items():
            if k in output_columns:
                out.loc[out[k].astype(str).str.strip().eq(""), k] = dv

        # Force provenance
        if "DOI" in output_columns:
            out["DOI"] = doi
        if "source_kind" in output_columns:
            out["source_kind"] = source_kind
        if "source_name" in output_columns:
            out["source_name"] = f"{dataset_name}::{s}"

        # Sample Name
        if "Sample Name" in output_columns:
            out["Sample Name"] = s

        # property and value from data
        if "property" in output_columns:
            out["property"] = p.y_property or "None Not Found"
        if "value" in output_columns and "y_value" in g.columns:
            out["value"] = g["y_value"].astype(str)

        # x mapping
        if p.x_maps_to and p.x_maps_to in output_columns and "x_value" in g.columns:
            out[p.x_maps_to] = g["x_value"].astype(str)

        all_out.append(out.reindex(columns=output_columns))

    return pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame(columns=output_columns)
