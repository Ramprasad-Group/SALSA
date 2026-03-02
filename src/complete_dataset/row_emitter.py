# row_emitter.py
from __future__ import annotations
import json
import os
import hashlib
from typing import Literal, Any, Dict, List, Optional, Tuple

import pandas as pd

OUTPUT_COLUMNS = [
  "Sample Name",
  "time(h)",
  "relative humidity(%)",
  "solvent type",
  "solvent molarity (M)",
  "degradation test temp (C)",
  "prop test temp (C)",
  "property",
  "value",
  "DOI",
  "source_kind",
  "source_name",
]

# These module-level defaults can be overridden at runtime via custom instructions
MISSING_VALUE_DEFAULTS = {
    "prop test temp (C)": "25 Not Found",
    "degradation test temp (C)": "None Not Found",
    "time(h)": "0 Not Found",
    "relative humidity(%)": "100 Not Found",
    "solvent type": "None Not Found",
    "solvent molarity (M)": "None Not Found",
}

DATA_PROCESSING_RULES = """\
- Measurements taken before or at the beginning of stability tests MUST have time(h) = 0"""

# ----------------------------
# Small cache (same approach as your plan cache)
# ----------------------------
def ensure_cache_dir(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)

def _safe_filename(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:180]

def _cache_path(cache_dir: str, key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(cache_dir, f"{_safe_filename(key)}__{h}.json")

def load_cached_rows(cache_dir: str, key: str) -> Optional[List[Dict[str, Any]]]:
    fp = _cache_path(cache_dir, key)
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r") as f:
            obj = json.load(f)
        # Support either {"rows":[...]} or [...]
        if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
            return obj["rows"]
        if isinstance(obj, list):
            return obj
        return None
    except Exception:
        return None

def save_cached_rows(cache_dir: str, key: str, rows: List[Dict[str, Any]]) -> None:
    fp = _cache_path(cache_dir, key)
    try:
        with open(fp, "w") as f:
            json.dump({"rows": rows}, f, indent=2)
    except Exception:
        pass

def _hash_payload(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ----------------------------
# Helpers
# ----------------------------
def _df_to_records_clean(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df2 = df.replace([pd.NA, float("inf"), float("-inf")], None)
    # Also convert numpy NaN to None
    df2 = df2.where(pd.notna(df2), None)
    return df2.to_dict(orient="records")

def _coerce_rows_schema(rows: List[Dict[str, Any]], doi: str, source_kind: str, source_name: str,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Light sanitation only: enforce columns, fill blanks, set provenance if missing.
    No "plan application"; we assume LLM already did the mapping.
    """
    cols = columns or OUTPUT_COLUMNS
    fixed: List[Dict[str, Any]] = []
    for r in rows or []:
        rr = {k: "" for k in cols}
        if isinstance(r, dict):
            for k in cols:
                v = r.get(k, "")
                rr[k] = "" if v is None else str(v)
        fixed.append(rr)
    return pd.DataFrame(fixed).reindex(columns=cols, fill_value="").fillna("").astype(str)

def subsample_figure_points(
    df: pd.DataFrame,
    sample_col: str = "series_label",
    step: int = 20,) -> pd.DataFrame:
    """
    For each unique sample (series_label), keep every `step`-th row,
    starting from the first row in that group.

    Assumes df already contains only ONE extracted figure dataset.
    """
    if df is None or df.empty or sample_col not in df.columns:
        return df

    out = []

    for _, g in df.groupby(sample_col, dropna=False):
        g = g.copy()

        # Ensure stable order:
        # Prefer numeric x_value ordering if possible
        if "x_value" in g.columns:
            xnum = pd.to_numeric(g["x_value"], errors="coerce")
            if xnum.notna().any():
                g["_xnum"] = xnum
                g = g.sort_values("_xnum", kind="mergesort").drop(columns="_xnum")

        # Keep every Nth row (includes first row)
        out.append(g.iloc[::step])

    return pd.concat(out, ignore_index=True) if out else df

# def create_table_prompt(payload, OUTPUT_COLUMNS):
#     """Create prompt for table extraction"""

#     parts = []

#     parts.append(f"""
#         Convert the given table into rows with exactly the specified output columns.

#         You may use:
#         - Table (columns + rows)
#         - Table caption
#         - Full paper text (if available)

#         OUTPUT_COLUMNS:
#         {json.dumps(OUTPUT_COLUMNS, indent=2)}

#         Table Columns: 
#         {json.dumps(payload.get("table_columns"), indent=2)},

#         Table Rows:
#         {json.dumps(payload.get("table_rows"), indent=2)},

#         Table Caption:
#         {json.dumps(payload.get("caption"), indent=2)},
#     """)

#     full_text = payload.get("full_text")
#     if full_text:
#         parts.append(f"""
#             Full Paper Text:
#             {json.dumps(full_text, indent=2)}
#         """)

#     prompt = "\n\n".join(parts)

#     # print("===TABLE PROMPT===", prompt)

#     return prompt

def create_table_prompt(payload, OUTPUT_COLUMNS):
    """
    User content: strict payload wrapper.
    Keep ALL logic/instructions in the system message; user message is data + tiny framing.
    """
    data = {
        "output_columns": OUTPUT_COLUMNS,
        "table_columns": payload.get("table_columns", []),
        "table_rows": payload.get("table_rows", []),
        "caption": payload.get("caption", ""),
        "table_name": payload.get("table_name", payload.get("label", "")),  # optional helper
        "full_text": payload.get("full_text", None),  # can be None
    }

    return (
        "INPUT_PAYLOAD_JSON (authoritative; do not invent fields outside this payload):\n"
        + json.dumps(data, ensure_ascii=False, indent=2)
    )

# def create_figure_prompt(payload: Dict[str, Any], OUTPUT_COLUMNS: List[str]) -> str:
#     """Create prompt for figure extraction (entire figure dataset)."""
#     parts: List[str] = []

#     parts.append(
#         f"""
#         Convert the given extracted figure dataset into rows with exactly the specified output columns.

#         You may use:
#         - Figure dataset (columns + points)
#         - Figure caption/snippets
#         - Full paper text (if available)

#         OUTPUT_COLUMNS:
#         {json.dumps(OUTPUT_COLUMNS, indent=2)}

#         FIGURE DATASET COLUMNS:
#         {json.dumps(payload.get("data_columns", []), indent=2)}

#         FIGURE DATASET ROWS:
#         {json.dumps(payload.get("data_rows", []), indent=2)}

#         FIGURE SNIPPETS (caption/context):
#         {json.dumps(payload.get("figure_snippets", []), indent=2)}
#         """.strip()
#         )

#     full_text = payload.get("full_text")
#     if full_text:
#         parts.append(
#             f"""
#             FULL PAPER TEXT:
#             {json.dumps(full_text, indent=2)}
#             """.strip()
#                     )

#     prompt = "\n\n".join(parts)
#     # print("===FIGURE PROMPT===\n", prompt)
#     return prompt

def create_figure_prompt(payload, OUTPUT_COLUMNS):
    data = {
        "output_columns": OUTPUT_COLUMNS,
        "data_columns": payload.get("dataset_columns", []),
        "data_rows": payload.get("dataset_rows", []),  # each entry should represent 1 point/row
        "figure_snippets": payload.get("dataset_snippets", []),
        "figure_name": payload.get("dataset_name", payload.get("figure_id", "")),
        "full_text": payload.get("full_text", None),
    }

    return (
        "INPUT_PAYLOAD_JSON (authoritative; do not invent fields outside this payload):\n"
        + json.dumps(data, ensure_ascii=False, indent=2)
    )


# ----------------------------
# Core LLM call: emits rows
# ----------------------------

JSONType = Literal["string", "number", "integer", "boolean", "null", "object", "array"]

def build_rows_json_schema(
    output_columns: List[str],
    *,
    schema_name: str = "aem_ocr_rows",
    strict: bool = True,
    field_type: JSONType = "string",
    allow_nulls: bool = False) -> Dict:
    """
    Build a strict JSON Schema of the form:
      { "rows": [ {<col>: <type>, ... } ] }

    - properties generated from output_columns
    - required = output_columns
    - additionalProperties = False everywhere
    """
    if not output_columns:
        raise ValueError("output_columns must be a non-empty list")

    def col_schema():
        if allow_nulls:
            # allow missing-like null but still require key
            return {"anyOf": [{"type": field_type}, {"type": "null"}]}
        return {"type": field_type}

    row_properties = {col: col_schema() for col in output_columns}

    return {
        "name": schema_name,
        "strict": strict,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["rows"],
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(output_columns),
                        "properties": row_properties,
                    },
                }
            },
        },
    }

def _build_missing_value_block(defaults: Dict[str, str]) -> str:
    """Build the missing-value rules section of the system prompt."""
    lines = []
    for col, default in defaults.items():
        lines.append(f'- {col}: "{default}"')
    return "\n        ".join(lines)


def infer_rows_for_table(
    client,
    doi: str,
    table_name: str,
    caption: str,
    df_table: pd.DataFrame,
    full_text: Optional[Any],
    cache_dir: str,
    max_rows_per_call: int = 60,
    *,
    output_columns: Optional[List[str]] = None,
    missing_value_defaults: Optional[Dict[str, str]] = None,
    extra_rules: str = "",
) -> Tuple[pd.DataFrame, Any]:
    """
    Returns (df_rows, meta). Sends a table to the LLM in one call and asks
    it to directly emit normalized dataset rows with exactly OUTPUT_COLUMNS.
    """
    cols = output_columns or OUTPUT_COLUMNS
    defaults = missing_value_defaults or MISSING_VALUE_DEFAULTS
    custom_rules = extra_rules.strip() if extra_rules else DATA_PROCESSING_RULES.strip()

    if df_table is None or df_table.empty:
        return pd.DataFrame(columns=cols), None

    # Convert the whole table to JSON-safe records (list[dict])
    records = _df_to_records_clean(df_table)

    payload: Dict[str, Any] = {
        "doi": doi,
        "source_kind": "table",
        "source_name": table_name,
        "table_name": table_name,
        "caption": caption,
        "table_columns": list(df_table.columns),
        "table_rows": records,
    }
    if full_text:
        payload["full_text"] = full_text
    else:
        payload["full_text"] = ""

    cache_key = f"emit_rows::table::{doi}::{table_name}::{_hash_payload(payload)}"
    cached = load_cached_rows(cache_dir, cache_key)
    if cached is not None:
        df_out = _coerce_rows_schema(
            cached, doi=doi, source_kind="table", source_name=table_name
        )
        return df_out, None

    missing_block = _build_missing_value_block(defaults)

    custom_rules_block = ""
    if custom_rules:
        custom_rules_block = f"\n        {custom_rules}"

    instructions = f"""
        You are an expert at converting scientific tables into structured tabular datasets.

        Your task:
        - Create ONE output row for EVERY individual data point.
        - A "data point" is defined as ONE (x_value, y_value) pair from the figure.
        - DO NOT aggregate, average, group, summarize, or combine points.

        SAMPLE NAME EXTRACTION - CRITICAL RULES:
        1. ALWAYS attempt to identify or construct a sample name for each data point
        2. Extract sample names from table columns, rows, or headers
        3. If sample names are implicit, construct from identifying columns (composition, ID, designation, etc.)
        4. Use descriptive identifiers based on varying parameters (e.g., "IEC_1.5", "80C_sample")
        5. ONLY leave Sample Name empty if absolutely no identifying information is available

        Output schema:
        Convert the data into rows with EXACTLY the following output_columns:
        {json.dumps(cols, indent=2)}

        Return ONLY valid JSON in this exact format:
        {{
        "rows": [
            {{ "<column>": "<value>", "...": "..." }},
            ...
        ]
        }}

        STRICT RULES (must follow):
        - Each row corresponds to exactly ONE data point.
        - Each row MUST contain ALL columns listed in output_columns.
        - No column may be omitted.
        - Do NOT nest arrays inside a row.
        - Do NOT create summary rows.
        - Do NOT merge multiple data points into one row.
        - Do not invent values.
        - Do not infer trends.
        - Do not perform any unit conversions.

        Allowed information sources:
        - (1) the provided table
        - (2) the table caption
        - (3) full text (if provided)

        Provenance (always fill exactly):
        - DOI = "{doi}"
        - source_kind = "table"
        - source_name = "{table_name}"

        Missing-value rules (use EXACT strings):
        {missing_block}

        IMPORTANT:
        - If a value is not explicitly stated, use the Not Found rules above.{custom_rules_block}

        If there are zero data points, return:
        {{ "rows": [] }}
        """.strip()

    prompt = create_table_prompt(payload, cols)

    json_schema = build_rows_json_schema(
        cols,
        schema_name="aem_ocr_rows",
        field_type="string",
        allow_nulls=False,
    )

    print('===TABLE INSTRUCTIONS===', instructions)
    print('===TABLE PROMPT===', prompt)
    print('===TABLE JSON_SCHEMA===', json_schema)

    text, meta = client.chat(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": json_schema},
        temperature=0.0,
        max_tokens=10000,
        return_meta=True,
    )

    print("====TABLE TEXT====", text)

    try:
        obj = json.loads(text) if isinstance(text, str) else text
    except Exception:
        obj = {}

    rows = obj.get("rows", []) if isinstance(obj, dict) else []
    if not isinstance(rows, list):
        rows = []

    save_cached_rows(cache_dir, cache_key, rows)

    df_out = _coerce_rows_schema(rows, doi=doi, source_kind="table", source_name=table_name)
    return df_out, meta


def infer_rows_for_figure_dataset(
    client,
    doi: str,
    dataset_name: str,
    figure_snips: List[Dict[str, Any]],
    df_points: pd.DataFrame,
    full_text: Optional[Any],
    cache_dir: str,
    max_points_per_call: int = 200,
    subsample_step: Optional[int] = None,
    *,
    output_columns: Optional[List[str]] = None,
    missing_value_defaults: Optional[Dict[str, str]] = None,
    extra_rules: str = "",
) -> Tuple[pd.DataFrame, Any]:
    """
    Returns (df_rows, meta). Sends the FULL extracted figure dataset to the LLM in one call and asks
    it to directly emit normalized dataset rows with exactly OUTPUT_COLUMNS.
    """
    cols = output_columns or OUTPUT_COLUMNS
    defaults = missing_value_defaults or MISSING_VALUE_DEFAULTS
    custom_rules = extra_rules.strip() if extra_rules else DATA_PROCESSING_RULES.strip()

    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=cols), None

    ensure_cache_dir(cache_dir)

    df_in = df_points
    records = _df_to_records_clean(df_in)

    payload: Dict[str, Any] = {
        "doi": doi,
        "source_kind": "figure",
        "source_name": dataset_name,
        "dataset_name": dataset_name,
        "dataset_snippets": figure_snips,
        "dataset_columns": list(df_in.columns),
        "dataset_rows": records,
        "full_text": full_text or "",
    }

    cache_key = f"emit_rows::figure::{doi}::{dataset_name}::{_hash_payload(payload)}"

    cached = load_cached_rows(cache_dir, cache_key)
    if cached is not None:
        df_out = _coerce_rows_schema(cached, doi=doi, source_kind="figure", source_name=dataset_name)
        return df_out, None

    missing_block = _build_missing_value_block(defaults)

    custom_rules_block = ""
    if custom_rules:
        custom_rules_block = f"\n        {custom_rules}"

    instructions = f"""
        You are an expert at converting scientific figure data into structured tabular datasets.

        Your task:
        - Create ONE output row for EVERY individual data point.
        - A "data point" is defined as ONE (x_value, y_value) pair from the figure.
        - DO NOT aggregate, average, group, summarize, or combine points.

        SAMPLE NAME EXTRACTION - CRITICAL RULES:
        1. ALWAYS attempt to identify or construct a sample name for each data point
        2. Priority order for finding sample names:
           a) Explicit sample names in caption or full text
           b) Series label from the legend (use as-is if it represents a sample)
           c) Construct from series_label if it contains identifying information (e.g., "IEC=1.5" → "Sample_IEC_1.5")
           d) Extract from dataset metadata or figure annotations
           e) Use parameter values to create descriptive identifier (e.g., "80C_60RH" for temp/humidity conditions)

        3. Examples of constructing sample names from extracted data:
           - series_label="IEC=1.5 mS/cm" → Sample Name: "IEC_1.5" or "Sample_1.5_mS_cm"
           - series_label="n=8" → Sample Name: "n8_spacer" or "C8_chain"
           - series_label="t=24h" AND time series → Sample Name: Use sample from caption if available
           - series_label="80°C" → Sample Name: "Sample_80C" (if temperature is the varying parameter)

        4. For time series where series_label is a time point:
           - Extract the actual sample name from caption/full text
           - Use the time value in the "time(h)" column
           - If no sample name in caption, create identifier like "Sample_TimeSeries_1"

        5. ONLY leave Sample Name empty as absolute last resort when:
           - No legend information available
           - No identifying information in caption
           - No parameter values to construct from

        6. When constructing sample names:
           - Use descriptive identifiers based on the varying parameter
           - Include units or values that distinguish samples
           - Be consistent across all data points from same series

        Output schema:
        Convert the data into rows with EXACTLY the following output_columns:
        {json.dumps(cols, indent=2)}

        Return ONLY valid JSON in this exact format:
        {{
        "rows": [
            {{ "<column>": "<value>", "...": "..." }},
            ...
        ]
        }}

        STRICT RULES (must follow):
        - Each row corresponds to exactly ONE data point.
        - Each row MUST contain ALL columns listed in output_columns.
        - No column may be omitted.
        - Do NOT nest arrays inside a row.
        - Do NOT create summary rows.
        - Do NOT merge multiple data points into one row.
        - Do not invent values.
        - Do not infer trends.
        - Do not perform any unit conversions.

        Allowed information sources:
        - (1) the provided figure data
        - (2) the figure caption
        - (3) full text (if provided)

        Provenance (always fill exactly):
        - DOI = "{doi}"
        - source_kind = "figure"
        - source_name = "{dataset_name}"

        Missing-value rules (use EXACT strings):
        {missing_block}

        IMPORTANT:
        - If a value is not explicitly stated, use the Not Found rules above.{custom_rules_block}

        If there are zero data points, return:
        {{ "rows": [] }}
        """.strip()


    prompt = create_figure_prompt(payload, cols)


    json_schema = build_rows_json_schema(
        cols,
        schema_name="aem_ocr_rows",
        field_type="string",
        allow_nulls=False,
    )

    print('===FIGURE INSTRUCTIONS===', instructions)
    print('===FIGURE PROMPT===', prompt)
    print('===FIGURE JSON_SCHEMA===', json_schema)



    text, meta = client.chat(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": json_schema},
        temperature=0.0,
        max_tokens=10000,
        return_meta=True,
    )

    print("====FIGURE TEXT====", text)

    try:
        obj = json.loads(text) if isinstance(text, str) else text
    except Exception as e:
        print(f"Failed to read JSON for DOI text: {text}: {e}")
        obj = {}

    rows = obj.get("rows", []) if isinstance(obj, dict) else []
    if not isinstance(rows, list):
        rows = []

    save_cached_rows(cache_dir, cache_key, rows)

    df_out = _coerce_rows_schema(rows, doi=doi, source_kind="figure", source_name=dataset_name)

    return df_out, meta