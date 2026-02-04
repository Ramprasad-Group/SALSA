# transform_plans.py
"""
LLM plan inference + deterministic application.

Design goals:
- GPT outputs compact PLAN JSON (semantics + mappings), not per-point rows.
- Python applies plan deterministically to *full-resolution* extracted points / table rows.
- time(h) is only populated when x_semantic == "time"; otherwise time(h) = "".
- Always preserve x axis as x_name/x_unit/x_value in output rows.

No extra dependencies beyond stdlib + pandas.
"""

from __future__ import annotations
import os
import json
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# ----------------------------
# Cache helpers
# ----------------------------
def ensure_plan_cache_dir(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)

def _safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:200]

def _cache_path(cache_dir: str, key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(cache_dir, f"{_safe_filename(key)}__{h}.json")

def load_cached_plan(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    fp = _cache_path(cache_dir, key)
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r") as f:
            obj = json.load(f)

        # Some older cache files may store a JSON string; try decoding again.
        if isinstance(obj, str):
            try:
                obj2 = json.loads(obj)
                obj = obj2
            except Exception:
                return None

        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def save_cached_plan(cache_dir: str, key: str, plan: Dict[str, Any]) -> None:
    fp = _cache_path(cache_dir, key)
    try:
        with open(fp, "w") as f:
            json.dump(plan, f, indent=2)
    except Exception:
        pass

# ----------------------------
# Heuristics (cheap guardrails)
# ----------------------------
_TIME_UNITS = {"h", "hr", "hrs", "hour", "hours", "min", "mins", "minute", "minutes", "s", "sec", "secs", "second", "seconds", "day", "days", "d"}
_TEMP_UNITS = {"c", "°c", "degc", "k", "kelvin"}
_RH_HINTS = {"rh", "relative humidity", "%rh"}

def heuristic_axis_semantics(x_name: str, x_unit: str) -> Dict[str, Any]:
    xn = (x_name or "").strip().lower()
    xu = (x_unit or "").strip().lower()

    # time
    if "time" in xn or xu in _TIME_UNITS:
        return {"x_semantic": "time", "confidence": 0.95, "reason": f"time-like label/unit ({x_name}, {x_unit})"}

    # temperature
    if "temp" in xn or "temperature" in xn or xu in _TEMP_UNITS:
        return {"x_semantic": "temperature", "confidence": 0.90, "reason": f"temp-like label/unit ({x_name}, {x_unit})"}

    # RH
    if any(h in xn for h in _RH_HINTS) or ("%rh" in xu):
        return {"x_semantic": "RH", "confidence": 0.85, "reason": f"RH-like label/unit ({x_name}, {x_unit})"}

    # IEC / water uptake / swelling / potential / current density quick hints
    if "iec" in xn:
        return {"x_semantic": "IEC", "confidence": 0.80, "reason": "x label contains IEC"}
    if "water uptake" in xn or xn == "wu":
        return {"x_semantic": "water_uptake", "confidence": 0.80, "reason": "x label suggests water uptake"}
    if "swelling" in xn:
        return {"x_semantic": "swelling", "confidence": 0.80, "reason": "x label suggests swelling"}
    if "potential" in xn or xn == "v" or xu == "v":
        return {"x_semantic": "potential", "confidence": 0.70, "reason": "x label/unit suggests potential"}
    if "current" in xn or "ma/cm" in xu or "a/cm" in xu:
        return {"x_semantic": "current_density", "confidence": 0.70, "reason": "x label/unit suggests current density"}

    return {"x_semantic": "other", "confidence": 0.50, "reason": "no strong heuristic match"}

# ----------------------------
# Unit conversion (minimal but safe)
# ----------------------------
_TIME_UNIT_TO_H = {
    "h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
    "min": 1/60, "mins": 1/60, "minute": 1/60, "minutes": 1/60,
    "s": 1/3600, "sec": 1/3600, "secs": 1/3600, "second": 1/3600, "seconds": 1/3600,
    "day": 24.0, "days": 24.0, "d": 24.0,
}

def to_hours(x: Any, unit: str) -> str:
    if x is None or x == "":
        return ""
    try:
        if isinstance(x, str):
            m = re.search(r"[-+]?\d*\.?\d+", x)
            if not m:
                return ""
            xv = float(m.group(0))
        else:
            xv = float(x)
    except Exception:
        return ""

    u = (unit or "").strip().lower()
    factor = _TIME_UNIT_TO_H.get(u)
    if factor is None:
        return ""
    return str(round(xv * factor, 6))

def normalize_property_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    # add/adjust for your ontology
    if "conduct" in s or ("oh" in s or "hydrox" in s or "ion"):
        return "anion conductivity"
    if "water uptake" in s or s in {"wu", "wateruptake"}:
        return "water uptake"
    if "swelling" in s:
        return "swelling"
    if "iec" in s:
        return "IEC"
    return name.strip()

def parse_temp_C_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(\d{1,3})\s*(?:°\s*C|°C|deg\s*C|C)\b", text)
    if m:
        return m.group(1)
    return None

def parse_rh_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(\d{1,3})\s*%\s*(?:rh|RH|relative humidity)", text)
    if m:
        return m.group(1)
    return None

# ----------------------------
# Payload builders
# ----------------------------
def build_figures_plan_payload(
    doi: str, 
    df_points: pd.DataFrame, 
    figure_snips: List[Dict[str,Any]], 
    max_series: int = 30,
    full_text: Optional[str] = None
    ) -> Dict[str, Any]:
    df = df_points.copy()
    # summarize axis names/units
    x_summary = df[["x_name","x_unit"]].drop_duplicates().head(10).to_dict("records")
    y_summary = df[["y_name","y_unit"]].drop_duplicates().head(10).to_dict("records")

    # sample series labels + a couple points each
    series_samples = []
    for s, g in df.groupby("series_label", dropna=False):
        g2 = g.head(2)
        series_samples.append({
            "series_label": str(s),
            "x_name": str(g2["x_name"].iloc[0]),
            "x_unit": str(g2["x_unit"].iloc[0]),
            "y_name": str(g2["y_name"].iloc[0]),
            "y_unit": str(g2["y_unit"].iloc[0]),
            "points": g2[["x_value","y_value"]].to_dict("records")
        })
        if len(series_samples) >= max_series:
            break

    
    payload = {
        "doi": doi,
        "x_axis_pairs": x_summary,
        "y_axis_pairs": y_summary,
        "series_samples": series_samples,
        "figure_snippets": figure_snips[:8],
    }

    # Add full text context (bounded)
    if full_text:
        payload["full_text"] = full_text

    return payload

def build_table_plan_payload(
    doi: str,
    table_name: str,
    table_columns: List[str],
    table_head: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    figures: List[Dict[str, Any]],
    full_text: Any,
    ) -> Dict[str, Any]:
        cap = (metadata or {}).get("caption") or (metadata or {}).get("table_caption") or ""

        return {
            "doi": doi,
            "table_name": table_name,
            "caption": cap[:1200],
            "columns": table_columns,
            "head_rows": table_head,
            "full_text": full_text,
        }

# ----------------------------
# LLM inference (returns (resp, plan))
# ----------------------------
def _parse_json_obj(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        # best-effort extract
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1 and b > a:
            try:
                return json.loads(s[a:b+1])
            except Exception:
                return {}
        return {}

def infer_figures_plan(client, payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], Any]:

    instructions = """
        You infer a transformation plan for an extracted FIGURE time-series (or other x-axis) dataset.

        Return ONLY valid JSON (no markdown). Required keys:

        1) scope: {doi, source_kind:"figure"}

        2) x_semantic: what the figure varies over.
        One of: ["time","temperature","RH","solvent_molarity","solvent_type","IEC","water_uptake","swelling","current_density","potential","other"]

        3) x_source: how to interpret x_value/x_name/x_unit.
        For figures, x is always from the extracted x columns, so use:
            {"from_x_axis": true}
        (You may still set {"empty":true} if x is unusable, but usually from_x_axis is correct.)

        4) x_unit: canonical unit string for x when meaningful:
        "h" | "C" | "%" | "M" | "V" | "mA/cm2" | "A/cm2" | "".
        (If x_semantic="time", set x_unit="h" and set units.time_to_h=true.)

        5) y_semantic_map (optional):
        A mapping from observed y_name patterns to canonical property names.
        Example:
        {
        "anion conductivity": ["oh conductivity","hydroxide conductivity","conductivity"],
        "water uptake": ["WU","water uptake"],
        "swelling": ["swelling","SR"]
        }

        6) constants: values that should apply to ALL points unless overridden.
        Each field uses {"const":"<value>"} or {"empty":true}.
        Fields (same as final dataset columns):
        "relative humidity(%)",
        "solvent type",
        "solvent molarity (M)",
        "degradation test temp (C)",
        "prop test temp (C)"

        7) derive_from_series_label (optional list of rules):
        Each rule: {"match":"<regex>", "set": {<field>: "<value or $1,$2 refs>"}}
        Use this to pull RH/temp/molarity/etc from series_label strings.
        Examples:
        - match "(?i)(\\d{1,3})\\s*%\\s*RH" -> set {"relative humidity(%)":"$1"}
        - match "(?i)(\\d{2,3})\\s*°?C" -> set {"prop test temp (C)":"$1"}
        - match "(?i)(\\d+(?:\\.\\d+)?)\\s*M\\s*(KOH|NaOH)" -> set {"solvent molarity (M)":"$1","solvent type":"$2"}

        8) units: {time_to_h: true/false}
        - If x_semantic == "time", set time_to_h true.
        - Otherwise set false.

        9) confidence: 0..1
        10) evidence: brief explanation of how you decided x_semantic and constants.

        Rules:
        - Do NOT output per-point rows.
        - Prefer axis label/unit cues + series_label patterns + figure captions + full_text.
        - Do NOT fabricate conditions not supported by captions/full_text/labels.
        - If unsure, set x_semantic="other", constants empty, and keep confidence low.
        """.strip()

    parts = []

    parts.append(f"""
    Based on the extracted figure x and y axis pairs, figure title & caption, and (if available) full paper text, infer a transformation plan for the extracted figure data.

    Extracted Figure Data X Axis Pairs:
    {json.dumps(payload.get("x_axis_pairs", []), indent=2)}

    Extracted Figure Data Y Axis Pairs:
    {json.dumps(payload.get("y_axis_pairs", []), indent=2)}

    Extracted Figure Title & Caption:
    {json.dumps(payload.get("figure_snippets", []), indent=2)}
    """)

    full_text = payload.get("full_text")
    if full_text:
        parts.append(f"""
    Full Paper Text:
    {json.dumps(full_text, indent=2)}
    """)

    content = "\n".join(parts)

    print("===FIGURE CONTENT===", content)


    text, meta = client.chat(
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": content}],
                
        #         [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        #     }
        # ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=5000,
        return_meta=True,
    )
    plan = _parse_json_obj(text)
    return text, plan, meta

def infer_tables_plan(client, payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    # instructions = """
    # You infer a TRANSFORMATION PLAN for an extracted TABLE.

    # Return ONLY valid JSON (no markdown). Required top-level keys:
    # - scope: {doi, table_name, source_kind:"table"}
    # - column_map: mapping to the normalized schema fields:
    #     "Sample Name": {"col": "<column name>"} OR {"const": "<value>"} OR {"empty": true}
    #     "time(h)": {"col": "<column name>", "unit_from": "<unit>", "unit_to": "h"} OR {"const": ""} OR {"empty": true}
    #     "property": {"col": "<column name>"} OR {"const": "<property name>"}
    #     "value": {"col": "<column name>"} OR {"empty": true}
    #     optional constants for:
    #     "relative humidity(%)","solvent type","solvent molarity (M)","degradation test temp (C)","prop test temp (C)"
    # - row_rules: optional instructions (e.g., if table is wide-format, how to unpivot)
    # - If the table is wide-format, you MUST use:
    #     - row_rules.unpivot = {id_cols: [...], value_cols: [...], var_name: "property", value_name: "value"}
    # - Do NOT return lists under keys named "property" or "value".
    # - confidence: 0..1
    # - evidence: brief explanation

    # Rules:
    # - Prefer direct column mappings using provided columns.
    # - If table is wide-format (multiple conditions across columns), specify row_rules.unpivot.
    # - Do NOT fabricate values not supported by the table/caption.
    # - 
    # """.strip()
    
    
    instructions = """
        You infer a transformation plan for an extracted table.

        Return ONLY valid JSON (no markdown). Required keys:

        1) scope: {doi, table_name, source_kind:"table"}

        2) x_semantic: what the table varies over.
        One of: ["time","temperature","RH","solvent_molarity","solvent_type","IEC","other"]

        3) - x_source: {"col":"<column>"} OR {"from_header": true} OR {"const":"<value>"} OR {"empty": true}
        
        4) - x_unit: "h" | "C" | "%" | "M" | ""

        5) column_map: mapping to normalized output fields. For each field:
        {"col":"<column>"} OR {"const":"<value>"} OR {"empty":true}
        Fields:
        "Sample Name","time(h)","relative humidity(%)","solvent type","solvent molarity (M)",
        "degradation test temp (C)","prop test temp (C)","property","value"

        6) row_rules (optional):
        - If wide-format, include:
            row_rules.unpivot = {id_cols:[...], value_cols:[...], var_name:"property", value_name:"value"}

        7) derive_from_property_label (optional list of rules):
        Each rule: {"match":"<regex>", "set": {<field>: "<value or $1,$2 group refs>"}}
        Use this to set time/temp/RH/solvent fields from header strings like "Before", "After", "80% RH", "1 M KOH".
        Example: match "(?i)before" -> set {"time(h)":"0"}.

        8) confidence: 0..1
        9) evidence: brief explanation

        Rules:
        - Prefer direct column mappings when possible.
        - If the independent variable is temperature/RH/molarity (not time), then time(h) should be empty unless explicitly derivable.
        - Do NOT fabricate values not supported by table/caption.
        """.strip()

    parts = []

    parts.append(f"""
    Based on the extracted table caption, columns, head rows, and (if available) full paper text, infer a transformation plan for the extracted table data:

    Extracted Table Caption:
    {json.dumps(payload.get("caption", ""), indent=2)}

    Extracted Table Columns:
    {json.dumps(payload.get("columns", []), indent=2)}

    Extracted Table Head Rows:
    {json.dumps(payload.get("head_rows", []), indent=2)}
    """)

    full_text = payload.get("full_text")
    if full_text:
        parts.append(f"""
    Full Paper Text:
    {json.dumps(full_text, indent=2)}
    """)

    content = "\n".join(parts)
    
    print("===TABLE CONTENT===", content)

    text, meta = client.chat(
        # messages=[
        #     {
        #         "role": "system",
        #         "content": instructions
        #     },
        #     {
        #         "role": "user",
        #         "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        #     }
        # ],
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": content}],

        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=5000,
        return_meta=True
    )
    plan = _parse_json_obj(text)
    return text, plan, meta

# ----------------------------
# Deterministic application
# ----------------------------
_OUTPUT_COLUMNS = [
    "Sample Name",
    "time(h)",
    "x_name",
    "x_unit",
    "x_value",
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
OUTPUT_COLUMNS = _OUTPUT_COLUMNS 


def apply_figure_plan(df_points: pd.DataFrame, plan: Dict[str, Any], doi: str) -> pd.DataFrame:
    df = df_points.copy()

    if not isinstance(plan, dict):
        plan = {}

    # --- semantics ---
    x_sem = (plan.get("x_semantic")
             or ((plan.get("global") or {}).get("default_x_semantic"))
             or "other")
    x_sem = x_sem or "other"

    # y ontology: new key or old key
    ont = (plan.get("y_semantic_map")
           or ((plan.get("global") or {}).get("property_ontology"))
           or {})
    if not isinstance(ont, dict):
        ont = {}

    # constants + derive rules (new)
    constants = plan.get("constants", {}) or {}
    if not isinstance(constants, dict):
        constants = {}
    derive_rules = plan.get("derive_from_series_label", []) or []
    if not isinstance(derive_rules, list):
        derive_rules = []

    ENV_FIELDS = [
        "relative humidity(%)",
        "solvent type",
        "solvent molarity (M)",
        "degradation test temp (C)",
        "prop test temp (C)",
    ]

    def normalize_prop(y_name: str) -> str:
        yn = (y_name or "").strip().lower()
        for canon, syns in ont.items():
            for s in (syns or []):
                if (s or "").strip().lower() in yn:
                    return canon
        return (y_name or "").strip()

    def _apply_derive_rules(label: str, rules: Any, row_out: Dict[str, str]) -> None:
        if not isinstance(rules, list) or not label:
            return
        supported = set(_OUTPUT_COLUMNS)
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            pat = rule.get("match")
            sets = rule.get("set")
            if not pat or not isinstance(sets, dict):
                continue
            try:
                m = re.search(pat, label)
            except re.error:
                continue
            if not m:
                continue

            last = m.lastindex or 0
            for k, templ in sets.items():
                if k not in supported or templ is None:
                    continue
                v = str(templ)
                for gi in range(1, 10):
                    v = v.replace(f"${gi}", m.group(gi) if gi <= last else "")
                row_out[k] = v

    def _apply_global_constants(row_out: Dict[str, str]) -> None:
        """
        Apply plan["constants"] in table-style:
          constants[field] can be {"const": "..."} or {"empty": true}
        """
        for k in ENV_FIELDS:
            spec = constants.get(k, {})
            if not isinstance(spec, dict):
                continue
            if spec.get("empty") is True:
                # explicitly empty means force blank
                row_out[k] = ""
            elif "const" in spec and spec["const"] is not None:
                row_out[k] = str(spec["const"])

    def _deterministic_fallback_parse(series_label: str, row_out: Dict[str, str]) -> None:
        """
        Final guardrail: parse RH/temp from label if still missing.
        """
        sl = series_label or ""
        if not row_out.get("relative humidity(%)"):
            rh = parse_rh_from_text(sl)
            if rh:
                row_out["relative humidity(%)"] = rh
        if not row_out.get("prop test temp (C)"):
            tc = parse_temp_C_from_text(sl)
            if tc:
                row_out["prop test temp (C)"] = tc

    rows = []
    for _, r in df.iterrows():
        row = {k: "" for k in _OUTPUT_COLUMNS}
        row["DOI"] = doi
        row["source_kind"] = "figure"
        row["source_name"] = ""  # you can set to dataset_name upstream if you have it

        # provenance / identity
        row["Sample Name"] = str(r.get("series_label", "") or "")
        row["x_name"] = str(r.get("x_name", "") or "")
        row["x_unit"] = str(r.get("x_unit", "") or "")
        xv = r.get("x_value", "")
        row["x_value"] = "" if pd.isna(xv) else str(xv)

        # property/value
        row["property"] = normalize_property_name(
            normalize_prop(str(r.get("y_name", "") or ""))
        )
        yv = r.get("y_value", "")
        row["value"] = "" if pd.isna(yv) or yv == "" else str(yv)

        # apply global constants (table-style)
        _apply_global_constants(row)

        # apply derive rules from series label (table-style)
        _apply_derive_rules(row["Sample Name"], derive_rules, row)

        # deterministic fallback parse (final guardrail)
        _deterministic_fallback_parse(row["Sample Name"], row)

        # time only if x_semantic says time AND unit convertible
        if x_sem == "time":
            row["time(h)"] = to_hours(r.get("x_value", ""), str(r.get("x_unit", "")))
        else:
            row["time(h)"] = ""

        rows.append(row)

    return pd.DataFrame(rows)

def _get_col(df: pd.DataFrame, colname: str) -> Optional[pd.Series]:
    if not colname:
        return None
    # direct match
    if colname in df.columns:
        return df[colname]
    # case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    if colname.lower() in lower_map:
        return df[lower_map[colname.lower()]]
    return None

def _norm_col(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # normalize common unicode variants
    s = s.replace("\u2212", "-")   # unicode minus
    s = s.replace("\u2013", "-")   # en-dash
    s = s.replace("\u2014", "-")   # em-dash
    s = s.replace("\u00b9", "1").replace("\u2071", "1")  # ¹ etc (optional)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _col_lookup(df: pd.DataFrame) -> Dict[str, str]:
    """
    Returns mapping: normalized_col -> actual_col (first occurrence wins).
    """
    m = {}
    for c in df.columns:
        k = _norm_col(c)
        if k and k not in m:
            m[k] = c
    return m

def _resolve_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """
    Map requested column names to actual df columns using normalized matching.
    Keeps only those that resolve.
    """
    lut = _col_lookup(df)
    out = []
    for c in cols or []:
        key = _norm_col(c)
        if key in lut:
            out.append(lut[key])
        else:
            # fallback: substring match
            for nk, real in lut.items():
                if key and (key in nk or nk in key):
                    out.append(real)
                    break
    # de-dup preserving order
    seen = set()
    dedup = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup

def apply_table_plan_to_df(df_table: pd.DataFrame, plan: Dict[str, Any], doi: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    ALLOWED_X_SEMANTICS = {
        "time", "temperature", "RH", "solvent_molarity", "solvent_type", "IEC", "other"
    }

    def _apply_derive_rules(prop_label: str, rules: Any, row_out: Dict[str, str]) -> None:
        """
        Apply derive_from_property_label rules to row_out based on prop_label.
        Supports $1, $2, ... backrefs.
        Ignores unknown fields.
        """
        if not isinstance(rules, list) or not prop_label:
            return

        supported = set(_OUTPUT_COLUMNS)
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            pat = rule.get("match")
            sets = rule.get("set")
            if not pat or not isinstance(sets, dict):
                continue
            try:
                m = re.search(pat, prop_label)
            except re.error:
                continue
            if not m:
                continue

            for k, templ in sets.items():
                if k not in supported:
                    continue
                if templ is None:
                    continue
                v = str(templ)
                # $1..$9 substitutions
                last = m.lastindex or 0
                for gi in range(1, 10):
                    v = v.replace(f"${gi}", m.group(gi) if gi <= last else "")
                row_out[k] = v

    def _x_value_from_source(r: pd.Series, x_source: Any) -> str:
        if not isinstance(x_source, dict):
            return ""
        if x_source.get("empty") is True:
            return ""
        if "const" in x_source:
            return "" if x_source["const"] is None else str(x_source["const"])
        if "col" in x_source:
            v = get_cell(r, x_source["col"])
            return "" if v is None else str(v)
        # from_header handled via derive rules (no direct value here)
        return ""

    ENV_FIELDS = [
        "relative humidity(%)",
        "solvent type",
        "solvent molarity (M)",
        "degradation test temp (C)",
        "prop test temp (C)",
    ]

    if df_table is None or df_table.empty:
        return out

    col_map = (plan or {}).get("column_map", {}) or {}

    derive_rules = (plan or {}).get("derive_from_property_label", []) or []
    x_sem = (plan or {}).get("x_semantic", "other") or "other"
    x_source = (plan or {}).get("x_source", {}) or {}
    x_unit = (plan or {}).get("x_unit", "") or ""

    # Optional: wide-format unpivot
    row_rules = (plan or {}).get("row_rules", {}) or {}
    if isinstance(row_rules, dict) and row_rules.get("unpivot"):
        up = row_rules["unpivot"]

        # If GPT returned: {"property":[...], "value":[...]} treat that as value_cols list
        if isinstance(up, dict) and "id_cols" not in up and "value_cols" not in up:
            # common wrong-but-reasonable format
            if isinstance(up.get("value"), list) and up.get("value"):
                up = {
                    "id_cols": ["Membrane"],          # best guess; will validate below
                    "value_cols": up["value"],
                    "var_name": "property",
                    "value_name": "value",
                }

        # ---- Standard melt expected format ----
        try:
            id_cols = up.get("id_cols", [])
            value_cols = up.get("value_cols", [])
            var_name = up.get("var_name", "property")
            value_name = up.get("value_name", "value")

            # Resolve id/value cols robustly (unicode/spacing/case)
            id_cols = _resolve_cols(df_table, id_cols)
            value_cols = _resolve_cols(df_table, value_cols)

            # Auto-detect id col if still missing
            if not id_cols:
                cand = [c for c in df_table.columns if df_table[c].dtype == object]
                if cand:
                    id_cols = [cand[0]]

            if id_cols and value_cols:
                df_table = df_table.melt(
                    id_vars=id_cols,
                    value_vars=value_cols,
                    var_name=var_name,
                    value_name=value_name
                )

                # If we successfully melted, the correct sources are the melt outputs
                # After melt, the only correct sources are the melt output columns.
                if var_name in df_table.columns:
                    col_map.setdefault("property", {})
                    if isinstance(col_map["property"], dict):
                        col_map["property"]["col"] = var_name

                if value_name in df_table.columns:
                    col_map.setdefault("value", {})
                    if isinstance(col_map["value"], dict):
                        col_map["value"]["col"] = value_name
        except Exception:
            pass


    # Build each output row deterministically
    for _, r in df_table.iterrows():
        row = {k: "" for k in _OUTPUT_COLUMNS}
        row["DOI"] = doi
        row["source_kind"] = "table"

        # Helper to resolve mapping entry
        def get_cell(row: pd.Series, col) -> Any:
            if col is None or col == "":
                return ""

            # If plan provides multiple candidate columns, try them in order.
            if isinstance(col, list):
                for c in col:
                    v = get_cell(row, c)
                    if v is None:
                        continue
                    if str(v).strip() != "":
                        return v
                return ""

            # Allow integer positional indexing
            if isinstance(col, int):
                try:
                    return row.iloc[col]
                except Exception:
                    return ""

            # Normal string lookup + case-insensitive fallback
            col = str(col)
            if col in row.index:
                return row[col]

            lm = {str(c).lower(): c for c in row.index}
            key = col.lower()
            if key in lm:
                return row[lm[key]]

            return ""

        def resolve(field: str) -> str:
            spec = col_map.get(field, {})
            if not isinstance(spec, dict):
                return ""
            if spec.get("empty") is True:
                return ""
            if "const" in spec:
                return "" if spec["const"] is None else str(spec["const"])
            if "col" in spec:
                s = get_cell(r, spec["col"])
                return "" if s is None else str(s)
            return ""

        # apply column_map const/col for env fields
        for k in ENV_FIELDS:
            if not row[k]:
                row[k] = resolve(k) or ""

        # sample
        row["Sample Name"] = resolve("Sample Name")

        # raw property/val
        raw_prop = resolve("property")
        raw_value = resolve("value")

        # 5) Apply derive_from_property_label rules using the *raw* property label
        # (this is where Before/After -> time, RH, molarity patterns, etc. happen)
        _apply_derive_rules(raw_prop, derive_rules, row)

        # Normalize property AFTER derivation
        row["property"] = normalize_property_name(raw_prop)

        # value
        if raw_value == "" or raw_value is None:
            row["value"] = ""
        else:
            try:
                row["value"] = str(float(raw_value))
            except Exception:
                row["value"] = str(raw_value)

        # time(h) with conversion if specified
        t_spec = col_map.get("time(h)", {})
        if not row.get("time(h)"):  # only fill if derive didn't set it
            if isinstance(t_spec, dict) and "col" in t_spec:
                raw_t = get_cell(r,t_spec["col"])
                unit_from = (t_spec.get("unit_from") or "").strip().lower() or "h"
                row["time(h)"] = to_hours(raw_t, unit_from)
            else:
                if not row.get("time"):
                    row["time(h)"] = resolve("time(h)")

        # x_semantic/x_source fallback for tables where x is time/temp/RH/molarity
        if not row.get("time(h)"):
            xv = _x_value_from_source(r, x_source)

            if x_sem == "time":
                row["time(h)"] = to_hours(xv, x_unit or "h")
            elif x_sem == "temperature" and not row.get("prop test temp (C)"):
                row["prop test temp (C)"] = xv
            elif x_sem == "RH" and not row.get("relative humidity(%)"):
                row["relative humidity(%)"] = xv
            elif x_sem == "solvent_molarity" and not row.get("solvent molarity (M)"):
                row["solvent molarity (M)"] = xv
            elif x_sem == "solvent_type" and not row.get("solvent type"):
                row["solvent type"] = xv

        # x_* are not meaningful for tables; leave blank
        row["x_name"] = ""
        row["x_unit"] = ""
        row["x_value"] = ""

        # If table didn't provide Sample Name but row has obvious id columns, you can keep blank here;
        # better to improve the LLM plan rather than guessing.
        out.append(row)

    # Drop rows that are completely empty of signal (no property/value/sample)
    cleaned = []
    for r in out:
        if (r.get("value","") == "" and r.get("property","") == "" and r.get("Sample Name","") == ""):
            continue
        cleaned.append(r)

    return cleaned
