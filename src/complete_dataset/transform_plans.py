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
    if "conduct" in s and ("oh" in s or "hydrox" in s):
        return "hydroxide conductivity"
    if ("ionic" in s and "conduct" in s) or ("conductivity" in s and "ion" in s):
        return "ionic conductivity"
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
def build_figures_global_plan_payload(
    doi: str, 
    df_points: pd.DataFrame, 
    figure_snips: List[Dict[str,Any]], 
    max_series: int = 30
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

    return {
        "doi": doi,
        "x_axis_pairs": x_summary,
        "y_axis_pairs": y_summary,
        "series_samples": series_samples,
        "figure_snippets": figure_snips[:8],
    }

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
        fig_snips = []
        for f in figures[:6]:
            md = f.get("metadata") or {}
            fig_cap = md.get("caption") or md.get("figure_caption") or ""
            fig_snips.append({"figure_name": f.get("figure_name", ""), "caption": fig_cap[:600]})

        ft_excerpt = ""
        if isinstance(full_text, dict):
            ft_excerpt = json.dumps(full_text, ensure_ascii=False)[:2000]
        elif isinstance(full_text, str):
            ft_excerpt = full_text[:2000]

        return {
            "doi": doi,
            "table_name": table_name,
            "caption": cap[:1200],
            "columns": table_columns,
            "head_rows": table_head,
            "figure_snippets": fig_snips,
            "full_text_excerpt": ft_excerpt,
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

def infer_figures_plan(client, payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:

    instructions = """
    Infer ONE global transformation plan for ALL extracted FIGURE datapoints in this DOI.

    Return ONLY valid JSON with keys:
    - scope: {doi, source_kind:"figure"}
    - global: {
        default_x_semantic: one of ["time","temperature","RH","IEC","water_uptake","swelling","current_density","potential","other"],
        property_ontology: {canonical_property: [synonyms...]},
        units: {time_to_h: true/false}
    }
    - series_overrides: list of objects:
        {match: "<regex applied to series_label>", constants: {<output field>: "<value>"}}
    (use overrides to pull RH/temp/etc from series labels when typical)
    - confidence: {x_semantic:0..1, y_semantic:0..1, context:0..1}
    - evidence: brief reasons

    Rules:
    - Do NOT output per-point rows.
    - Prefer using axis label/unit cues + series_label patterns + captions.
    - If unsure, set default_x_semantic="other".
    """.strip()

    text, meta = client.chat(
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=5000,
        return_meta=True,
    )
    plan = _parse_json_obj(text)
    return text, plan, meta

def infer_tables_plan(client, payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    instructions = """
    You infer a TRANSFORMATION PLAN for an extracted TABLE.

    Return ONLY valid JSON (no markdown). Required top-level keys:
    - scope: {doi, table_name, source_kind:"table"}
    - column_map: mapping to the normalized schema fields:
        "Sample Name": {"col": "<column name>"} OR {"const": "<value>"} OR {"empty": true}
        "time(h)": {"col": "<column name>", "unit_from": "<unit>", "unit_to": "h"} OR {"const": ""} OR {"empty": true}
        "property": {"col": "<column name>"} OR {"const": "<property name>"}
        "value": {"col": "<column name>"} OR {"empty": true}
        optional constants for:
        "relative humidity(%)","solvent type","solvent concentration (%)","degradation test temp (C)","prop test temp (C)"
    - row_rules: optional instructions (e.g., if table is wide-format, how to unpivot)
    - If the table is wide-format, you MUST use:
        - row_rules.unpivot = {id_cols: [...], value_cols: [...], var_name: "property", value_name: "value"}
    - Do NOT return lists under keys named "property" or "value".
    - confidence: 0..1
    - evidence: brief explanation

    Rules:
    - Prefer direct column mappings using provided columns.
    - If table is wide-format (multiple conditions across columns), specify row_rules.unpivot.
    - Do NOT fabricate values not supported by the table/caption.
    """.strip()

    text, meta = client.chat(
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
            }
        ],

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
    "solvent concentration (%)",
    "degradation test temp (C)",
    "prop test temp (C)",
    "property",
    "value",
    "DOI",
    "source_kind",
    "source_name",
]
OUTPUT_COLUMNS = _OUTPUT_COLUMNS 


def apply_global_figure_plan(df_points: pd.DataFrame, plan: Dict[str, Any], doi: str) -> pd.DataFrame:
    df = df_points.copy()

    # default x semantic decision (time vs other)
    default_x_sem = ((plan or {}).get("global", {}) or {}).get("default_x_semantic", "other")

    # property ontology
    ont = ((plan or {}).get("global", {}) or {}).get("property_ontology", {}) or {}

    def normalize_prop(y_name: str) -> str:
        yn = (y_name or "").strip().lower()
        for canon, syns in ont.items():
            for s in syns:
                if (s or "").strip().lower() in yn:
                    return canon
        # fallback: keep original but cleaned
        return (y_name or "").strip()

    # series overrides (regex on series_label)
    if not isinstance(plan, dict):
        plan = {}

    overrides = plan.get("series_overrides", []) or []
    if not isinstance(overrides, list):
        overrides = []

    def constants_for_series(series_label: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        sl = series_label or ""

        for ov in overrides:
            if not isinstance(ov, dict):
                continue
            try:
                if re.match(ov.get("match", "$^"), sl):
                    consts = ov.get("constants", {}) or {}
                    if isinstance(consts, dict):
                        out.update({k: str(v) for k, v in consts.items() if v is not None})
            except Exception:
                continue

        # deterministic parsing fallback
        if "relative humidity(%)" not in out:
            rh = parse_rh_from_text(sl)
            if rh:
                out["relative humidity(%)"] = rh
        if "prop test temp (C)" not in out:
            tc = parse_temp_C_from_text(sl)
            if tc:
                out["prop test temp (C)"] = tc

        return out

    rows = []
    for _, r in df.iterrows():
        row = {k: "" for k in _OUTPUT_COLUMNS}
        row["DOI"] = doi
        row["source_kind"] = "figure"
        row["source_name"] = ""  # optionally fill if you can associate

        row["Sample Name"] = str(r.get("series_label",""))
        row["x_name"] = str(r.get("x_name",""))
        row["x_unit"] = str(r.get("x_unit",""))
        row["x_value"] = "" if pd.isna(r.get("x_value","")) else str(r.get("x_value",""))

        # time only if semantic says time AND unit convertible
        if default_x_sem == "time":
            row["time(h)"] = to_hours(r.get("x_value",""), str(r.get("x_unit","")))
        else:
            row["time(h)"] = ""

        row["property"] = normalize_prop(str(r.get("y_name","")))
        yv = r.get("y_value","")
        row["value"] = "" if pd.isna(yv) or yv == "" else str(yv)

        consts = constants_for_series(row["Sample Name"])
        for k,v in consts.items():
            row[k] = str(v)

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

def apply_table_plan_to_df(df_table: pd.DataFrame, plan: Dict[str, Any], doi: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if df_table is None or df_table.empty:
        return out

    col_map = (plan or {}).get("column_map", {}) or {}
    const = (plan or {}).get("constants", {}) or {}

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

            # Validate columns exist; auto-detect id col if needed
            if not id_cols or any(c not in df_table.columns for c in id_cols):
                # pick a reasonable ID column: first non-numeric-ish column
                cand = [c for c in df_table.columns if df_table[c].dtype == object]
                if cand:
                    id_cols = [cand[0]]

            # keep only value cols that exist
            value_cols = [c for c in value_cols if c in df_table.columns]

            if id_cols and value_cols:
                df_table = df_table.melt(
                    id_vars=id_cols,
                    value_vars=value_cols,
                    var_name=var_name,
                    value_name=value_name
                )

                # After melt, set column_map to use var_name/value_name if it was pointing to property/value
                # (helps when GPT wrote {"col":"property"} / {"col":"value"}.)
                # We'll handle in resolve() by reading from r.get(spec["col"]).
        except Exception:
            pass


    # Build each output row deterministically
    for _, r in df_table.iterrows():
        row = {k: "" for k in _OUTPUT_COLUMNS}
        row["DOI"] = doi

        # constants
        for k in ["relative humidity(%)", "solvent type", "solvent concentration (%)",
                  "degradation test temp (C)", "prop test temp (C)"]:
            if k in const and const[k] is not None:
                row[k] = str(const[k])

        # Helper to resolve mapping entry
        def get_cell(row: pd.Series, col: str) -> Any:
            if not col:
                return ""
            if col in row.index:
                return row[col]
            # case-insensitive match within the row index
            lm = {str(c).lower(): c for c in row.index}
            key = str(col).lower()
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


        # sample
        row["Sample Name"] = resolve("Sample Name")

        # property
        prop = resolve("property")
        row["property"] = normalize_property_name(prop)

        # value
        val = resolve("value")
        if val == "" or val is None:
            row["value"] = ""
        else:
            try:
                row["value"] = str(float(val))
            except Exception:
                row["value"] = str(val)

        # time(h) with conversion if specified
        t_spec = col_map.get("time(h)", {})
        if isinstance(t_spec, dict) and "col" in t_spec:
            raw_t = r.get(t_spec["col"], "")
            unit_from = (t_spec.get("unit_from") or "").strip().lower()
            # treat blank unit_from as "h" only if column name implies hour
            if not unit_from:
                unit_from = "h"
            row["time(h)"] = to_hours(raw_t, unit_from)
        else:
            row["time(h)"] = resolve("time(h)")  # const/empty

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
