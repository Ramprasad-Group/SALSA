#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

# Update if you use other models
PRICES_PER_1M = {
    # standard pricing (not priority tier)
    "gpt-4o": {"in": 2.50, "out": 10.00, "cached_in": 2.50},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60, "cached_in": 0.075},
    "gpt-4.1-mini": {"in": 0.4, "out":1.60, "cached_in":0.10},
    "gpt-4.1-nano": {"in": 0.10, "out":0.4, "cached_in":0.025},
    "gpt-4.1": {"in": 2.00, "out":8.00, "cached_in":0.5},
    "gpt-5-nano": {"in": 0.05, "out":0.40, "cached_in":0.005},
    "gpt-5-mini": {"in": 0.25, "out":2.00, "cached_in":0.025},
    "gpt-5": {"in": 1.25, "out":10.00, "cached_in":0.125},
    "gpt-5.1": {"in": 1.25, "out":10.00, "cached_in":0.125},
    "gpt-5.2": {"in": 1.75, "out":14.00, "cached_in":0.175},
    # No cached-input discount listed for pro; cached tokens are priced as input.
    "gpt-5.2-pro": {"in": 21.00, "out":168.00},
    "gpt-5.4": {"in": 2.50, "out":15.00, "cached_in":0.25},
    "gpt-5.4-mini": {"in": 0.75, "out":4.50, "cached_in":0.075},
    "gpt-5.4-nano": {"in": 0.20, "out":1.25, "cached_in":0.02},
    "gpt-5.4-pro": {"in": 30.00, "out":180.00},
    "gpt-5.5": {"in": 5.00, "out":30.00, "cached_in":0.50},
    "gpt-5.5-pro": {"in": 30.00, "out":180.00},
}

STAGE_LABELS_BY_KIND = {
    "table_xml": "Table recreation",
    "table_html": "Table recreation",
    "filter_figure": "Figure filtering",
    "filter_table": "Table filtering",
    "direct_llm_figure_data": "Figure data extraction",
    "chemical_composition_relevant_content": "Chemical composition: find relevant content",
    "chemical_composition_table": "Chemical composition: extract composition",
    "emit_rows_figure": "Complete dataset from figures",
    "emit_rows_table": "Complete dataset from tables",
}

def canonical_model(name: str) -> str:
    # Your logs show "gpt-4o-2024-08-06" → treat as "gpt-4o"
    if not name:
        return "unknown"
    elif name.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    elif name.startswith("gpt-4o"):
        return "gpt-4o"
    elif name.startswith("gpt-4.1-mini"):
        return "gpt-4.1-mini"
    elif name.startswith("gpt-4.1-nano"):
        return "gpt-4.1-nano"
    elif name.startswith("gpt-4.1"):
        return "gpt-4.1"
    elif name.startswith("gpt-5-mini"):
        return "gpt-5-mini"
    elif name.startswith("gpt-5-nano"):
        return "gpt-5-nano"
    elif name.startswith("gpt-5.5-pro"):
        return "gpt-5.5-pro"
    elif name.startswith("gpt-5.5"):
        return "gpt-5.5"
    elif name.startswith("gpt-5.4-mini"):
        return "gpt-5.4-mini"
    elif name.startswith("gpt-5.4-nano"):
        return "gpt-5.4-nano"
    elif name.startswith("gpt-5.4-pro"):
        return "gpt-5.4-pro"
    elif name.startswith("gpt-5.4"):
        return "gpt-5.4"
    elif name.startswith("gpt-5.2-pro"):
        return "gpt-5.2-pro"
    elif name.startswith("gpt-5.2"):
        return "gpt-5.2"
    elif name.startswith("gpt-5.1"):
        return "gpt-5.1"
    elif name.startswith("gpt-5"):
        return "gpt-5"
    else:
        return "unknown"
    return name

def stage_label_for_record(rec: dict) -> str:
    extra = rec.get("extra") or {}
    if not isinstance(extra, dict):
        return "Other LLM usage"

    kind = str(extra.get("kind") or "").strip()
    if kind in STAGE_LABELS_BY_KIND:
        return STAGE_LABELS_BY_KIND[kind]
    if kind:
        return kind.replace("_", " ").title()
    return "Other LLM usage"

def _blank_totals() -> dict:
    return {"input": 0, "cached": 0, "output": 0, "n": 0}

def _add_record_totals(totals: dict, rec: dict) -> None:
    totals["input"] += int(rec.get("input_tokens", 0) or 0)
    totals["cached"] += int(rec.get("cached_tokens", 0) or 0)
    totals["output"] += int(rec.get("output_tokens", 0) or 0)
    totals["n"] += 1

def _cost_for_totals(model: str, totals: dict) -> float:
    prices = PRICES_PER_1M.get(model)
    if not prices:
        return 0.0
    non_cached_in = max(0, totals["input"] - totals["cached"])
    return (
        (non_cached_in / 1e6) * prices["in"]
        + (totals["cached"] / 1e6) * prices.get("cached_in", prices["in"])
        + (totals["output"] / 1e6) * prices["out"]
    )

def _public_totals(totals: dict, cost: float) -> dict:
    return {
        "calls": totals["n"],
        "input": totals["input"],
        "cached": totals["cached"],
        "output": totals["output"],
        "cost": cost,
    }

def main(path: str):
    totals = defaultdict(_blank_totals)
    stage_model_totals = defaultdict(lambda: defaultdict(_blank_totals))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = canonical_model(rec.get("model", ""))
            _add_record_totals(totals[m], rec)
            _add_record_totals(stage_model_totals[stage_label_for_record(rec)][m], rec)

    grand_total = 0.0
    print("\nPer-stage totals:")
    for stage, models in stage_model_totals.items():
        stage_calls = stage_input = stage_cached = stage_output = 0
        stage_cost = 0.0
        model_parts = []
        for m, t in models.items():
            model_cost = _cost_for_totals(m, t)
            stage_calls += t["n"]
            stage_input += t["input"]
            stage_cached += t["cached"]
            stage_output += t["output"]
            stage_cost += model_cost
            model_parts.append(f"{m}: calls={t['n']}, cost=${model_cost:.6f}")
        model_summary = "; ".join(model_parts)
        print(
            f"- {stage}: calls={stage_calls}, input={stage_input}, "
            f"cached={stage_cached}, output={stage_output}, cost=${stage_cost:.6f}"
            f" ({model_summary})"
        )

    print("\nPer-model totals:")
    for m, t in totals.items():
        prices = PRICES_PER_1M.get(m)
        if not prices:
            print(f"- {m}: {t}  (NO PRICES CONFIGURED)")
            continue

        cost = _cost_for_totals(m, t)
        grand_total += cost

        print(f"- {m}: calls={t['n']}, input={t['input']}, cached={t['cached']}, output={t['output']}, cost=${cost:.6f}")

    print(f"\nGrand total: ${grand_total:.6f}\n")

def summarise_usage(path: str) -> dict:
    """
    Return a dict with per-model breakdown and grand total cost.

    Returns:
        {
            "models": {
                "<model>": {"calls": N, "input": N, "cached": N, "output": N, "cost": float},
                ...
            },
            "stages": {
                "<stage>": {"calls": N, "input": N, "cached": N, "output": N, "cost": float},
                ...
            },
            "grand_total": float,
        }
    """
    from pathlib import Path as _P
    if not _P(path).exists():
        return {"models": {}, "stages": {}, "grand_total": 0.0}

    totals = defaultdict(_blank_totals)
    stage_model_totals = defaultdict(lambda: defaultdict(_blank_totals))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = canonical_model(rec.get("model", ""))
            _add_record_totals(totals[m], rec)
            _add_record_totals(stage_model_totals[stage_label_for_record(rec)][m], rec)

    grand_total = 0.0
    models_out = {}
    for m, t in totals.items():
        cost = _cost_for_totals(m, t)
        grand_total += cost
        models_out[m] = _public_totals(t, cost)

    stages_out = {}
    for stage, models in stage_model_totals.items():
        stage_totals = _blank_totals()
        stage_cost = 0.0
        stage_models_out = {}
        for m, t in models.items():
            model_cost = _cost_for_totals(m, t)
            stage_totals["n"] += t["n"]
            stage_totals["input"] += t["input"]
            stage_totals["cached"] += t["cached"]
            stage_totals["output"] += t["output"]
            stage_cost += model_cost
            stage_models_out[m] = _public_totals(t, model_cost)
        stages_out[stage] = _public_totals(stage_totals, stage_cost)
        stages_out[stage]["models"] = stage_models_out

    return {"models": models_out, "stages": stages_out, "grand_total": grand_total}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sum_cost.py /path/to/usage.jsonl")
        sys.exit(1)
    main(sys.argv[1])
