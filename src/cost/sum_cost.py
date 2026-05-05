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

def main(path: str):
    totals = defaultdict(lambda: {"input": 0, "cached": 0, "output": 0, "n": 0})

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = canonical_model(rec.get("model", ""))
            totals[m]["input"] += int(rec.get("input_tokens", 0) or 0)
            totals[m]["cached"] += int(rec.get("cached_tokens", 0) or 0)
            totals[m]["output"] += int(rec.get("output_tokens", 0) or 0)
            totals[m]["n"] += 1

    grand_total = 0.0
    print("\nPer-model totals:")
    for m, t in totals.items():
        prices = PRICES_PER_1M.get(m)
        if not prices:
            print(f"- {m}: {t}  (NO PRICES CONFIGURED)")
            continue

        non_cached_in = max(0, t["input"] - t["cached"])
        cost = (non_cached_in / 1e6) * prices["in"] + (t["cached"] / 1e6) * prices.get("cached_in", prices["in"]) + (t["output"] / 1e6) * prices["out"]
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
            "grand_total": float,
        }
    """
    from pathlib import Path as _P
    if not _P(path).exists():
        return {"models": {}, "grand_total": 0.0}

    totals = defaultdict(lambda: {"input": 0, "cached": 0, "output": 0, "n": 0})

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = canonical_model(rec.get("model", ""))
            totals[m]["input"] += int(rec.get("input_tokens", 0) or 0)
            totals[m]["cached"] += int(rec.get("cached_tokens", 0) or 0)
            totals[m]["output"] += int(rec.get("output_tokens", 0) or 0)
            totals[m]["n"] += 1

    grand_total = 0.0
    models_out = {}
    for m, t in totals.items():
        prices = PRICES_PER_1M.get(m)
        cost = 0.0
        if prices:
            non_cached_in = max(0, t["input"] - t["cached"])
            cost = (
                (non_cached_in / 1e6) * prices["in"]
                + (t["cached"] / 1e6) * prices.get("cached_in", prices["in"])
                + (t["output"] / 1e6) * prices["out"]
            )
        grand_total += cost
        models_out[m] = {
            "calls": t["n"],
            "input": t["input"],
            "cached": t["cached"],
            "output": t["output"],
            "cost": cost,
        }

    return {"models": models_out, "grand_total": grand_total}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sum_cost.py /path/to/usage.jsonl")
        sys.exit(1)
    main(sys.argv[1])
