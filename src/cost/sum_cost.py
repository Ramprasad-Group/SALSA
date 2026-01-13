#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

# Update if you use other models
PRICES_PER_1M = {
    # standard pricing (not priority tier)
    "gpt-4o": {"in": 2.50, "out": 10.00, "cached_in": 2.50},  # set cached_in if you enable caching pricing later
}

def canonical_model(name: str) -> str:
    # Your logs show "gpt-4o-2024-08-06" → treat as "gpt-4o"
    if not name:
        return "unknown"
    if name.startswith("gpt-4o"):
        return "gpt-4o"
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sum_cost.py /path/to/usage.jsonl")
        sys.exit(1)
    main(sys.argv[1])
