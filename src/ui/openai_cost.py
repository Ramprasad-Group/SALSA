# SALSA/src/openai_cost_ledger.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class UsageRecord:
    ts: float
    run_id: str
    doi: Optional[str]
    artifact_id: Optional[str] # figure id / table id / etc
    model: Optional[str]
    service_tier: Optional[str]
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    extra: Dict[str, Any]

def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def extract_usage_fields(resp: Any) -> Dict[str, int]:
    """
    Supports:
      - Responses API: resp.usage is a dict-like with *_tokens_details
      - Chat Completions: resp.usage is an object with prompt_tokens / completion_tokens / total_tokens
    """
    usage = getattr(resp, "usage", None)

    # -------- Case 0: meta dict from LLMClient.chat(return_meta=True) --------
    if isinstance(resp, dict) and "usage" in resp:
        u = resp.get("usage") or {}

        # OpenAI: usage might be an object (prompt_tokens/...)
        if not isinstance(u, dict):
            prompt_tokens = _safe_int(getattr(u, "prompt_tokens", 0))
            completion_tokens = _safe_int(getattr(u, "completion_tokens", 0))
            total_tokens = _safe_int(getattr(u, "total_tokens", prompt_tokens + completion_tokens))
            return dict(
                input_tokens=prompt_tokens,
                cached_tokens=0,
                output_tokens=completion_tokens,
                reasoning_tokens=0,
                total_tokens=total_tokens,
            )

        # Ollama: usage dict already normalized by our wrapper
        input_tokens = _safe_int(u.get("input_tokens", 0))
        output_tokens = _safe_int(u.get("output_tokens", 0))
        total_tokens = _safe_int(u.get("total_tokens", input_tokens + output_tokens))
        return dict(
            input_tokens=input_tokens,
            cached_tokens=_safe_int(u.get("cached_tokens", 0)),
            output_tokens=output_tokens,
            reasoning_tokens=_safe_int(u.get("reasoning_tokens", 0)),
            total_tokens=total_tokens,
        )

    # -------- Case 1: Responses API (dict-like) --------
    if isinstance(usage, dict):
        itd = usage.get("input_tokens_details", {}) or {}
        otd = usage.get("output_tokens_details", {}) or {}

        input_tokens = _safe_int(usage.get("input_tokens", 0))
        output_tokens = _safe_int(usage.get("output_tokens", 0))
        total_tokens = _safe_int(usage.get("total_tokens", input_tokens + output_tokens))

        cached_tokens = _safe_int(itd.get("cached_tokens", 0))
        reasoning_tokens = _safe_int(otd.get("reasoning_tokens", 0))

        return dict(
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
        )

    # -------- Case 2: Chat Completions usage object --------
    # OpenAI ChatCompletions typically reports:
    #   usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    if usage is not None:
        prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
        total_tokens = _safe_int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens))

        # ChatCompletions doesn't expose cached/reasoning breakdown (usually)
        return dict(
            input_tokens=prompt_tokens,
            cached_tokens=0,
            output_tokens=completion_tokens,
            reasoning_tokens=0,
            total_tokens=total_tokens,
        )

    # -------- Fallback --------
    return dict(
        input_tokens=0,
        cached_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        total_tokens=0,
    )

DEFAULT_COST_ROOT = Path(__file__).resolve().parent.parent  # <repo>/src

def append_usage(
    *,
    run_id: str,
    resp: Any,
    doi: Optional[str] = None,
    artifact_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Appends one JSON line to <run_dir>/cost/usage.jsonl
    """

    run_dir = Path(os.getenv("SALSA_RUN_DIR", str(DEFAULT_COST_ROOT)))
    out_dir = run_dir / "cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "usage.jsonl"

    u = extract_usage_fields(resp)
    rec = UsageRecord(
        ts=time.time(),
        run_id=run_id,
        doi=doi,
        artifact_id=artifact_id,
        model=(resp.get("model") if isinstance(resp, dict) else getattr(resp, "model", None)),
        service_tier=(resp.get("service_tier") if isinstance(resp, dict) else getattr(resp, "service_tier", None)),
        input_tokens=u["input_tokens"],
        cached_tokens=u["cached_tokens"],
        output_tokens=u["output_tokens"],
        reasoning_tokens=u["reasoning_tokens"],
        total_tokens=u["total_tokens"],
        extra=extra or {},
    )

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    return out_path


def run_id_from_env(default: str = "unknown") -> str:
    return os.getenv("SALSA_RUN_ID", default)
