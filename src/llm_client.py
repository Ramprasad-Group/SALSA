# SALSA/src/llm_client.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# ---- Config ----

@dataclass(frozen=True)
class LLMConfig:
    provider: str                 # "ollama" | "openai"
    model: str                    # e.g. "llama3:8b" or "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    ollama_host: Optional[str] = None


def load_llm_config() -> LLMConfig:
    provider = os.environ.get("SALSA_LLM_PROVIDER", "ollama").strip().lower()
    model = os.environ.get("SALSA_LLM_MODEL", "").strip()

    if provider in ("openai", "gpt"):
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
        if not model:
            # model = os.environ.get("OPENAI_MODEL", "gpt-4o").strip()
            print("=====NO OPENAI MODEL=====")
            quit()
        return LLMConfig(provider="openai", model=model, base_url=base_url, api_key=api_key)

    # default: ollama
    # accept OLLAMA_HOST as either "127.0.0.1:11434" or "http://127.0.0.1:11434"
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    base_url = host.rstrip("/") + "/v1"

    if not model:
        # model = os.environ.get("OLLAMA_MODEL", "llama3:8b").strip()
        print("=====NO OLLAMA MODEL=====")
        quit()

    # api_key can be any non-empty string for Ollama compatibility
    return LLMConfig(provider="ollama", model=model, base_url=base_url, api_key="ollama")


def load_processing_llm_config() -> Optional[LLMConfig]:
    """
    Build an LLMConfig for data processing (dataset completion and chemical extraction) from dedicated env vars.
    Returns None if no processing-specific config is set (falls back to general LLM).
    """
    p_provider = os.environ.get("SALSA_PROCESSING_LLM_PROVIDER", "").strip().lower()
    p_model = os.environ.get("SALSA_PROCESSING_LLM_MODEL", "").strip()

    if not p_model:
        return None  # no override; use general LLM

    if p_provider in ("openai", "gpt", ""):
        base_url = os.environ.get(
            "SALSA_PROCESSING_OPENAI_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ).strip()
        api_key = os.environ.get(
            "SALSA_PROCESSING_OPENAI_API_KEY",
            os.environ.get("OPENAI_API_KEY", ""),
        ).strip() or None
        return LLMConfig(provider="openai", model=p_model, base_url=base_url, api_key=api_key)

    # ollama
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    base_url = host.rstrip("/") + "/v1"
    return LLMConfig(provider="ollama", model=p_model, base_url=base_url, api_key="ollama")


# ---- Unified interface ----

class LLMClient:
    """
    A tiny wrapper that exposes one method: `chat(messages, **kwargs)`
    and returns text.
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        from openai import OpenAI

        # Works for:
        # - OpenAI (base_url=https://api.openai.com/v1)
        # - Ollama (base_url=http://127.0.0.1:11434/v1)

        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    @property
    def model(self) -> str:
        return self.cfg.model

    @property
    def provider(self) -> str:
        return self.cfg.provider

    def _is_gpt5_model(self) -> bool:
        return self.cfg.provider == "openai" and self.cfg.model.startswith("gpt-5")

    def _effective_temperature(self, temperature: float) -> Optional[float]:
        if self.cfg.provider == "openai" and self.cfg.model.startswith("gpt-5.5"):
            return 1.0
        if self._is_gpt5_model():
            return None
        return temperature

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        return_meta: bool = False,
    ) -> str:
        """
        messages: [{"role":"system"|"user"|"assistant", "content": "..."}]
        The user content may also be a multimodal OpenAI-compatible content list.
        Returns assistant text.
        If return_meta=True, returns (text, meta) where meta includes raw resp + token counts when available.
        """

        if self.cfg.provider in ("openai", "ollama"):
            effective_temperature = self._effective_temperature(temperature)
            kwargs = {
                "model": self.cfg.model,
                "messages": messages,
            }
            if effective_temperature is not None:
                kwargs["temperature"] = effective_temperature
            if max_tokens is not None:
                if self._is_gpt5_model():
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                kwargs["response_format"] = response_format

            resp = self._client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""

            if not return_meta:
                return text

            usage = getattr(resp, "usage", None)
            meta = {
                "provider": self.cfg.provider,
                "model": getattr(resp, "model", self.cfg.model),
                "usage": usage,
                "_raw": resp,
            }
            return text, meta


        # if self.cfg.provider == "openai":
        #     # Newer OpenAI SDK supports responses API and chat.completions depending on version.
        #     # This is the most broadly compatible call pattern for "OpenAI-compatible" endpoints.
        #     kwargs: Dict[str, Any] = {"model": self.cfg.model, "messages": messages, "temperature": temperature}
        #     if max_tokens is not None:
        #         kwargs["max_tokens"] = max_tokens

        #     # If you want JSON-only output, you can enforce response_format when supported
        #     # (won't work for every compatible endpoint).
        #     if response_format is not None:
        #         kwargs["response_format"] = response_format

        #     resp = self._client.chat.completions.create(**kwargs)
        #     text = resp.choices[0].message.content or ""

        #     if not return_meta:
        #         return text

        #     # OpenAI: resp.usage has prompt_tokens/completion_tokens/total_tokens
        #     usage = getattr(resp, "usage", None)
        #     meta = {
        #         "provider": "openai",
        #         "model": getattr(resp, "model", self.cfg.model),
        #         "service_tier": getattr(resp, "service_tier", None),
        #         "usage": usage,   # keep the real usage object for your existing extractor
        #         "_raw": resp,
        #     }
        #     return text, meta

        # # ollama
        # kwargs2: Dict[str, Any] = {
        #     "model": self.cfg.model,
        #     "messages": messages,
        #     "options": {"temperature": temperature},
        # }
        # if max_tokens is not None:
        #     kwargs2["options"]["num_predict"] = max_tokens

        # # Ollama returns dict-like response
        # resp = self._client.chat(**kwargs2)
        # text = (resp.get("message") or {}).get("content", "")

        # if not return_meta:
        #     return text

        # # Ollama: token counts are in prompt_eval_count (input) and eval_count (output)
        # # per Ollama API docs. :contentReference[oaicite:0]{index=0}
        # meta = {
        #     "provider": "ollama",
        #     "model": resp.get("model", self.cfg.model),
        #     "usage": {
        #         "input_tokens": int(resp.get("prompt_eval_count") or 0),
        #         "output_tokens": int(resp.get("eval_count") or 0),
        #         "total_tokens": int((resp.get("prompt_eval_count") or 0)) + int((resp.get("eval_count") or 0)),
        #     },
        #     "_raw": resp,
        # }
        # return text, meta


# ---- Convenience helpers ----

_global_client: Optional[LLMClient] = None

# def get_llm_client(force_new: bool = False) -> LLMClient:
#     """
#     Singleton-style: build once per process. Great for Streamlit + CLI.
#     Set force_new=True if you need to re-read env in the same process.
#     """
#     global _global_client
#     if _global_client is None or force_new:
#         cfg = load_llm_config()
#         _global_client = LLMClient(cfg)
#     return _global_client

def get_llm_client(
    *,
    force_new: bool = False,
    override: Optional[LLMConfig] = None) -> LLMClient:
    """
    If override is provided, do NOT use the global singleton.
    """
    global _global_client

    if override is not None:
        return LLMClient(override)

    if _global_client is None or force_new:
        cfg = load_llm_config()
        _global_client = LLMClient(cfg)

    return _global_client
