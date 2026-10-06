# SALSA/src/llm_client.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# ---- Config ----

@dataclass(frozen=True)
class LLMConfig:
    provider: str                 # "ollama" | "openai"
    model: str                    # e.g. "llama3:8b" or "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_num_ctx: Optional[int] = None


EnvName = Union[str, Tuple[str, ...]]


def _normalize_env_names(names: EnvName) -> tuple[str, ...]:
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def _load_optional_int_env(*names: str) -> Optional[int]:
    """Return the first valid positive integer environment variable value."""
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            print(f"[llm_client] WARNING: ignoring invalid integer for {name}: {raw!r}")
            continue
        if value > 0:
            return value
        print(f"[llm_client] WARNING: ignoring non-positive integer for {name}: {raw!r}")
    return None


def _load_first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return default


def _load_llm_config_from_env(
    *,
    provider_env: EnvName,
    model_env: EnvName,
    openai_base_url_envs: tuple[str, ...],
    openai_api_key_envs: tuple[str, ...],
    ollama_host_envs: tuple[str, ...],
    ollama_num_ctx_envs: tuple[str, ...],
    default_provider: str,
    required: bool,
) -> Optional[LLMConfig]:
    provider = _load_first_env(
        *_normalize_env_names(provider_env),
        default=default_provider,
    ).strip().lower()
    model_names = _normalize_env_names(model_env)
    model = _load_first_env(*model_names).strip()

    if not model:
        if required:
            model_list = ", ".join(model_names)
            if provider in ("openai", "gpt"):
                raise RuntimeError(
                    f"No OpenAI model configured. Set one of: {model_list}"
                )
            raise RuntimeError(
                f"No Ollama model configured. Set one of: {model_list}"
            )
        return None

    if provider in ("openai", "gpt"):
        base_url = _load_first_env(
            *openai_base_url_envs,
            default="https://api.openai.com/v1",
        ).strip()
        api_key = _load_first_env(*openai_api_key_envs).strip() or None
        return LLMConfig(
            provider="openai",
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    host = _load_first_env(*ollama_host_envs, default="http://127.0.0.1:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    base_url = host.rstrip("/") + "/v1"
    ollama_num_ctx = _load_optional_int_env(*ollama_num_ctx_envs)
    return LLMConfig(
        provider="ollama",
        model=model,
        base_url=base_url,
        api_key="ollama",
        ollama_host=host,
        ollama_num_ctx=ollama_num_ctx,
    )


def load_llm_config() -> LLMConfig:
    cfg = _load_llm_config_from_env(
        provider_env="SALSA_LLM_PROVIDER",
        model_env="SALSA_LLM_MODEL",
        openai_base_url_envs=("OPENAI_BASE_URL",),
        openai_api_key_envs=("OPENAI_API_KEY",),
        ollama_host_envs=("OLLAMA_HOST",),
        ollama_num_ctx_envs=("SALSA_OLLAMA_NUM_CTX", "OLLAMA_NUM_CTX"),
        default_provider="ollama",
        required=True,
    )
    assert cfg is not None
    return cfg


def load_optional_llm_config_from_env(
    *,
    provider_env: EnvName,
    model_env: EnvName,
    openai_base_url_envs: tuple[str, ...],
    openai_api_key_envs: tuple[str, ...],
    ollama_host_envs: tuple[str, ...],
    ollama_num_ctx_envs: tuple[str, ...],
    default_provider: str = "openai",
) -> Optional[LLMConfig]:
    return _load_llm_config_from_env(
        provider_env=provider_env,
        model_env=model_env,
        openai_base_url_envs=openai_base_url_envs,
        openai_api_key_envs=openai_api_key_envs,
        ollama_host_envs=ollama_host_envs,
        ollama_num_ctx_envs=ollama_num_ctx_envs,
        default_provider=default_provider,
        required=False,
    )


def load_processing_llm_config() -> Optional[LLMConfig]:
    """
    Build an LLMConfig for data processing (dataset completion and chemical extraction) from dedicated env vars.
    Returns None if no processing-specific config is set (falls back to general LLM).
    """
    return load_optional_llm_config_from_env(
        provider_env="SALSA_PROCESSING_LLM_PROVIDER",
        model_env="SALSA_PROCESSING_LLM_MODEL",
        openai_base_url_envs=(
            "SALSA_PROCESSING_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        ),
        openai_api_key_envs=(
            "SALSA_PROCESSING_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ),
        ollama_host_envs=("OLLAMA_HOST",),
        ollama_num_ctx_envs=(
            "SALSA_PROCESSING_OLLAMA_NUM_CTX",
            "SALSA_OLLAMA_NUM_CTX",
            "OLLAMA_NUM_CTX",
        ),
        default_provider="openai",
    )


# ---- Unified interface ----

class LLMClient:
    """
    A tiny wrapper that exposes one method: `chat(messages, **kwargs)`
    and returns text.
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        from observability import setup_tracing
        from openai import OpenAI

        # No-op unless PHOENIX_COLLECTOR_ENDPOINT is set.
        setup_tracing()

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
            if self.cfg.provider == "ollama" and self.cfg.ollama_num_ctx:
                kwargs["extra_body"] = {
                    "options": {
                        "num_ctx": self.cfg.ollama_num_ctx,
                    }
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


# ---- Convenience helpers ----

_global_client: Optional[LLMClient] = None
_override_clients: Dict[LLMConfig, LLMClient] = {}

def get_llm_client(
    *,
    force_new: bool = False,
    override: Optional[LLMConfig] = None) -> LLMClient:
    """
    If override is provided, reuse a cached client per distinct config
    instead of the global singleton (LLMConfig is a frozen/hashable dataclass).
    """
    global _global_client

    if override is not None:
        if force_new or override not in _override_clients:
            _override_clients[override] = LLMClient(override)
        return _override_clients[override]

    if _global_client is None or force_new:
        cfg = load_llm_config()
        _global_client = LLMClient(cfg)

    return _global_client
