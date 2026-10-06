# SALSA/src/observability.py
"""
Optional OpenTelemetry tracing of LLM calls, exported to Arize Phoenix.

Disabled unless PHOENIX_COLLECTOR_ENDPOINT is set, e.g.

    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
    PHOENIX_PROJECT_NAME=SALSA        # optional, defaults to "SALSA"
    PHOENIX_API_KEY=...               # optional, only if Phoenix auth is on

Every OpenAI-SDK call (OpenAI and Ollama both go through it in llm_client)
is then recorded as an OpenInference LLM span with prompts, outputs and
token counts.

This lives outside llm_client so that importlib.reload(llm_client) in the
UI doesn't instrument the SDK twice.
"""
from __future__ import annotations

import os
import threading

_lock = threading.Lock()
_initialized = False


def _traces_endpoint(base: str) -> str:
    base = base.strip().rstrip("/")
    if base.endswith("/v1/traces"):
        return base
    return base + "/v1/traces"


def setup_tracing() -> bool:
    """Instrument the OpenAI SDK once per process. Returns True if tracing is on."""
    global _initialized

    with _lock:
        if _initialized:
            return True

        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
        if not endpoint:
            return False

        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as e:
            print(f"[observability] WARNING: tracing disabled, missing package: {e}")
            return False

        project = os.environ.get("PHOENIX_PROJECT_NAME", "").strip() or "SALSA"
        headers = {}
        api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"

        provider = TracerProvider(
            resource=Resource.create({"openinference.project.name": project})
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=_traces_endpoint(endpoint), headers=headers)
            )
        )
        trace.set_tracer_provider(provider)
        OpenAIInstrumentor().instrument(tracer_provider=provider)

        _initialized = True
        print(f"[observability] tracing LLM calls to {endpoint} (project {project!r})")
        return True
