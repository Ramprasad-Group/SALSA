# SALSA/src/ui/streamlit_app.py
from __future__ import annotations
import json
import inspect
import pickle
import threading
import traceback
import tempfile
import io
from http.cookies import SimpleCookie
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from PIL import Image
from image_render import is_valid_image_path, open_image_safe
from ui_common import (
    load_csv as _load_csv,
    load_emit_rows,
    section_dropdown,
    st_image_auto,
)
import figures_view
from pipeline import (
    ADMIN_INPUT_ITEM_LIMIT,
    DEFAULT_INPUT_ITEM_LIMIT,
    _subprocess_env_for_llm_step,
)
from discover import discover_run
from schemas import RunResult
from run_progress import (
    RUN_STATUS_FILE_NAME,
    _now_eastern_iso,
    clear_run_cancellation,
    read_run_status,
    request_run_cancellation,
    write_run_status,
)
from intervention import latest_plextract_run_dir
from intervention import (
    _try_find_source_image,
    get_lineformer_dir,
    load_series_catalog,
    load_mapping,
    save_mapping,
    save_mask_override,
    write_series_catalog_override,
    list_override_masks,
    apply_overrides_to_plextract,
    override_lineformer_dir,
)
from run_plextract_manual import run_plextract_on_crops as _run_plextract_on_crops
from custom_instructions import (
    DEFAULT_TABLE_EXTRACTION_RULES,
    default_instructions,
)
from canvas_geometry import fabric_circle_center
import numpy as np
import cv2
from cropping import save_crops
from run_plextract_manual import run_plextract_on_manual_crops
from axis_config import (
    load_axis_config, save_axis_config, reset_axis_config,
    list_presets as list_axis_presets,
    save_preset as save_axis_preset,
    load_preset as load_axis_preset,
    delete_preset as delete_axis_preset,
)
import math
import sys
import subprocess
import sys
import base64
from typing import List, Optional
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence, Optional, Any
from urllib.request import urlopen
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None
try:
    from dateutil import tz as dateutil_tz
except Exception:
    dateutil_tz = None
from auth import (
    AuthStore,
    data_root_from_env,
    is_admin_user,
    run_belongs_to_user,
    safe_child_path,
    send_account_approved_email,
    send_account_request_email,
    send_password_reset_email,
    send_run_notification_email,
    user_workspaces_root,
    validate_password,
    write_run_manifest,
)
from llm_step_config import (
    GENERAL_LLM_ENV_VARS,
    LLM_STEP_CHEM_COMPOSITION,
    LLM_STEP_CHEM_RELEVANT,
    LLM_STEP_COMPLETE_DATASET,
    LLM_STEP_EXTRACT_TABLES,
    LLM_STEP_FIGURE_DATA,
    LLM_STEP_FILTER_FIGURES,
    LLM_STEP_FILTER_TABLES,
    LLM_STEP_LABELS,
    LLM_STEP_ENV_PREFIXES,
    step_env_names,
)

SALSA_LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "salsa_icon.png"
PUBLIC_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "public"
PUBLISHER_TDM_ENV_NAMES = {
    "openalex_api_key": ("OPENALEX_API_KEY",),
    "elsevier_api_key": ("ELSEVIER_API_KEY", "ELS_API_KEY"),
    "wiley_tdm_token": ("WILEY_TDM_TOKEN", "WILEY_API_KEY"),
    "springer_nature_api_key": ("SPRINGER_NATURE_API_KEY", "SPRINGER_API_KEY"),
    "rsc_tdm_token": ("RSC_TDM_TOKEN", "RSC_API_KEY"),
    "ncbi_api_key": ("NCBI_API_KEY", "PUBMED_API_KEY"),
    "ieee_api_key": ("IEEE_API_KEY", "IEEE_XPLORE_API_KEY"),
}
LEGACY_PROCESSING_LLM_ENV_VARS = (
    "SALSA_PROCESSING_LLM_PROVIDER",
    "SALSA_PROCESSING_LLM_MODEL",
    "SALSA_PROCESSING_OPENAI_BASE_URL",
    "SALSA_PROCESSING_OPENAI_API_KEY",
    "SALSA_PROCESSING_OLLAMA_NUM_CTX",
)
_INITIAL_PUBLISHER_TDM_ENV = {
    env_name: os.environ.get(env_name)
    for env_names in PUBLISHER_TDM_ENV_NAMES.values()
    for env_name in env_names
}

# Make cost module importable
_cost_dir = os.path.join(os.path.dirname(__file__), "..", "cost")
if _cost_dir not in sys.path:
    sys.path.insert(0, _cost_dir)
from sum_cost import summarise_usage

#-------------------
#-----WORKSPACE-----
#-------------------

_WORKSPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

def _list_subdirs(p: Path) -> list[str]:
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])

def _ignored_workspace_or_run_name(name: str) -> bool:
    return name in {"axis_presets", "instruction_presets", "settings_presets", "custom_instructions"}

def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _run_custom_instructions_path(run_dir: Path) -> Path | None:
    run_snapshot = run_dir / "custom_instructions.json"
    if run_snapshot.exists():
        return run_snapshot
    workspace_default = run_dir.parent / "custom_instructions" / "default.json"
    return workspace_default if workspace_default.exists() else None

def _run_custom_instructions_label(run_dir: Path) -> tuple[str, str]:
    manifest = _read_json_file(run_dir / "custom_instructions_manifest.json")
    name = str(manifest.get("name") or "").strip()
    if name:
        return name, "Run snapshot"

    status = read_run_status(run_dir / RUN_STATUS_FILE_NAME)
    name = str(status.get("custom_instructions_name") or "").strip()
    if name:
        return name, "Run snapshot"

    if (run_dir / "custom_instructions.json").exists():
        return "Custom instructions", "Run snapshot"
    if (run_dir.parent / "custom_instructions" / "default.json").exists():
        return "Workspace default", "Legacy fallback; exact run snapshot is unavailable"
    return "Not recorded", "No custom instruction file found"

def _render_run_custom_instructions(run_dir: Path) -> None:
    label, source = _run_custom_instructions_label(run_dir)
    st.info(f"Custom instructions: **{label}**")

    ci_path = _run_custom_instructions_path(run_dir)
    if ci_path is None:
        st.caption(source)
        return

    with st.expander("View custom instructions used", expanded=False):
        st.caption(source)
        st.json(_read_json_file(ci_path))

def _run_status_outcome_message(status: dict[str, Any], run_id: str) -> tuple[str, str]:
    state = str(status.get("state") or "").strip().lower()
    stage = _display_stage_label(status.get("stage"))
    error = str(status.get("error") or "").strip()
    updated = _format_eastern_timestamp(status.get("updated_at"))
    warnings = status.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []
    warnings = [str(item).strip() for item in warnings if str(item).strip()]

    if state == "done":
        if warnings:
            message = f"Run **{run_id}** completed with warnings."
            message += "\n\n" + "\n".join(f"- {warning}" for warning in warnings)
            kind = "warning"
        else:
            message = f"Run **{run_id}** completed successfully."
            kind = "success"
        if updated:
            message += f"\n\nCompleted (ET): `{updated}`"
        return kind, message
    if state == "error":
        message = f"Run **{run_id}** failed."
        if stage:
            message += f"\n\nStage: **{stage}**"
        if updated:
            message += f"\n\nUpdated (ET): `{updated}`"
        if error:
            message += f"\n\nError: {error}"
        return "error", message
    if state == "cancelled":
        message = f"Run **{run_id}** was cancelled."
        if updated:
            message += f"\n\nUpdated (ET): `{updated}`"
        return "warning", message
    if state == "interrupted":
        message = f"Run **{run_id}** was interrupted."
        if stage:
            message += f"\n\nLast stage: **{stage}**"
        if updated:
            message += f"\n\nUpdated (ET): `{updated}`"
        if error:
            message += f"\n\nReason: {error}"
        return "warning", message
    return "", ""


def _render_run_status_outcome(run_id: str, status: dict[str, Any]) -> None:
    kind, message = _run_status_outcome_message(status, run_id)
    if not message:
        return
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)
    else:
        st.warning(message)

    traceback_text = str(status.get("traceback") or "").strip()
    if traceback_text:
        with st.expander("Traceback", expanded=False):
            st.code(traceback_text, language="text")

    if "notification_sent" in status:
        if status.get("notification_sent"):
            st.caption("Completion email notification sent.")
        else:
            notification_error = str(status.get("notification_error") or "").strip()
            if notification_error:
                st.warning(f"Completion email notification was not sent: {notification_error}")
            else:
                st.warning("Completion email notification was not sent.")

def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(size, 0))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024

def _dir_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not Path(current_root, d).is_symlink()
        ]
        for filename in filenames:
            path = Path(current_root) / filename
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    return total

def _path_mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""

def _count_subdirs(path: Path) -> int:
    try:
        return sum(1 for child in path.iterdir() if child.is_dir())
    except OSError:
        return 0

def _is_safe_workspace_name(name: str) -> bool:
    """
    Restrict workspace names so users can't do path traversal like ../../etc.
    """
    name = (name or "").strip()
    if not name:
        return False
    if "/" in name or "\\" in name:
        return False
    return bool(_WORKSPACE_RE.match(name))

def _ensure_workspace(root: Path, name: str) -> Path:
    """
    Create (if needed) and return <root>/<name>.
    """
    ws = (root / name).resolve()
    # Enforce that ws is inside root even after resolve()
    root_res = root.resolve()
    if root_res not in ws.parents and ws != root_res:
        raise ValueError(f"Workspace path escapes root: {ws}")
    ws.mkdir(parents=True, exist_ok=True)
    return ws

def _default_workspace_name() -> str:
    return "workspace_1"

def _default_ollama_models_dir() -> Path:
    """
    Programmatically discover the Ollama models directory.

    Resolution order:
    1. ``OLLAMA_MODELS`` environment variable (explicit override).
    2. Ask the running ``ollama`` daemon via ``ollama show --modelpath`` (most
       authoritative – reflects the daemon's *actual* storage location).
    3. Parse the systemd unit file for an ``Environment=OLLAMA_MODELS=…`` line.
    4. Probe well-known filesystem candidates.
    5. Fall back to ``~/.ollama/models`` and create it if it doesn't exist.
    """
    # 1. Environment variable
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        p = Path(env).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2. Ask the CLI – ``ollama show --modelpath <any-model>`` prints the
    #    blob-storage root.  We don't need a real model name; even when the
    #    command fails it can reveal the base path in its error text.  A
    #    lighter alternative is to inspect ``ollama list`` output and derive
    #    the path, but ``OLLAMA_MODELS`` from the daemon env is more direct.
    try:
        # ``ollama`` respects OLLAMA_HOST, so this talks to the right daemon.
        out = subprocess.check_output(
            ["ollama", "list"], text=True, stderr=subprocess.STDOUT, timeout=5
        )
        # If ``ollama list`` succeeds the daemon is up; the daemon's storage
        # dir is authoritative.  Try to read it from the process environment.
        models_dir = _ollama_models_dir_from_proc()
        if models_dir and models_dir.is_dir():
            return models_dir
    except Exception:
        pass

    # 3. systemd unit file (common on Linux when Ollama was installed via the
    #    official installer script).
    for unit_path in (
        Path("/etc/systemd/system/ollama.service"),
        Path("/lib/systemd/system/ollama.service"),
    ):
        try:
            if unit_path.is_file():
                text = unit_path.read_text()
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("Environment=") and "OLLAMA_MODELS" in line:
                        # e.g. Environment="OLLAMA_MODELS=/data/ollama/models"
                        _, _, val = line.partition("OLLAMA_MODELS=")
                        val = val.strip().strip('"').strip("'")
                        if val:
                            p = Path(val)
                            if p.is_dir():
                                return p
        except Exception:
            pass

    # 4. Well-known candidates
    candidates = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
        Path("/opt/ollama/models"),
    ]
    for c in candidates:
        try:
            if c.exists():
                return c
        except (PermissionError, OSError):
            continue

    # 5. Fallback – create the user-local directory
    fallback = Path.home() / ".ollama" / "models"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

def _ollama_models_dir_from_proc() -> Optional[Path]:
    """
    Try to read ``OLLAMA_MODELS`` from the running ``ollama`` daemon process's
    environment on Linux (``/proc/<pid>/environ``).  Returns *None* on failure.
    """
    try:
        pids = subprocess.check_output(
            ["pgrep", "-x", "ollama"], text=True, timeout=3
        ).split()
        for pid in pids:
            env_file = Path(f"/proc/{pid}/environ")
            if env_file.exists():
                env_bytes = env_file.read_bytes()
                for entry in env_bytes.split(b"\x00"):
                    if entry.startswith(b"OLLAMA_MODELS="):
                        val = entry.split(b"=", 1)[1].decode()
                        p = Path(val)
                        if p.is_dir():
                            return p
    except Exception:
        pass
    return None

def _default_ollama_host() -> str:
    """
    Programmatically discover the Ollama host endpoint.

    Resolution order:
    1. ``OLLAMA_HOST`` environment variable.
    2. ``OLLAMA_HOST`` from the running daemon's process environment (Linux).
    3. ``OLLAMA_HOST`` from the systemd unit file.
    4. Default: ``http://127.0.0.1:11434`` (standard Ollama port).
    """
    # 1. Environment variable
    env = os.environ.get("OLLAMA_HOST")
    if env:
        host = env.strip()
        if not host.startswith("http"):
            host = "http://" + host
        return host

    # 2. Daemon process environment (Linux)
    try:
        pids = subprocess.check_output(
            ["pgrep", "-x", "ollama"], text=True, timeout=3
        ).split()
        for pid in pids:
            env_file = Path(f"/proc/{pid}/environ")
            if env_file.exists():
                env_bytes = env_file.read_bytes()
                for entry in env_bytes.split(b"\x00"):
                    if entry.startswith(b"OLLAMA_HOST="):
                        val = entry.split(b"=", 1)[1].decode().strip()
                        if not val.startswith("http"):
                            val = "http://" + val
                        return val
    except Exception:
        pass

    # 3. systemd unit file
    for unit_path in (
        Path("/etc/systemd/system/ollama.service"),
        Path("/lib/systemd/system/ollama.service"),
    ):
        try:
            if unit_path.is_file():
                text = unit_path.read_text()
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("Environment=") and "OLLAMA_HOST" in line:
                        _, _, val = line.partition("OLLAMA_HOST=")
                        val = val.strip().strip('"').strip("'")
                        if val:
                            if not val.startswith("http"):
                                val = "http://" + val
                            return val
        except Exception:
            pass

    # 4. Standard default
    return "http://127.0.0.1:11434"

def _normalize_ollama_host(host: str) -> str:
    value = (host or "").strip() or _default_ollama_host()
    if not value.startswith("http"):
        value = "http://" + value
    return value.rstrip("/")


@st.cache_data(show_spinner=False, ttl=5)
def _ollama_list_models_via_api(host: str) -> Optional[List[str]]:
    """
    Query the Ollama HTTP API directly so the UI reflects whether the daemon is
    actually reachable, not just whether models exist on disk.
    """
    normalized_host = _normalize_ollama_host(host)
    try:
        with urlopen(f"{normalized_host}/api/tags", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = []
        for item in payload.get("models", []):
            name = str(item.get("name", "")).strip()
            if name:
                models.append(name)
        return models
    except Exception:
        return None


def _ollama_list_models_via_cli(host: str) -> Optional[List[str]]:
    """
    Best source of truth: `ollama list` returns the models actually available to the ollama daemon.
    """
    try:
        # Note: OLLAMA_HOST can be set in env to point to a non-default daemon
        cli_env = os.environ.copy()
        cli_env["OLLAMA_HOST"] = _normalize_ollama_host(host)
        out = subprocess.check_output(
            ["ollama", "list"],
            text=True,
            stderr=subprocess.STDOUT,
            env=cli_env,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not lines:
            return []
        # Expect header like: NAME ID SIZE MODIFIED
        # Models in first column, e.g. "llama3:8b"
        models = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=5)
def _ollama_server_is_available(host: str) -> bool:
    normalized_host = _normalize_ollama_host(host)
    if _ollama_list_models_via_api(normalized_host) is not None:
        return True
    return _ollama_list_models_via_cli(normalized_host) is not None

def _ollama_list_models_from_dir(models_dir: Path) -> List[str]:
    """
    Fallback: attempt to infer installed models from the local models directory.
    This is less reliable than `ollama list`, but helps when CLI isn't available.
    """
    try:
        # Typical structure includes "manifests" + "blobs"
        manifests = models_dir / "manifests"
        if not manifests.exists():
            return []
        # manifests can be nested; filenames represent tags.
        # We'll return relative leaf paths that look like repo/model:tag-ish.
        found = []
        for p in manifests.rglob("*"):
            if p.is_file():
                rel = p.relative_to(manifests).as_posix()
                # Heuristic: keep only reasonably short “tag” paths
                if len(rel.split("/")) >= 2:
                    found.append(rel)
        # Deduplicate + shorten presentation
        found = sorted(set(found))
        return found
    except Exception:
        return []



def render_salsa_brand_header() -> None:
    if not SALSA_LOGO_PATH.exists():
        st.title("SALSA - Semi-Autonomous Literature Scraping Assistant")
        return

    _, logo_col, _ = st.columns([1, 6, 1])
    with logo_col:
        st_image_auto(str(SALSA_LOGO_PATH), full_width=True)


# Thin aliases so the rest of this module keeps its existing call sites while the
# implementations live in image_render (and are unit-testable without Streamlit).
_is_valid_image_path = is_valid_image_path
_open_image_safe = open_image_safe


def doi_to_safe(doi: str) -> str:
    return doi.replace("/", "@").strip()

@st.cache_data(show_spinner=False)
def _cached_summarise_usage(usage_path_str: str, mtime_ns: int, size: int) -> dict:
    # mtime_ns/size are part of the cache key purely to invalidate the cache
    # when usage.jsonl actually changes (e.g. a still-running pipeline appending
    # to it); summarise_usage itself always reads from usage_path_str.
    return summarise_usage(usage_path_str)


def _render_cost_summary(run_dir: Path):
    """Display cost summary for a run if usage.jsonl exists."""
    usage_path = run_dir / "cost" / "usage.jsonl"
    if not usage_path.exists():
        return
    usage_stat = usage_path.stat()
    summary = _cached_summarise_usage(str(usage_path), usage_stat.st_mtime_ns, usage_stat.st_size)
    if not summary["models"]:
        return
    with st.expander("Cost Summary", expanded=False):
        stages = summary.get("stages") or {}
        if stages:
            stage_rows = []
            for stage_name, info in stages.items():
                stage_models = info.get("models") or {}
                model_bits = [
                    (
                        f"{model_name if model_name != 'unknown' else 'unknown model'} "
                        f"({model_info['calls']} calls, ${model_info['cost']:.4f})"
                    )
                    for model_name, model_info in stage_models.items()
                ]
                stage_rows.append(
                    {
                        "Stage": stage_name,
                        "Model(s)": "; ".join(model_bits) if model_bits else "",
                        "Calls": info["calls"],
                        "Input": info["input"],
                        "Cached": info["cached"],
                        "Output": info["output"],
                        "Cost": f"${info['cost']:.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)
        st.metric("Total", f"${summary['grand_total']:.4f}")

def completed_dataset_csv_path(run_dir: Path, doi: str) -> Path:
    """
    complete-dataset.py writes:
      <run_dir>/complete_dataset/<modified_doi>_dataset.csv
    where modified_doi is DOI with '/' -> '@'
    """
    modified = doi_to_safe(doi)
    return run_dir / "complete_dataset" / f"{modified}_dataset.csv"

def completed_dataset_json_path(run_dir: Path, doi: str) -> Path:
    """
    Also written by complete-dataset.py:
      <run_dir>/complete_dataset/<modified_doi>_complete.json
    """
    modified = doi_to_safe(doi)
    return run_dir / "complete_dataset" / f"{modified}_complete.json"

def find_emit_rows_for_source(
    run_dir: Path,
    doi: str,
    source_kind: str,
    source_name: str
) -> Path | None:
    """
    Find the emit_rows cache file for a specific table/figure.

    Cache key format used by row_emitter.py:
      emit_rows::{source_kind}::{doi}::{source_name}::{hash}

    The _safe_filename() in row_emitter replaces non-alphanumeric chars with '_',
    then _cache_path appends '__<sha256[:20]>.json'.  So the on-disk name is:
      emit_rows__{source_kind}__{sanitized_doi}_{sanitized_source_name}__{sha256}__{sha256}.json
    (double underscores between cache_key segments, single underscores within segments)

    Returns path to cache file if found, None otherwise.
    """
    cache_dir = run_dir / "complete_dataset" / "row_cache"
    if not cache_dir.exists():
        return None

    # Build the exact prefix that row_emitter._safe_filename would produce from
    # the cache key "emit_rows::{source_kind}::{doi}::{source_name}::".
    # _safe_filename replaces any non-[a-zA-Z0-9._-] with '_', so :: -> __ and / -> _
    def _safe_filename_compat(s: str) -> str:
        """Reproduce row_emitter._safe_filename logic."""
        import re as _re
        return _re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:180]

    # The cache key always starts with "emit_rows::{kind}::{doi}::{name}::"
    # After _safe_filename this becomes "emit_rows__{kind}__{doi_safe}__{name_safe}__"
    source_names = [source_name]
    source_stem = Path(str(source_name)).stem
    if source_stem and source_stem not in source_names:
        source_names.append(source_stem)
    safe_prefixes = [
        _safe_filename_compat(f"emit_rows::{source_kind}::{doi}::{name}::")
        for name in source_names
    ]

    candidates = []
    for cache_file in cache_dir.iterdir():
        if cache_file.suffix != ".json":
            continue
        # Files are named: <safe_prefix><hash>__<sha256>.json
        # Check that the filename starts with a compatible safe_prefix.
        if any(
            cache_file.stem.startswith(prefix) or cache_file.name.startswith(prefix)
            for prefix in safe_prefixes
        ):
            candidates.append(cache_file)

    # Return most recent if multiple found
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None

def collect_all_emit_rows(run_dir: Path, doi: str, doi_item) -> pd.DataFrame:
    """
    Collect and concatenate all LLM-generated emit_rows from row_cache
    for all tables, figures, and subfigures for the given DOI.

    This creates a reactive dataset that updates whenever emit_rows are regenerated.
    """
    all_dfs = []
    safe_doi = doi.replace("/", "@").strip()
    seen_source_names: set[str] = set()

    def _add_emit(source_kind: str, source_name: str):
        if source_name in seen_source_names:
            return
        seen_source_names.add(source_name)
        emit_path = find_emit_rows_for_source(run_dir, doi, source_kind, source_name)
        if emit_path:
            df = load_emit_rows(emit_path)
            if df is not None and not df.empty:
                all_dfs.append(df)

    # Collect emit_rows from all tables
    for table in doi_item.tables:
        _add_emit("table", table.id)

    # Collect emit_rows from all figures
    for figure in doi_item.figures:
        _add_emit("figure", figure.id)

    # Collect emit_rows from subfigures in both plextract workdirs.
    # Subfigures can appear in:
    #   1. <run_dir>/plextract/results/<n>/output/<safe_doi>/<subfig_name>/
    #   2. <run_dir>/interventions/interventions_plextract/results/<n>/output/<safe_doi>/<subfig_name>/
    _workdirs_to_scan = [
        run_dir / "plextract",
        run_dir / "interventions" / "interventions_plextract",
    ]
    for wd in _workdirs_to_scan:
        if not wd.exists():
            continue
        latest = latest_plextract_run_dir(wd)
        if not latest:
            continue
        out_base = latest / "output" / safe_doi
        if not out_base.exists():
            continue
        for subfig_dir in out_base.iterdir():
            if subfig_dir.is_dir() and "_subfig_" in subfig_dir.name:
                _add_emit("figure", subfig_dir.name)

    # Concatenate all dataframes
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove exact duplicates
    combined = combined.drop_duplicates()

    return combined

def _publisher_name_from_doi(doi: str) -> str:
    doi = str(doi).lower()
    prefix_publishers = (
        (("10.1016/", "10.1006/"), "Elsevier"),
        (("10.1002/",), "Wiley"),
        (("10.1007/", "10.1023/"), "Springer"),
        (("10.1038/",), "Nature"),
        (("10.1021/",), "ACS"),
        (("10.1039/",), "RSC"),
        (("10.1103/",), "APS"),
        (("10.1063/",), "AIP Publishing"),
        (("10.1088/",), "IOP"),
        (("10.1149/",), "ECS"),
        (("10.1109/",), "IEEE"),
    )
    return next(
        (publisher for prefixes, publisher in prefix_publishers if doi.startswith(prefixes)),
        "Unknown",
    )


def build_doi_stats(rr: RunResult, *, doi_order: list[str] | None = None) -> pd.DataFrame:
    def _publisher_for_doi(doi_item) -> str:
        for key in ("publisher", "Publisher", "directory"):
            value = doi_item.extra.get(key)
            if value not in (None, "") and str(value).lower() != "docling_pdf":
                return str(value)
        for collection in (doi_item.tables or [], doi_item.figures or []):
            for item in collection:
                value = item.extra.get("publisher") or item.extra.get("Publisher")
                if value not in (None, "") and str(value).lower() != "docling_pdf":
                    return str(value)
        return _publisher_name_from_doi(doi_item.doi)

    def _chemical_sample_count(doi_item) -> int:
        csv_path = doi_item.extra.get("chemical_composition_csv")
        if csv_path:
            try:
                return int(len(pd.read_csv(csv_path)))
            except Exception:
                pass

        json_path = doi_item.extra.get("chemical_composition_json")
        if json_path:
            try:
                data = json.loads(Path(json_path).read_text(encoding="utf-8"))
                samples = data.get("samples") if isinstance(data, dict) else None
                if isinstance(samples, list):
                    return len(samples)
            except Exception:
                pass

        return 0

    rows = []
    for d in rr.dois:
        safe = doi_to_safe(d.doi)
        rows.append({
            "DOI": d.doi,
            "Publisher": _publisher_for_doi(d),
            "Tables": len(d.tables or []),
            "Figures": len(d.figures or []),
            "Chemical Samples": _chemical_sample_count(d),
            "_doi_safe": safe,  # hidden helper
        })

    df = pd.DataFrame(rows)
    # --- preserve paste order if provided ---
    if doi_order:
        order_map = {doi: i for i, doi in enumerate(doi_order)}
        df["_order"] = df["DOI"].map(order_map).fillna(10**9).astype(int)
        df = df.sort_values("_order", kind="stable").drop(columns=["_order"]).reset_index(drop=True)

    
    return df

def _ollama_models() -> List[str]:
    host = _normalize_ollama_host(st.session_state.get("ollama_host", default_ollama_host))
    api_models = _ollama_list_models_via_api(host)
    if api_models is not None:
        return api_models
    cli_models = _ollama_list_models_via_cli(host)
    if cli_models is not None:
        return cli_models
    models_dir = Path(
        st.session_state.get("ollama_models_dir", str(default_ollama_dir))
    )
    manifest_models = _ollama_list_models_from_dir(models_dir)
    models = []
    for manifest_model in manifest_models:
        parts = manifest_model.split("/")
        if len(parts) >= 2:
            models.append(f"{parts[-2]}:{parts[-1]}")
    return sorted(set(models))

_STEP_LLM_RENDER_ORDER = [
    LLM_STEP_EXTRACT_TABLES,
    LLM_STEP_CHEM_RELEVANT,
    LLM_STEP_CHEM_COMPOSITION,
    LLM_STEP_FILTER_FIGURES,
    LLM_STEP_FILTER_TABLES,
    LLM_STEP_FIGURE_DATA,
    LLM_STEP_COMPLETE_DATASET,
]


def _ensure_shared_llm_settings_in_state() -> None:
    if "openai_base_url" not in st.session_state:
        st.session_state["openai_base_url"] = os.environ.get(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        )
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""
    if "elsevier_api_key" not in st.session_state:
        st.session_state["elsevier_api_key"] = ""
    if "openalex_api_key" not in st.session_state:
        st.session_state["openalex_api_key"] = ""
    if "scopus_inst_token" not in st.session_state:
        st.session_state["scopus_inst_token"] = ""
    if "wiley_tdm_token" not in st.session_state:
        st.session_state["wiley_tdm_token"] = ""
    if "springer_nature_api_key" not in st.session_state:
        st.session_state["springer_nature_api_key"] = ""
    if "rsc_tdm_token" not in st.session_state:
        st.session_state["rsc_tdm_token"] = ""
    if "ncbi_api_key" not in st.session_state:
        st.session_state["ncbi_api_key"] = ""
    if "ieee_api_key" not in st.session_state:
        st.session_state["ieee_api_key"] = ""
    if "ollama_host" not in st.session_state:
        st.session_state["ollama_host"] = os.environ.get(
            "OLLAMA_HOST",
            _default_ollama_host(),
        )
    if "ollama_models_dir" not in st.session_state:
        st.session_state["ollama_models_dir"] = os.environ.get(
            "OLLAMA_MODELS",
            str(_default_ollama_models_dir()),
        )
    if "ollama_num_ctx" not in st.session_state:
        try:
            st.session_state["ollama_num_ctx"] = int(
                os.environ.get("SALSA_OLLAMA_NUM_CTX", "16384")
            )
        except ValueError:
            st.session_state["ollama_num_ctx"] = 16384


def _sync_publisher_tdm_env_from_session() -> None:
    _ensure_shared_llm_settings_in_state()
    for session_key, env_names in PUBLISHER_TDM_ENV_NAMES.items():
        value = str(st.session_state.get(session_key, "")).strip()
        for env_name in env_names:
            if value:
                os.environ[env_name] = value
            elif _INITIAL_PUBLISHER_TDM_ENV.get(env_name) is not None:
                os.environ[env_name] = _INITIAL_PUBLISHER_TDM_ENV[env_name] or ""
            else:
                os.environ.pop(env_name, None)


def _step_provider_key(step_key: str) -> str:
    return f"{step_key}_llm_provider"


def _step_model_key(step_key: str) -> str:
    return f"{step_key}_llm_model"


def _step_provider_model_memory_key(step_key: str, provider_label: str) -> str:
    suffix = "ollama" if provider_label == "Ollama" else "openai"
    return f"{step_key}_llm_model_remembered_{suffix}"


def _remember_step_model_for_provider(step_key: str, provider_label: str) -> None:
    value = str(st.session_state.get(_step_model_key(step_key), "")).strip()
    if value:
        st.session_state[_step_provider_model_memory_key(step_key, provider_label)] = value


def _on_step_model_change(step_key: str, provider_label: str) -> None:
    _keep_create_settings_preset_open()
    provider_model_key = _step_provider_model_memory_key(step_key, provider_label)
    st.session_state[_step_model_key(step_key)] = str(
        st.session_state.get(provider_model_key, "")
    ).strip()


def _on_manual_ollama_model_change(step_key: str) -> None:
    _keep_create_settings_preset_open()
    provider_model_key = _step_provider_model_memory_key(step_key, "Ollama")
    manual_model_key = f"{provider_model_key}_manual"
    value = str(st.session_state.get(manual_model_key, "")).strip()
    st.session_state[provider_model_key] = value
    st.session_state[_step_model_key(step_key)] = value


def _on_step_provider_change(step_key: str) -> None:
    """Swap the shared model widget state when a step changes provider."""
    _keep_create_settings_preset_open()
    provider_key = _step_provider_key(step_key)
    model_key = _step_model_key(step_key)
    new_provider = st.session_state.get(provider_key, "Chat GPT")
    previous_provider_key = f"{provider_key}_previous"
    previous_provider = st.session_state.get(previous_provider_key, "Chat GPT")

    _remember_step_model_for_provider(step_key, previous_provider)
    remembered = st.session_state.get(
        _step_provider_model_memory_key(step_key, new_provider)
    )
    if remembered:
        st.session_state[model_key] = remembered
    else:
        st.session_state.pop(model_key, None)
    st.session_state[previous_provider_key] = new_provider


def _step_provider_label(step_key: str) -> str:
    return st.session_state.get(_step_provider_key(step_key), "Chat GPT")


def _step_model_value(step_key: str) -> str:
    return str(st.session_state.get(_step_model_key(step_key), "")).strip()


def _clear_general_llm_env(env: dict[str, str]) -> None:
    for key in GENERAL_LLM_ENV_VARS:
        env.pop(key, None)


def _clear_step_llm_env(step_key: str) -> None:
    for key in step_env_names(step_key):
        os.environ.pop(key, None)


def _sync_step_llm_env_from_session(step_key: str) -> None:
    _ensure_shared_llm_settings_in_state()
    _clear_step_llm_env(step_key)

    model = _step_model_value(step_key)
    if not model:
        return

    provider = _step_provider_label(step_key)
    prefix = LLM_STEP_ENV_PREFIXES[step_key]
    if provider == "Ollama":
        os.environ[f"{prefix}_LLM_PROVIDER"] = "ollama"
        os.environ[f"{prefix}_LLM_MODEL"] = model
        os.environ[f"{prefix}_OLLAMA_HOST"] = st.session_state.get("ollama_host", default_ollama_host)
        os.environ[f"{prefix}_OLLAMA_MODELS"] = st.session_state.get(
            "ollama_models_dir",
            str(default_ollama_dir),
        )
        os.environ[f"{prefix}_OLLAMA_NUM_CTX"] = str(
            int(st.session_state.get("ollama_num_ctx", 16384))
        )
    else:
        os.environ[f"{prefix}_LLM_PROVIDER"] = "openai"
        os.environ[f"{prefix}_LLM_MODEL"] = model
        os.environ[f"{prefix}_OPENAI_BASE_URL"] = st.session_state.get(
            "openai_base_url",
            "https://api.openai.com/v1",
        )
        api_key = str(st.session_state.get("openai_api_key", "")).strip()
        if api_key:
            os.environ[f"{prefix}_OPENAI_API_KEY"] = api_key


def _sync_all_step_llm_envs_from_session() -> None:
    _clear_general_llm_env(os.environ)
    for step_key in _STEP_LLM_RENDER_ORDER:
        _sync_step_llm_env_from_session(step_key)


def _complete_dataset_env_for_current_settings(rr: RunResult) -> dict[str, str]:
    _sync_all_step_llm_envs_from_session()
    env = _subprocess_env_for_llm_step(LLM_STEP_COMPLETE_DATASET)
    for key in LEGACY_PROCESSING_LLM_ENV_VARS:
        env.pop(key, None)
    ci = st.session_state.get("custom_instructions")
    if not isinstance(ci, dict):
        ci = default_instructions()
    ci_path = rr.run_dir / "interventions" / "current_complete_dataset_settings.json"
    ci_path.parent.mkdir(parents=True, exist_ok=True)
    ci_path.write_text(json.dumps(ci, indent=2) + "\n", encoding="utf-8")
    env["SALSA_CUSTOM_INSTRUCTIONS"] = str(ci_path)
    env["SALSA_RUN_ID"] = rr.run_id
    env["SALSA_RUN_DIR"] = str(rr.run_dir)
    return env


def _step_has_configured_model(step_key: str) -> bool:
    return bool(_step_model_value(step_key))


def _render_step_llm_selector(
    step_key: str,
    *,
    ollama_available: bool,
    ollama_models: Sequence[str],
    caption: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    _ensure_shared_llm_settings_in_state()

    st.subheader(title or LLM_STEP_LABELS[step_key])
    if caption:
        st.caption(caption)

    provider_key = _step_provider_key(step_key)
    model_key = _step_model_key(step_key)
    previous_provider_key = f"{provider_key}_previous"
    current_provider = st.session_state.get(provider_key, "Chat GPT")
    if previous_provider_key not in st.session_state:
        st.session_state[previous_provider_key] = current_provider
    provider_options = ["Chat GPT", "Ollama"]
    provider = st.radio(
        "Provider",
        options=provider_options,
        index=provider_options.index(st.session_state[provider_key])
        if st.session_state.get(provider_key) in provider_options
        else 0,
        key=provider_key,
        horizontal=True,
        on_change=_on_step_provider_change,
        args=(step_key,),
    )

    if provider == "Ollama":
        models = list(ollama_models)
        provider_model_key = _step_provider_model_memory_key(step_key, "Ollama")
        remembered_model = str(
            st.session_state.get(provider_model_key, "")
        ).strip()
        if remembered_model and remembered_model not in models:
            models.insert(0, remembered_model)
        if provider_model_key not in st.session_state:
            current_model = str(st.session_state.get(model_key, "")).strip()
            if current_model in models:
                st.session_state[provider_model_key] = current_model
        if models:
            selected_model = st.selectbox(
                "Model",
                options=models,
                index=0,
                key=provider_model_key,
                on_change=lambda: _on_step_model_change(step_key, "Ollama"),
            )
        else:
            manual_model_key = f"{provider_model_key}_manual"
            if manual_model_key not in st.session_state:
                st.session_state[manual_model_key] = str(
                    st.session_state.get(provider_model_key, "")
                )
            selected_model = st.text_input(
                "Model (no models detected)",
                key=manual_model_key,
                on_change=lambda: _on_manual_ollama_model_change(step_key),
            )
            st.session_state[provider_model_key] = str(selected_model).strip()
            st.caption("Tip: run `ollama pull <model>` then refresh.")
    else:
        provider_model_key = _step_provider_model_memory_key(step_key, "Chat GPT")
        if provider_model_key not in st.session_state:
            current_model = str(st.session_state.get(model_key, "")).strip()
            if current_model in _openai_models:
                st.session_state[provider_model_key] = current_model
        if st.session_state.get(provider_model_key) not in _openai_models:
            st.session_state.pop(provider_model_key, None)
        selected_model = st.selectbox(
            "Model",
            options=_openai_models,
            index=0,
            key=provider_model_key,
            on_change=lambda: _on_step_model_change(step_key, "Chat GPT"),
        )
    st.session_state[model_key] = str(selected_model).strip()


def _active_llm_steps(
    *,
    run_extract_tables: bool,
    use_docling_pdf_uploads: bool = False,
    run_chemical_composition: bool,
    run_filter_figures: bool,
    run_filter_tables: bool,
    run_plextract: bool,
    bypass_ocr_with_llm: bool,
    run_complete_dataset: bool,
) -> list[str]:
    steps: list[str] = []
    if run_extract_tables and not use_docling_pdf_uploads:
        steps.append(LLM_STEP_EXTRACT_TABLES)
    if run_chemical_composition:
        steps.extend([LLM_STEP_CHEM_RELEVANT, LLM_STEP_CHEM_COMPOSITION])
    if run_filter_figures:
        steps.append(LLM_STEP_FILTER_FIGURES)
    if run_filter_tables:
        steps.append(LLM_STEP_FILTER_TABLES)
    if run_plextract and bypass_ocr_with_llm:
        steps.append(LLM_STEP_FIGURE_DATA)
    if run_complete_dataset:
        steps.append(LLM_STEP_COMPLETE_DATASET)
    return steps


def _missing_llm_steps(step_keys: Sequence[str]) -> list[str]:
    return [
        LLM_STEP_LABELS[step_key]
        for step_key in step_keys
        if not _step_has_configured_model(step_key)
    ]


def _any_step_uses_ollama(step_keys: Sequence[str]) -> bool:
    return any(_step_provider_label(step_key) == "Ollama" for step_key in step_keys)


def _missing_openai_api_key_steps(step_keys: Sequence[str]) -> list[str]:
    if str(st.session_state.get("openai_api_key", "")).strip():
        return []
    return [
        LLM_STEP_LABELS[step_key]
        for step_key in step_keys
        if _step_provider_label(step_key) == "Chat GPT"
    ]


def _llm_step_caption(step_key: str) -> str:
    captions = {
        LLM_STEP_EXTRACT_TABLES: "Used when parsing table content with the LLM.",
        LLM_STEP_CHEM_RELEVANT: "Finds the text, tables, and schemes relevant to chemical composition.",
        LLM_STEP_CHEM_COMPOSITION: "Uses the selected relevant evidence to extract sample compositions.",
        LLM_STEP_FILTER_FIGURES: "Decides which figures are relevant downstream.",
        LLM_STEP_FILTER_TABLES: "Decides which tables are relevant downstream.",
        LLM_STEP_FIGURE_DATA: "Used only when bypassing OCR and extracting figure data directly with the LLM.",
        LLM_STEP_COMPLETE_DATASET: "Builds the final dataset rows.",
    }
    return captions.get(step_key, "")


def _looks_like_doi_scope(value: str) -> bool:
    return bool(re.fullmatch(r"\d+\s+DOIs?", value.strip()))


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except OSError:
        return False
    return True


def _run_status_is_active(status: dict[str, Any]) -> bool:
    return status.get("state") in {"running", "cancelling"} and _pid_is_alive(status.get("pid"))


def _format_eastern_timestamp(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return raw
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if ZoneInfo is not None:
        eastern_tz = ZoneInfo("America/New_York")
    elif dateutil_tz is not None:
        eastern_tz = dateutil_tz.gettz("America/New_York") or timezone(timedelta(hours=-5), "ET")
    else:
        eastern_tz = timezone(timedelta(hours=-5), "ET")
    eastern = dt.astimezone(eastern_tz)
    hour = eastern.strftime("%I").lstrip("0") or "0"
    return (
        f"{eastern:%Y-%m-%d} {hour}:{eastern:%M:%S} "
        f"{eastern:%p} {eastern:%Z}"
    )


def _display_stage_label(stage: Any) -> str:
    label = str(stage or "Running")
    return label.replace(" with Docling", "")


def _mark_inactive_run_status(status_path: Path, status: dict[str, Any]) -> dict[str, Any]:
    state = status.get("state")
    pid = status.get("pid")
    if state not in {"running", "cancelling"} or _pid_is_alive(pid):
        return status
    write_run_status(
        status_path,
        state="interrupted",
        stage="Interrupted",
        error=f"Last recorded pipeline process is no longer running (pid: {pid}).",
    )
    return read_run_status(status_path)


def _run_progress_status_text(status: dict[str, Any]) -> str:
    stage = _display_stage_label(status.get("stage"))
    doi = str(status.get("doi") or "").strip()
    lines = [f"Running stage: **{stage}**"]
    if doi:
        label = "Scope" if _looks_like_doi_scope(doi) else "Current DOI"
        lines.append(f"{label}: `{doi}`")
    figure = str(status.get("figure") or "").strip()
    if figure:
        lines.append(f"Current figure: `{figure}`")
    updated_at = status.get("updated_at")
    if updated_at:
        lines.append(f"Updated (ET): `{_format_eastern_timestamp(updated_at)}`")
    return "\n\n".join(lines)


def _render_run_progress_status(status: dict[str, Any]) -> None:
    st.info(_run_progress_status_text(status))


def _workspace_display_name_for_run(run_dir: Path, workspaces_root: Path) -> str:
    try:
        rel = run_dir.resolve().relative_to(workspaces_root.resolve())
        workspace_name = rel.parts[0] if rel.parts else run_dir.parent.name
    except Exception:
        workspace_name = run_dir.parent.name
    if any(ch.isupper() for ch in workspace_name):
        return workspace_name
    return workspace_name.replace("_", " ").replace("-", " ").title()


def _render_run_query(run_dir: Path) -> None:
    query = _read_json_file(run_dir / "run_manifest.json").get("query")
    if not isinstance(query, dict):
        return
    text = str(query.get("text") or "")
    if not text.strip():
        return
    provider = str(query.get("provider") or "Literature").strip()
    st.subheader("Literature query")
    st.caption(f"Provider: {provider}")
    st.code(text, language=None)
    settings = query.get("settings")
    if isinstance(settings, dict) and settings:
        st.caption("Query settings")
        st.json(settings)


@st.cache_data(show_spinner=False, ttl=30)
def _admin_run_records(data_root: Path, users: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for user in users:
        profile_slug = str(user["profile_slug"])
        user_workspaces = user_workspaces_root(data_root, user)
        for workspace_dir in sorted(
            [p for p in user_workspaces.iterdir() if p.is_dir() and not _ignored_workspace_or_run_name(p.name)],
            key=lambda p: p.name.lower(),
        ):
            for run_dir in sorted(
                [p for p in workspace_dir.iterdir() if p.is_dir() and not _ignored_workspace_or_run_name(p.name)],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                status = read_run_status(run_dir / RUN_STATUS_FILE_NAME)
                manifest = _read_json_file(run_dir / "run_manifest.json")
                doi_count = _count_subdirs(run_dir / "reparsed")
                records.append({
                    "user_id": int(user["id"]),
                    "username": user["username"],
                    "email": user["email"],
                    "display_name": user["display_name"],
                    "role": user["role"],
                    "profile_slug": profile_slug,
                    "workspace": workspace_dir.name,
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "owner_username": manifest.get("owner_username") or user["username"],
                    "created_at": manifest.get("created_at") or "",
                    "state": status.get("state") or "unknown",
                    "stage": status.get("stage") or "",
                    "doi": status.get("doi") or "",
                    "updated_at": status.get("updated_at") or "",
                    "last_modified": _path_mtime_iso(run_dir),
                    "doi_count": doi_count,
                    "size_bytes": _dir_size_bytes(run_dir),
                })
    return records


@st.cache_data(show_spinner=False, ttl=30)
def _admin_user_stats(users: Sequence[Any], run_records: Sequence[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    records_by_user: dict[int, list[dict[str, Any]]] = {}
    for record in run_records:
        records_by_user.setdefault(int(record["user_id"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for user in users:
        user_id = int(user["id"])
        user_records = records_by_user.get(user_id, [])
        workspaces_root_for_user = user_workspaces_root(data_root, user)
        workspaces = [
            p for p in workspaces_root_for_user.iterdir()
            if p.is_dir() and not _ignored_workspace_or_run_name(p.name)
        ]
        state_counts: dict[str, int] = {}
        for record in user_records:
            state = str(record.get("state") or "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1
        last_activity = max(
            [str(r.get("updated_at") or r.get("last_modified") or "") for r in user_records],
            default="",
        )
        total_size = sum(int(r.get("size_bytes") or 0) for r in user_records)
        rows.append({
            "username": user["username"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "institution": user["institution"],
            "display_name": user["display_name"],
            "role": user["role"],
            "workspaces": len(workspaces),
            "runs": len(user_records),
            "done": state_counts.get("done", 0),
            "running": state_counts.get("running", 0) + state_counts.get("cancelling", 0),
            "errors": state_counts.get("error", 0),
            "interrupted": state_counts.get("interrupted", 0),
            "total_dois": sum(int(r.get("doi_count") or 0) for r in user_records),
            "disk": _format_bytes(total_size),
            "last_activity": last_activity,
        })
    return rows


def _render_admin_page(
    *,
    auth_store: AuthStore,
    data_root: Path,
    current_user: Any,
) -> bool:
    st.header("Admin")
    st.caption("Manage SALSA profiles, roles, run access, and per-user activity.")

    approval_notice = st.session_state.pop("admin_account_approval_notice", None)
    if approval_notice:
        st.success(approval_notice["message"])
        if not approval_notice["notification_sent"]:
            st.warning(
                "The account is approved, but the approval email could not be sent. "
                "Check the SMTP and SALSA_PUBLIC_URL settings."
            )

    pending_users = auth_store.list_pending_users()
    users = auth_store.list_users()
    # sqlite3.Row isn't hashable by Streamlit's cache hasher, so pass plain dicts
    # into the cached admin data functions instead of the Row objects themselves.
    users_for_cache = [dict(user) for user in users]
    run_records = _admin_run_records(data_root, users_for_cache)
    stats_rows = _admin_user_stats(users_for_cache, run_records, data_root)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Users", len(users))
    metric_cols[1].metric("Admins", sum(1 for user in users if user["role"] == "admin"))
    metric_cols[2].metric("Pending", len(pending_users))
    metric_cols[3].metric("Runs", len(run_records))
    metric_cols[4].metric("Storage", _format_bytes(sum(int(r.get("size_bytes") or 0) for r in run_records)))

    st.subheader("Account Requests")
    if pending_users:
        request_rows = [
            {
                "name": f"{user['first_name']} {user['last_name']}".strip(),
                "username": user["username"],
                "email": user["email"],
                "institution": user["institution"],
                "requested_at": user["created_at"],
            }
            for user in pending_users
        ]
        st.dataframe(pd.DataFrame(request_rows), use_container_width=True, hide_index=True)
        request_options = {
            f"{user['username']} ({user['email']})": user
            for user in pending_users
        }
        request_label = st.selectbox(
            "Pending request",
            options=list(request_options.keys()),
            key="admin_pending_user",
        )
        requested_user = request_options[request_label]
        approve_col, reject_col = st.columns(2)
        if approve_col.button(
            "Approve account",
            type="primary",
            key=f"admin_approve_{requested_user['id']}",
        ):
            try:
                approved_user = auth_store.approve_account(
                    int(requested_user["id"]),
                    int(current_user["id"]),
                )
            except (PermissionError, ValueError) as exc:
                st.error(str(exc))
            else:
                notification_sent = False
                try:
                    notification_sent = send_account_approved_email(approved_user)
                except Exception as exc:
                    print(f"[SALSA auth] Account approval email failed: {exc}")
                st.session_state["admin_account_approval_notice"] = {
                    "message": f"Approved {requested_user['username']}.",
                    "notification_sent": notification_sent,
                }
                st.rerun()
        if reject_col.button(
            "Reject request",
            key=f"admin_reject_{requested_user['id']}",
        ):
            try:
                auth_store.reject_account(
                    int(requested_user["id"]),
                    int(current_user["id"]),
                )
            except (PermissionError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.success(f"Rejected {requested_user['username']}.")
                st.rerun()
    else:
        st.info("No pending account requests.")

    st.subheader("User Statistics")
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No users found.")

    st.subheader("Role Management")
    selected_user = None
    user_options = {
        f"{user['username']} ({user['email']})": user
        for user in users
    }
    if user_options:
        selected_user_label = st.selectbox(
            "Profile",
            options=list(user_options.keys()),
            key="admin_role_user",
        )
        selected_user = user_options[selected_user_label]
        role_options = ["admin", "user"]
        selected_role = st.selectbox(
            "Role",
            options=role_options,
            index=role_options.index(selected_user["role"]) if selected_user["role"] in role_options else 1,
            key=f"admin_role_value_{selected_user['id']}",
        )
        if st.button("Update role", type="primary", key="admin_update_role"):
            try:
                auth_store.set_user_role(int(selected_user["id"]), selected_role)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(f"Updated {selected_user['username']} to {selected_role}.")
                st.rerun()

    if selected_user is None:
        return False

    st.subheader("Delete Account")
    st.warning(
        "Deleting an account permanently removes its login and sessions. "
        "Workspace files are preserved on the server."
    )
    delete_confirmation = st.text_input(
        f"Type {selected_user['username']} to confirm",
        key=f"admin_delete_confirmation_{selected_user['id']}",
    )
    deleting_self = int(selected_user["id"]) == int(current_user["id"])
    if deleting_self:
        st.caption("You cannot delete your own account.")
    if st.button(
        "Delete account",
        key=f"admin_delete_user_{selected_user['id']}",
        disabled=deleting_self or delete_confirmation != selected_user["username"],
    ):
        try:
            auth_store.delete_user(
                int(selected_user["id"]),
                int(current_user["id"]),
            )
        except (PermissionError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Deleted account {selected_user['username']}.")
            st.rerun()

    selected_run_records = [
        record for record in run_records
        if int(record["user_id"]) == int(selected_user["id"])
    ]
    st.subheader(f"Runs for {selected_user['username']}")
    if selected_run_records:
        runs_df = pd.DataFrame(selected_run_records)
        display_df = runs_df[[
            "workspace",
            "run_id",
            "state",
            "stage",
            "doi_count",
            "updated_at",
            "last_modified",
            "run_dir",
        ]].copy()
        display_df.insert(5, "disk", runs_df["size_bytes"].map(lambda n: _format_bytes(int(n or 0))))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        run_options = {
            f"{record['workspace']} / {record['run_id']} [{record['state']}]": record
            for record in selected_run_records
        }
        selected_run_label = st.selectbox(
            "Open run",
            options=list(run_options.keys()),
            key="admin_open_run",
        )
        if st.button("Load selected run", key="admin_load_run"):
            selected_record = run_options[selected_run_label]
            try:
                run_path = safe_child_path(
                    data_root,
                    "users",
                    str(selected_record["profile_slug"]),
                    "workspaces",
                    str(selected_record["workspace"]),
                    str(selected_record["run_id"]),
                )
                st.session_state["rr"] = discover_run(run_path)
                st.session_state["admin_loaded_run_owner"] = (
                    f"{selected_record['username']} / {selected_record['workspace']}"
                )
                st.session_state["admin_show_loaded_run"] = True
                st.success(f"Loaded {selected_record['run_id']} from {selected_record['username']}.")
            except Exception as exc:
                st.error(f"Could not load run: {exc}")
        if st.session_state.get("admin_show_loaded_run") and st.session_state.get("rr") is not None:
            owner = st.session_state.get("admin_loaded_run_owner", "")
            st.info(f"Showing loaded run below: **{owner}**")
            if st.button("Stay on admin dashboard only", key="admin_hide_loaded_run"):
                st.session_state["admin_show_loaded_run"] = False
                st.rerun()
            return True
    else:
        st.info("No runs found for this profile yet.")

    return False


def _render_guide_page() -> None:
    st.header("Guide")
    st.caption("A practical walkthrough for setting up SALSA runs, tuning settings, and correcting figure extraction results.")

    quick_start_tab, settings_tab, review_tab, interventions_tab = st.tabs(
        ["Start a run", "Settings presets", "Review results", "Figure interventions"]
    )

    with quick_start_tab:
        st.subheader("Start a new run")
        st.markdown(
            """
1. Choose or create a workspace in the sidebar.
2. In **Input Source**, choose **DOI(s)**, **PDF(s)**, or **Figure(s)**.
3. Provide the inputs:
   - **DOI(s):** paste one DOI per line.
   - **PDF(s):** upload PDF files. Optionally add one document ID per PDF so outputs use a readable identifier instead of the filename.
   - **Figure(s):** upload image files. Optionally add a figure set ID and one caption per uploaded figure.
4. In **Run configuration**, enter a **Run Name** if you want a stable name. Leave it blank to use a timestamp.
5. Click **Run pipeline**.

Run names may contain letters, numbers, underscores, periods, and dashes. A named run is easiest to find later from **Load an existing run**.
            """
        )
        st.subheader("Input choice tips")
        st.markdown(
            """
- Use **DOI(s)** when SALSA should download and parse papers for you.
- Use **PDF(s)** when you already have the papers locally or want to process documents that are not easy to fetch by DOI.
- Use **Figure(s)** when you only need figure data extraction from uploaded images. In this mode, table extraction and chemical composition extraction are disabled because there is no full paper context.
            """
        )

    with settings_tab:
        st.subheader("Use bundled presets")
        st.markdown(
            f"""
SALSA creates two starter settings presets:

- **{DEFAULT_SETTINGS_PRESET_NAME}:** conservative general-purpose settings. It extracts tables, figures, OCR figure data, and the completed dataset. Chemical composition plus figure and table filtering are off by default.
- **{EXAMPLE_AEM_SETTINGS_PRESET_NAME}:** an example configuration for AEM degradation workflows. It enables figure and table filtering and uses stronger LLM choices for selected steps.

Open **Load Settings Preset**, choose a preset, and click **Load**. The currently loaded preset appears at the bottom of that expander.
            """
        )

        st.subheader("Adjust cost and performance")
        st.markdown(
            """
Open **Create/Modify Settings Preset** to edit the active configuration before a run.

- Turn off stages you do not need. Each disabled LLM stage reduces cost and run time.
- Use **Chat GPT** models for managed API calls, or **Ollama** models when you want local inference and have Ollama running.
- Smaller models such as `gpt-4.1-mini` are typically cheaper and faster.
- Larger or more capable models can improve extraction quality on difficult papers, noisy tables, and nuanced filtering tasks.
- For Ollama-backed steps, increase **Ollama context window (num_ctx)** when papers are being truncated, at the cost of more memory and slower local inference.
- Keep **Use LLM to extract data from figures** off when you want the editable OCR/intervention workflow. Turn it on when you prefer direct LLM figure extraction and do not need crop, axis, or series edits.
            """
        )

        st.subheader("Save custom settings")
        st.markdown(
            """
1. Edit stage checkboxes, LLM providers/models, and instruction text in **Create/Modify Settings Preset**.
2. Add or edit **Data processing rules** and output column defaults if the completed dataset should follow a specific schema.
3. Enter a **Settings preset name**.
4. Click **Save**.

Saved presets are shared across your workspaces, so you can reload the same setup for future runs.
            """
        )

    with review_tab:
        st.subheader("Load and inspect existing runs")
        st.markdown(
            """
Use **Load an existing run** in the sidebar:

1. Pick the workspace to browse.
2. Select a **Run ID**.
3. Click **Load run**.

After loading, SALSA shows the run ID, workspace, cost summary, and a **Processed DOIs** table. Select a DOI, then switch between **Parsed text**, **Tables**, **Figures**, **Chemical Composition**, and **Dataset**.
            """
        )

        st.subheader("What to check")
        st.markdown(
            """
- **Tables:** inspect extracted table files and any LLM-generated dataset rows.
- **Figures:** inspect the original figure image, OCR-extracted data, and completed-dataset rows.
- **Dataset:** when **Agglomerate run datasets** was selected, review and download the run-wide LLM table/figure data and chemical-composition CSVs.
- **Cost summary:** compare stage-level usage when changing presets or models.
            """
        )

    with interventions_tab:
        st.subheader("Open figure interventions")
        st.markdown(
            """
1. Load a completed or partially completed run.
2. Select a DOI and open **Figures**.
3. For a figure, click **Load Interventions**.

Interventions are saved under the run, so you can load the same run later and continue from the saved corrections.
            """
        )

        st.subheader("Crop figures")
        st.markdown(
            """
Use **Crop Figure** when one image contains multiple charts or panels.

1. Draw rectangles around each chart.
2. Use **Move/resize rectangles** to refine the crop boxes.
3. Click **Save crops**.
4. Run OCR data extraction on the crops so each subfigure gets its own extracted dataset.
            """
        )

        st.subheader("Modify axes and titles")
        st.markdown(
            """
Open **Axis Configuration** to correct axis titles, axis type, and calibration.

- Edit X and Y axis titles and click **Save axis titles**.
- For calibration, click Xmin, Xmax, Ymin, and Ymax on the image.
- Enter the real axis values, choose linear or log10 axes, then click **Save calibration override from clicks**.
            """
        )

        st.subheader("Modify series")
        st.markdown(
            """
Open **Series Management** to correct which plotted marks belong to each data series.

- Add, rename, or remove series as needed.
- Assign or correct detected masks/points for each series.
- Save the series mapping before rerunning extraction.
            """
        )

        st.subheader("Rerun extraction after edits")
        st.markdown(
            """
- Click **Rerun OCR data extraction** after crop, axis, or series changes. This reapplies overrides and regenerates OCR-derived figure data.
- Click **Run LLM dataset generation** after OCR data looks correct. This regenerates completed-dataset rows for the selected figure or subfigure.

If the LLM rerun is unavailable, check that **Complete dataset** has an LLM model configured in the sidebar and that an API key is present for Chat GPT-backed steps.
            """
        )


def _render_public_setup_page() -> None:
    st.header("Setup instructions")
    setup_path = PUBLIC_DOCS_DIR / "setup.md"
    try:
        setup_markdown = setup_path.read_text(encoding="utf-8")
    except OSError:
        st.error("The setup instructions are temporarily unavailable.")
        return
    st.markdown(setup_markdown)


def _render_public_docs_navigation(active_page: str) -> None:
    with st.sidebar:
        st.header("SALSA documentation")
        st.caption("Public documentation — no account required")
        if active_page != "guide" and st.button(
            "User guide", use_container_width=True, key="public_nav_guide"
        ):
            _set_query_param("page", "guide")
            st.rerun()
        if active_page != "setup" and st.button(
            "Setup instructions", use_container_width=True, key="public_nav_setup"
        ):
            _set_query_param("page", "setup")
            st.rerun()
        st.divider()
        if st.button("Sign in / Open SALSA", type="primary", use_container_width=True):
            _clear_query_param("page")
            st.rerun()


def _render_public_page(page: str) -> None:
    _render_public_docs_navigation(page)
    if page == "guide":
        _render_guide_page()
    else:
        _render_public_setup_page()


def clear_processing_llm_env_if_disabled() -> None:
    """Prevent stale processing-model env vars from affecting in-process reruns."""
    if st.session_state.get("use_separate_processing_llm"):
        return
    for key in LEGACY_PROCESSING_LLM_ENV_VARS:
        os.environ.pop(key, None)


_CI_COLUMN_NAME = "column name"
_CI_DEFAULT_VALUE = "default value"
_CI_COLUMN_DEFAULTS_SOURCE_KEY = "ci_column_defaults_table_source"
_CI_COLUMN_DEFAULTS_VERSION_KEY = "ci_column_defaults_table_version"
_CI_CHEM_COLUMN_DEFAULTS_SOURCE_KEY = "ci_chemical_column_defaults_table_source"
_CI_CHEM_COLUMN_DEFAULTS_VERSION_KEY = "ci_chemical_column_defaults_table_version"
_CI_PENDING_APPLY_KEY = "ci_pending_apply"
_CI_FLASH_MESSAGE_KEY = "ci_flash_message"
_CI_EDITOR_KEYS = (
    "ci_table_extraction",
    "ci_figure_filter",
    "ci_table_filter",
    "ci_chemical_composition",
    "ci_figure_data",
    "ci_data_processing",
    _CI_COLUMN_DEFAULTS_SOURCE_KEY,
    _CI_COLUMN_DEFAULTS_VERSION_KEY,
    _CI_CHEM_COLUMN_DEFAULTS_SOURCE_KEY,
    _CI_CHEM_COLUMN_DEFAULTS_VERSION_KEY,
)


def _column_defaults_table_from_instructions(
    ci: dict[str, Any],
    *,
    columns_key: str = "output_columns",
    defaults_key: str = "missing_value_defaults",
) -> pd.DataFrame:
    output_columns = ci.get(columns_key, [])
    if isinstance(output_columns, str):
        output_columns = [c.strip() for c in output_columns.splitlines() if c.strip()]

    missing_defaults = ci.get(defaults_key, {})
    if isinstance(missing_defaults, str):
        try:
            missing_defaults = json.loads(missing_defaults)
        except Exception:
            missing_defaults = {}

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_column in output_columns or []:
        column_name = str(raw_column).strip()
        if not column_name or column_name in seen:
            continue
        seen.add(column_name)
        default_value = ""
        if isinstance(missing_defaults, dict):
            raw_default = missing_defaults.get(column_name, "")
            default_value = "" if raw_default is None else str(raw_default)
        rows.append(
            {
                _CI_COLUMN_NAME: column_name,
                _CI_DEFAULT_VALUE: default_value,
            }
        )

    if isinstance(missing_defaults, dict):
        for raw_column, raw_default in missing_defaults.items():
            column_name = str(raw_column).strip()
            if not column_name or column_name in seen:
                continue
            seen.add(column_name)
            rows.append(
                {
                    _CI_COLUMN_NAME: column_name,
                    _CI_DEFAULT_VALUE: "" if raw_default is None else str(raw_default),
                }
            )

    return pd.DataFrame(rows, columns=[_CI_COLUMN_NAME, _CI_DEFAULT_VALUE])


def _normalize_column_defaults_table(
    table_value: Any,
) -> tuple[list[str], dict[str, str]]:
    if isinstance(table_value, pd.DataFrame):
        records = table_value.to_dict(orient="records")
    elif isinstance(table_value, list):
        records = table_value
    else:
        records = []

    output_columns: list[str] = []
    missing_defaults: dict[str, str] = {}
    seen: set[str] = set()

    for record in records:
        raw_column = record.get(_CI_COLUMN_NAME, "")
        raw_default = record.get(_CI_DEFAULT_VALUE, "")

        column_name = "" if pd.isna(raw_column) else str(raw_column).strip()
        if not column_name:
            continue

        if column_name not in seen:
            output_columns.append(column_name)
            seen.add(column_name)

        default_value = "" if pd.isna(raw_default) else str(raw_default).strip()
        if default_value:
            missing_defaults[column_name] = default_value
        else:
            missing_defaults.pop(column_name, None)

    return output_columns, missing_defaults


def _custom_instructions_from_editor_state(
    column_defaults_table: Any,
    chemical_column_defaults_table: Any,
) -> dict[str, Any]:
    output_columns, missing_defaults = _normalize_column_defaults_table(column_defaults_table)
    chemical_columns, chemical_defaults = _normalize_column_defaults_table(
        chemical_column_defaults_table
    )
    return {
        "table_extraction_rules": st.session_state.get("ci_table_extraction", ""),
        "figure_filter_criteria": st.session_state.get("ci_figure_filter", ""),
        "table_filter_criteria": st.session_state.get("ci_table_filter", ""),
        "chemical_composition_rules": st.session_state.get("ci_chemical_composition", ""),
        "chemical_composition_columns": chemical_columns,
        "chemical_composition_defaults": chemical_defaults,
        "figure_data_extraction_rules": st.session_state.get("ci_figure_data", ""),
        "data_processing_rules": st.session_state.get("ci_data_processing", ""),
        "output_columns": output_columns,
        "missing_value_defaults": missing_defaults,
    }


def _ci_column_defaults_widget_key() -> str:
    version = int(st.session_state.get(_CI_COLUMN_DEFAULTS_VERSION_KEY, 0))
    return f"ci_column_defaults_table_widget_{version}"


def _ci_chemical_column_defaults_widget_key() -> str:
    version = int(st.session_state.get(_CI_CHEM_COLUMN_DEFAULTS_VERSION_KEY, 0))
    return f"ci_chemical_column_defaults_table_widget_{version}"


def _apply_custom_instruction_state(
    ci: dict[str, Any],
    *,
    preset_name: Optional[str],
) -> None:
    st.session_state["custom_instructions"] = ci
    st.session_state["current_preset_name"] = preset_name
    st.session_state["ci_table_extraction"] = ci.get("table_extraction_rules", "")
    st.session_state["ci_figure_filter"] = ci.get("figure_filter_criteria", "")
    st.session_state["ci_table_filter"] = ci.get("table_filter_criteria", "")
    st.session_state["ci_chemical_composition"] = ci.get("chemical_composition_rules", "")
    composition_columns = ci.get("chemical_composition_columns")
    if composition_columns is None:
        composition_columns = default_instructions().get(
            "chemical_composition_columns", []
        )
    st.session_state["ci_figure_data"] = ci.get("figure_data_extraction_rules", "")
    st.session_state["ci_data_processing"] = ci.get("data_processing_rules", "")
    if _CI_COLUMN_DEFAULTS_VERSION_KEY not in st.session_state:
        st.session_state[_CI_COLUMN_DEFAULTS_VERSION_KEY] = 0
    else:
        st.session_state[_CI_COLUMN_DEFAULTS_VERSION_KEY] += 1
    if _CI_CHEM_COLUMN_DEFAULTS_VERSION_KEY not in st.session_state:
        st.session_state[_CI_CHEM_COLUMN_DEFAULTS_VERSION_KEY] = 0
    else:
        st.session_state[_CI_CHEM_COLUMN_DEFAULTS_VERSION_KEY] += 1
    st.session_state[_CI_COLUMN_DEFAULTS_SOURCE_KEY] = _column_defaults_table_from_instructions(ci)
    chemical_ci = dict(ci)
    chemical_ci["chemical_composition_columns"] = composition_columns
    st.session_state[_CI_CHEM_COLUMN_DEFAULTS_SOURCE_KEY] = (
        _column_defaults_table_from_instructions(
            chemical_ci,
            columns_key="chemical_composition_columns",
            defaults_key="chemical_composition_defaults",
        )
    )
    st.session_state["ci_new_preset_name"] = preset_name or ""


def _queue_custom_instruction_state(
    ci: dict[str, Any],
    *,
    preset_name: Optional[str],
    flash_message: Optional[str] = None,
) -> None:
    st.session_state[_CI_PENDING_APPLY_KEY] = {
        "ci": ci,
        "preset_name": preset_name,
    }
    if flash_message:
        st.session_state[_CI_FLASH_MESSAGE_KEY] = flash_message


def _apply_pending_custom_instruction_state() -> Optional[str]:
    pending = st.session_state.pop(_CI_PENDING_APPLY_KEY, None)
    if isinstance(pending, dict):
        ci = pending.get("ci")
        if isinstance(ci, dict):
            _apply_custom_instruction_state(
                ci,
                preset_name=pending.get("preset_name"),
            )
    return st.session_state.pop(_CI_FLASH_MESSAGE_KEY, None)


SETTINGS_PRESETS_DIR_NAME = "settings_presets"
DEFAULT_SETTINGS_PRESET_NAME = "default"
EXAMPLE_AEM_SETTINGS_PRESET_NAME = "example_aem_degradation"
_SETTINGS_PENDING_APPLY_KEY = "settings_pending_apply"
_SETTINGS_FLASH_MESSAGE_KEY = "settings_flash_message"

_STAGE_STATE_KEYS = {
    "exclude_full_text": "settings_exclude_full_text",
    "run_extract_tables": "settings_run_extract_tables",
    "run_extract_figures": "settings_run_extract_figures",
    "run_pdf_page_ocr": "settings_run_pdf_page_ocr",
    "run_filter_tables": "settings_run_filter_tables",
    "run_filter_figures": "settings_run_filter_figures",
    "run_plextract": "settings_run_plextract",
    "bypass_ocr_with_llm": "settings_bypass_ocr_with_llm",
    "run_chemical_composition": "settings_run_chemical_composition",
    "run_complete_dataset": "settings_run_complete_dataset",
    "run_aggregation": "settings_run_aggregation",
}

_DEFAULT_STAGE_SETTINGS = {
    "exclude_full_text": True,
    "run_extract_tables": True,
    "run_extract_figures": True,
    "run_pdf_page_ocr": False,
    "run_filter_tables": False,
    "run_filter_figures": False,
    "run_plextract": True,
    "bypass_ocr_with_llm": False,
    "run_chemical_composition": False,
    "run_complete_dataset": True,
    "run_aggregation": False,
}

_DEFAULT_OUTPUT_COLUMNS = [
    "Sample Name",
    "independent variable",
    "independent value",
    "independent unit",
    "dependant variable",
    "dependent value",
    "dependent unit",
    "DOI",
    "source_kind",
    "source_name",
]

_LEGACY_DEFAULT_OUTPUT_COLUMNS = [
    "Sample Name",
    "independent variable",
    "independent value",
    "independent unit",
    "property",
    "value",
    "unit",
    "DOI",
    "source_kind",
    "source_name",
]

_AEM_EXAMPLE_STAGE_SETTINGS = {
    **_DEFAULT_STAGE_SETTINGS,
    "run_filter_figures": True,
    "run_filter_tables": True,
    "run_chemical_composition": True,
}

DEFAULT_SETTINGS_LLM_MODEL = "gpt-4.1-mini"
AEM_EXAMPLE_LLM_MODELS = {
    LLM_STEP_EXTRACT_TABLES: "gpt-4.1",
    LLM_STEP_CHEM_RELEVANT: "gpt-4.1",
    LLM_STEP_CHEM_COMPOSITION: "gpt-5.5",
    LLM_STEP_FILTER_FIGURES: "gpt-4.1-mini",
    LLM_STEP_FILTER_TABLES: "gpt-4.1",
    LLM_STEP_FIGURE_DATA: "gpt-5.5",
    LLM_STEP_COMPLETE_DATASET: "gpt-4.1",
}


def _default_settings_instructions() -> dict[str, Any]:
    return {
        "table_extraction_rules": DEFAULT_TABLE_EXTRACTION_RULES,
        "figure_filter_criteria": "",
        "table_filter_criteria": "",
        "chemical_composition_rules": "",
        "figure_data_extraction_rules": "",
        "data_processing_rules": "",
        "output_columns": list(_DEFAULT_OUTPUT_COLUMNS),
        "missing_value_defaults": {},
    }


def _settings_presets_dir(workspaces_root: Path) -> Path:
    d = Path(workspaces_root) / SETTINGS_PRESETS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_settings_presets(workspaces_root: Path) -> list[str]:
    return sorted(
        p.stem for p in _settings_presets_dir(workspaces_root).glob("*.json") if p.is_file()
    )


def _save_settings_preset(workspaces_root: Path, name: str, settings: dict[str, Any]) -> Path:
    path = _settings_presets_dir(workspaces_root) / f"{name}.json"
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return path


def _load_settings_preset(workspaces_root: Path, name: str) -> dict[str, Any]:
    path = _settings_presets_dir(workspaces_root) / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return _normalize_settings_preset(data)


def _settings_preset_exists(workspaces_root: Path, name: str) -> bool:
    return (_settings_presets_dir(workspaces_root) / f"{name}.json").is_file()


def _settings_default_llm_steps() -> dict[str, dict[str, str]]:
    return {
        step_key: {"provider": "Chat GPT", "model": DEFAULT_SETTINGS_LLM_MODEL}
        for step_key in _STEP_LLM_RENDER_ORDER
    }


def _aem_example_llm_steps() -> dict[str, dict[str, str]]:
    return {
        step_key: {
            "provider": "Chat GPT",
            "model": AEM_EXAMPLE_LLM_MODELS.get(step_key, DEFAULT_SETTINGS_LLM_MODEL),
        }
        for step_key in _STEP_LLM_RENDER_ORDER
    }


def _make_settings_preset(
    *,
    stages: dict[str, bool],
    custom_instructions: dict[str, Any],
    llm_steps: Optional[dict[str, dict[str, str]]] = None,
    include_session_llms: bool = False,
    openai_api_key: str = "",
    elsevier_api_key: str = "",
    wiley_tdm_token: str = "",
    springer_nature_api_key: str = "",
    rsc_tdm_token: str = "",
    ncbi_api_key: str = "",
    ieee_api_key: str = "",
    wos_api_key: str = "",
    openalex_api_key: str = "",
    scopus_inst_token: str = "",
) -> dict[str, Any]:
    llm_steps = dict(llm_steps or _settings_default_llm_steps())
    if include_session_llms:
        for step_key in _STEP_LLM_RENDER_ORDER:
            llm_steps[step_key] = {
                "provider": _step_provider_label(step_key),
                "model": _step_model_value(step_key),
            }

    return {
        "version": 1,
        "openai_api_key": openai_api_key,
        "elsevier_api_key": elsevier_api_key,
        "wiley_tdm_token": wiley_tdm_token,
        "springer_nature_api_key": springer_nature_api_key,
        "rsc_tdm_token": rsc_tdm_token,
        "ncbi_api_key": ncbi_api_key,
        "ieee_api_key": ieee_api_key,
        "wos_api_key": wos_api_key,
        "openalex_api_key": openalex_api_key,
        "scopus_inst_token": scopus_inst_token,
        "openai_base_url": st.session_state.get("openai_base_url", "https://api.openai.com/v1")
        if include_session_llms
        else "https://api.openai.com/v1",
        "ollama_host": st.session_state.get("ollama_host", default_ollama_host)
        if include_session_llms and "default_ollama_host" in globals()
        else "",
        "ollama_models_dir": st.session_state.get("ollama_models_dir", str(default_ollama_dir))
        if include_session_llms and "default_ollama_dir" in globals()
        else "",
        "ollama_num_ctx": int(st.session_state.get("ollama_num_ctx", 16384))
        if include_session_llms
        else 16384,
        "stages": dict(stages),
        "llm_steps": llm_steps,
        "custom_instructions": dict(custom_instructions),
    }


def _normalize_settings_preset(data: dict[str, Any]) -> dict[str, Any]:
    default = _make_settings_preset(
        stages=_DEFAULT_STAGE_SETTINGS,
        custom_instructions=_default_settings_instructions(),
    )
    if not isinstance(data, dict):
        return default

    stages = data.get("stages", {})
    if isinstance(stages, dict):
        default["stages"].update(
            {
                key: bool(stages.get(key, default["stages"][key]))
                for key in _DEFAULT_STAGE_SETTINGS
            }
        )

    llm_steps = data.get("llm_steps", {})
    if isinstance(llm_steps, dict):
        for step_key in _STEP_LLM_RENDER_ORDER:
            raw = llm_steps.get(step_key, {})
            if isinstance(raw, dict):
                provider = str(raw.get("provider") or "Chat GPT")
                default["llm_steps"][step_key] = {
                    "provider": provider if provider in {"Chat GPT", "Ollama"} else "Chat GPT",
                    "model": str(raw.get("model") or "").strip(),
                }

    ci = data.get("custom_instructions", {})
    if isinstance(ci, dict):
        merged_ci = _default_settings_instructions()
        merged_ci.update(ci)
        default["custom_instructions"] = merged_ci

    for key in (
        "openai_api_key",
        "elsevier_api_key",
        "wiley_tdm_token",
        "springer_nature_api_key",
        "rsc_tdm_token",
        "ncbi_api_key",
        "ieee_api_key",
        "wos_api_key",
        "openalex_api_key",
        "scopus_inst_token",
        "openai_base_url",
        "ollama_host",
        "ollama_models_dir",
    ):
        if key in data:
            default[key] = str(data.get(key) or "")
    if not default["openai_base_url"]:
        default["openai_base_url"] = "https://api.openai.com/v1"
    try:
        default["ollama_num_ctx"] = int(data.get("ollama_num_ctx", default["ollama_num_ctx"]))
    except Exception:
        pass

    return default


def _current_stage_settings_from_state() -> dict[str, bool]:
    return {
        stage_name: bool(st.session_state.get(state_key, default_value))
        for stage_name, state_key in _STAGE_STATE_KEYS.items()
        for default_value in [_DEFAULT_STAGE_SETTINGS[stage_name]]
    }


def _preserve_settings_editor_state() -> None:
    """Keep conditionally rendered settings widgets from being garbage-collected."""
    keys = {
        *_STAGE_STATE_KEYS.values(),
        *_CI_EDITOR_KEYS,
        "openai_api_key",
        "wos_api_key",
        "openalex_api_key",
        "scopus_inst_token",
        "elsevier_api_key",
        "wiley_tdm_token",
        "springer_nature_api_key",
        "rsc_tdm_token",
        "ncbi_api_key",
        "ieee_api_key",
        "settings_editor_section",
    }
    for step_key in _STEP_LLM_RENDER_ORDER:
        provider_key = _step_provider_key(step_key)
        keys.update(
            {
                provider_key,
                f"{provider_key}_previous",
                _step_model_key(step_key),
                _step_provider_model_memory_key(step_key, "Chat GPT"),
                _step_provider_model_memory_key(step_key, "Ollama"),
                f"{_step_provider_model_memory_key(step_key, 'Ollama')}_manual",
            }
        )
    for key in keys:
        if key in st.session_state:
            # Reassigning a widget-backed key detaches it from Streamlit's
            # end-of-run widget cleanup when its settings section is hidden.
            st.session_state[key] = st.session_state[key]


def _current_settings_preset_from_state(current_ci: dict[str, Any]) -> dict[str, Any]:
    _ensure_shared_llm_settings_in_state()
    return _make_settings_preset(
        stages=_current_stage_settings_from_state(),
        custom_instructions=current_ci,
        include_session_llms=True,
        openai_api_key=str(st.session_state.get("openai_api_key", "")).strip(),
        elsevier_api_key=str(st.session_state.get("elsevier_api_key", "")).strip(),
        wiley_tdm_token=str(st.session_state.get("wiley_tdm_token", "")).strip(),
        springer_nature_api_key=str(st.session_state.get("springer_nature_api_key", "")).strip(),
        rsc_tdm_token=str(st.session_state.get("rsc_tdm_token", "")).strip(),
        ncbi_api_key=str(st.session_state.get("ncbi_api_key", "")).strip(),
        ieee_api_key=str(st.session_state.get("ieee_api_key", "")).strip(),
        wos_api_key=str(st.session_state.get("wos_api_key", "")).strip(),
        openalex_api_key=str(st.session_state.get("openalex_api_key", "")).strip(),
        scopus_inst_token=str(st.session_state.get("scopus_inst_token", "")).strip(),
    )


def _apply_settings_preset_state(settings: dict[str, Any], *, preset_name: Optional[str]) -> None:
    settings = _normalize_settings_preset(settings)
    _ensure_shared_llm_settings_in_state()
    for stage_name, state_key in _STAGE_STATE_KEYS.items():
        st.session_state[state_key] = bool(
            settings.get("stages", {}).get(stage_name, _DEFAULT_STAGE_SETTINGS[stage_name])
        )

    st.session_state["openai_api_key"] = str(settings.get("openai_api_key") or "")
    st.session_state["elsevier_api_key"] = str(settings.get("elsevier_api_key") or "")
    st.session_state["wiley_tdm_token"] = str(settings.get("wiley_tdm_token") or "")
    st.session_state["springer_nature_api_key"] = str(
        settings.get("springer_nature_api_key") or ""
    )
    st.session_state["rsc_tdm_token"] = str(settings.get("rsc_tdm_token") or "")
    st.session_state["ncbi_api_key"] = str(settings.get("ncbi_api_key") or "")
    st.session_state["ieee_api_key"] = str(settings.get("ieee_api_key") or "")
    st.session_state["wos_api_key"] = str(settings.get("wos_api_key") or "")
    st.session_state["openalex_api_key"] = str(settings.get("openalex_api_key") or "")
    st.session_state["scopus_inst_token"] = str(settings.get("scopus_inst_token") or "")
    st.session_state["openai_base_url"] = str(
        settings.get("openai_base_url") or "https://api.openai.com/v1"
    )
    if settings.get("ollama_host"):
        st.session_state["ollama_host"] = str(settings.get("ollama_host") or "")
    if settings.get("ollama_models_dir"):
        st.session_state["ollama_models_dir"] = str(settings.get("ollama_models_dir") or "")
    st.session_state["ollama_num_ctx"] = int(settings.get("ollama_num_ctx", 16384))

    for step_key, config in settings.get("llm_steps", {}).items():
        if step_key not in _STEP_LLM_RENDER_ORDER or not isinstance(config, dict):
            continue
        provider = str(config.get("provider") or "Chat GPT")
        provider_key = _step_provider_key(step_key)
        st.session_state[provider_key] = provider
        st.session_state[f"{provider_key}_previous"] = provider
        model = str(config.get("model") or "").strip()
        provider_model_key = _step_provider_model_memory_key(step_key, provider)
        if model:
            st.session_state[_step_model_key(step_key)] = model
            st.session_state[provider_model_key] = model
            if provider == "Ollama":
                st.session_state[f"{provider_model_key}_manual"] = model
        else:
            st.session_state.pop(_step_model_key(step_key), None)
            st.session_state.pop(provider_model_key, None)
            if provider == "Ollama":
                st.session_state.pop(f"{provider_model_key}_manual", None)

    _apply_custom_instruction_state(
        settings.get("custom_instructions", _default_settings_instructions()),
        preset_name=preset_name,
    )
    st.session_state["current_settings_preset_name"] = preset_name
    st.session_state["settings_new_preset_name"] = preset_name or ""


def _queue_settings_preset_state(
    settings: dict[str, Any],
    *,
    preset_name: Optional[str],
    flash_message: Optional[str] = None,
) -> None:
    st.session_state[_SETTINGS_PENDING_APPLY_KEY] = {
        "settings": settings,
        "preset_name": preset_name,
    }
    if flash_message:
        st.session_state[_SETTINGS_FLASH_MESSAGE_KEY] = flash_message


def _apply_pending_settings_preset_state() -> Optional[str]:
    pending = st.session_state.pop(_SETTINGS_PENDING_APPLY_KEY, None)
    if isinstance(pending, dict):
        settings = pending.get("settings")
        if isinstance(settings, dict):
            _apply_settings_preset_state(
                settings,
                preset_name=pending.get("preset_name"),
            )
    return st.session_state.pop(_SETTINGS_FLASH_MESSAGE_KEY, None)


def _bootstrap_settings_presets_for_workspaces(workspaces_root: Path) -> None:
    if not _settings_preset_exists(workspaces_root, DEFAULT_SETTINGS_PRESET_NAME):
        _save_settings_preset(
            workspaces_root,
            DEFAULT_SETTINGS_PRESET_NAME,
            _make_settings_preset(
                stages=_DEFAULT_STAGE_SETTINGS,
                custom_instructions=_default_settings_instructions(),
            ),
        )
    else:
        default_settings = _load_settings_preset(
            workspaces_root,
            DEFAULT_SETTINGS_PRESET_NAME,
        )
        default_stages = default_settings.setdefault("stages", {})
        default_custom_instructions = default_settings.setdefault(
            "custom_instructions",
            {},
        )
        default_columns = default_custom_instructions.get("output_columns")
        needs_update = False
        if default_stages.get("run_chemical_composition") is not False:
            default_stages["run_chemical_composition"] = False
            needs_update = True
        if not default_columns or default_columns == _LEGACY_DEFAULT_OUTPUT_COLUMNS:
            default_custom_instructions["output_columns"] = list(_DEFAULT_OUTPUT_COLUMNS)
            needs_update = True
        if needs_update:
            _save_settings_preset(
                workspaces_root,
                DEFAULT_SETTINGS_PRESET_NAME,
                default_settings,
            )
    if not _settings_preset_exists(workspaces_root, EXAMPLE_AEM_SETTINGS_PRESET_NAME):
        _save_settings_preset(
            workspaces_root,
            EXAMPLE_AEM_SETTINGS_PRESET_NAME,
            _make_settings_preset(
                stages=_AEM_EXAMPLE_STAGE_SETTINGS,
                custom_instructions=default_instructions(),
                llm_steps=_aem_example_llm_steps(),
            ),
        )


@st.cache_resource(show_spinner=False)
def _bootstrap_settings_presets_for_profile_cached(data_root_str: str, profile_slug: str) -> bool:
    # Cached per (data_root, profile_slug) for the life of the process: bootstrapping
    # walks/reads/writes that profile's workspace on disk, so without caching this
    # redid that I/O for every registered user on every Streamlit rerun by anyone.
    profile_workspaces_root = Path(data_root_str) / "users" / profile_slug / "workspaces"
    profile_workspaces_root.mkdir(parents=True, exist_ok=True)
    _bootstrap_settings_presets_for_workspaces(profile_workspaces_root)
    return True


def _bootstrap_settings_presets_for_profiles(data_root: Path, users: Sequence[Any]) -> None:
    data_root_str = str(data_root)
    for user in users:
        try:
            _bootstrap_settings_presets_for_profile_cached(data_root_str, str(user["profile_slug"]))
        except Exception:
            continue


def _keep_create_settings_preset_open() -> None:
    st.session_state["create_settings_preset_expanded"] = True


def _keep_profile_expanded() -> None:
    st.session_state["profile_expanded"] = True


def _keep_load_settings_preset_open() -> None:
    st.session_state["load_settings_preset_expanded"] = True


def _settings_checkbox(label: str, state_key: str, default: bool, **kwargs: Any) -> bool:
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    kwargs.setdefault("on_change", _keep_create_settings_preset_open)
    return bool(st.checkbox(label, key=state_key, **kwargs))


def _render_instruction_text_area(label: str, state_key: str, height: int = 120) -> None:
    with st.expander(label, expanded=False):
        st.text_area(
            label,
            height=height,
            key=state_key,
            label_visibility="collapsed",
            on_change=_keep_create_settings_preset_open,
        )


# SET OLLAMA DIR AND HOST (computed outside expander for reuse)
default_ollama_dir = _default_ollama_models_dir()
default_ollama_host = _default_ollama_host()

_openai_models = [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]

_AUTH_SESSION_COOKIE_NAME = "salsa_session"
_AUTH_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30
MAX_INPUT_ITEMS = DEFAULT_INPUT_ITEM_LIMIT


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _query_param(name: str) -> str:
    try:
        value = st.query_params.get(name, "")
    except Exception:
        params = st.experimental_get_query_params()
        value = params.get(name, [""])
        if isinstance(value, list):
            value = value[0] if value else ""
    if isinstance(value, list):
        return str(value[0] if value else "")
    return str(value or "")


def _set_query_param(name: str, value: str) -> None:
    try:
        st.query_params[name] = value
    except Exception:
        params = st.experimental_get_query_params()
        params[name] = value
        st.experimental_set_query_params(**params)


def _clear_query_param(name: str) -> None:
    try:
        if name in st.query_params:
            del st.query_params[name]
    except Exception:
        params = st.experimental_get_query_params()
        params.pop(name, None)
        st.experimental_set_query_params(**params)


def _cookie_from_header(cookie_header: str, name: str) -> str:
    if not cookie_header:
        return ""
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return ""
    morsel = cookie.get(name)
    return morsel.value if morsel is not None else ""


def _request_cookie_value(name: str) -> str:
    context = getattr(st, "context", None)
    if context is not None:
        try:
            cookies = getattr(context, "cookies", None)
            if cookies and cookies.get(name):
                return str(cookies.get(name) or "")
        except Exception:
            pass
        try:
            headers = getattr(context, "headers", None)
            if headers:
                value = _cookie_from_header(str(headers.get("cookie") or headers.get("Cookie") or ""), name)
                if value:
                    return value
        except Exception:
            pass

    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers

        headers = _get_websocket_headers() or {}
        return _cookie_from_header(str(headers.get("Cookie") or headers.get("cookie") or ""), name)
    except Exception:
        return ""


def _render_cookie_script(*, token: str = "", clear: bool = False) -> None:
    cookie_name = json.dumps(_AUTH_SESSION_COOKIE_NAME)
    cookie_writer = """
        function writeCookie(value) {
            try {
                window.parent.document.cookie = value;
            } catch (err) {}
            try {
                document.cookie = value;
            } catch (err) {}
        }
    """
    if clear:
        script = f"""
        <script>
        {cookie_writer}
        writeCookie({cookie_name} + "=; Max-Age=0; Path=/; SameSite=Lax");
        window.setTimeout(function() {{
            window.parent.location.reload();
        }}, 50);
        </script>
        """
    else:
        cookie_value = json.dumps(token)
        max_age = int(_AUTH_SESSION_MAX_AGE_SECONDS)
        script = f"""
        <script>
        {cookie_writer}
        writeCookie({cookie_name} + "=" + encodeURIComponent({cookie_value}) + "; Max-Age={max_age}; Path=/; SameSite=Lax");
        </script>
        """
    components.html(script, height=1, width=1)


def _render_pending_auth_cookie_scripts() -> None:
    token = st.session_state.pop("pending_auth_cookie_token", "")
    clear_cookie = bool(st.session_state.pop("pending_clear_auth_cookie", False))
    if clear_cookie:
        if _request_cookie_value(_AUTH_SESSION_COOKIE_NAME):
            _render_cookie_script(clear=True)
    elif token:
        _render_cookie_script(token=str(token))


def _set_auth_session(auth_store: AuthStore, user: Any) -> None:
    token = auth_store.create_auth_session_token(int(user["id"]))
    st.session_state["auth_user_id"] = int(user["id"])
    st.session_state["auth_session_token"] = token
    st.session_state["pending_auth_cookie_token"] = token


def _clear_auth_session(auth_store: Optional[AuthStore] = None) -> None:
    token = str(
        st.session_state.get("auth_session_token")
        or _request_cookie_value(_AUTH_SESSION_COOKIE_NAME)
        or ""
    )
    if auth_store is not None and token:
        auth_store.revoke_auth_session_token(token)
    st.session_state["pending_clear_auth_cookie"] = True
    for key in (
        "auth_user_id",
        "auth_session_token",
        "rr",
        "workspace_name",
        "load_workspace",
        "load_run_id",
        "pending_auth_cookie_token",
        _ACTIVE_PIPELINE_RUN_KEY if "_ACTIVE_PIPELINE_RUN_KEY" in globals() else "",
    ):
        if key:
            st.session_state.pop(key, None)


def _current_user(auth_store: AuthStore):
    user_id = st.session_state.get("auth_user_id")
    token = str(
        st.session_state.get("auth_session_token")
        or _request_cookie_value(_AUTH_SESSION_COOKIE_NAME)
        or ""
    )

    if user_id:
        user = auth_store.get_user(int(user_id))
    elif token:
        user = auth_store.user_from_auth_session_token(token)
        if user is not None:
            st.session_state["auth_user_id"] = int(user["id"])
            st.session_state["auth_session_token"] = token
    else:
        user = None

    if user is None and user_id:
        _clear_auth_session()
    elif user is None and token:
        _clear_auth_session()
        _render_cookie_script(clear=True)
    return user


def _render_initial_admin_setup(auth_store: AuthStore) -> None:
    st.subheader("Create administrator account")
    st.info(
        "Welcome to SALSA. No administrator account exists yet. Create one to "
        "finish setting up this installation. The administrator approves new "
        "account requests and manages users."
    )
    with st.form("initial_admin_form"):
        username = st.text_input("Username", key="initial_admin_username")
        email = st.text_input("Email", key="initial_admin_email")
        first_name = st.text_input("First name", key="initial_admin_first_name")
        last_name = st.text_input("Last name", key="initial_admin_last_name")
        institution = st.text_input("Institution", key="initial_admin_institution")
        password = st.text_input("Password", type="password", key="initial_admin_pw")
        password_again = st.text_input(
            "Confirm password",
            type="password",
            key="initial_admin_pw_again",
        )
        submitted = st.form_submit_button("Create administrator account", type="primary")
    if submitted:
        try:
            if password != password_again:
                raise ValueError("Passwords do not match.")
            user = auth_store.create_initial_admin(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                institution=institution,
                password=password,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            _set_auth_session(auth_store, user)
            st.rerun()


def _render_auth_gate(auth_store: AuthStore):
    user = _current_user(auth_store)
    if user is not None:
        return user

    if auth_store.needs_initial_admin():
        _render_initial_admin_setup(auth_store)
        st.stop()

    st.subheader("Sign in")
    st.info(
        "At this time, SALSA is available only to members of the Ramprasad Group. "
        "For access inquiries, contact wschertzer3@gatech.edu."
    )
    guide_col, setup_col = st.columns(2)
    with guide_col:
        if st.button("Read the user guide", use_container_width=True):
            _set_query_param("page", "guide")
            st.rerun()
    with setup_col:
        if st.button("View setup instructions", use_container_width=True):
            _set_query_param("page", "setup")
            st.rerun()
    login_tab, create_tab, forgot_tab, reset_tab = st.tabs(
        ["Login", "Create Account", "Forgot Password", "Reset Password"]
    )

    with login_tab:
        with st.form("login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            user = auth_store.authenticate(identifier, password)
            if user is None:
                st.error("Invalid username/email or password.")
            else:
                _set_auth_session(auth_store, user)
                st.rerun()

    with create_tab:
        if not _env_truthy("SALSA_AUTH_ALLOW_SIGNUP", True):
            st.info("Account creation is disabled on this deployment.")
        else:
            with st.form("create_account_form"):
                username = st.text_input("Username", key="signup_username")
                email = st.text_input("Email", key="signup_email")
                first_name = st.text_input("First name", key="signup_first_name")
                last_name = st.text_input("Last name", key="signup_last_name")
                institution = st.text_input("Institution", key="signup_institution")
                password = st.text_input("Password", type="password", key="signup_pw")
                password_again = st.text_input(
                    "Confirm password",
                    type="password",
                    key="signup_pw_again",
                )
                submitted = st.form_submit_button("Create account", type="primary")
            if submitted:
                try:
                    if password != password_again:
                        raise ValueError("Passwords do not match.")
                    user = auth_store.request_account(
                        username=username,
                        email=email,
                        first_name=first_name,
                        last_name=last_name,
                        institution=institution,
                        password=password,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    notification_sent = False
                    notification_error = ""
                    try:
                        notification_sent = send_account_request_email(
                            auth_store.list_admin_emails(),
                            user,
                        )
                    except Exception as exc:
                        notification_error = str(exc)
                        print(f"[SALSA auth] Account request email failed: {exc}")
                    st.success(
                        "Account request submitted. An administrator must approve it "
                        "before you can sign in."
                    )
                    if not notification_sent:
                        if "550" in notification_error and "5.7.1" in notification_error:
                            st.warning(
                                "The request was saved and is visible to administrators, "
                                "but Georgia Tech's outbound relay rejected this server. "
                                "The host polymergenome.storm.gatech.edu and source IP "
                                "143.215.76.191 must be approved as an Authorized Sender."
                            )
                        else:
                            st.warning(
                                "The request was saved and is visible to administrators, "
                                "but the administrator email notification could not be sent."
                            )

    with forgot_tab:
        smtp_configured = bool(os.environ.get("SALSA_SMTP_HOST") and os.environ.get("SALSA_SMTP_FROM"))
        print_reset_links = _env_truthy("SALSA_AUTH_PRINT_RESET_LINKS", False)
        if smtp_configured:
            st.caption("A one-time reset token will be emailed if the account exists.")
        elif print_reset_links:
            st.caption("A one-time reset token will be written to the SALSA service logs.")
        else:
            st.warning(
                "Password reset delivery is not configured. Set SMTP environment "
                "variables, or set SALSA_AUTH_PRINT_RESET_LINKS=1 for internal-only "
                "log-based resets."
            )
        with st.form("forgot_password_form"):
            identifier = st.text_input("Username or email", key="forgot_identifier")
            submitted = st.form_submit_button("Send reset token")
        if submitted:
            if not (smtp_configured or print_reset_links):
                st.error("Password reset delivery is not configured for this deployment.")
            else:
                user = auth_store.find_user(identifier)
                delivery_failed = False
                delivery_error = ""
                if user is not None:
                    token = auth_store.create_password_reset_token(int(user["id"]))
                    mailed = False
                    if smtp_configured:
                        try:
                            mailed = send_password_reset_email(user["email"], token)
                        except Exception as exc:
                            delivery_failed = True
                            delivery_error = str(exc)
                            print(f"[SALSA auth] Password reset email failed: {exc}")
                    if print_reset_links and not mailed:
                        print(
                            "[SALSA auth] Password reset token for "
                            f"{user['username']}: {token}"
                        )
                if delivery_failed and not print_reset_links:
                    if "550" in delivery_error and "5.7.1" in delivery_error:
                        st.error(
                            "Georgia Tech's outbound relay rejected this server. "
                            "The host polymergenome.storm.gatech.edu and source IP "
                            "143.215.76.191 must be approved as an Authorized Sender."
                        )
                    else:
                        st.error(
                            "Reset email could not be sent. Check the SMTP settings "
                            "and service logs."
                        )
                elif smtp_configured:
                    st.success("If that account exists, a reset email has been sent.")
                else:
                    st.success("If that account exists, a reset token was written to the service logs.")
            if not smtp_configured and not print_reset_links:
                st.info(
                    "The service reads deployment settings from "
                    "`/home/wschertzer3/SALSA/.env.production`."
                )
            elif not smtp_configured and print_reset_links:
                st.info("Check the service logs with `journalctl -u salsa`.")
            elif smtp_configured and not print_reset_links:
                st.info(
                    "If email delivery still fails, enable `SALSA_AUTH_PRINT_RESET_LINKS=1` "
                    "temporarily to write fallback reset links to the service logs."
                )

    with reset_tab:
        with st.form("reset_password_form"):
            identifier = st.text_input(
                "Username or email",
                value=_query_param("reset_identifier"),
                key="reset_identifier",
            )
            token = st.text_input(
                "Reset token",
                value=_query_param("reset_token"),
                key="reset_token",
            )
            password = st.text_input("New password", type="password", key="reset_pw")
            password_again = st.text_input(
                "Confirm new password",
                type="password",
                key="reset_pw_again",
            )
            submitted = st.form_submit_button("Reset password", type="primary")
        if submitted:
            try:
                if password != password_again:
                    raise ValueError("Passwords do not match.")
                ok = auth_store.reset_password_with_token(
                    identifier=identifier,
                    token=token,
                    new_password=password,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                if ok:
                    st.success("Password reset. You can log in now.")
                    _clear_query_param("reset_token")
                    _clear_query_param("reset_identifier")
                else:
                    st.error("Invalid or expired reset token.")

    st.stop()



st.set_page_config(page_title="SALSA UI", layout="wide")
render_salsa_brand_header()

st.markdown("""
<style>
/* Make the expander content scrollable and resizable */
div[data-testid="stExpanderDetails"] {
    max-height: 800px;
    overflow-y: auto;
    padding-right: 10px;
    resize: vertical;
    min-height: 100px;
}
</style>
""", unsafe_allow_html=True)

public_page = _query_param("page").strip().lower()
if public_page in {"guide", "setup"}:
    _render_public_page(public_page)
    st.stop()

@st.cache_resource(show_spinner=False)
def _get_auth_store(db_path_str: str) -> AuthStore:
    # AuthStore.__init__ opens sqlite and runs the full schema script; cache the
    # instance per process instead of paying that on every Streamlit rerun.
    store = AuthStore(Path(db_path_str))
    store.bootstrap_default_admin()
    return store


scripts_root = (Path(__file__).resolve().parent / "..").resolve()
salsa_data_root = data_root_from_env(scripts_root / "ui_runs")
auth_db_path = Path(os.environ.get("SALSA_AUTH_DB", str(salsa_data_root / "auth.sqlite3")))
auth_store = _get_auth_store(str(auth_db_path))
if _query_param("salsa_session") and not st.session_state.get("_legacy_auth_query_cleared"):
    st.session_state["_legacy_auth_query_cleared"] = True
    _clear_query_param("salsa_session")
_render_pending_auth_cookie_scripts()
current_user = _render_auth_gate(auth_store)
MAX_INPUT_ITEMS = ADMIN_INPUT_ITEM_LIMIT if is_admin_user(current_user) else DEFAULT_INPUT_ITEM_LIMIT
workspaces_root = user_workspaces_root(salsa_data_root, current_user)
_bootstrap_settings_presets_for_profiles(salsa_data_root, auth_store.list_users())
show_admin_page = False
_ACTIVE_PIPELINE_RUN_KEY = "active_pipeline_run"

if _ACTIVE_PIPELINE_RUN_KEY in st.session_state:
    st.session_state.pop("rr", None)

# ---------------- Sidebar ----------------
with st.sidebar:
    if "salsa_active_page" not in st.session_state:
        st.session_state["salsa_active_page"] = "tool"

    if st.session_state["salsa_active_page"] == "guide":
        if st.button("Back to SALSA tool", use_container_width=True):
            st.session_state["salsa_active_page"] = "tool"
            st.rerun()
    else:
        if st.button("Guide", use_container_width=True):
            st.session_state["salsa_active_page"] = "guide"
            st.rerun()

    st.divider()

    with st.expander(
        "Profile",
        expanded=bool(st.session_state.get("profile_expanded", False)),
    ):
        profile_label = current_user["display_name"] or current_user["username"]
        st.write(f"Signed in as **{profile_label}**")
        st.caption(current_user["email"])
        st.caption(f"Role: **{current_user['role']}**")
        if st.button("Logout", on_click=_keep_profile_expanded):
            _clear_auth_session(auth_store)
            st.rerun()

        with st.form("change_password_form"):
            current_password = st.text_input(
                "Current password",
                type="password",
                key="change_pw_current",
            )
            new_password = st.text_input(
                "New password",
                type="password",
                key="change_pw_new",
            )
            new_password_again = st.text_input(
                "Confirm new password",
                type="password",
                key="change_pw_new_again",
            )
            submitted = st.form_submit_button(
                "Update password", on_click=_keep_profile_expanded
            )
        if submitted:
            try:
                if new_password != new_password_again:
                    raise ValueError("Passwords do not match.")
                validate_password(new_password)
                if auth_store.authenticate(current_user["username"], current_password) is None:
                    raise ValueError("Current password is incorrect.")
                auth_store.set_password(int(current_user["id"]), new_password)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Password updated.")

    st.divider()
    if is_admin_user(current_user):
        selected_page = st.radio(
            "View",
            options=["Run workspace", "Admin"],
            horizontal=True,
            key="salsa_sidebar_view",
        )
        show_admin_page = selected_page == "Admin"
        st.divider()

    # Choose/create workspace
    st.header("Workspace")

    # List existing workspaces (filter out shared config directories)
    existing_workspaces = [
        ws for ws in _list_subdirs(workspaces_root)
        if not _ignored_workspace_or_run_name(ws)
    ]

    # Apply pending selection (but DON'T write workspace_name)
    pending = st.session_state.pop("pending_workspace_name", None)

    # Determine what we want selected this run
    desired = pending or st.session_state.get("workspace_name")
    if desired not in existing_workspaces:
        desired = existing_workspaces[0] if existing_workspaces else _default_workspace_name()

    options = existing_workspaces if existing_workspaces else [desired]
    index = options.index(desired) if desired in options else 0

    st.selectbox(
        "Active workspace",
        options=options,
        index=index,
        key="workspace_name",
    )


    # Allow creation of a new workspace
    new_ws = st.text_input("Create new workspace", value="", key="new_workspace_name")
    col_ws_a, col_ws_b = st.columns([1, 1])

    with col_ws_a:
        if st.button("Create workspace"):
            if not _is_safe_workspace_name(new_ws):
                st.error("Invalid workspace name. Use letters/numbers plus _ . - (max 64 chars).")
            else:
                _ensure_workspace(workspaces_root, new_ws)
                st.session_state["pending_workspace_name"] = new_ws
                st.rerun()

    with col_ws_b:
        if st.button("Refresh workspaces"):
            st.rerun()

    # Now this is always the true selected value
    active_workspace_name = st.session_state["workspace_name"]
    active_workspace_dir = _ensure_workspace(workspaces_root, active_workspace_name)
    run_root = active_workspace_dir

    st.divider()
    st.header("Input Source")
    input_source = st.radio(
        "Input source",
        options=["DOI(s)", "Query", "PDF(s)", "Figure(s)"],
        horizontal=True,
        key="input_source",
        label_visibility="collapsed",
    )
    use_docling_pdf_uploads = input_source == "PDF(s)"
    use_figure_uploads = input_source == "Figure(s)"
    use_query = input_source == "Query"
    use_file_uploads = use_docling_pdf_uploads or use_figure_uploads

    st.divider()
    st.header("Settings")
    _bootstrap_settings_presets_for_workspaces(workspaces_root)
    settings_flash_message = _apply_pending_settings_preset_state()
    if "custom_instructions" not in st.session_state:
        _apply_settings_preset_state(
            _load_settings_preset(workspaces_root, DEFAULT_SETTINGS_PRESET_NAME),
            preset_name=DEFAULT_SETTINGS_PRESET_NAME,
        )
    elif any(key not in st.session_state for key in _CI_EDITOR_KEYS):
        _apply_custom_instruction_state(
            st.session_state["custom_instructions"],
            preset_name=st.session_state.get("current_preset_name"),
        )
    _preserve_settings_editor_state()
    if settings_flash_message:
        st.success(settings_flash_message)

    existing_settings_presets = _list_settings_presets(workspaces_root)

    show_settings_editor = st.toggle(
        "Create/Modify Settings Preset",
        key="create_settings_preset_expanded",
    )
    if show_settings_editor:
        _ensure_shared_llm_settings_in_state()

        settings_section_options = [
            "API Keys",
            "Pipeline Stages",
            "LLM Models & Instructions",
        ]
        if st.session_state.get("settings_editor_section") not in settings_section_options:
            st.session_state["settings_editor_section"] = settings_section_options[0]
        settings_editor_section = st.radio(
            "Settings section",
            options=settings_section_options,
            key="settings_editor_section",
            horizontal=True,
            label_visibility="collapsed",
            on_change=_keep_create_settings_preset_open,
        )
        stage_settings = _current_stage_settings_from_state()
        exclude_full_text = stage_settings["exclude_full_text"]
        run_extract_tables = stage_settings["run_extract_tables"] and not use_figure_uploads
        run_extract_figures = stage_settings["run_extract_figures"] or use_figure_uploads
        run_pdf_page_ocr = stage_settings["run_pdf_page_ocr"] if use_docling_pdf_uploads else False
        run_filter_tables = stage_settings["run_filter_tables"] and not use_figure_uploads
        run_filter_figures = stage_settings["run_filter_figures"]
        run_plextract = stage_settings["run_plextract"]
        bypass_ocr_with_llm = stage_settings["bypass_ocr_with_llm"]
        run_chemical_composition = (
            stage_settings["run_chemical_composition"] and not use_figure_uploads
        )
        run_complete_dataset = stage_settings["run_complete_dataset"]
        run_aggregation = stage_settings["run_aggregation"]
        ci = st.session_state["custom_instructions"]
        ci_column_defaults_table = st.session_state.get(
            _CI_COLUMN_DEFAULTS_SOURCE_KEY,
            _column_defaults_table_from_instructions(ci),
        )
        chemical_column_defaults_table = st.session_state.get(
            _CI_CHEM_COLUMN_DEFAULTS_SOURCE_KEY,
            _column_defaults_table_from_instructions(
                ci,
                columns_key="chemical_composition_columns",
                defaults_key="chemical_composition_defaults",
            ),
        )
        if settings_editor_section == "API Keys":
            st.text_input(
                "OpenAI API key",
                key="openai_api_key",
                type="password",
                placeholder="sk-...",
                help="Required for every selected stage that uses Chat GPT.",
                on_change=_keep_create_settings_preset_open,
            )
            st.text_input(
                "WOS API Expanded key",
                key="wos_api_key",
                type="password",
                placeholder="Clarivate API key",
                help="Web of Science API.",
                on_change=_keep_create_settings_preset_open,
            )
            # query_key_cols = st.columns(2)
            # with query_key_cols[0]:
            st.text_input(
                "OpenAlex API key",
                key="openalex_api_key",
                type="password",
                help=(
                    "Used for OpenAlex keyword/semantic queries and cached full-text "
                    "downloads. OpenAlex applies its own API usage limits."
                ),
                on_change=_keep_create_settings_preset_open,
            )
            # with query_key_cols[1]:
            st.text_input(
                "Scopus API key",
                key="scopus_inst_token",
                type="password",
                on_change=_keep_create_settings_preset_open,
            )
            st.caption(
                "SALSA tries OpenAlex cached full text first, then documented publisher "
                "APIs. DOI redirects do not grant full-text access; upload an authorized "
                "PDF when neither source can provide one."
            )
            st.caption(
                "Sources with no API Key required",
                help=(
                    "APS Harvest, PubMed Central, and PLOS. An NCBI API key is "
                    "optional for higher PubMed Central request limits."
                ),
            )
            publisher_key_cols = st.columns(2)
            with publisher_key_cols[0]:
                st.text_input(
                    "Elsevier API key",
                    key="elsevier_api_key",
                    type="password",
                    placeholder="Elsevier API key",
                    on_change=_keep_create_settings_preset_open,
                )
            with publisher_key_cols[1]:
                st.text_input(
                    "Wiley API Key",
                    key="wiley_tdm_token",
                    type="password",
                    placeholder="Wiley API Key",
                    on_change=_keep_create_settings_preset_open,
                )
            publisher_key_cols = st.columns(2)
            with publisher_key_cols[0]:
                st.text_input(
                    "Springer API key",
                    key="springer_nature_api_key",
                    type="password",
                    placeholder="Springer API key",
                    on_change=_keep_create_settings_preset_open,
                )
            with publisher_key_cols[1]:
                st.text_input(
                    "IEEE Xplore API key",
                    key="ieee_api_key",
                    type="password",
                    placeholder="IEEE Xplore API key",
                    on_change=_keep_create_settings_preset_open,
                )
            publisher_key_cols = st.columns(2)
            with publisher_key_cols[0]:
                st.text_input(
                    "RSC API Key",
                    key="rsc_tdm_token",
                    type="password",
                    placeholder="RSC API Key",
                    on_change=_keep_create_settings_preset_open,
                )
            with publisher_key_cols[1]:
                st.text_input(
                    "NCBI API key",
                    key="ncbi_api_key",
                    type="password",
                    placeholder="Optional PubMed/PMC key",
                    on_change=_keep_create_settings_preset_open,
                )


        if settings_editor_section == "Pipeline Stages":
            if use_figure_uploads:
                run_extract_figures = True
                run_extract_tables = False
                run_pdf_page_ocr = False
                run_chemical_composition = False
                run_filter_tables = False
                exclude_full_text = True
                st.checkbox("Exclude full text from context", value=True, disabled=True, key="settings_upload_exclude_full_text")
                st.checkbox("Extract tables", value=False, disabled=True, key="settings_upload_extract_tables")
                st.checkbox("Extract figures", value=True, disabled=True, key="settings_upload_extract_figures")
                st.checkbox("Filter tables", value=False, disabled=True, key="settings_upload_filter_tables")
            else:
                exclude_full_text = _settings_checkbox(
                    "Exclude full text from context",
                    _STAGE_STATE_KEYS["exclude_full_text"],
                    _DEFAULT_STAGE_SETTINGS["exclude_full_text"],
                )
                run_extract_tables = _settings_checkbox(
                    "Extract tables",
                    _STAGE_STATE_KEYS["run_extract_tables"],
                    _DEFAULT_STAGE_SETTINGS["run_extract_tables"],
                )
                run_extract_figures = _settings_checkbox(
                    "Extract figures",
                    _STAGE_STATE_KEYS["run_extract_figures"],
                    _DEFAULT_STAGE_SETTINGS["run_extract_figures"],
                )
                if use_docling_pdf_uploads:
                    run_pdf_page_ocr = _settings_checkbox(
                        "OCR scanned PDF pages",
                        _STAGE_STATE_KEYS["run_pdf_page_ocr"],
                        _DEFAULT_STAGE_SETTINGS["run_pdf_page_ocr"],
                        help=(
                            "Use Docling OCR for scanned/image-only uploaded PDFs. "
                            "Leave off for normal text PDFs; figure-data OCR is controlled "
                            "by Extract data from figures."
                        ),
                    )
                else:
                    run_pdf_page_ocr = False
                run_filter_tables = _settings_checkbox(
                    "Filter tables",
                    _STAGE_STATE_KEYS["run_filter_tables"],
                    _DEFAULT_STAGE_SETTINGS["run_filter_tables"],
                )

            run_filter_figures = _settings_checkbox(
                "Filter figures",
                _STAGE_STATE_KEYS["run_filter_figures"],
                _DEFAULT_STAGE_SETTINGS["run_filter_figures"],
            )
            run_plextract = _settings_checkbox(
                "Extract data from figures",
                _STAGE_STATE_KEYS["run_plextract"],
                _DEFAULT_STAGE_SETTINGS["run_plextract"],
            )
            _, figure_data_llm_col = st.columns([0.08, 0.92])
            with figure_data_llm_col:
                bypass_ocr_with_llm = _settings_checkbox(
                    "Use LLM to extract data from figures",
                    _STAGE_STATE_KEYS["bypass_ocr_with_llm"],
                    _DEFAULT_STAGE_SETTINGS["bypass_ocr_with_llm"],
                    help="Writes an LLM-digitized figure dataset. Cannot crop images or modify series/axes in this mode",
                )
            if use_figure_uploads:
                st.checkbox(
                    "Extract chemical composition",
                    value=False,
                    disabled=True,
                    key="settings_upload_chemical_composition",
                )
                run_chemical_composition = False
            else:
                run_chemical_composition = _settings_checkbox(
                    "Extract chemical composition",
                    _STAGE_STATE_KEYS["run_chemical_composition"],
                    _DEFAULT_STAGE_SETTINGS["run_chemical_composition"],
                )
            run_complete_dataset = _settings_checkbox(
                "Complete dataset",
                _STAGE_STATE_KEYS["run_complete_dataset"],
                _DEFAULT_STAGE_SETTINGS["run_complete_dataset"],
            )
            run_aggregation = _settings_checkbox(
                "Agglomerate run datasets",
                _STAGE_STATE_KEYS["run_aggregation"],
                _DEFAULT_STAGE_SETTINGS["run_aggregation"],
                help="After dataset completion, create one table/figure CSV and one chemical-composition CSV for the run.",
            )


        if settings_editor_section == "LLM Models & Instructions":
            ollama_host = _normalize_ollama_host(
                st.session_state.get("ollama_host", default_ollama_host)
            )
            ollama_available = _ollama_server_is_available(ollama_host)
            if not ollama_available:
                st.info(
                    f"No Ollama server is running at `{ollama_host}`. "
                    "Start Ollama to enable Ollama model selection."
                )

            llm_steps_to_render = _active_llm_steps(
                run_extract_tables=run_extract_tables,
                use_docling_pdf_uploads=use_file_uploads,
                run_chemical_composition=run_chemical_composition,
                run_filter_figures=run_filter_figures,
                run_filter_tables=run_filter_tables and not use_figure_uploads,
                run_plextract=run_plextract,
                bypass_ocr_with_llm=bypass_ocr_with_llm,
                run_complete_dataset=run_complete_dataset,
            )
            active_steps_use_ollama = _any_step_uses_ollama(llm_steps_to_render)
            ollama_models = _ollama_models()
            if ollama_available and active_steps_use_ollama:
                st.number_input(
                    "Ollama context window (num_ctx)",
                    min_value=4096,
                    step=1024,
                    value=int(st.session_state.get("ollama_num_ctx", 16384)),
                    key="ollama_num_ctx",
                    help="Shared across all Ollama-backed steps.",
                    on_change=_keep_create_settings_preset_open,
                )

            if run_extract_tables and not use_file_uploads:
                with st.container():
                    _render_step_llm_selector(
                        LLM_STEP_EXTRACT_TABLES,
                        ollama_available=ollama_available,
                        ollama_models=ollama_models,
                        caption=_llm_step_caption(LLM_STEP_EXTRACT_TABLES),
                        title=f"{LLM_STEP_LABELS[LLM_STEP_EXTRACT_TABLES]} settings",
                    )
                    _render_instruction_text_area(
                        "Table extraction rules",
                        "ci_table_extraction",
                    )
                st.divider()

            chemical_column_defaults_table = st.session_state.get(
                _CI_CHEM_COLUMN_DEFAULTS_SOURCE_KEY,
                _column_defaults_table_from_instructions(
                    ci,
                    columns_key="chemical_composition_columns",
                    defaults_key="chemical_composition_defaults",
                ),
            )
            if run_chemical_composition:
                with st.container():
                    _render_step_llm_selector(
                        LLM_STEP_CHEM_RELEVANT,
                        ollama_available=ollama_available,
                        ollama_models=ollama_models,
                        caption=_llm_step_caption(LLM_STEP_CHEM_RELEVANT),
                        title=f"{LLM_STEP_LABELS[LLM_STEP_CHEM_RELEVANT]} settings",
                    )
                    st.divider()
                    _render_step_llm_selector(
                        LLM_STEP_CHEM_COMPOSITION,
                        ollama_available=ollama_available,
                        ollama_models=ollama_models,
                        caption=_llm_step_caption(LLM_STEP_CHEM_COMPOSITION),
                        title=f"{LLM_STEP_LABELS[LLM_STEP_CHEM_COMPOSITION]} settings",
                    )
                    _render_instruction_text_area(
                        "Chemical composition instructions",
                        "ci_chemical_composition",
                        height=120,
                    )
                    with st.expander(
                        "Required chemical composition columns and missing-value defaults",
                        expanded=False,
                    ):
                        chemical_column_defaults_table = st.data_editor(
                            chemical_column_defaults_table,
                            hide_index=True,
                            num_rows="dynamic",
                            use_container_width=True,
                            key=_ci_chemical_column_defaults_widget_key(),
                            on_change=_keep_create_settings_preset_open,
                        )
                st.divider()

            if run_filter_figures:
                with st.container():
                    _render_step_llm_selector(
                        LLM_STEP_FILTER_FIGURES,
                        ollama_available=ollama_available,
                        ollama_models=ollama_models,
                        caption=_llm_step_caption(LLM_STEP_FILTER_FIGURES),
                        title=f"{LLM_STEP_LABELS[LLM_STEP_FILTER_FIGURES]} settings",
                    )
                    _render_instruction_text_area(
                        "Figure filtering criteria",
                        "ci_figure_filter",
                        height=150,
                    )
                st.divider()

            if run_filter_tables and not use_figure_uploads:
                with st.container():
                    _render_step_llm_selector(
                        LLM_STEP_FILTER_TABLES,
                        ollama_available=ollama_available,
                        ollama_models=ollama_models,
                        caption=_llm_step_caption(LLM_STEP_FILTER_TABLES),
                        title=f"{LLM_STEP_LABELS[LLM_STEP_FILTER_TABLES]} settings",
                    )
                    _render_instruction_text_area(
                        "Table filtering criteria",
                        "ci_table_filter",
                        height=150,
                    )
                st.divider()

            if run_plextract and bypass_ocr_with_llm:
                with st.container():
                    _render_step_llm_selector(
                        LLM_STEP_FIGURE_DATA,
                        ollama_available=ollama_available,
                        ollama_models=ollama_models,
                        caption=_llm_step_caption(LLM_STEP_FIGURE_DATA),
                        title=f"{LLM_STEP_LABELS[LLM_STEP_FIGURE_DATA]} settings",
                    )
                    _render_instruction_text_area(
                        "Figure data extraction instructions",
                        "ci_figure_data",
                        height=120,
                    )
                st.divider()

            ci_column_defaults_table = st.session_state.get(
                _CI_COLUMN_DEFAULTS_SOURCE_KEY,
                _column_defaults_table_from_instructions(ci),
            )
            if run_complete_dataset:
                with st.container():
                    _render_step_llm_selector(
                        LLM_STEP_COMPLETE_DATASET,
                        ollama_available=ollama_available,
                        ollama_models=ollama_models,
                        caption=_llm_step_caption(LLM_STEP_COMPLETE_DATASET),
                        title=f"{LLM_STEP_LABELS[LLM_STEP_COMPLETE_DATASET]} settings",
                    )
                    _render_instruction_text_area(
                        "Data processing rules",
                        "ci_data_processing",
                    )
                    with st.expander(
                        "Output columns and missing-value defaults",
                        expanded=False,
                    ):
                        ci_column_defaults_table = st.data_editor(
                            st.session_state.get(
                                _CI_COLUMN_DEFAULTS_SOURCE_KEY,
                                _column_defaults_table_from_instructions(ci),
                            ),
                            hide_index=True,
                            num_rows="dynamic",
                            use_container_width=True,
                            key=_ci_column_defaults_widget_key(),
                            on_change=_keep_create_settings_preset_open,
                        )

        current_ci = _custom_instructions_from_editor_state(
            ci_column_defaults_table,
            chemical_column_defaults_table,
        )
        st.session_state["custom_instructions"] = current_ci

        st.subheader("Save Preset")
        col_name, col_save_btn = st.columns([3, 1])
        with col_name:
            preset_name = st.text_input(
                "Settings preset name",
                value=st.session_state.get("current_settings_preset_name", ""),
                key="settings_new_preset_name",
                placeholder="Enter a name for this settings preset",
                on_change=_keep_create_settings_preset_open,
            )
        with col_save_btn:
            st.write("")
            st.write("")
            if st.button("Save", type="primary", key="save_settings_preset_btn") and preset_name.strip():
                clean_preset_name = preset_name.strip()
                current_settings = _make_settings_preset(
                    stages={
                        "exclude_full_text": exclude_full_text,
                        "run_extract_tables": run_extract_tables,
                        "run_extract_figures": run_extract_figures,
                        "run_pdf_page_ocr": run_pdf_page_ocr,
                        "run_filter_tables": run_filter_tables and not use_figure_uploads,
                        "run_filter_figures": run_filter_figures,
                        "run_plextract": run_plextract,
                        "bypass_ocr_with_llm": bypass_ocr_with_llm,
                        "run_chemical_composition": run_chemical_composition,
                        "run_complete_dataset": run_complete_dataset,
                        "run_aggregation": run_aggregation,
                    },
                    custom_instructions=current_ci,
                    include_session_llms=True,
                    openai_api_key=str(st.session_state.get("openai_api_key", "")).strip(),
                    elsevier_api_key=str(st.session_state.get("elsevier_api_key", "")).strip(),
                    wiley_tdm_token=str(st.session_state.get("wiley_tdm_token", "")).strip(),
                    springer_nature_api_key=str(
                        st.session_state.get("springer_nature_api_key", "")
                    ).strip(),
                    rsc_tdm_token=str(st.session_state.get("rsc_tdm_token", "")).strip(),
                    ncbi_api_key=str(st.session_state.get("ncbi_api_key", "")).strip(),
                    ieee_api_key=str(st.session_state.get("ieee_api_key", "")).strip(),
                    wos_api_key=str(st.session_state.get("wos_api_key", "")).strip(),
                    openalex_api_key=str(st.session_state.get("openalex_api_key", "")).strip(),
                    scopus_inst_token=str(st.session_state.get("scopus_inst_token", "")).strip(),
                )
                _save_settings_preset(workspaces_root, clean_preset_name, current_settings)
                _queue_settings_preset_state(
                    current_settings,
                    preset_name=clean_preset_name,
                    flash_message=f"Saved settings preset: {clean_preset_name}",
                )
                st.rerun()

    with st.expander(
        "Load Settings Preset",
        expanded=bool(st.session_state.get("load_settings_preset_expanded", False)),
    ):
        if existing_settings_presets:
            col_load_a, col_load_b = st.columns([3, 1])
            with col_load_a:
                settings_preset_choice = st.selectbox(
                    "Saved settings preset",
                    options=[""] + existing_settings_presets,
                    key="settings_preset_choice",
                    on_change=_keep_load_settings_preset_open,
                )
            with col_load_b:
                st.write("")
                st.write("")
                if st.button(
                    "Load",
                    type="secondary",
                    key="load_settings_preset_btn",
                    on_click=_keep_load_settings_preset_open,
                ) and settings_preset_choice:
                    loaded_settings = _load_settings_preset(workspaces_root, settings_preset_choice)
                    _queue_settings_preset_state(
                        loaded_settings,
                        preset_name=settings_preset_choice,
                        flash_message=f"Loaded settings preset: {settings_preset_choice}",
                    )
                    st.rerun()
        else:
            st.info("No settings presets saved yet.")

        if st.session_state.get("current_settings_preset_name"):
            st.caption(f"Currently loaded: **{st.session_state['current_settings_preset_name']}**")
        else:
            st.caption("Currently loaded: **Unsaved settings**")

    st.divider()
    with st.expander("Load an existing run", expanded=False):

        # Choose which workspace to browse when loading runs (filter out shared config directories)
        load_ws_options = [
            ws for ws in _list_subdirs(workspaces_root)
            if not _ignored_workspace_or_run_name(ws)
        ]

        if load_ws_options:
            # ensure state is valid
            if st.session_state.get("load_workspace") not in load_ws_options:
                st.session_state["load_workspace"] = (
                    active_workspace_name if active_workspace_name in load_ws_options else load_ws_options[0]
                )

            load_workspace_name = st.selectbox(
                "Browse workspace",
                options=load_ws_options,
                key="load_workspace",   # <-- single source of truth
            )

            load_workspace_dir = _ensure_workspace(workspaces_root, load_workspace_name)

            existing_runs = sorted(
                [
                    p.name
                    for p in load_workspace_dir.iterdir()
                    if p.is_dir() and not _ignored_workspace_or_run_name(p.name)
                ],
                reverse=True,
            )
            pick = st.selectbox("Run ID", options=[""] + existing_runs, key="load_run_id")

            if st.button("Load run") and pick:
                try:
                    run_path = safe_child_path(load_workspace_dir, pick)
                    if not run_belongs_to_user(run_path, workspaces_root, current_user):
                        raise PermissionError("This run does not belong to the signed-in user.")
                    rr = discover_run(run_path)
                except Exception as exc:
                    st.error(f"Could not load run: {exc}")
                else:
                    st.session_state["rr"] = rr
        else:
            st.info("No workspaces exist yet. Create one above, then run the pipeline.")

class _RunCancelEvent:
    def __init__(self, run_dir: Path):
        self._event = threading.Event()
        self._run_dir = run_dir

    def is_set(self) -> bool:
        return self._event.is_set() or (self._run_dir / "cancel_requested").exists()

    def set(self) -> None:
        self._event.set()
        request_run_cancellation(self._run_dir)


def _rerun_app() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
    rerun()


def _clear_loaded_run_for_pipeline_launch() -> None:
    st.session_state.pop("rr", None)


def _pipeline_status_snapshot(worker: dict[str, Any]) -> dict[str, Any]:
    with worker["lock"]:
        snapshot = dict(worker["status"])
    status_path = worker.get("status_path")
    if status_path:
        disk_status = read_run_status(status_path)
        # The on-disk status file is JSON, so object values such as RunResult are
        # stringified there. Keep the in-memory object for UI follow-up actions.
        disk_status.pop("result", None)
        snapshot.update(disk_status)
    return snapshot


def _set_pipeline_status(worker: dict[str, Any], **updates: Any) -> None:
    with worker["lock"]:
        worker["status"].update(updates)
    status_path = worker.get("status_path")
    if status_path:
        disk_updates = dict(updates)
        proc = worker.get("process")
        if proc is not None and "pid" not in disk_updates:
            disk_updates["pid"] = proc.pid
        result = disk_updates.pop("result", None)
        if result is not None:
            disk_updates["result_run_id"] = getattr(result, "run_id", None)
            disk_updates["result_run_dir"] = str(getattr(result, "run_dir", ""))
        write_run_status(status_path, **disk_updates)


def _notify_run_finished(
    worker: dict[str, Any],
    auth_context: dict[str, Any],
    *,
    state: str,
    rr: RunResult | None = None,
    error: str = "",
) -> None:
    if not auth_context.get("notify_on_completion"):
        return
    email = str(auth_context.get("email") or "").strip()
    if not email:
        _set_pipeline_status(
            worker,
            notification_sent=False,
            notification_error="No email address is available for this user.",
        )
        return

    run_id = str(
        getattr(rr, "run_id", None)
        or auth_context.get("run_id")
        or ""
    )
    workspace = str(
        auth_context.get("workspace")
        or (getattr(rr, "run_dir", Path()).parent.name if rr is not None else "")
    )
    run_dir = str(getattr(rr, "run_dir", "") or auth_context.get("run_dir") or "")
    try:
        sent = send_run_notification_email(
            email,
            display_name=str(auth_context.get("display_name") or auth_context.get("username") or ""),
            run_id=run_id,
            workspace=workspace,
            state=state,
            processed_count=len(rr.dois) if rr is not None else None,
            failure_count=len(rr.failures) if rr is not None else None,
            error=error,
            run_dir=run_dir,
        )
    except Exception as exc:
        _set_pipeline_status(
            worker,
            notification_sent=False,
            notification_error=str(exc),
        )
        print(f"[SALSA email] Run notification failed: {exc}")
        return

    _set_pipeline_status(
        worker,
        notification_sent=bool(sent),
        notification_error="" if sent else "Email delivery is not configured or no recipient was available.",
    )


def _start_pipeline_worker(pipeline_kwargs: dict[str, Any], dois: list[str]) -> dict[str, Any]:
    lock = threading.Lock()
    pipeline_payload = dict(pipeline_kwargs)
    auth_context = dict(pipeline_payload.pop("_auth_context", {}))
    run_dir = safe_child_path(Path(pipeline_kwargs["run_root"]), str(pipeline_kwargs["run_id"]))
    clear_run_cancellation(run_dir)
    cancel_event = _RunCancelEvent(run_dir)
    status_path = run_dir / RUN_STATUS_FILE_NAME
    status: dict[str, Any] = {
        "run_id": str(pipeline_kwargs.get("run_id") or ""),
        "state": "running",
        "stage": "Starting pipeline",
        "doi": dois[0] if len(dois) == 1 else f"{len(dois)} DOIs",
        "result": None,
        "error": None,
        "traceback": None,
        "updated_at": _now_eastern_iso(),
    }
    worker: dict[str, Any] = {
        "lock": lock,
        "cancel_event": cancel_event,
        "status": status,
        "status_path": status_path,
        "dois": list(dois),
    }

    payload = {
        "pipeline_kwargs": pipeline_payload,
        "auth_context": auth_context,
    }
    payload_file = tempfile.NamedTemporaryFile(
        prefix="salsa_pipeline_",
        suffix=".pkl",
        dir="/tmp",
        delete=False,
    )
    with payload_file:
        pickle.dump(payload, payload_file)

    scripts_root = Path(pipeline_kwargs["scripts_root"]).resolve()
    worker_script = Path(__file__).resolve().with_name("pipeline_worker.py")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [p for p in existing_pythonpath.split(os.pathsep) if p]
    scripts_root_text = str(scripts_root)
    if scripts_root_text not in pythonpath_parts:
        env["PYTHONPATH"] = os.pathsep.join([scripts_root_text] + pythonpath_parts)

    run_dir.mkdir(parents=True, exist_ok=True)
    if auth_context:
        write_run_manifest(
            run_dir,
            user_id=auth_context.get("user_id"),
            username=str(auth_context.get("username") or ""),
            workspace=str(auth_context.get("workspace") or run_dir.parent.name),
            run_id=str(auth_context.get("run_id") or run_dir.name),
            query_provider=str(auth_context.get("query_provider") or ""),
            query=str(auth_context.get("query") or ""),
            query_settings=dict(auth_context.get("query_settings") or {}),
        )
    log_path = run_dir / "pipeline_worker.log"
    try:
        log_file = log_path.open("ab")
    except OSError:
        log_file = subprocess.DEVNULL

    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker_script), payload_file.name],
            cwd=str(scripts_root),
            env=env,
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        if log_file is not subprocess.DEVNULL:
            log_file.close()
    except Exception as exc:
        if log_file is not subprocess.DEVNULL:
            log_file.close()
        try:
            Path(payload_file.name).unlink()
        except OSError:
            pass
        _set_pipeline_status(
            worker,
            state="error",
            error=str(exc),
            traceback=traceback.format_exc(),
            stage="Failed to start",
        )
        return worker

    status["pid"] = proc.pid
    worker["process"] = proc
    worker["log_path"] = log_path
    return worker


def _render_active_pipeline_run() -> None:
    worker = st.session_state.get(_ACTIVE_PIPELINE_RUN_KEY)
    if not worker:
        return

    st.session_state.pop("rr", None)
    snapshot = _pipeline_status_snapshot(worker)
    state = snapshot.get("state", "running")

    proc = worker.get("process")
    thread = worker.get("thread")
    is_alive = (
        proc.poll() is None
        if proc is not None
        else bool(thread is not None and thread.is_alive())
    )

    if is_alive:
        if worker["cancel_event"].is_set():
            st.warning("Cancellation requested. Stopping the active pipeline step...")
        elif st.button("Cancel Run", type="secondary"):
            worker["cancel_event"].set()
            _set_pipeline_status(worker, state="cancelling", stage="Cancelling")
            st.warning("Cancellation requested. Stopping the active pipeline step...")

        snapshot = _pipeline_status_snapshot(worker)
        st.info(_run_progress_status_text(snapshot))
        st.caption("This run is still active. Use Refresh status to update progress.")
        if st.button("Refresh status", key="refresh_active_pipeline_status"):
            _rerun_app()
        st.stop()

    snapshot = _pipeline_status_snapshot(worker)
    state = snapshot.get("state", state)
    if state in {"running", "cancelling"} and proc is not None and proc.returncode is not None:
        log_hint = f" See {worker.get('log_path')} for worker output." if worker.get("log_path") else ""
        _set_pipeline_status(
            worker,
            state="error",
            stage="Failed",
            error=f"Pipeline worker exited unexpectedly (exit code {proc.returncode}).{log_hint}",
        )
        snapshot = _pipeline_status_snapshot(worker)
        state = snapshot.get("state", "error")
    st.session_state.pop(_ACTIVE_PIPELINE_RUN_KEY, None)
    if state == "done":
        rr_done = snapshot.get("result")
        if rr_done is None and snapshot.get("result_run_dir"):
            try:
                rr_done = discover_run(Path(str(snapshot["result_run_dir"])))
            except Exception:
                rr_done = None
        if rr_done is not None:
            st.session_state["rr"] = rr_done
            _render_run_status_outcome(rr_done.run_id, snapshot)
            _render_cost_summary(rr_done.run_dir)
    elif state == "cancelled":
        _render_run_status_outcome(
            str(snapshot.get("result_run_id") or snapshot.get("run_id") or ""),
            snapshot,
        )
    elif state == "error":
        _render_run_status_outcome(
            str(snapshot.get("result_run_id") or snapshot.get("run_id") or ""),
            snapshot,
        )


# ---------------- Run pipeline ----------------
_render_active_pipeline_run()

if st.session_state.get("salsa_active_page") == "guide":
    _render_guide_page()
    st.stop()

if show_admin_page:
    show_loaded_run = _render_admin_page(
        auth_store=auth_store,
        data_root=salsa_data_root,
        current_user=current_user,
    )
    if not show_loaded_run:
        st.stop()
    st.divider()

_loaded_progress_rr: RunResult | None = st.session_state.get("rr")
if _loaded_progress_rr is not None:
    _loaded_progress_status_path = _loaded_progress_rr.run_dir / RUN_STATUS_FILE_NAME
    _loaded_progress_status = _mark_inactive_run_status(
        _loaded_progress_status_path,
        read_run_status(_loaded_progress_status_path),
    )
    if _run_status_is_active(_loaded_progress_status):
        st.info(f"Run ID: **{_loaded_progress_rr.run_id}**")
        st.info(
            "Workspace: "
            f"**{_workspace_display_name_for_run(_loaded_progress_rr.run_dir, workspaces_root)}**"
        )
        _render_run_query(_loaded_progress_rr.run_dir)
        _render_run_custom_instructions(_loaded_progress_rr.run_dir)
        st.info(_run_progress_status_text(_loaded_progress_status))
        if _loaded_progress_status.get("state") == "cancelling":
            st.warning("Cancellation requested. Stopping the active pipeline step...")
        elif st.button(
            "Cancel Loaded Run",
            type="secondary",
            key=f"cancel_loaded_run_{_loaded_progress_rr.run_id}",
        ):
            request_run_cancellation(_loaded_progress_rr.run_dir)
            write_run_status(
                _loaded_progress_status_path,
                state="cancelling",
                stage="Cancelling",
            )
            st.warning("Cancellation requested. Stopping the active pipeline step...")
        st.caption("This run is still active. Use Refresh status to update progress.")
        if st.button(
            "Refresh status",
            key=f"refresh_loaded_progress_{_loaded_progress_rr.run_id}",
        ):
            _rerun_app()
        st.stop()

if use_query:
    st.subheader("Literature query")
    query_provider = st.selectbox(
        "Query provider",
        options=["OpenAlex", "Web of Science", "Elsevier Scopus"],
        key="query_provider",
    )
    query_examples = {
        "OpenAlex": ("e.g. anion exchange membranes, dielectric constant, ROMP ", "Plain-text search across OpenAlex works."),
        "Web of Science": ('TS=("anion exchange membrane")', "Web of Science advanced-search syntax."),
        "Elsevier Scopus": ('TITLE-ABS-KEY("anion exchange membrane")', "Scopus Search API query syntax."),
    }
    query_placeholder, query_help = query_examples[query_provider]
    literature_query_text = st.text_area(
        f"{query_provider} query",
        height=140,
        placeholder=query_placeholder,
        help=query_help,
    )
    openalex_publisher = ""
    openalex_open_access = "Any"
    openalex_sort_label = "Relevance (highest first)"
    openalex_top_dois = MAX_INPUT_ITEMS
    if query_provider == "OpenAlex":
        openalex_filter_cols = st.columns(2)
        with openalex_filter_cols[0]:
            openalex_search_mode = st.selectbox(
                "Search mode",
                options=["Semantic", "Keyword"],
                help=(
                    "Semantic search matches meaning and works best with a detailed "
                    "research description. Keyword search matches terms in indexed text."
                ),
            )
            openalex_sort_label = st.selectbox(
                "Result ranking",
                options=["Relevance (highest first)", "Most cited", "Newest"],
            )
            openalex_publisher = st.text_input(
                "Publisher (optional)",
                placeholder="e.g. Elsevier",
            )
        with openalex_filter_cols[1]:
            openalex_open_access = st.selectbox(
                "Open-access filter",
                options=["Any", "Open access only", "Closed access only"],
            )
            openalex_top_dois = int(st.number_input(
                "Top DOIs to use",
                min_value=1,
                max_value=min(
                    MAX_INPUT_ITEMS,
                    50 if openalex_search_mode == "Semantic" else MAX_INPUT_ITEMS,
                ),
                value=min(
                    MAX_INPUT_ITEMS,
                    50 if openalex_search_mode == "Semantic" else MAX_INPUT_ITEMS,
                ),
                step=1,
                help=(
                    "SALSA follows the selected ranking and keeps this many unique "
                    "DOI-bearing works. OpenAlex semantic search is limited to 50 results."
                ),
            ))
        st.caption(f"The top {openalex_top_dois} matching DOIs will be passed into this SALSA run.")
    else:
        st.caption(f"At most {MAX_INPUT_ITEMS} matching records will be passed into this SALSA run.")
    uploaded_pdf_files = []
    uploaded_pdf_ids_text = ""
    uploaded_figure_files = []
    uploaded_figure_document_id = ""
    uploaded_figure_captions_text = ""
    doi_text = ""
elif use_docling_pdf_uploads:
    st.subheader("Input PDF(s)")
    uploaded_pdf_files = st.file_uploader(
        f"Upload PDF file(s), max {MAX_INPUT_ITEMS}",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploaded_pdf_files",
    )
    if uploaded_pdf_files and len(uploaded_pdf_files) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} PDFs.")
    uploaded_pdf_ids_text = st.text_area(
        "Document IDs (optional, one per PDF)",
        height=100,
        help="Leave blank to use uploaded/<filename>. These IDs are used anywhere SALSA normally shows a DOI.",
    )
    doi_text = ""
    literature_query_text = ""
elif use_figure_uploads:
    st.subheader("Input Figure(s)")
    uploaded_figure_files = st.file_uploader(
        f"Upload figure image file(s), max {MAX_INPUT_ITEMS}",
        type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
        key="uploaded_figure_files",
    )
    if uploaded_figure_files and len(uploaded_figure_files) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} figure images.")
    uploaded_figure_document_id = st.text_input(
        "Figure set ID (optional)",
        value="",
        help="Leave blank to use uploaded/<filename> for one figure or uploaded/figures for multiple figures.",
    )
    uploaded_figure_captions_text = st.text_area(
        "Figure captions (optional, one per uploaded figure)",
        height=120,
    )
    uploaded_pdf_files = []
    uploaded_pdf_ids_text = ""
    doi_text = ""
    literature_query_text = ""
else:
    uploaded_pdf_files = []
    uploaded_pdf_ids_text = ""
    uploaded_figure_files = []
    uploaded_figure_document_id = ""
    uploaded_figure_captions_text = ""
    st.subheader("Input DOI(s)")
    doi_text = st.text_area(
        f"One DOI per line, max {MAX_INPUT_ITEMS} "
        "(to increase quota, please reach out to SALSA creators at "
        "wschertzer3@gatech.edu)",
        height=140,
    )
    literature_query_text = ""

st.subheader("Run configuration")
default_run_id = st.session_state.get("pending_run_id", "")
run_id_input = st.text_input(
    "Run Name (optional)",
    value=default_run_id,
    help="Leave blank to auto-generate timestamp-based run name",
)
smtp_configured_for_runs = bool(
    os.environ.get("SALSA_SMTP_HOST") and os.environ.get("SALSA_SMTP_FROM")
)
notify_run_complete = st.checkbox(
    "Email me when this run finishes",
    value=bool(smtp_configured_for_runs),
    key="notify_run_complete",
    disabled=not smtp_configured_for_runs,
    help=(
        f"Send a completion, failure, or cancellation notice to {current_user['email']}."
        if smtp_configured_for_runs
        else "Email delivery is not configured on this SALSA server."
    ),
)
if not smtp_configured_for_runs:
    st.caption("Run notification emails are unavailable until SMTP is configured.")

st.session_state["pending_run_id"] = run_id_input


if st.button(
    "Run pipeline",
    type="primary",
    on_click=_clear_loaded_run_for_pipeline_launch,
):
    uploaded_pdfs = []
    uploaded_figures = []
    figure_document_id = ""
    if use_query:
        if not literature_query_text.strip():
            st.error(f"Enter a {query_provider} query.")
            st.stop()
        try:
            _src_root = Path(__file__).resolve().parent.parent
            if str(_src_root) not in sys.path:
                sys.path.insert(0, str(_src_root))
            if query_provider == "Web of Science":
                from wos_query import WebOfScienceClient

                client = WebOfScienceClient(
                    api_key=str(st.session_state.get("wos_api_key", "")).strip()
                )
            elif query_provider == "OpenAlex":
                from literature_query import OpenAlexClient

                client = OpenAlexClient(
                    api_key=str(st.session_state.get("openalex_api_key", "")).strip()
                )
            else:
                from literature_query import ScopusClient

                client = ScopusClient(
                    api_key=str(st.session_state.get("elsevier_api_key", "")).strip(),
                    inst_token=str(st.session_state.get("scopus_inst_token", "")).strip(),
                )
            with st.spinner(f"Querying {query_provider} for DOIs..."):
                if query_provider == "OpenAlex":
                    openalex_sort = {
                        "Relevance (highest first)": "relevance_score:desc",
                        "Most cited": "cited_by_count:desc",
                        "Newest": "publication_date:desc",
                    }[openalex_sort_label]
                    openalex_oa_value = {
                        "Any": None,
                        "Open access only": True,
                        "Closed access only": False,
                    }[openalex_open_access]
                    dois = client.search_dois(
                        literature_query_text,
                        max_dois=openalex_top_dois,
                        publisher=openalex_publisher,
                        open_access=openalex_oa_value,
                        sort=openalex_sort,
                        search_mode=openalex_search_mode.lower(),
                    )
                else:
                    dois = client.search_dois(
                        literature_query_text,
                        max_records=MAX_INPUT_ITEMS,
                    )
        except (ValueError, RuntimeError) as exc:
            st.error(f"{query_provider} query failed: {exc}")
            st.stop()
        if not dois:
            st.error(f"The {query_provider} query returned no records with DOIs.")
            st.stop()
        st.success(f"Found {len(dois)} unique DOIs in {query_provider}.")
    elif use_docling_pdf_uploads:
        id_lines = [line.strip() for line in uploaded_pdf_ids_text.splitlines() if line.strip()]
        for idx, uploaded_file in enumerate(uploaded_pdf_files or []):
            document_id = id_lines[idx] if idx < len(id_lines) else None
            uploaded_pdfs.append({
                "filename": uploaded_file.name,
                "content": uploaded_file.getvalue(),
                "document_id": document_id,
            })
        dois = [
            item["document_id"]
            or f"uploaded/{re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(item['filename']).stem).strip('._-') or 'uploaded_pdf'}"
            for item in uploaded_pdfs
        ]
    elif use_figure_uploads:
        caption_lines = uploaded_figure_captions_text.splitlines()
        figure_document_id = uploaded_figure_document_id.strip()
        for idx, uploaded_file in enumerate(uploaded_figure_files or []):
            caption = caption_lines[idx].strip() if idx < len(caption_lines) else ""
            uploaded_figures.append({
                "filename": uploaded_file.name,
                "content": uploaded_file.getvalue(),
                "caption": caption,
            })
        if figure_document_id:
            dois = [figure_document_id]
        elif len(uploaded_figures) == 1:
            first_name = uploaded_figures[0]["filename"] if uploaded_figures else "uploaded_figure"
            dois = [f"uploaded/{re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(first_name).stem).strip('._-') or 'uploaded_figure'}"]
        else:
            dois = ["uploaded/figures"] if uploaded_figures else []
    else:
        dois = [d.strip() for d in doi_text.splitlines() if d.strip()]
    st.session_state["doi_order"] = dois
    if use_docling_pdf_uploads and not uploaded_pdfs:
        st.error("Please upload at least one PDF.")
    elif use_docling_pdf_uploads and len(uploaded_pdfs) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} PDFs.")
    elif use_figure_uploads and not uploaded_figures:
        st.error("Please upload at least one figure image.")
    elif use_figure_uploads and len(uploaded_figures) > MAX_INPUT_ITEMS:
        st.error(f"Upload at most {MAX_INPUT_ITEMS} figure images.")
    elif not use_file_uploads and not dois:
        st.error("Please paste at least one DOI.")
    elif not use_file_uploads and len(dois) > MAX_INPUT_ITEMS:
        st.error(f"Enter at most {MAX_INPUT_ITEMS} DOIs.")
    else:
        active_llm_steps = _active_llm_steps(
            run_extract_tables=run_extract_tables,
            use_docling_pdf_uploads=use_file_uploads,
            run_chemical_composition=run_chemical_composition,
            run_filter_figures=run_filter_figures,
            run_filter_tables=run_filter_tables and not use_figure_uploads,
            run_plextract=run_plextract,
            bypass_ocr_with_llm=bypass_ocr_with_llm,
            run_complete_dataset=run_complete_dataset,
        )
        missing_llm_steps = _missing_llm_steps(active_llm_steps)
        missing_openai_api_key_steps = _missing_openai_api_key_steps(active_llm_steps)
        if missing_llm_steps:
            st.error(
                "Configure an LLM model for these stages before running: "
                + ", ".join(missing_llm_steps)
            )
        elif missing_openai_api_key_steps:
            st.error(
                "Enter an OpenAI API key in the sidebar before running these Chat GPT stages: "
                + ", ".join(missing_openai_api_key_steps)
            )
        else:
            custom_run_id = run_id_input.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

            _sync_all_step_llm_envs_from_session()
            _sync_publisher_tdm_env_from_session()

            try:
                target = safe_child_path(run_root, custom_run_id)
            except PermissionError as exc:
                st.error(f"Invalid run name: {exc}")
                st.stop()

            if not _WORKSPACE_RE.match(custom_run_id):
                st.error("Run name may only contain letters, numbers, underscores, dots, or dashes.")
                st.stop()

            if custom_run_id:
                if target.exists():
                    st.error(
                        f"Run directory already exists: {target}. "
                        "Choose a different run name or delete the existing run."
                    )
                    st.stop()

            # --- Save custom instructions for this run ---
            _run_ci = st.session_state.get("custom_instructions", default_instructions())
            # Parse output_columns if still a string
            if isinstance(_run_ci.get("output_columns"), str):
                _run_ci["output_columns"] = [c.strip() for c in _run_ci["output_columns"].splitlines() if c.strip()]
            if isinstance(_run_ci.get("missing_value_defaults"), str):
                try:
                    _run_ci["missing_value_defaults"] = json.loads(_run_ci["missing_value_defaults"])
                except Exception:
                    pass

            # Debug: show what preset is being used
            preset_name = (
                st.session_state.get("current_settings_preset_name")
                or st.session_state.get("current_preset_name")
                or "custom"
            )
            if preset_name and _settings_preset_exists(workspaces_root, preset_name):
                saved_preset_ci = _load_settings_preset(
                    workspaces_root,
                    preset_name,
                ).get("custom_instructions", {})
                if saved_preset_ci != _run_ci:
                    preset_name = f"{preset_name} (modified)"
            st.info(f"Using custom instructions: **{preset_name}**")
            with st.expander("View custom instructions being used", expanded=False):
                st.json(_run_ci)

            pipeline_kwargs = dict(
                dois=dois,
                run_root=run_root,
                scripts_root=scripts_root,
                run_id=custom_run_id,
                run_extract_figures=run_extract_figures,
                run_extract_tables=run_extract_tables,
                run_pdf_page_ocr=run_pdf_page_ocr if use_docling_pdf_uploads else False,
                run_filter_figures=run_filter_figures,
                run_filter_tables=run_filter_tables and not use_figure_uploads,
                run_plextract=run_plextract and not bypass_ocr_with_llm,
                run_llm_figure_extraction=run_plextract and bypass_ocr_with_llm,
                run_complete_dataset=run_complete_dataset,
                run_aggregation=run_aggregation,
                skip_llm_figures=not bypass_ocr_with_llm,
                run_chemical_composition=run_chemical_composition,
                exclude_full_text=exclude_full_text,
                custom_instructions=_run_ci,
                custom_instructions_name=preset_name,
                uploaded_pdfs=uploaded_pdfs if use_docling_pdf_uploads else None,
                uploaded_figures=uploaded_figures if use_figure_uploads else None,
                uploaded_figure_document_id=figure_document_id if use_figure_uploads else None,
                input_item_limit=MAX_INPUT_ITEMS,
                _auth_context={
                    "user_id": int(current_user["id"]),
                    "username": current_user["username"],
                    "display_name": current_user["display_name"] or current_user["username"],
                    "email": current_user["email"],
                    "workspace": active_workspace_name,
                    "run_id": custom_run_id,
                    "run_dir": str(target),
                    "notify_on_completion": bool(notify_run_complete),
                    "query_provider": query_provider if use_query else "",
                    "query": literature_query_text if use_query else "",
                    "query_settings": (
                        {
                            "search_mode": openalex_search_mode,
                            "result_ranking": openalex_sort_label,
                            "publisher": openalex_publisher,
                            "open_access_filter": openalex_open_access,
                            "max_results": openalex_top_dois,
                        }
                        if use_query and query_provider == "OpenAlex"
                        else (
                            {"max_results": MAX_INPUT_ITEMS}
                            if use_query
                            else {}
                        )
                    ),
                },
            )
            st.session_state.pop("rr", None)
            st.session_state[_ACTIVE_PIPELINE_RUN_KEY] = _start_pipeline_worker(
                pipeline_kwargs,
                dois,
            )
            _rerun_app()
            st.stop()

rr: RunResult | None = st.session_state.get("rr")
if not rr:
    st.stop()

st.info(f"Run ID: **{rr.run_id}**")
st.info(f"Workspace: **{_workspace_display_name_for_run(rr.run_dir, workspaces_root)}**")
_render_run_query(rr.run_dir)
_render_run_custom_instructions(rr.run_dir)
_loaded_run_status_path = rr.run_dir / RUN_STATUS_FILE_NAME
_loaded_run_status = _mark_inactive_run_status(
    _loaded_run_status_path,
    read_run_status(_loaded_run_status_path),
)
if _run_status_is_active(_loaded_run_status):
    st.info(_run_progress_status_text(_loaded_run_status))
    if _loaded_run_status.get("state") == "cancelling":
        st.warning("Cancellation requested. Stopping the active pipeline step...")
    elif st.button(
        "Cancel Loaded Run",
        type="secondary",
        key=f"cancel_loaded_run_{rr.run_id}",
    ):
        request_run_cancellation(rr.run_dir)
        write_run_status(
            _loaded_run_status_path,
            state="cancelling",
            stage="Cancelling",
        )
        st.warning("Cancellation requested. Stopping the active pipeline step...")
    st.caption("This run is still active. Use Refresh status to update progress.")
    if st.button("Refresh status", key=f"refresh_loaded_run_{rr.run_id}"):
        _rerun_app()
    st.stop()
_render_run_status_outcome(rr.run_id, _loaded_run_status)
_render_cost_summary(rr.run_dir)

if rr.failures:
    st.subheader("Failed DOIs")
    st.caption(
        "The most specific failure found for each DOI is shown below."
    )
    failed_rows = [
        {
            "DOI": failure.doi,
            "Publisher": (
                _publisher_name_from_doi(failure.doi)
                if str(failure.publisher).strip().lower() in {"", "unknown", "docling_pdf"}
                else failure.publisher
            ),
            "Stage": failure.stage,
            "Reason": failure.message,
        }
        for failure in rr.failures
    ]
    st.dataframe(
        pd.DataFrame(failed_rows),
        use_container_width=True,
        hide_index=True,
    )

# ---------------- DOI summary table + DOI selector ----------------
st.subheader("Processed DOIs")

df_stats = build_doi_stats(rr)

# Render as interactive table (row click -> selects DOI when supported)
df_view = df_stats.drop(columns=["_doi_safe"], errors="ignore")
_doi_table_key = "doi_summary_table"
_doi_table_applied_key = "_doi_summary_table_applied_doi"

# Streamlit row selection support varies by version; use feature detection.
df_selected_doi = None
try:
    sig = inspect.signature(st.dataframe)
    if "on_select" in sig.parameters and "selection_mode" in sig.parameters:
        evt = st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=_doi_table_key,
        )
        if evt and "selection" in evt and evt["selection"].get("rows"):
            i = evt["selection"]["rows"][0]
            df_selected_doi = df_stats.loc[i, "DOI"]
    else:
        st.dataframe(df_view, use_container_width=True, hide_index=True)
except Exception:
    st.dataframe(df_view, use_container_width=True, hide_index=True)

# Keep selected DOI in session_state so table + dropdown stay in sync
doi_options = [d.doi for d in rr.dois]

if not doi_options:
    st.warning("No DOIs found in the loaded run. The run may be empty or incomplete.")
    st.stop()

if st.session_state.get("selected_doi") not in doi_options:
    st.session_state["selected_doi"] = doi_options[0]

if df_selected_doi in doi_options and st.session_state.get(_doi_table_applied_key) != df_selected_doi:
    # Only let the summary table drive selection when the clicked row changes.
    # Otherwise a stale selected row can overwrite the dropdown's first change
    # on the next rerun.
    st.session_state["selected_doi"] = df_selected_doi
    st.session_state[_doi_table_applied_key] = df_selected_doi
elif not df_selected_doi:
    st.session_state.pop(_doi_table_applied_key, None)

selected_doi = st.selectbox(
    "Select DOI",
    options=doi_options,
    key="selected_doi",
)

doi_item = next((d for d in rr.dois if d.doi == selected_doi), None)

if doi_item is None:
    st.error(f"Could not find DOI '{selected_doi}' in the loaded run. The data may have changed since this session was saved.")
    st.stop()

# ---------------- DOI detail view (under DOI dropdown) ----------------
_aggregated_data_path = rr.run_dir / "aggregated" / "tables_figures.csv"
_aggregated_composition_path = rr.run_dir / "aggregated" / "chemical_compositions.csv"
_doi_view_options = ["Parsed text", "Tables", "Figures", "Chemical Composition", "Dataset"]
_doi_view_key = "selected_doi_detail_view"
if st.session_state.get(_doi_view_key) not in _doi_view_options:
    st.session_state[_doi_view_key] = _doi_view_options[0]

if hasattr(st, "segmented_control"):
    selected_doi_view = st.segmented_control(
        "View",
        options=_doi_view_options,
        key=_doi_view_key,
        label_visibility="collapsed",
    )
else:
    selected_doi_view = st.radio(
        "View",
        options=_doi_view_options,
        key=_doi_view_key,
        horizontal=True,
        label_visibility="collapsed",
    )
selected_doi_view = selected_doi_view or st.session_state[_doi_view_key]

if selected_doi_view == "Parsed text":
    if doi_item.parsed_json_path and doi_item.parsed_json_path.exists():
        st.json(json.loads(doi_item.parsed_json_path.read_text()))
    else:
        st.warning("No parsed.json found for this DOI in the run workspace.")

if selected_doi_view == "Tables":
    if not doi_item.tables:
        st.warning("No tables found.")
    for idx, t in enumerate(doi_item.tables):
        # Add divider between tables
        if idx > 0:
            st.divider()

        st.markdown(f"### {t.id}")
        with st.expander("Table", expanded=False):
            if t.caption:
                st.markdown(f"**Caption:** {t.caption}")
            if t.table_path and t.table_path.exists():
                suf = t.table_path.suffix.lower()
                if suf in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                    st_image_auto(str(t.table_path), full_width=True)
                elif suf == ".csv":
                    df = pd.read_csv(t.table_path)
                    st.data_editor(df, use_container_width=True, key=f"table_{selected_doi}_{t.id}")
                else:
                    st.code(t.table_path.read_text(errors="ignore")[:20000])

        # --- Display LLM-generated emit_rows from row_cache ---
        emit_rows_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "table", t.id)
        if emit_rows_path:
            df_emit = load_emit_rows(emit_rows_path)
            if df_emit is not None and not df_emit.empty:
                with section_dropdown("LLM-Generated Dataset", expanded=False):
                    st.caption(f"**Source:** `{emit_rows_path.relative_to(rr.run_dir)}`")
                    st.data_editor(df_emit, use_container_width=True, hide_index=True, key=f"table_emit_{selected_doi}_{t.id}")

                    csv_data = df_emit.to_csv(index=False)
                    st.download_button(
                        label="Download emit_rows CSV",
                        data=csv_data,
                        file_name=f"{t.id}_emit_rows.csv",
                        mime="text/csv",
                    )
        else:
            # Check if row_cache exists but no data found for this table
            row_cache_dir = rr.run_dir / "complete_dataset" / "row_cache"
            if row_cache_dir.exists():
                with section_dropdown("LLM-Generated Dataset", expanded=False):
                    st.info(f"No row_cache data found for {t.id}. Cache may not have been generated yet.")

        # Metadata expander
        if t.metadata_path and t.metadata_path.exists():
            with st.expander("Metadata", expanded=False):
                st.json(json.loads(t.metadata_path.read_text()))

if selected_doi_view == "Figures":
    figures_view.render_figures_view(
        rr=rr,
        doi_item=doi_item,
        deps=figures_view.FiguresViewDeps(
            run_dir=rr.run_dir,
            scripts_root=scripts_root,
            selected_doi=selected_doi,
            safe_doi=doi_to_safe(selected_doi),
            find_emit_rows_for_source=find_emit_rows_for_source,
            step_has_configured_model=_step_has_configured_model,
            missing_openai_api_key_steps=_missing_openai_api_key_steps,
            complete_dataset_env=lambda: _complete_dataset_env_for_current_settings(rr),
            run_plextract_on_crops=_run_plextract_on_crops,
            llm_step_complete_dataset=LLM_STEP_COMPLETE_DATASET,
        ),
        find_source_image=_try_find_source_image,
    )

if selected_doi_view == "Dataset":
    st.subheader("Run datasets")
    st.caption("These files contain data from the entire run, not only the selected DOI.")

    if st.button(
        "Agglomerate current run data",
        type="primary",
        key=f"agglomerate_loaded_run_{rr.run_id}",
        help=(
            "Rebuild both run-wide CSVs from the current LLM datasets and chemical "
            "compositions. New LLM-generated figure datasets replace their original versions."
        ),
    ):
        try:
            _src_root_for_aggregation = Path(__file__).resolve().parent.parent
            if str(_src_root_for_aggregation) not in sys.path:
                sys.path.insert(0, str(_src_root_for_aggregation))
            from run_aggregation import aggregate_run as _aggregate_loaded_run

            with st.spinner("Agglomerating current run data..."):
                _aggregation_result = _aggregate_loaded_run(rr.run_dir)
            _load_csv.clear()
            st.success(
                f"Agglomerated {_aggregation_result.data_rows:,} table/figure rows and "
                f"{_aggregation_result.composition_rows:,} chemical-composition rows."
            )
            for _aggregation_warning in _aggregation_result.warnings:
                st.warning(_aggregation_warning)
        except Exception as _aggregation_exc:
            st.error(f"Agglomeration failed: {_aggregation_exc}")

    for title, csv_path, download_name in (
        ("Tables and figures", _aggregated_data_path, "tables_figures.csv"),
        ("Chemical compositions", _aggregated_composition_path, "chemical_compositions.csv"),
    ):
        st.markdown(f"#### {title}")
        if not csv_path.exists():
            st.info(f"No {title.lower()} aggregate was created for this run.")
            continue
        try:
            aggregate_df = _load_csv(str(csv_path))
            st.caption(f"{len(aggregate_df):,} rows · {len(aggregate_df.columns):,} columns")
            st.dataframe(aggregate_df, use_container_width=True, hide_index=True)
            st.download_button(
                f"Download {download_name}",
                data=csv_path.read_bytes(),
                file_name=download_name,
                mime="text/csv",
                key=f"download_aggregate_{download_name}",
            )
        except Exception as exc:
            st.warning(f"Could not load {csv_path.name}: {exc}")

if selected_doi_view == "Chemical Composition":
    st.subheader("Chemical Composition")

    # --- Add src/ to path for chemical_composition imports ---
    _src_for_chem = (Path(__file__).resolve().parent / "..").resolve()
    if str(_src_for_chem) not in sys.path:
        sys.path.insert(0, str(_src_for_chem))

    import importlib
    import llm_client as _llm_client_module
    from chemical_composition import extract_composition as _cc_module

    # Streamlit keeps imported modules alive across reruns. Reload the LLM client
    # and chemistry extractor so model fixes are picked up without restarting.
    importlib.reload(_llm_client_module)
    _cc_module = importlib.reload(_cc_module)

    _cc_identify = _cc_module.identify_relevant_content
    _cc_extract = _cc_module.extract_samples_and_compositions
    _cc_run = _cc_module.run_chemical_composition_extraction

    def _cc_build_table_fallback(samples: list[dict], *, doi: str = ""):
        def _num(value):
            text = "" if value is None else str(value).strip()
            if not text:
                return ""
            if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
                return f"{float(text):.12g}"
            assignment = re.search(r"(?:^|[\s,(;])(?:x|y|z|n|m)\s*=\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
            if assignment:
                value_f = float(assignment.group(1))
                return f"{(value_f / 100.0 if value_f > 1 else value_f):.12g}"
            exp = re.search(r"(?:experimental|experimentally|exp\.?)\D{0,60}?([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
            if exp:
                value_f = float(exp.group(1))
                return f"{(value_f / 100.0 if value_f > 1 else value_f):.12g}"
            pct = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*(?:%|mol\s*%|wt\s*%|vol\s*%)", text, flags=re.I)
            if pct:
                return f"{(float(pct[0]) / 100.0):.12g}"
            ctx = re.search(r"\b(?:dob|degree|graft(?:ing|ed)?|content|loading|ratio|fraction|functionalization)\b\D{0,120}?([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
            if ctx:
                value_f = float(ctx.group(1))
                return f"{(value_f / 100.0 if 1 < value_f <= 100 else value_f):.12g}"
            return ""

        def _smiles(value, *, polymer=False):
            smiles = "" if value is None else str(value).strip()
            if not smiles:
                return ""
            if polymer and "*" not in smiles:
                return ""
            return smiles

        max_polymer = 3
        max_additive = 3
        normalized = []
        for sample in samples:
            polymers = sample.get("polymer_components") or sample.get("co_monomers") or []
            additives = sample.get("additive_components") or []
            if not additives and isinstance(sample.get("additive"), dict):
                additive = sample["additive"]
                if additive.get("name") or additive.get("content") or additive.get("smiles"):
                    additives = [additive]
            polymers = [p for p in polymers if isinstance(p, dict)]
            additives = [a for a in additives if isinstance(a, dict)]
            normalized.append((sample.get("sample_name", ""), polymers, additives))
            max_polymer = max(max_polymer, len(polymers))
            max_additive = max(max_additive, len(additives))

        columns = (
            ["DOI", "sample_name"]
            + [f"smiles{i}" for i in range(1, max_polymer + 1)]
            + [f"c{i}" for i in range(1, max_polymer + 1)]
            + [f"additive_smiles{i}" for i in range(1, max_additive + 1)]
            + [f"additive_c{i}" for i in range(1, max_additive + 1)]
            + ["additive_name_1", "additive_name_2"]
        )
        rows = []
        for sample_name, polymers, additives in normalized:
            row = {column: "" for column in columns}
            row["DOI"] = doi
            row["sample_name"] = sample_name
            for idx, component in enumerate(polymers, start=1):
                row[f"smiles{idx}"] = _smiles(component.get("smiles", ""), polymer=True)
                row[f"c{idx}"] = _num(
                    component.get("ratio")
                    or component.get("content")
                    or component.get("amount")
                    or component.get("ratio_text")
                    or component.get("content_text")
                )
            for idx, component in enumerate(additives, start=1):
                row[f"additive_smiles{idx}"] = _smiles(component.get("smiles", ""))
                row[f"additive_c{idx}"] = _num(
                    component.get("content")
                    or component.get("ratio")
                    or component.get("amount")
                    or component.get("content_text")
                    or component.get("ratio_text")
                )
                if idx <= 2:
                    row[f"additive_name_{idx}"] = str(component.get("name", "") or "")
            rows.append(row)
        return rows, columns

    _cc_build_table = getattr(
        _cc_module,
        "build_composition_table_rows",
        _cc_build_table_fallback,
    )
    from chemical_composition.drawing_tool import (
        render_jsme_editor_local,
        render_ketcher_editor,
        load_drawn_structures,
        save_drawn_structure,
        delete_drawn_structure,
    )

    _cc_safe = doi_to_safe(selected_doi)

    # Show previously extracted results if they exist
    cc_json_path = doi_item.extra.get("chemical_composition_json")
    cc_csv_path = doi_item.extra.get("chemical_composition_csv")

    if cc_json_path and Path(cc_json_path).exists():
        cc_data = json.loads(Path(cc_json_path).read_text())

        # --- Samples table ---
        with st.expander("Extracted Sample Composition Table", expanded=True):
            samples = cc_data.get("samples", [])
            if samples:
                try:
                    _rows, _cols = _cc_build_table(samples, doi=cc_data.get("doi", selected_doi))
                    df_cc = pd.DataFrame(_rows, columns=_cols)
                    st.dataframe(df_cc, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download samples CSV",
                        data=df_cc.to_csv(index=False).encode("utf-8"),
                        file_name=f"{_cc_safe}_samples.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(f"Failed to build samples table from JSON: {e}")
            elif cc_csv_path and Path(cc_csv_path).exists():
                try:
                    df_cc = pd.read_csv(cc_csv_path)
                    st.dataframe(df_cc, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download samples CSV",
                        data=Path(cc_csv_path).read_bytes(),
                        file_name=f"{_cc_safe}_samples.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(f"Failed to load samples CSV: {e}")
            else:
                st.info("No samples extracted.")

        # --- Relevant content (with full table/figure rendering) ---
        with st.expander("Relevant Content", expanded=False):
            relevant = cc_data.get("relevant_content", [])
            if relevant:
                # First priority: chemical_composition directory (preserved copies)
                _rc_chem_figures_dir = rr.run_dir / "chemical_composition" / _cc_safe / "figures"
                _rc_chem_tables_dir = rr.run_dir / "chemical_composition" / _cc_safe / "tables"
                # Fallback: figures_tables directory (may have been filtered)
                _rc_figures_dir = rr.run_dir / "figures_tables" / "figures" / _cc_safe
                _rc_tables_dir = rr.run_dir / "figures_tables" / "tables" / _cc_safe

                # Gather all figure image paths (prioritizing chemical_composition directory)
                _rc_all_fig_paths: dict[str, Path] = {}
                # First, check chemical_composition directory
                if _rc_chem_figures_dir.exists():
                    for _fp in _rc_chem_figures_dir.iterdir():
                        if _fp.is_file() and _fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                            _rc_all_fig_paths[_fp.stem.lower().replace(" ", "").replace("_", "")] = _fp
                # Then, add from figures_tables (if not already present)
                if _rc_figures_dir.exists():
                    for _fp in _rc_figures_dir.iterdir():
                        if _fp.is_file() and _fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                            _norm_key = _fp.stem.lower().replace(" ", "").replace("_", "")
                            if _norm_key not in _rc_all_fig_paths:
                                _rc_all_fig_paths[_norm_key] = _fp

                # Also include figures already loaded on the DOI item
                for _fig in doi_item.figures:
                    if _fig.figure_path and _fig.figure_path.exists():
                        _norm_key = _fig.id.lower().replace(" ", "").replace("_", "")
                        if _norm_key not in _rc_all_fig_paths:
                            _rc_all_fig_paths[_norm_key] = _fig.figure_path

                # Gather all table file paths (prioritizing chemical_composition directory)
                _rc_all_table_paths: dict[str, Path] = {}
                _rc_all_table_captions: dict[str, str] = {}
                # First, check chemical_composition directory
                if _rc_chem_tables_dir.exists():
                    for _tp in _rc_chem_tables_dir.iterdir():
                        if _tp.is_file() and not _tp.name.endswith("_metadata.json"):
                            _norm_id = _tp.stem.lower().replace(" ", "").replace("_", "")
                            _rc_all_table_paths[_norm_id] = _tp
                            # Try to load caption from sidecar metadata
                            _meta_p = _tp.with_name(_tp.stem + "_metadata.json")
                            if _meta_p.exists():
                                try:
                                    _rc_all_table_captions[_norm_id] = json.loads(_meta_p.read_text()).get("caption", "")
                                except Exception:
                                    pass
                # Then, add from figures_tables (if not already present)
                if _rc_tables_dir.exists():
                    for _tp in _rc_tables_dir.iterdir():
                        if _tp.is_file() and not _tp.name.endswith("_metadata.json"):
                            _norm_id = _tp.stem.lower().replace(" ", "").replace("_", "")
                            if _norm_id not in _rc_all_table_paths:
                                _rc_all_table_paths[_norm_id] = _tp
                                # Try to load caption from sidecar metadata
                                _meta_p = _tp.with_name(_tp.stem + "_metadata.json")
                                if _meta_p.exists():
                                    try:
                                        _rc_all_table_captions[_norm_id] = json.loads(_meta_p.read_text()).get("caption", "")
                                    except Exception:
                                        pass

                # Also include tables already loaded on the DOI item
                for _tbl in doi_item.tables:
                    _norm_id = _tbl.id.lower().replace(" ", "").replace("_", "")
                    if _tbl.table_path and _tbl.table_path.exists():
                        if _norm_id not in _rc_all_table_paths:
                            _rc_all_table_paths[_norm_id] = _tbl.table_path
                    if _tbl.caption and _norm_id not in _rc_all_table_captions:
                        _rc_all_table_captions[_norm_id] = _tbl.caption

                for i, item_rc in enumerate(relevant):
                    st.markdown(
                        f"**{item_rc.get('source_type', '')}** — "
                        f"{item_rc.get('source_id', '')}"
                    )
                    # Show relevance as wrapped text
                    _rel_text = item_rc.get("relevance", "")
                    if _rel_text:
                        st.markdown(f"_{_rel_text}_")

                    _src_type = item_rc.get("source_type", "").lower()
                    _src_id = item_rc.get("source_id", "").lower().replace(" ", "")
                    _src_id_norm = _src_id.replace("_", "")

                    # Display the referenced figure/scheme image if available
                    if _src_type in ("figure/scheme", "figure", "scheme"):
                        _matched_fig = False
                        for _fkey, _fpath in _rc_all_fig_paths.items():
                            if _src_id_norm and (_src_id_norm in _fkey or _fkey in _src_id_norm):
                                _rc_img_col, _ = st.columns([1, 2])
                                with _rc_img_col:
                                    st_image_auto(str(_fpath), full_width=True)
                                _matched_fig = True
                                break
                        # Suppress "not found" message - item may have been filtered intentionally

                    # Display the referenced table if available
                    if _src_type in ("table",):
                        _matched_tbl = False
                        for _tkey, _tpath in _rc_all_table_paths.items():
                            if _src_id_norm and (_src_id_norm in _tkey or _tkey in _src_id_norm):
                                _tbl_caption = _rc_all_table_captions.get(_tkey, "")
                                if _tbl_caption:
                                    st.caption(_tbl_caption)
                                _suf = _tpath.suffix.lower()
                                if _suf in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                                    st_image_auto(str(_tpath), full_width=True)
                                elif _suf == ".csv":
                                    try:
                                        _tdf = pd.read_csv(_tpath)
                                        st.dataframe(_tdf, use_container_width=True, hide_index=True)
                                    except Exception:
                                        st.code(_tpath.read_text(errors="ignore")[:20000])
                                else:
                                    st.code(_tpath.read_text(errors="ignore")[:20000])
                                _matched_tbl = True
                                break
                        # Suppress "not found" message - item may have been filtered intentionally

                    if item_rc.get("excerpt"):
                        # Use markdown for better text wrapping (no horizontal scrolling)
                        st.markdown(f"<div style='white-space: pre-wrap; word-wrap: break-word;'>{item_rc['excerpt'][:500]}</div>", unsafe_allow_html=True)
                    if i < len(relevant) - 1:
                        st.divider()
            else:
                st.info("No relevant content identified.")

    else:
        st.info(
            "No chemical composition data found for this DOI.\n\n"
            "Enable **Extract chemical composition** in the sidebar and re-run, "
            "or use the **Run Extraction** dropdown below to extract on demand."
        )

    # --- Chemical Structure Drawing Tools ---
    with st.expander("Draw Chemical Structures - JSME", expanded=False):
        render_jsme_editor_local(height=550, key=f"jsme_{_cc_safe}")

    with st.expander("Draw Chemical Structures - Ketcher", expanded=False):
        render_ketcher_editor(height=720, key=f"ketcher_{_cc_safe}")

    with st.expander("Save Drawn Structure", expanded=False):
        draw_sample_name = st.text_input(
            "Sample name for this structure",
            key=f"draw_sample_{_cc_safe}",
            placeholder="e.g. QPPO-TMA-40",
        )
        draw_smiles = st.text_input(
            "SMILES (paste from editor or type manually)",
            key=f"draw_smiles_{_cc_safe}",
            placeholder="e.g. C(C(=O)O)N",
        )
        draw_notes = st.text_input(
            "Notes (optional)",
            key=f"draw_notes_{_cc_safe}",
            placeholder="e.g. backbone polymer for sample set A",
        )

        if st.button("Save structure", type="primary", key=f"save_drawn_{_cc_safe}"):
            if not draw_sample_name.strip():
                st.error("Please provide a sample name.")
            else:
                save_drawn_structure(
                    run_dir=rr.run_dir,
                    safe_doi=_cc_safe,
                    sample_name=draw_sample_name.strip(),
                    smiles=draw_smiles.strip(),
                    notes=draw_notes.strip(),
                )
                st.success(f"Saved structure for '{draw_sample_name.strip()}'")
                st.rerun()

    # --- User-Drawn Structures ---
    with st.expander("User-Drawn Structures", expanded=False):
        drawn = load_drawn_structures(rr.run_dir, _cc_safe)
        if drawn:
            for idx, ds in enumerate(drawn):
                cols = st.columns([3, 2, 1])
                with cols[0]:
                    st.markdown(f"**{ds.get('sample_name', 'Unnamed')}**")
                    if ds.get("smiles"):
                        st.code(ds["smiles"], language=None)
                with cols[1]:
                    if ds.get("notes"):
                        st.caption(ds["notes"])
                with cols[2]:
                    if st.button("Delete", key=f"del_drawn_{_cc_safe}_{idx}"):
                        delete_drawn_structure(rr.run_dir, _cc_safe, idx)
                        st.rerun()
        else:
            st.info("No user-drawn structures saved yet. Use the drawing tool below.")

    # --- Run Extraction ---
    with st.expander("Run Chemical Composition Extraction", expanded=False):
        st.caption(
            "Run the LLM-based extraction on demand for this DOI. "
            "This will analyse the parsed text, tables, and figure captions "
            "to identify chemical structures, sample names, and compositions."
        )

        if st.button("Extract chemical composition", type="primary", key=f"run_cc_{_cc_safe}"):
            try:
                with st.spinner("Extracting chemical composition..."):
                    parsed_json = doi_item.parsed_json_path
                    tables_d = rr.run_dir / "figures_tables" / "tables" / _cc_safe
                    figures_d = rr.run_dir / "figures_tables" / "figures" / _cc_safe
                    missing_chem_steps = _missing_llm_steps(
                        [LLM_STEP_CHEM_RELEVANT, LLM_STEP_CHEM_COMPOSITION]
                    )
                    if missing_chem_steps:
                        raise RuntimeError(
                            "Configure an LLM model for these chemistry stages first: "
                            + ", ".join(missing_chem_steps)
                        )
                    missing_openai_api_key_steps = _missing_openai_api_key_steps(
                        [LLM_STEP_CHEM_RELEVANT, LLM_STEP_CHEM_COMPOSITION]
                    )
                    if missing_openai_api_key_steps:
                        raise RuntimeError(
                            "Enter an OpenAI API key in the sidebar before running these Chat GPT chemistry stages: "
                            + ", ".join(missing_openai_api_key_steps)
                        )
                    _sync_all_step_llm_envs_from_session()

                    result = _cc_run(
                        run_dir=rr.run_dir,
                        safe_doi=_cc_safe,
                        parsed_json_path=parsed_json,
                        tables_dir=tables_d if tables_d.exists() else None,
                        figures_dir=figures_d if figures_d.exists() else None,
                        composition_rules=st.session_state.get("custom_instructions", {}).get(
                            "chemical_composition_rules",
                            "",
                        ),
                        required_columns=st.session_state.get("custom_instructions", {}).get(
                            "chemical_composition_columns"
                        ),
                        required_defaults=st.session_state.get("custom_instructions", {}).get(
                            "chemical_composition_defaults"
                        ),
                    )

                st.success("Extraction complete!")

                # Show results inline
                samples = result.get("samples", [])
                st.markdown(f"**Found {len(samples)} sample(s)**")
                if samples:
                    for sample_idx, sample in enumerate(samples, start=1):
                        sample_name = sample.get("sample_name", "Unknown")
                        st.markdown(f"**Sample {sample_idx}: {sample_name}**")
                        st.json(sample)

                relevant = result.get("relevant_content", [])
                st.markdown(f"**Identified {len(relevant)} relevant content section(s)**")

                st.info("Reload the run or switch tabs to see the full results.")

            except Exception as e:
                st.error(f"Chemical composition extraction failed: {e}")
