# AEM_OCR/src/ui/streamlit_app.py
from __future__ import annotations
import json
import inspect
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pipeline import run_full_pipeline
from discover import discover_run
from schemas import RunResult
from intervention import find_interventions, latest_plextract_run_dir
from intervention import (
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
from run_plextract_manual import run_plextract_on_crops
from custom_instructions import (
    default_instructions,
    list_presets,
    save_preset,
    load_preset,
    delete_preset,
)
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
import os
import sys
import subprocess
import sys
from typing import List, Optional
import re
from datetime import datetime

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
    # nice default when user hasn't chosen anything yet
    return datetime.now().strftime("workspace_%Y%m%d")

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

def _ollama_list_models_via_cli() -> List[str]:
    """
    Best source of truth: `ollama list` returns the models actually available to the ollama daemon.
    """
    try:
        # Note: OLLAMA_HOST can be set in env to point to a non-default daemon
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
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
        return []

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

def st_image_auto(img, *, full_width=True, **kwargs):
    """
    Streamlit image compat:
      - newer streamlit: use_container_width
      - older streamlit: use_column_width
    Avoid passing both.
    """
    sig = inspect.signature(st.image)
    kwargs.pop("use_container_width", None)
    kwargs.pop("use_column_width", None)
    if "use_container_width" in sig.parameters:
        return st.image(img, use_container_width=full_width, **kwargs)
    return st.image(img, use_column_width=full_width, **kwargs)

def doi_to_safe(doi: str) -> str:
    return doi.replace("/", "@").strip()

def _render_cost_summary(run_dir: Path):
    """Display cost summary for a run if usage.jsonl exists."""
    usage_path = run_dir / "cost" / "usage.jsonl"
    if not usage_path.exists():
        return
    summary = summarise_usage(str(usage_path))
    if not summary["models"]:
        return
    with st.expander("Cost Summary", expanded=False):
        for model_name, info in summary["models"].items():
            label = model_name if model_name != "unknown" else "unknown model"
            st.markdown(
                f"**{label}**: {info['calls']} calls, "
                f"{info['input']:,} input tokens, "
                f"{info['cached']:,} cached, "
                f"{info['output']:,} output tokens "
                f"— **${info['cost']:.4f}**"
            )
        st.markdown(f"### Total: **${summary['grand_total']:.4f}**")

@st.cache_data(show_spinner=False)
def _cached_flagged(run_dir_str: str):
    # cache intervention scan so it doesn't re-run constantly
    run_dir = Path(run_dir_str)
    return find_interventions(run_dir)

@st.cache_data(show_spinner=False)
def _load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

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
    cache_key_prefix = f"emit_rows::{source_kind}::{doi}::{source_name}::"
    safe_prefix = _safe_filename_compat(cache_key_prefix)

    candidates = []
    for cache_file in cache_dir.iterdir():
        if cache_file.suffix != ".json":
            continue
        # Files are named: <safe_prefix><hash>__<sha256>.json
        # Check that the filename starts with our safe_prefix
        if cache_file.stem.startswith(safe_prefix) or cache_file.name.startswith(safe_prefix):
            candidates.append(cache_file)

    # Return most recent if multiple found
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def load_emit_rows(cache_path: Path | None) -> pd.DataFrame | None:
    """
    Load emit_rows from cache file and return as DataFrame.

    Cache file format: {"rows": [...], "metadata": {...}}
    """
    if not cache_path or not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)

        # Support both {"rows": [...]} and [...]
        if isinstance(data, dict) and "rows" in data:
            rows = data["rows"]
        elif isinstance(data, list):
            rows = data
        else:
            return None

        if not rows:
            return None

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Failed to load emit_rows: {e}")
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


def build_doi_stats(rr: RunResult, *, include_interventions: bool, doi_order: list[str] | None = None) -> pd.DataFrame:
    rows = []
    flagged = []
    if include_interventions:
        flagged = _cached_flagged(str(rr.run_dir))

    # count interventions per safe_doi
    flagged_counts = {}
    if flagged:
        for x in flagged:
            flagged_counts[x.safe_doi] = flagged_counts.get(x.safe_doi, 0) + 1

    for d in rr.dois:
        safe = doi_to_safe(d.doi)
        rows.append({
            "DOI": d.doi,
            "Figures": len(d.figures or []),
            "Tables": len(d.tables or []),
            "Interventions Needed": flagged_counts.get(safe, 0) if include_interventions else None,
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
    models = _ollama_list_models_via_cli()
    if not models:
        models = _ollama_list_models_from_dir(Path(str(default_ollama_dir)))
    return models

# SET OLLAMA DIR AND HOST (computed outside expander for reuse)
default_ollama_dir = _default_ollama_models_dir()
default_ollama_host = _default_ollama_host()

_openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini", "gpt-5-nano", "gpt-5"]



st.set_page_config(page_title="AEM_OCR UI", layout="wide")
st.title("AEM_OCR")

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

# ---------------- Sidebar ----------------
with st.sidebar:

    # Set workspace variables – resolve relative to this file, not cwd
    scripts_root = (Path(__file__).resolve().parent / "..").resolve()
    workspaces_root = Path(str(scripts_root / "ui_runs"))
    workspaces_root.mkdir(parents=True, exist_ok=True)

    st.divider()

    # Choose/create workspace
    st.header("Workspace")

    # List existing workspaces (filter out axis_presets and instruction_presets)
    existing_workspaces = [
        ws for ws in _list_subdirs(workspaces_root)
        if ws not in ("axis_presets", "instruction_presets")
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
    st.header("Select Stages")
    with st.expander("Stages", expanded=False):
        run_download_doi = st.checkbox("Download DOIs", value=True)
        run_extract_figures = st.checkbox("Extract figures", value=True)
        run_extract_tables = st.checkbox("Extract tables", value=True)
        run_chemical_composition = st.checkbox("Extract chemical composition", value=True)
        run_filter_figures = st.checkbox("Filter figures", value=True)
        run_filter_tables = st.checkbox("Filter tables", value=True)
        run_plextract = st.checkbox("Extract data from figures", value=True)
        run_complete_dataset = st.checkbox("Complete dataset", value=True)
        exclude_full_text = st.checkbox("Exclude full text from context", value=True)

    st.divider()
    st.header("Select LLM(s)")
    with st.expander("LLM", expanded=False):
        # --- General LLM (used for filtering, figure planning, etc.) ---
        st.subheader("General LLM")
        st.caption("Used for figure/table filtering, figure planning, and other pre-processing steps.")

        provider = st.radio(
            "Provider",
            options=["Ollama", "Chat GPT"],
            index=0,
            key="llm_provider",
        )

        if provider == "Ollama":
            ollama_models_dir = Path(str(default_ollama_dir))
            ollama_host = default_ollama_host

            models = _ollama_models()
            if models:
                # Clear stale value from a previous provider so index=0 takes effect.
                if st.session_state.get("llm_model") not in models:
                    st.session_state.pop("llm_model", None)
                selected_model = st.selectbox("Model", options=models, index=0, key="llm_model")
            else:
                selected_model = st.text_input(
                    "Model (no models detected)",
                    value="",
                    key="llm_model",
                )
                st.caption("Tip: run `ollama pull <model>` then refresh.")
        else:
            # Read OpenAI config from environment (not displayed in UI)
            if "openai_base_url" not in st.session_state:
                st.session_state["openai_base_url"] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if "openai_api_key" not in st.session_state:
                st.session_state["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
            # Clear stale value from a previous provider so index=0 (gpt-4o) takes effect.
            if st.session_state.get("llm_model") not in _openai_models:
                st.session_state.pop("llm_model", None)
            selected_model = st.selectbox("Model", options=_openai_models, index=0, key="llm_model")

        # --- Data Processing LLM ---
        st.subheader("Data Processing LLM")
        st.caption("Used for dataset completion and chemical extraction. Can differ from the general LLM.")

        use_separate_processing_llm = st.checkbox(
            "Use a different LLM for data processing",
            value=False,
            key="use_separate_processing_llm",
        )

        if use_separate_processing_llm:
            processing_provider = st.radio(
                "Processing provider",
                options=["Ollama", "Chat GPT"],
                index=1,
                key="processing_llm_provider",
            )

            if processing_provider == "Ollama":
                models = _ollama_models()
                if models:
                    if st.session_state.get("processing_llm_model") not in models:
                        st.session_state.pop("processing_llm_model", None)
                    st.selectbox("Processing model", options=models, index=0, key="processing_llm_model")
                else:
                    st.text_input("Processing model (no models detected)", value="", key="processing_llm_model")
            else:
                # Read processing OpenAI config from environment (not displayed in UI)
                if "processing_openai_base_url" not in st.session_state:
                    st.session_state["processing_openai_base_url"] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                if "processing_openai_api_key" not in st.session_state:
                    st.session_state["processing_openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
                if st.session_state.get("processing_llm_model") not in _openai_models:
                    st.session_state.pop("processing_llm_model", None)
                st.selectbox("Processing model", options=_openai_models, index=0, key="processing_llm_model")

    st.divider()
    st.header("Custom Instructions")
    if "custom_instructions" not in st.session_state:
        st.session_state["custom_instructions"] = default_instructions()
        st.session_state["current_preset_name"] = None

    ci = st.session_state["custom_instructions"]

    # --- Load presets ---
    existing_presets = list_presets(workspaces_root)
    with st.expander("Load Preset Custom Instructions", expanded = False):
        if existing_presets:
            col_load_a, col_load_b = st.columns([3, 1])
            with col_load_a:
                preset_choice = st.selectbox(
                    "Select preset to load",
                    options=[""] + existing_presets,
                    key="ci_preset_choice",
                )
            with col_load_b:
                if st.button("Load", type="secondary") and preset_choice:
                    loaded = load_preset(workspaces_root, preset_choice)
                    st.session_state["custom_instructions"] = loaded
                    st.session_state["current_preset_name"] = preset_choice
                    st.success(f"Loaded preset: {preset_choice}")
                    st.rerun()
        else:
            st.info("No presets saved yet. Create your first preset below.")

        # Show currently loaded preset
        if st.session_state.get("current_preset_name"):
            st.caption(f"Currently editing: **{st.session_state['current_preset_name']}**")
        else:
            st.caption("Currently editing: **Unsaved changes (default)**")

    # --- Instruction editors ---
    with st.expander("Table extraction rules", expanded=False):
        st.caption(
            "Extra rules for how raw HTML/XML tables are parsed into structured data. "
            "E.g. how merged cells or multi-level headers should be handled."
        )
        st.text_area(
            "Table extraction extra rules",
            value=ci.get("table_extraction_rules", ""),
            height=120,
            key="ci_table_extraction",
        )

    with st.expander("Figure filtering criteria", expanded=False):
        st.caption(
            "Criteria the LLM uses to decide whether to keep or discard a figure. "
            "E.g. 'only keep line charts showing conductivity vs time'."
        )
        st.text_area(
            "Figure filter criteria",
            value=ci.get("figure_filter_criteria", ""),
            height=150,
            key="ci_figure_filter",
        )

    with st.expander("Table filtering criteria", expanded=False):
        st.caption(
            "Criteria the LLM uses to decide whether to keep or discard a table."
        )
        st.text_area(
            "Table filter criteria",
            value=ci.get("table_filter_criteria", ""),
            height=150,
            key="ci_table_filter",
        )

    with st.expander("Data processing rules", expanded=False):
        st.caption(
            "Extra rules applied when the LLM processes data (converts tables/figures into "
            "final dataset rows, extracts chemical compositions). E.g. 'before = time 0'."
        )
        st.text_area(
            "Data processing extra rules",
            value=ci.get("data_processing_rules", ""),
            height=120,
            key="ci_data_processing",
        )

    with st.expander("Output columns", expanded=False):
        st.caption(
            "One column name per line for the final dataset. "
            "Change this to capture different properties."
        )
        st.text_area(
            "Output columns (one per line)",
            value="\n".join(ci.get("output_columns", [])),
            height=160,
            key="ci_output_columns",
        )

    with st.expander("Missing-value defaults", expanded=False):
        st.caption(
            "JSON dict of column name to default value when not found. "
            "E.g. {\"time(h)\": \"0 Not Found\"}"
        )
        st.text_area(
            "Missing-value defaults (JSON)",
            value=json.dumps(ci.get("missing_value_defaults", {}), indent=2),
            height=160,
            key="ci_missing_defaults",
        )

    with st.expander("Save Custom Instructions", expanded=False):

        st.subheader("Save Preset")
        col_name, col_save_btn, col_del_btn = st.columns([3, 1, 1])

        with col_name:
            # Default to current preset name if editing an existing one
            default_name = st.session_state.get("current_preset_name", "")
            preset_name = st.text_input(
                "Preset name",
                value=default_name,
                key="ci_new_preset_name",
                placeholder="Enter a name for this preset"
            )

        with col_save_btn:
            st.write("")  # spacing
            st.write("")  # spacing
            if st.button("Save", type="primary") and preset_name.strip():
                # Read all fields from session state
                _ci_to_save = {}
                _ci_to_save["table_extraction_rules"] = st.session_state.get("ci_table_extraction", "")
                _ci_to_save["figure_filter_criteria"] = st.session_state.get("ci_figure_filter", "")
                _ci_to_save["table_filter_criteria"] = st.session_state.get("ci_table_filter", "")
                _ci_to_save["data_processing_rules"] = st.session_state.get("ci_data_processing", "")

                raw_cols = st.session_state.get("ci_output_columns", "")
                _ci_to_save["output_columns"] = [c.strip() for c in raw_cols.splitlines() if c.strip()]
                try:
                    _ci_to_save["missing_value_defaults"] = json.loads(
                        st.session_state.get("ci_missing_defaults", "{}")
                    )
                except json.JSONDecodeError:
                    st.error("Missing-value defaults must be valid JSON.")
                    _ci_to_save["missing_value_defaults"] = {}

                save_preset(workspaces_root, preset_name.strip(), _ci_to_save)
                st.session_state["current_preset_name"] = preset_name.strip()
                st.success(f"Saved preset: {preset_name.strip()}")
                st.rerun()

        with col_del_btn:
            st.write("")  # spacing
            st.write("")  # spacing
            if existing_presets and st.button("Delete", type="secondary") and st.session_state.get("ci_preset_choice"):
                delete_preset(workspaces_root, st.session_state["ci_preset_choice"])
                if st.session_state.get("current_preset_name") == st.session_state["ci_preset_choice"]:
                    st.session_state["current_preset_name"] = None
                st.success(f"Deleted preset: {st.session_state['ci_preset_choice']}")
                st.rerun()

        # --- Reset to defaults button ---
        if st.button("Reset to defaults", type="secondary"):
            st.session_state["custom_instructions"] = default_instructions()
            st.session_state["current_preset_name"] = None
            st.rerun()

    st.divider()
    st.header("Axis & Units Configuration")

    # Initialize axis config in session state
    if "axis_config" not in st.session_state:
        # Try to load from active workspace if available from session state
        _active_rr = st.session_state.get("rr")
        if _active_rr and hasattr(_active_rr, 'run_dir') and _active_rr.run_dir:
            st.session_state["axis_config"] = load_axis_config(_active_rr.run_dir)
        else:
            st.session_state["axis_config"] = load_axis_config(workspaces_root)
        st.session_state["current_axis_preset_name"] = None

    _ax_cfg = st.session_state["axis_config"]

    # --- Load axis presets ---
    _active_rr = st.session_state.get("rr")
    workspace_for_axis = (_active_rr.run_dir if (_active_rr and hasattr(_active_rr, 'run_dir') and _active_rr.run_dir)
                          else workspaces_root)
    existing_axis_presets = list_axis_presets(workspace_for_axis)

    with st.expander("Load Axis Preset", expanded=False):
        if existing_axis_presets:
            col_ax_load_a, col_ax_load_b = st.columns([3, 1])
            with col_ax_load_a:
                axis_preset_choice = st.selectbox(
                    "Select axis preset to load",
                    options=[""] + existing_axis_presets,
                    key="axis_preset_choice",
                )
            with col_ax_load_b:
                if st.button("Load", type="secondary", key="load_axis_preset_btn") and axis_preset_choice:
                    loaded = load_axis_preset(workspace_for_axis, axis_preset_choice)
                    if loaded:
                        st.session_state["axis_config"] = loaded
                        st.session_state["current_axis_preset_name"] = axis_preset_choice
                        st.success(f"Loaded axis preset: {axis_preset_choice}")
                        st.rerun()
        else:
            st.info("No axis presets saved yet. Create your first preset below.")

        # Show currently loaded preset
        if st.session_state.get("current_axis_preset_name"):
            st.caption(f"Currently editing: **{st.session_state['current_axis_preset_name']}**")
        else:
            st.caption("Currently editing: **Unsaved changes (default)**")

    # --- Axis config editors in expanders ---
    with st.expander("Unit Aliases", expanded=False):
        st.caption("Map raw unit strings (e.g. `sec`, `hrs`) to their canonical form (e.g. `s`, `h`).")
        _ua = _ax_cfg.get("unit_aliases", {})
        _ua_rows = [{"raw": k, "canonical": v} for k, v in sorted(_ua.items())]
        _ua_df = pd.DataFrame(_ua_rows) if _ua_rows else pd.DataFrame(columns=["raw", "canonical"])
        _ua_edited = st.data_editor(
            _ua_df,
            num_rows="dynamic",
            use_container_width=True,
            key="sidebar_ax_unit_aliases_editor",
        )
        # Store edited data in session state
        if st.button("Apply changes", key="apply_unit_aliases"):
            new_ua = {}
            for _, row in _ua_edited.iterrows():
                raw = str(row.get("raw", "")).strip()
                can = str(row.get("canonical", "")).strip()
                if raw and can:
                    new_ua[raw] = can
            st.session_state["axis_config"]["unit_aliases"] = new_ua
            st.success("Applied unit alias changes")

    with st.expander("Unit Dimensions", expanded=False):
        st.caption("Map canonical unit strings to a dimension name (e.g. `s` -> `time`, `mS/cm` -> `conductivity`).")
        _ud = _ax_cfg.get("unit_dimensions", {})
        _ud_rows = [{"unit": k, "dimension": v} for k, v in sorted(_ud.items())]
        _ud_df = pd.DataFrame(_ud_rows) if _ud_rows else pd.DataFrame(columns=["unit", "dimension"])
        _ud_edited = st.data_editor(
            _ud_df,
            num_rows="dynamic",
            use_container_width=True,
            key="sidebar_ax_unit_dims_editor",
        )
        if st.button("Apply changes", key="apply_unit_dims"):
            new_ud = {}
            for _, row in _ud_edited.iterrows():
                unit = str(row.get("unit", "")).strip()
                dim = str(row.get("dimension", "")).strip()
                if unit and dim:
                    new_ud[unit] = dim
            st.session_state["axis_config"]["unit_dimensions"] = new_ud
            st.success("Applied unit dimension changes")

    with st.expander("Canonical Targets", expanded=False):
        st.caption("For each dimension, choose the canonical target unit that values are converted to.")
        _ct = _ax_cfg.get("canonical_targets", {})
        _ct_rows = [{"dimension": k, "target_unit": v} for k, v in sorted(_ct.items())]
        _ct_df = pd.DataFrame(_ct_rows) if _ct_rows else pd.DataFrame(columns=["dimension", "target_unit"])
        _ct_edited = st.data_editor(
            _ct_df,
            num_rows="dynamic",
            use_container_width=True,
            key="sidebar_ax_canon_targets_editor",
        )
        if st.button("Apply changes", key="apply_canon_targets"):
            new_ct = {}
            for _, row in _ct_edited.iterrows():
                dim = str(row.get("dimension", "")).strip()
                target = str(row.get("target_unit", "")).strip()
                if dim and target:
                    new_ct[dim] = target
            st.session_state["axis_config"]["canonical_targets"] = new_ct
            st.success("Applied canonical target changes")

    # --- Save axis preset ---
    with st.expander("Save Axis Preset", expanded=False):
        st.subheader("Save Preset")
        col_ax_name, col_ax_save, col_ax_del = st.columns([3, 1, 1])

        with col_ax_name:
            default_ax_name = st.session_state.get("current_axis_preset_name", "")
            axis_preset_name = st.text_input(
                "Axis preset name",
                value=default_ax_name,
                key="axis_new_preset_name",
                placeholder="Enter a name for this axis preset"
            )

        with col_ax_save:
            st.write("")  # spacing
            st.write("")  # spacing
            if st.button("Save", type="primary", key="save_axis_preset_btn") and axis_preset_name.strip():
                save_axis_preset(workspace_for_axis, axis_preset_name.strip(), st.session_state["axis_config"])
                # Also save to the workspace
                save_axis_config(workspace_for_axis, st.session_state["axis_config"])
                st.session_state["current_axis_preset_name"] = axis_preset_name.strip()
                st.success(f"Saved axis preset: {axis_preset_name.strip()}")
                st.rerun()

        with col_ax_del:
            st.write("")  # spacing
            st.write("")  # spacing
            if existing_axis_presets and st.button("Delete", type="secondary", key="delete_axis_preset_btn") and st.session_state.get("axis_preset_choice"):
                delete_axis_preset(workspace_for_axis, st.session_state["axis_preset_choice"])
                if st.session_state.get("current_axis_preset_name") == st.session_state["axis_preset_choice"]:
                    st.session_state["current_axis_preset_name"] = None
                st.success(f"Deleted axis preset: {st.session_state['axis_preset_choice']}")
                st.rerun()

        # --- Reset to defaults button ---
        if st.button("Reset axis config to defaults", type="secondary", key="reset_axis_btn"):
            st.session_state["axis_config"] = reset_axis_config(workspace_for_axis)
            st.session_state["current_axis_preset_name"] = None
            st.success("Reset axis configuration to defaults")
            st.rerun()

    st.divider()
    with st.expander("Load an existing run", expanded=False):

        # Choose which workspace to browse when loading runs (filter out axis_presets and instruction_presets)
        load_ws_options = [
            ws for ws in _list_subdirs(workspaces_root)
            if ws not in ("axis_presets", "instruction_presets")
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

            existing_runs = sorted([p.name for p in load_workspace_dir.iterdir() if p.is_dir()], reverse=True)
            pick = st.selectbox("Run ID", options=[""] + existing_runs, key="load_run_id")

            if st.button("Load run") and pick:
                rr = discover_run(load_workspace_dir / pick)
                st.session_state["rr"] = rr
        else:
            st.info("No workspaces exist yet. Create one above, then run the pipeline.")

# ---------------- Run pipeline ----------------
st.subheader("Input DOIs")
doi_text = st.text_area("One DOI per line", height=140)

st.subheader("Run configuration")
default_run_id = st.session_state.get("pending_run_id", "")
run_id_input = st.text_input(
    "Run ID (optional)",
    value=default_run_id,
    help="Leave blank to auto-generate timestamp-based run ID",
)

st.session_state["pending_run_id"] = run_id_input


if st.button("Run pipeline", type="primary"):
    dois = [d.strip() for d in doi_text.splitlines() if d.strip()]
    st.session_state["doi_order"] = dois
    if not dois:
        st.error("Please paste at least one DOI.")
    else:
        with st.spinner("Running your pipeline end-to-end…"):
            custom_run_id = run_id_input.strip() or None

            # --- Apply general LLM configuration for this run ---
            provider = st.session_state.get("llm_provider", "Ollama")
            model = st.session_state.get("llm_model", "")

            if provider == "Ollama":
                os.environ["AEM_OCR_LLM_PROVIDER"] = "ollama"
                os.environ["AEM_OCR_LLM_MODEL"] = model
                os.environ["OLLAMA_HOST"] = st.session_state.get("ollama_host", _default_ollama_host())
                os.environ["OLLAMA_MODELS"] = st.session_state.get("ollama_models_dir", str(_default_ollama_models_dir()))
            else:
                os.environ["AEM_OCR_LLM_PROVIDER"] = "openai"
                os.environ["AEM_OCR_LLM_MODEL"] = model
                os.environ["OPENAI_BASE_URL"] = st.session_state.get("openai_base_url", "https://api.openai.com/v1")
                if st.session_state.get("openai_api_key"):
                    os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]

            # --- Apply data processing LLM configuration ---
            if st.session_state.get("use_separate_processing_llm"):
                p_provider = st.session_state.get("processing_llm_provider", "Chat GPT")
                p_model = st.session_state.get("processing_llm_model", "")
                if p_provider == "Ollama":
                    os.environ["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "ollama"
                    os.environ["AEM_OCR_PROCESSING_LLM_MODEL"] = p_model
                else:
                    os.environ["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "openai"
                    os.environ["AEM_OCR_PROCESSING_LLM_MODEL"] = p_model
                    p_base = st.session_state.get("processing_openai_base_url", "https://api.openai.com/v1")
                    p_key = st.session_state.get("processing_openai_api_key", "")
                    if p_base:
                        os.environ["AEM_OCR_PROCESSING_OPENAI_BASE_URL"] = p_base
                    if p_key:
                        os.environ["AEM_OCR_PROCESSING_OPENAI_API_KEY"] = p_key
            else:
                # Clear processing-specific env vars so processing uses the general LLM
                for k in ["AEM_OCR_PROCESSING_LLM_PROVIDER", "AEM_OCR_PROCESSING_LLM_MODEL",
                           "AEM_OCR_PROCESSING_OPENAI_BASE_URL", "AEM_OCR_PROCESSING_OPENAI_API_KEY"]:
                    os.environ.pop(k, None)

            if custom_run_id:
                target = run_root / custom_run_id
                if target.exists():
                    st.warning(f"Run directory already exists and may be overwritten: {target}")


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
            preset_name = st.session_state.get("current_preset_name", "defaults")
            st.info(f"Using custom instructions: **{preset_name}**")
            with st.expander("View custom instructions being used", expanded=False):
                st.json(_run_ci)

            rr = run_full_pipeline(
                dois=dois,
                run_root=run_root,
                scripts_root=scripts_root,
                run_id=custom_run_id,
                run_extract_figures=run_extract_figures,
                run_extract_tables=run_extract_tables,
                run_filter_figures=run_filter_figures,
                run_filter_tables=run_filter_tables,
                run_plextract=run_plextract,
                run_complete_dataset=run_complete_dataset,
                run_chemical_composition=run_chemical_composition,
                exclude_full_text=exclude_full_text,
                custom_instructions=_run_ci,
            )

        st.session_state["rr"] = rr
        st.success(f"Done. Run ID: {rr.run_id}")
        _render_cost_summary(rr.run_dir)

rr: RunResult | None = st.session_state.get("rr")
if not rr:
    st.stop()

st.info(f"Run ID: **{rr.run_id}**")
st.info(f"Workspace: **{rr.run_dir}**")
_render_cost_summary(rr.run_dir)

# ---------------- DOI summary table + DOI selector ----------------
st.subheader("Processed DOIs")

# Let user decide whether to compute interventions (can be expensive)
colA, colB = st.columns([1, 3])
with colA:
    compute_interventions = st.checkbox("Include intervention counts", value=False)
with colB:
    if compute_interventions:
        st.caption("Intervention counts are cached per run directory once computed.")
    else:
        st.caption("Turn on to identify figures needing manual intervention.")

df_stats = build_doi_stats(rr, include_interventions=compute_interventions)

# Render as interactive table (row click -> selects DOI when supported)
df_view = df_stats.drop(columns=["_doi_safe"], errors="ignore")

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

if "selected_doi" not in st.session_state:
    st.session_state["selected_doi"] = doi_options[0]

if df_selected_doi:
    st.session_state["selected_doi"] = df_selected_doi

selected_doi = st.selectbox(
    "Select DOI",
    options=doi_options,
    index=doi_options.index(st.session_state["selected_doi"]) if st.session_state["selected_doi"] in doi_options else 0,
)
st.session_state["selected_doi"] = selected_doi

doi_item = next((d for d in rr.dois if d.doi == selected_doi), None)

if doi_item is None:
    st.error(f"Could not find DOI '{selected_doi}' in the loaded run. The data may have changed since this session was saved.")
    st.stop()

# ---------------- Tabs (under DOI dropdown) ----------------
tab_text, tab_figs, tab_tables, tab_chem, tab_xy = st.tabs(
    ["Parsed text", "Figures", "Tables", "Chemical Composition", "Dataset"]
)

with tab_text:
    if doi_item.parsed_json_path and doi_item.parsed_json_path.exists():
        st.json(json.loads(doi_item.parsed_json_path.read_text()))
    else:
        st.warning("No parsed.json found for this DOI in the run workspace.")

with tab_figs:
    if not doi_item.figures:
        st.warning("No figures found.")
    # Add width control for figure images
    fig_img_width = st.slider(
        "Figure image width (px)",
        min_value=200,
        max_value=1200,
        value=600,
        step=50,
        key=f"fig_img_width_{selected_doi}"
    )
    for idx, f in enumerate(doi_item.figures):
        # Add divider between figures
        if idx > 0:
            st.divider()

        # Figure header and basic info (not in expander)
        st.markdown(f"### {f.id}")

        # ---- Per-figure variable setup ----
        _safe_doi_er = doi_to_safe(selected_doi)
        _fig_id_er = f.id
        _fig_stem_er = Path(f.id).stem
        _img_name_er = Path(f.id).name

        # Find source image
        _source_img_er = f.figure_path
        if _source_img_er is None or not _source_img_er.exists():
            _filtered_fig_root_er = rr.run_dir / "figures_tables" / "figures"
            from intervention import _try_find_source_image as _tfs_er
            _source_img_er = _tfs_er(_filtered_fig_root_er, _safe_doi_er, _fig_id_er)

        # Find plextract figure_dir; plextract output dirs include the file extension
        # (e.g. "Figure_9.jpg/") so we match by stem, not exact name.
        _plextract_fig_dir_er = None
        _latest_plextract_er = latest_plextract_run_dir(rr.run_dir / "plextract")
        if _latest_plextract_er:
            _plex_out_base_er = _latest_plextract_er / "output" / _safe_doi_er
            if _plex_out_base_er.exists():
                for _cand_er in _plex_out_base_er.iterdir():
                    if _cand_er.is_dir() and Path(_cand_er.name).stem == _fig_id_er:
                        _plextract_fig_dir_er = _cand_er
                        break

        # Per-figure intervention plextract workdir
        _per_fig_int_wd_er = rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions"
        _int_fig_dir_er = None
        if _per_fig_int_wd_er.exists():
            _lat_int_er = latest_plextract_run_dir(_per_fig_int_wd_er)
            if _lat_int_er:
                _int_out_base_er = _lat_int_er / "output" / _safe_doi_er
                if _int_out_base_er.exists():
                    for _cand2_er in _int_out_base_er.iterdir():
                        if _cand2_er.is_dir() and Path(_cand2_er.name).stem == _fig_id_er:
                            _int_fig_dir_er = _cand2_er
                            break

        # crops output directory
        _crops_out_dir_er = rr.run_dir / "interventions" / _safe_doi_er / _img_name_er

        # ---- Collect axis config subfig dirs ----
        # Match by exact name, by stem (handles extension-bearing plextract dirs),
        # or by subfigure prefix.
        _ax_subfig_options_er: list = []
        _ax_subfig_dirs_er: list = []
        for _ax_wd_er in [
            rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions",
            rr.run_dir / "interventions" / "interventions_plextract",
            rr.run_dir / "plextract",
        ]:
            if not _ax_wd_er.exists():
                continue
            _ax_lat_er = latest_plextract_run_dir(_ax_wd_er)
            if not _ax_lat_er:
                continue
            _ax_out_base_er = _ax_lat_er / "output" / _safe_doi_er
            if not _ax_out_base_er.exists():
                continue
            for _ax_d_er in sorted(_ax_out_base_er.iterdir()):
                if _ax_d_er.is_dir() and (
                    _ax_d_er.name == _fig_id_er or
                    Path(_ax_d_er.name).stem == _fig_id_er or
                    _ax_d_er.name.startswith(_fig_stem_er + "_subfig_")
                ):
                    if _ax_d_er.name not in _ax_subfig_options_er:
                        _ax_subfig_options_er.append(_ax_d_er.name)
                        _ax_subfig_dirs_er.append(_ax_d_er)

        # ---- 1. Figure image expander ----
        with st.expander(f"Figure: {f.id}", expanded=False):
            if f.caption:
                st.markdown(f"**Caption:** {f.caption}")
            if _source_img_er and _source_img_er.exists():
                st.image(str(_source_img_er), width=fig_img_width)
            elif f.figure_path and f.figure_path.exists():
                st.image(str(f.figure_path), width=fig_img_width)
                st.caption(str(f.figure_path))
            else:
                st.warning("No source image found for this figure.")

        # ---- 2. Crop Figure expander ----
        with st.expander("Crop Figure", expanded=False):
            if _source_img_er and _source_img_er.exists():
                _pil_img_er = Image.open(_source_img_er).convert("RGB")
                _orig_w_er, _orig_h_er = _pil_img_er.size

                _default_crop_w_er = min(_orig_w_er, 600)
                _crop_w_er = st.slider(
                    "Display width (px)", 200, min(_orig_w_er, 1500), _default_crop_w_er, 50,
                    key=f"er_crop_w_{_safe_doi_er}_{_fig_id_er}"
                )
                _display_h_er = int(_orig_h_er * (_crop_w_er / _orig_w_er))

                _draw_mode_er = st.radio(
                    "Canvas mode", ["rect", "transform"],
                    format_func=lambda x: "Draw new rectangles" if x == "rect" else "Edit/reshape rectangles",
                    key=f"er_canvas_mode_{_safe_doi_er}_{_fig_id_er}", horizontal=True
                )

                st.image(str(_source_img_er), width=_crop_w_er)
                st.write("**Draw crop regions on the canvas below** — coordinates match the image above.")
                _canvas_er = st_canvas(
                    background_color="#f0f2f6",
                    fill_color="rgba(255, 165, 0, 0.3)", stroke_color="#e63946", stroke_width=2,
                    update_streamlit=True, height=_display_h_er, width=_crop_w_er,
                    drawing_mode=_draw_mode_er,
                    key=f"er_canvas_{_safe_doi_er}_{_fig_id_er}",
                )

                _scale_x_er = _orig_w_er / _crop_w_er
                _scale_y_er = _orig_h_er / _display_h_er
                _rects_er = []
                if _canvas_er.json_data and "objects" in _canvas_er.json_data:
                    for _obj_er in _canvas_er.json_data["objects"]:
                        if _obj_er.get("type") != "rect":
                            continue
                        _l_er = float(_obj_er.get("left", 0))
                        _t_er = float(_obj_er.get("top", 0))
                        _w_obj_er = float(_obj_er.get("width", 0)) * float(_obj_er.get("scaleX", 1))
                        _h_obj_er = float(_obj_er.get("height", 0)) * float(_obj_er.get("scaleY", 1))
                        _rects_er.append((
                            int(_l_er * _scale_x_er), int(_t_er * _scale_y_er),
                            int((_l_er + _w_obj_er) * _scale_x_er), int((_t_er + _h_obj_er) * _scale_y_er)
                        ))

                st.write(f"Rectangles drawn: **{len(_rects_er)}**")

                _cc1_er, _cc2_er = st.columns(2)
                with _cc1_er:
                    if st.button("Save crops", disabled=len(_rects_er) == 0, key=f"er_save_crops_{_safe_doi_er}_{_fig_id_er}"):
                        _saved_er = save_crops(_source_img_er, _rects_er, _crops_out_dir_er, prefix=_fig_stem_er)
                        st.success(f"Saved {len(_saved_er)} crops to `{_crops_out_dir_er}`")
                        _sc = st.columns(min(len(_saved_er), 4)) if _saved_er else []
                        for _si, _p in enumerate(_saved_er):
                            with _sc[_si % len(_sc)]:
                                st.image(str(_p), width=200)
                                st.caption(_p.name)

                _existing_crops_er = sorted(_crops_out_dir_er.glob(f"{_fig_stem_er}_subfig_*.png")) if _crops_out_dir_er.exists() else []
                if _existing_crops_er:
                    st.write(f"**Existing crops:** {len(_existing_crops_er)}")
                    _crop_cols_er = st.columns(min(len(_existing_crops_er), 6))
                    for _ci, _cp in enumerate(_existing_crops_er):
                        with _crop_cols_er[_ci % len(_crop_cols_er)]:
                            st.image(str(_cp), width=150)
                            st.caption(_cp.name)

                    _cl1_er, _cl2_er = st.columns(2)
                    with _cl1_er:
                        if st.button("Clear all crops", key=f"er_clear_crops_{_safe_doi_er}_{_fig_id_er}"):
                            import shutil as _shutil_er
                            for _cp in _existing_crops_er:
                                _cp.unlink(missing_ok=True)
                            st.success("Cleared existing crops.")
                            st.rerun()
                    with _cl2_er:
                        if st.button("Re-run extraction on existing crops", type="primary", key=f"er_rerun_crops_{_safe_doi_er}_{_fig_id_er}"):
                            try:
                                with st.spinner("Running plextract on crops..."):
                                    run_plextract_on_manual_crops(
                                        scripts_root=scripts_root,
                                        run_dir=rr.run_dir,
                                        safe_doi=_safe_doi_er,
                                        figure_id=_img_name_er,
                                    )
                                st.success("Done. Subfigures updated.")
                                st.rerun()
                            except Exception as _e_er:
                                st.error(f"Extraction failed: {_e_er}")
                else:
                    st.info("No crops saved yet. Draw rectangles above to create crops.")
            else:
                st.warning("No source image found for this figure.")

        # ---- 3. Axis Configuration expander ----
        with st.expander("Axis Configuration", expanded=False):
            if not _ax_subfig_dirs_er:
                st.info("No plextract output found for this figure. Run plextract extraction first.")
            else:
                if len(_ax_subfig_dirs_er) > 1:
                    _ax_sel_name_er = st.selectbox(
                        "Select figure/subfigure to configure",
                        _ax_subfig_options_er,
                        key=f"er_ax_sel_{_safe_doi_er}_{_fig_id_er}"
                    )
                    _ax_sel_dir_er = _ax_subfig_dirs_er[_ax_subfig_options_er.index(_ax_sel_name_er)]
                else:
                    _ax_sel_dir_er = _ax_subfig_dirs_er[0]
                    _ax_sel_name_er = _ax_subfig_options_er[0]

                _ax_chartdete_er = _ax_sel_dir_er / "chartdete"

                # --- Axis Titles ---
                st.markdown("**Axis Titles**")
                _ax_title_override_er = _ax_chartdete_er / "axis_title_override.json"
                _ax_title_ocr_er = _ax_chartdete_er / "axis_title_texts.json"
                _ax_x_title_er = ""
                _ax_y_title_er = ""
                if _ax_title_override_er.exists():
                    try:
                        _ax_ov_er = json.loads(_ax_title_override_er.read_text())
                        _ax_x_title_er = _ax_ov_er.get("x_title", "")
                        _ax_y_title_er = _ax_ov_er.get("y_title", "")
                    except Exception:
                        pass
                elif _ax_title_ocr_er.exists():
                    try:
                        _ax_titles_dict_er = json.loads(_ax_title_ocr_er.read_text())
                        for _ap_er, _at_er in _ax_titles_dict_er.items():
                            _ab_er = Path(_ap_er).name.lower()
                            if any(x in _ab_er for x in ("x_", "x-", "_x.", "xtitle", "xlabel")):
                                _ax_x_title_er = _at_er
                            elif any(x in _ab_er for x in ("y_", "y-", "_y.", "ytitle", "ylabel")):
                                _ax_y_title_er = _at_er
                        if not _ax_x_title_er and not _ax_y_title_er and len(_ax_titles_dict_er) == 2:
                            _ax_vals_er = list(_ax_titles_dict_er.values())
                            _ax_x_title_er = _ax_vals_er[0]
                            _ax_y_title_er = _ax_vals_er[1]
                    except Exception:
                        pass

                _col_xt_er, _col_yt_er = st.columns(2)
                with _col_xt_er:
                    _new_x_title_er = st.text_input("X axis title", value=_ax_x_title_er, key=f"er_x_title_{_safe_doi_er}_{_ax_sel_name_er}")
                with _col_yt_er:
                    _new_y_title_er = st.text_input("Y axis title", value=_ax_y_title_er, key=f"er_y_title_{_safe_doi_er}_{_ax_sel_name_er}")

                if st.button("Save axis titles", key=f"er_save_titles_{_safe_doi_er}_{_ax_sel_name_er}"):
                    _ax_chartdete_er.mkdir(parents=True, exist_ok=True)
                    _ax_title_override_er.write_text(json.dumps({"x_title": _new_x_title_er, "y_title": _new_y_title_er}, indent=2))
                    st.success("Axis title overrides saved. These will be used by finalize_labels when you rerun dataset generation.")

                # --- Axis Calibration ---
                st.markdown("**Axis Calibration**")
                st.caption("Set the min/max values for each axis. Changes are preserved as overrides and survive plextract reruns.")
                _pix2val_override_er = _ax_chartdete_er / "pix2val_override.json"
                _pix2val_orig_er = _ax_chartdete_er / "pix2val.json"
                _bb_json_er = _ax_chartdete_er / "bounding_boxes.json"

                _pix2val_data_er = None
                _pix2val_source_er = ""
                if _pix2val_override_er.exists():
                    try:
                        _pix2val_data_er = json.loads(_pix2val_override_er.read_text())
                        _pix2val_source_er = "(using override)"
                    except Exception:
                        pass
                if _pix2val_data_er is None and _pix2val_orig_er.exists():
                    try:
                        _pix2val_data_er = json.loads(_pix2val_orig_er.read_text())
                        _pix2val_source_er = "(from OCR)"
                    except Exception:
                        pass

                if _pix2val_data_er:
                    st.caption(f"Calibration data {_pix2val_source_er}")

                    # Try to derive human-readable min/max from pix2val + bounding box
                    _plot_area_er = None
                    if _bb_json_er.exists():
                        try:
                            _bbs_er = json.loads(_bb_json_er.read_text())
                            _areas_er = sorted(_bbs_er.get("plot_area", []), key=lambda el: el[4] if len(el) > 4 else -1, reverse=True)
                            if _areas_er:
                                _plot_area_er = _areas_er[0]
                        except Exception:
                            pass

                    _xmap_er = _pix2val_data_er.get("x", {})
                    _ymap_er = _pix2val_data_er.get("y", {})
                    _xtype_er = _xmap_er.get("type", "linear")
                    _ytype_er = _ymap_er.get("type", "linear")
                    _xa_er = float(_xmap_er.get("a", 1.0))
                    _xb_er = float(_xmap_er.get("b", 0.0))
                    _ya_er = float(_ymap_er.get("a", 1.0))
                    _yb_er = float(_ymap_er.get("b", 0.0))
                    _domain_er = _pix2val_data_er.get("domain", {})

                    # Compute current min/max from plot area + coefficients
                    _x_val_min_er = _x_val_max_er = _y_val_min_er = _y_val_max_er = None
                    if _plot_area_er and len(_plot_area_er) >= 4:
                        _px1_er = float(_plot_area_er[0])
                        _py1_er = float(_plot_area_er[1])
                        _px2_er = float(_plot_area_er[2])
                        _py2_er = float(_plot_area_er[3])
                        if _xtype_er == "linear":
                            _x_val_min_er = _xa_er * _px1_er + _xb_er
                            _x_val_max_er = _xa_er * _px2_er + _xb_er
                        elif _xtype_er == "log10":
                            _x_val_min_er = 10 ** (_xa_er * _px1_er + _xb_er)
                            _x_val_max_er = 10 ** (_xa_er * _px2_er + _xb_er)
                        if _ytype_er == "linear":
                            _y_val_min_er = _ya_er * _py2_er + _yb_er
                            _y_val_max_er = _ya_er * _py1_er + _yb_er
                        elif _ytype_er == "log10":
                            _y_val_min_er = 10 ** (_ya_er * _py2_er + _yb_er)
                            _y_val_max_er = 10 ** (_ya_er * _py1_er + _yb_er)

                    # Fall back to domain field if no plot area
                    if _x_val_min_er is None and "x" in _domain_er and len(_domain_er["x"]) >= 2:
                        _x_val_min_er, _x_val_max_er = _domain_er["x"][0], _domain_er["x"][1]
                    if _y_val_min_er is None and "y" in _domain_er and len(_domain_er["y"]) >= 2:
                        _y_val_min_er, _y_val_max_er = _domain_er["y"][0], _domain_er["y"][1]

                    _col_ax_type_er, _col_ay_type_er = st.columns(2)
                    with _col_ax_type_er:
                        _new_xtype_er = st.selectbox("X axis type", ["linear", "log10"], index=0 if _xtype_er == "linear" else 1, key=f"er_xtype_{_safe_doi_er}_{_ax_sel_name_er}")
                    with _col_ay_type_er:
                        _new_ytype_er = st.selectbox("Y axis type", ["linear", "log10"], index=0 if _ytype_er == "linear" else 1, key=f"er_ytype_{_safe_doi_er}_{_ax_sel_name_er}")

                    _col_xmin_er, _col_xmax_er, _col_ymin_er, _col_ymax_er = st.columns(4)
                    with _col_xmin_er:
                        _new_x_min_er = st.text_input("X min", value=f"{_x_val_min_er:.6g}" if _x_val_min_er is not None else "", key=f"er_x_min_{_safe_doi_er}_{_ax_sel_name_er}")
                    with _col_xmax_er:
                        _new_x_max_er = st.text_input("X max", value=f"{_x_val_max_er:.6g}" if _x_val_max_er is not None else "", key=f"er_x_max_{_safe_doi_er}_{_ax_sel_name_er}")
                    with _col_ymin_er:
                        _new_y_min_er = st.text_input("Y min", value=f"{_y_val_min_er:.6g}" if _y_val_min_er is not None else "", key=f"er_y_min_{_safe_doi_er}_{_ax_sel_name_er}")
                    with _col_ymax_er:
                        _new_y_max_er = st.text_input("Y max", value=f"{_y_val_max_er:.6g}" if _y_val_max_er is not None else "", key=f"er_y_max_{_safe_doi_er}_{_ax_sel_name_er}")

                    if st.button("Save calibration overrides", key=f"er_save_calib_{_safe_doi_er}_{_ax_sel_name_er}"):
                        try:
                            _nxmin_er = float(_new_x_min_er) if _new_x_min_er.strip() else None
                            _nxmax_er = float(_new_x_max_er) if _new_x_max_er.strip() else None
                            _nymin_er = float(_new_y_min_er) if _new_y_min_er.strip() else None
                            _nymax_er = float(_new_y_max_er) if _new_y_max_er.strip() else None

                            _new_pix2val_er = dict(_pix2val_data_er)
                            _new_pix2val_er["x"] = dict(_xmap_er)
                            _new_pix2val_er["y"] = dict(_ymap_er)
                            _new_pix2val_er["x"]["type"] = _new_xtype_er
                            _new_pix2val_er["y"]["type"] = _new_ytype_er

                            # Recompute a, b from new min/max if plot area is available
                            if _plot_area_er and len(_plot_area_er) >= 4 and None not in (_nxmin_er, _nxmax_er, _nymin_er, _nymax_er):
                                _px1n_er = float(_plot_area_er[0])
                                _py1n_er = float(_plot_area_er[1])
                                _px2n_er = float(_plot_area_er[2])
                                _py2n_er = float(_plot_area_er[3])
                                if _new_xtype_er == "linear" and _px2n_er != _px1n_er:
                                    _nxa_er = (_nxmax_er - _nxmin_er) / (_px2n_er - _px1n_er)
                                    _nxb_er = _nxmin_er - _nxa_er * _px1n_er
                                    _new_pix2val_er["x"]["a"] = _nxa_er
                                    _new_pix2val_er["x"]["b"] = _nxb_er
                                elif _new_xtype_er == "log10" and _px2n_er != _px1n_er and _nxmin_er > 0 and _nxmax_er > 0:
                                    _nxa_er = (math.log10(_nxmax_er) - math.log10(_nxmin_er)) / (_px2n_er - _px1n_er)
                                    _nxb_er = math.log10(_nxmin_er) - _nxa_er * _px1n_er
                                    _new_pix2val_er["x"]["a"] = _nxa_er
                                    _new_pix2val_er["x"]["b"] = _nxb_er
                                if _new_ytype_er == "linear" and _py2n_er != _py1n_er:
                                    _nya_er = (_nymax_er - _nymin_er) / (_py1n_er - _py2n_er)
                                    _nyb_er = _nymin_er - _nya_er * _py2n_er
                                    _new_pix2val_er["y"]["a"] = _nya_er
                                    _new_pix2val_er["y"]["b"] = _nyb_er
                                elif _new_ytype_er == "log10" and _py2n_er != _py1n_er and _nymin_er > 0 and _nymax_er > 0:
                                    _nya_er = (math.log10(_nymax_er) - math.log10(_nymin_er)) / (_py1n_er - _py2n_er)
                                    _nyb_er = math.log10(_nymin_er) - _nya_er * _py2n_er
                                    _new_pix2val_er["y"]["a"] = _nya_er
                                    _new_pix2val_er["y"]["b"] = _nyb_er

                            _new_pix2val_er["domain"] = {
                                "x": [_nxmin_er, _nxmax_er] if None not in (_nxmin_er, _nxmax_er) else _domain_er.get("x", [None, None]),
                                "y": [_nymin_er, _nymax_er] if None not in (_nymin_er, _nymax_er) else _domain_er.get("y", [None, None]),
                            }
                            _ax_chartdete_er.mkdir(parents=True, exist_ok=True)
                            _pix2val_override_er.write_text(json.dumps(_new_pix2val_er, indent=2))
                            st.success("Calibration override saved.")
                        except Exception as _e_ax_er:
                            st.error(f"Failed to save calibration: {_e_ax_er}")

                    if _pix2val_override_er.exists():
                        if st.button("Clear calibration override (revert to OCR)", key=f"er_clear_calib_{_safe_doi_er}_{_ax_sel_name_er}"):
                            _pix2val_override_er.unlink()
                            st.success("Cleared calibration override.")
                            st.rerun()
                else:
                    st.info("No pix2val calibration data found. Run plextract extraction first.")

        # --- Extracted dataset table (plextract final_dataset.csv) ---
        final_csv = None
        if isinstance(getattr(f, "extra", None), dict):
            final_csv = f.extra.get("final_dataset_csv")

        # --- Display extracted data in expander ---
        with st.expander("Extracted Data", expanded=False):
            if final_csv and Path(final_csv).exists():
                try:
                    df = _load_csv(final_csv)
                    st.data_editor(df, use_container_width=True, hide_index=True, key=f"fig_extracted_{selected_doi}_{f.id}")
                    st.caption(final_csv)
                except Exception as e:
                    st.warning(f"Failed to load extracted dataset:\n`{final_csv}`\n\n{e}")
            else:
                st.info("No extracted dataset found for this figure.")

        # --- Display LLM-generated emit_rows from row_cache ---
        emit_rows_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "figure", f.id)
        if emit_rows_path:
            df_emit = load_emit_rows(emit_rows_path)
            if df_emit is not None and not df_emit.empty:
                with st.expander("LLM-Generated Dataset Rows", expanded=False):
                    st.markdown("*Normalized dataset rows generated by LLM from extracted figure data*")
                    st.caption(f"**Source:** `{emit_rows_path.relative_to(rr.run_dir)}`")
                    st.data_editor(df_emit, use_container_width=True, hide_index=True, key=f"fig_emit_{selected_doi}_{f.id}")

                    csv_data = df_emit.to_csv(index=False)
                    st.download_button(
                        label="Download emit_rows CSV",
                        data=csv_data,
                        file_name=f"{f.id}_emit_rows.csv",
                        mime="text/csv",
                    )
        else:
            # Check if row_cache exists but no data found for this figure
            row_cache_dir = rr.run_dir / "complete_dataset" / "row_cache"
            if row_cache_dir.exists():
                with st.expander("LLM-Generated Dataset Rows", expanded=False):
                    st.info(f"No row_cache data found for {f.id}. Cache may not have been generated yet.")

        # --- Display cropped subfigures from plextract and interventions ---
        # Collect subfigure dirs from all relevant workdirs, deduplicating by name.
        # Per-figure plextract_interventions takes highest priority (most up-to-date),
        # then the legacy shared interventions_plextract, then original plextract.
        safe_doi = selected_doi.replace("/", "@").strip()
        fig_stem = Path(f.id).stem
        _subfig_map: dict[str, Path] = {}  # name -> dir (first match wins)

        from intervention import latest_plextract_run_dir

        # Build priority-ordered list of workdirs to search for subfigures.
        _workdirs_priority: list[Path] = []

        # 1. Per-figure plextract_interventions: each figure dir under
        #    interventions/<safe_doi>/ may have its own plextract_interventions/
        #    subdir when the user ran "Re-run extraction" for that specific figure.
        _int_doi_dir = rr.run_dir / "interventions" / safe_doi
        _per_fig_plextract: Path | None = None
        if _int_doi_dir.exists():
            for _int_fig_d in _int_doi_dir.iterdir():
                if _int_fig_d.is_dir() and Path(_int_fig_d.name).stem == fig_stem:
                    _candidate = _int_fig_d / "plextract_interventions"
                    if _candidate.exists():
                        _per_fig_plextract = _candidate
                        _workdirs_priority.append(_candidate)
                    break

        # 2. Legacy shared interventions_plextract (backward compatibility).
        _workdirs_priority.append(rr.run_dir / "interventions" / "interventions_plextract")

        # 3. Original plextract output.
        _workdirs_priority.append(rr.run_dir / "plextract")

        for _wd in _workdirs_priority:
            if not _wd.exists():
                continue
            _latest = latest_plextract_run_dir(_wd)
            if not _latest:
                continue
            _out_base = _latest / "output" / safe_doi
            if not _out_base.exists():
                continue
            for _d in _out_base.iterdir():
                if _d.is_dir() and _d.name.startswith(fig_stem + "_subfig_"):
                    # First match wins (per-figure intervention has priority)
                    if _d.name not in _subfig_map:
                        _subfig_map[_d.name] = _d

        subfig_dirs = sorted(_subfig_map.values(), key=lambda d: d.name)

        # Set latest_manual / out_base for use by the data completion button and
        # the original-image lookup below.  Prefer the per-figure workdir, then
        # fall back to the shared interventions_plextract, then plain plextract.
        latest_manual = None
        out_base = None
        for _wd in [
            _per_fig_plextract,
            rr.run_dir / "interventions" / "interventions_plextract",
            rr.run_dir / "plextract",
        ]:
            if _wd is None or not _wd.exists():
                continue
            _lat = latest_plextract_run_dir(_wd)
            if _lat:
                _ob = _lat / "output" / safe_doi
                if _ob.exists():
                    latest_manual = _lat
                    out_base = _ob
                    break

        if subfig_dirs:
                # Header for subfigures section
                st.markdown("**Cropped Subfigures**")

                for subfig_dir in subfig_dirs:
                    # Subfigure header with data completion button
                    col_header, col_button = st.columns([3, 1])
                    with col_header:
                        st.markdown(f"### {subfig_dir.name}")
                    with col_button:
                        if st.button("Run Data Completion", key=f"complete_{subfig_dir.name}", help="Run complete_dataset.py to extract dataset rows from this subfigure"):
                            with st.spinner(f"Running data completion for {subfig_dir.name}..."):
                                try:
                                    # Verify index.csv exists
                                    reparsed_dir = rr.run_dir / "reparsed"
                                    index_csv_path = reparsed_dir / "index.csv"

                                    if not index_csv_path.exists():
                                        error_msg = f"**Required file not found:** `{index_csv_path.relative_to(rr.run_dir)}`\n\n"

                                        if not reparsed_dir.exists():
                                            error_msg += "The `reparsed/` directory doesn't exist. "
                                        else:
                                            error_msg += "The `reparsed/` directory exists but `index.csv` is missing. "

                                        error_msg += ("This file is created during the 'Download DOIs' stage of the pipeline.\n\n"
                                                     "**To fix this:**\n"
                                                     "1. Run the full pipeline with 'Download DOIs' enabled, OR\n"
                                                     "2. If you've already downloaded papers, check if `index.csv` exists in a different workspace")

                                        st.error(error_msg)
                                        continue

                                    # Create temporary CSV with this DOI
                                    import tempfile
                                    import subprocess

                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                                        tmp.write(f"DOI\n{selected_doi}\n")
                                        tmp_path = tmp.name

                                    # Run complete_dataset.py
                                    complete_dataset_script = Path(__file__).parent.parent / "complete_dataset" / "complete_dataset.py"

                                    # Determine the best extracted_figures_root for this subfigure.
                                    # The subfig_dir already points to the correct output directory.
                                    _efr = subfig_dir.parent.parent  # .../output
                                    cmd = [
                                        "python", str(complete_dataset_script),
                                        "--file_path", tmp_path,
                                        "--output_dir", str(rr.run_dir / "complete_dataset"),
                                        "--index_csv_path", str(index_csv_path),
                                        "--figures_tables_dir", str(rr.run_dir / "figures_tables"),
                                        "--extracted_figures_root", str(_efr),
                                        "--target_figure", subfig_dir.name,
                                    ]

                                    # Pass LLM configuration environment variables to subprocess
                                    env = os.environ.copy()

                                    # Apply LLM configuration from session state
                                    provider = st.session_state.get("llm_provider", "Ollama")
                                    model = st.session_state.get("llm_model", "")

                                    if provider == "Ollama":
                                        env["AEM_OCR_LLM_PROVIDER"] = "ollama"
                                        env["AEM_OCR_LLM_MODEL"] = model
                                        env["OLLAMA_HOST"] = st.session_state.get("ollama_host", "http://127.0.0.1:11434")
                                        ollama_models_dir = st.session_state.get("ollama_models_dir", "")
                                        if ollama_models_dir:
                                            env["OLLAMA_MODELS"] = ollama_models_dir
                                    else:
                                        env["AEM_OCR_LLM_PROVIDER"] = "openai"
                                        env["AEM_OCR_LLM_MODEL"] = model
                                        env["OPENAI_BASE_URL"] = st.session_state.get("openai_base_url", "https://api.openai.com/v1")
                                        if st.session_state.get("openai_api_key"):
                                            env["OPENAI_API_KEY"] = st.session_state["openai_api_key"]

                                    # Apply data processing LLM configuration if separate
                                    if st.session_state.get("use_separate_processing_llm"):
                                        p_provider = st.session_state.get("processing_llm_provider", "Chat GPT")
                                        p_model = st.session_state.get("processing_llm_model", "")
                                        if p_provider == "Ollama":
                                            env["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "ollama"
                                            env["AEM_OCR_PROCESSING_LLM_MODEL"] = p_model
                                        else:
                                            env["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "openai"
                                            env["AEM_OCR_PROCESSING_LLM_MODEL"] = p_model
                                            p_base = st.session_state.get("processing_openai_base_url", "https://api.openai.com/v1")
                                            p_key = st.session_state.get("processing_openai_api_key", "")
                                            if p_base:
                                                env["AEM_OCR_PROCESSING_OPENAI_BASE_URL"] = p_base
                                            if p_key:
                                                env["AEM_OCR_PROCESSING_OPENAI_API_KEY"] = p_key
                                    else:
                                        # Clear processing-specific env vars so processing uses the general LLM
                                        for k in ["AEM_OCR_PROCESSING_LLM_PROVIDER", "AEM_OCR_PROCESSING_LLM_MODEL",
                                                   "AEM_OCR_PROCESSING_OPENAI_BASE_URL", "AEM_OCR_PROCESSING_OPENAI_API_KEY"]:
                                            env.pop(k, None)

                                    # Validate that a model is configured
                                    if not model and not st.session_state.get("processing_llm_model"):
                                        st.error("No LLM model configured. Please configure LLM settings in the sidebar before running data completion.")
                                        Path(tmp_path).unlink(missing_ok=True)
                                        continue

                                    # Set custom instructions path if it exists in the workspace
                                    ci_path = rr.run_dir.parent / "custom_instructions" / "default.json"
                                    if ci_path.exists():
                                        env["AEM_OCR_CUSTOM_INSTRUCTIONS"] = str(ci_path)

                                    # Set run ID and run directory for cost tracking
                                    env["AEM_OCR_RUN_ID"] = rr.run_id
                                    env["AEM_OCR_RUN_DIR"] = str(rr.run_dir)

                                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

                                    # Clean up temp file
                                    Path(tmp_path).unlink(missing_ok=True)

                                    if result.returncode == 0:
                                        st.success(f"Data completion completed for {subfig_dir.name}! Refresh to see updated row_cache data.")
                                        st.rerun()
                                    else:
                                        st.error(f"Data completion failed:\n```\n{result.stderr}\n```")
                                except subprocess.TimeoutExpired:
                                    st.error("Data completion timed out (>5 minutes)")
                                except Exception as e:
                                    st.error(f"Error running data completion: {e}")

                    with st.expander(f"### {subfig_dir.name}"):
                        # Show original cropped image instead of recreated image.
                        # subfig_dir.name already includes the extension (e.g.
                        # "Figure_5_subfig_0.png"), so do NOT append ".png" again.
                        _subfig_img_found = False
                        if latest_manual:
                            input_dir = latest_manual / "input" / safe_doi
                            original_img = input_dir / subfig_dir.name
                            if original_img.exists():
                                st.image(str(original_img), width=fig_img_width, caption="Original cropped image")
                                _subfig_img_found = True
                        if not _subfig_img_found:
                            # Fallback: check interventions crop directory
                            _crops_dir = rr.run_dir / "interventions" / safe_doi
                            for _fig_d in (_crops_dir.iterdir() if _crops_dir.exists() else []):
                                _candidate = _fig_d / subfig_dir.name
                                if _candidate.exists():
                                    st.image(str(_candidate), width=fig_img_width, caption="Original cropped image")
                                    _subfig_img_found = True
                                    break
                        if not _subfig_img_found:
                            # Fallback to recreated image
                            final_dir = subfig_dir / "final"
                            if final_dir.exists():
                                final_img = final_dir / "final_image.png"
                                if final_img.exists():
                                    st.image(str(final_img), width=fig_img_width, caption="Recreated image (original not found)")

                    # Show extracted dataset in expander
                    final_dir = subfig_dir / "final"
                    final_csv = final_dir / "final_dataset.csv"
                    if final_csv.exists():
                        with st.expander("Extracted Data", expanded=False):
                            try:
                                df_subfig = pd.read_csv(final_csv)
                                st.data_editor(df_subfig, use_container_width=True, hide_index=True, key=f"subfig_extracted_{selected_doi}_{subfig_dir.name}")
                            except Exception as e:
                                st.warning(f"Failed to load: {e}")

                        # Show emit_rows for this subfigure from row_cache
                        emit_rows_path_subfig = find_emit_rows_for_source(
                            rr.run_dir, selected_doi, "figure", subfig_dir.name
                        )
                        if emit_rows_path_subfig:
                            df_emit_subfig = load_emit_rows(emit_rows_path_subfig)
                            if df_emit_subfig is not None and not df_emit_subfig.empty:
                                with st.expander("LLM-Generated Dataset Rows", expanded=False):
                                    st.caption(f"**Source:** `{emit_rows_path_subfig.relative_to(rr.run_dir)}`")
                                    st.data_editor(df_emit_subfig, use_container_width=True, hide_index=True, key=f"subfig_emit_{selected_doi}_{subfig_dir.name}")

                    # Add separator between subfigures if there are multiple
                    if subfig_dir != subfig_dirs[-1]:
                        st.markdown("---")

        # ========================
        # SECTION: Series Management
        # ========================
        with st.expander("Series Management", expanded=False):
            # Determine which plextract workdir to use for series management.
            # Search across all relevant workdirs in priority order:
            #   1. Per-figure interventions plextract
            #   2. Shared interventions plextract
            #   3. Main plextract output
            _sm_int_wd_er = rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions"

            _sm_fig_dir_er = _plextract_fig_dir_er
            _sm_fig_id_er = _fig_id_er
            _sm_source_img_er = _source_img_er

            _sm_workdirs_to_search_er = [
                _sm_int_wd_er,
                rr.run_dir / "interventions" / "interventions_plextract",
                rr.run_dir / "plextract",
            ]
            _sm_subfig_dirs_er = []
            for _sm_wd_check_er in _sm_workdirs_to_search_er:
                if not _sm_wd_check_er.exists():
                    continue
                _sm_lat_check_er = latest_plextract_run_dir(_sm_wd_check_er)
                if not _sm_lat_check_er:
                    continue
                _sm_out_base_check_er = _sm_lat_check_er / "output" / _safe_doi_er
                if not _sm_out_base_check_er.exists():
                    continue
                _found_er = sorted([
                    d for d in _sm_out_base_check_er.iterdir()
                    if d.is_dir() and d.name.startswith(_fig_stem_er + "_subfig_")
                ])
                if _found_er:
                    _sm_subfig_dirs_er = _found_er
                    break  # use the first workdir that has subfigures

            if _sm_subfig_dirs_er:
                st.info(f"Found {len(_sm_subfig_dirs_er)} cropped subfigure(s). Select one to manage its series:")
                _sm_subfig_opts_er = [d.name for d in _sm_subfig_dirs_er]
                _sm_sel_subfig_er = st.selectbox(
                    "Select subfigure for series management",
                    _sm_subfig_opts_er,
                    key=f"er_series_subfig_{_safe_doi_er}_{_fig_id_er}"
                )
                _sm_subfig_idx_er = _sm_subfig_opts_er.index(_sm_sel_subfig_er)
                _sm_fig_dir_er = _sm_subfig_dirs_er[_sm_subfig_idx_er]
                _sm_fig_id_er = _sm_subfig_dirs_er[_sm_subfig_idx_er].name

                # Try to find source image for this subfigure
                _crops_dir_sm = rr.run_dir / "interventions" / _safe_doi_er / _img_name_er
                _subfig_src_er = _crops_dir_sm / f"{_fig_stem_er}_subfig_{_sm_subfig_idx_er}.png"
                if _subfig_src_er.exists():
                    _sm_source_img_er = _subfig_src_er
                else:
                    _run_root_sm = _sm_fig_dir_er.parent.parent.parent
                    _input_cand_sm = _run_root_sm / "input" / _safe_doi_er / _sm_fig_id_er
                    if _input_cand_sm.exists():
                        _sm_source_img_er = _input_cand_sm

            _sm_lf_dir_er = get_lineformer_dir(_sm_fig_dir_er) if _sm_fig_dir_er else None

            # If the lineformer dir doesn't exist, create the override dir so that
            # users can create new series from scratch without running extraction first.
            _sm_lf_override_dir_er = override_lineformer_dir(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
            if _sm_lf_dir_er is None or not _sm_lf_dir_er.exists():
                _sm_lf_override_dir_er.mkdir(parents=True, exist_ok=True)
                _sm_lf_cat_path_er = _sm_lf_override_dir_er / "series_catalog.json"
                if not _sm_lf_cat_path_er.exists():
                    _sm_lf_cat_path_er.write_text("[]")
                _sm_lf_dir_er = _sm_lf_override_dir_er
                st.info("No plextract lineformer output found for this figure. You can still create new series below.")

            if True:
                _sm_series_items_er = load_series_catalog(_sm_lf_dir_er)
                if not _sm_series_items_er:
                    st.info("No line series masks found. LineFormer may not have detected any series yet — use 'Create new series' below.")

                _sm_mapping_er = load_mapping(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                _sm_names_er = _sm_mapping_er.get("series_names", {})
                _sm_deleted_er = set(int(x) for x in _sm_mapping_er.get("deleted", []))
                _sm_override_masks_er = list_override_masks(rr.run_dir, _safe_doi_er, _sm_fig_id_er)

                # Include newly-created series
                _sm_orig_indices_er = {s.series_index for s in _sm_series_items_er}
                for _idx_er, _mask_path_er in sorted(_sm_override_masks_er.items()):
                    if _idx_er not in _sm_orig_indices_er:
                        try:
                            _mimg_er = Image.open(_mask_path_er).convert("L")
                            _marr_er = np.array(_mimg_er)
                            _ys_er, _xs_er = np.where(_marr_er > 0)
                            _cx_er = float(np.mean(_xs_er)) if len(_xs_er) > 0 else 0.0
                            _cy_er = float(np.mean(_ys_er)) if len(_ys_er) > 0 else 0.0
                        except Exception:
                            _cx_er, _cy_er = 0.0, 0.0
                        from intervention import SeriesItem as _SeriesItem_er
                        _sm_series_items_er.append(_SeriesItem_er(
                            series_index=_idx_er, mask_path=_mask_path_er, centroid=(_cx_er, _cy_er)
                        ))

                _SERIES_COLORS_ER = [
                    (255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 165, 0),
                    (148, 0, 211), (0, 206, 209), (255, 20, 147), (128, 128, 0),
                    (0, 255, 127), (100, 149, 237),
                ]

                # Load series labels from plextract final_dataset.csv
                _plextract_series_labels_er: dict = {}
                _final_csv_er = (_sm_fig_dir_er / "final" / "final_dataset.csv") if _sm_fig_dir_er else None
                if _final_csv_er and _final_csv_er.exists():
                    try:
                        _fdf_er = pd.read_csv(_final_csv_er)
                        if "series_label" in _fdf_er.columns:
                            for _sl in _fdf_er["series_label"].dropna().unique():
                                _sl_str = str(_sl).strip()
                                try:
                                    _si = int(_sl_str)
                                    _plextract_series_labels_er[_si] = _sl_str
                                except ValueError:
                                    _si = len(_plextract_series_labels_er)
                                    _plextract_series_labels_er[_si] = _sl_str
                    except Exception:
                        pass

                def _display_name_er(idx):
                    if str(idx) in _sm_names_er and _sm_names_er[str(idx)] != f"Series {idx}":
                        return _sm_names_er[str(idx)]
                    if idx in _plextract_series_labels_er:
                        return _plextract_series_labels_er[idx]
                    return f"Series {idx}"

                if _sm_source_img_er and _sm_source_img_er.exists():
                    _base_img_er = Image.open(_sm_source_img_er).convert("RGBA")
                    _w_er, _h_er = _base_img_er.size
                else:
                    st.warning("No source image for series overlay.")
                    _base_img_er = None
                    _w_er, _h_er = 600, 400

                _series_w_er = st.slider(
                    "Series overlay image width (px)", 200, min(_w_er, 1500), min(_w_er, 600), 50,
                    key=f"er_series_w_{_safe_doi_er}_{_sm_fig_id_er}"
                )
                _disp_w_er = _series_w_er
                _disp_h_er = int(_h_er * (_disp_w_er / _w_er))

                # Build overlay
                _overlay_er = Image.new("RGBA", (_w_er, _h_er), (0, 0, 0, 0))
                _legend_lines_er = []
                for _s_er in sorted(_sm_series_items_er, key=lambda x: x.series_index):
                    _i_er = _s_er.series_index
                    if _i_er in _sm_deleted_er:
                        continue
                    _color_er = _SERIES_COLORS_ER[_i_er % len(_SERIES_COLORS_ER)]
                    _mp_er = _sm_override_masks_er.get(_i_er, _s_er.mask_path)
                    try:
                        _mask_er = Image.open(_mp_er).convert("L")
                        _marr2_er = np.array(_mask_er)
                        _cl_er = np.zeros((*_marr2_er.shape, 4), dtype=np.uint8)
                        _cl_er[_marr2_er > 0] = (*_color_er, 140)
                        _overlay_er = Image.alpha_composite(_overlay_er, Image.fromarray(_cl_er))
                    except Exception:
                        pass
                    _hex_c_er = "#{:02x}{:02x}{:02x}".format(*_color_er)
                    _legend_lines_er.append(f'<span style="color:{_hex_c_er}; font-weight:bold;">&#9632;</span> {_display_name_er(_i_er)}')

                if _base_img_er is not None:
                    _composite_er = Image.alpha_composite(_base_img_er, _overlay_er).convert("RGB")
                    _composite_disp_er = _composite_er.resize((_disp_w_er, _disp_h_er))
                else:
                    _composite_disp_er = None

                # Edit or Create Series Mask
                st.write("**Edit or Create Series Mask**")
                _active_series_er = [s for s in _sm_series_items_er if s.series_index not in _sm_deleted_er]
                _existing_indices_er = [s.series_index for s in _sm_series_items_er]
                _next_idx_er = max(_existing_indices_er) + 1 if _existing_indices_er else 0

                _mode_options_er = ["Edit existing series", "Create new series"]
                _mode_er = st.radio(
                    "Mode", _mode_options_er,
                    key=f"er_series_mode_{_safe_doi_er}_{_sm_fig_id_er}", horizontal=True,
                )
                _is_creating_er = (_mode_er == "Create new series")
                _cur_mask_img_er = None

                if _is_creating_er:
                    _selected_idx_er = _next_idx_er
                    _series_name_er = st.text_input(
                        "Name for new series",
                        value=f"Series {_next_idx_er}",
                        key=f"er_new_series_name_{_safe_doi_er}_{_sm_fig_id_er}",
                    )
                else:
                    if not _active_series_er:
                        st.info("No active series to edit.")
                        _selected_idx_er = None
                        _series_name_er = ""
                    else:
                        _sel_opts_er = [f"{_display_name_er(s.series_index)}" for s in _active_series_er]
                        _sel_ui_er = st.selectbox(
                            "Select series to edit",
                            range(len(_active_series_er)),
                            format_func=lambda i: _sel_opts_er[i],
                            key=f"er_mask_edit_sel_{_safe_doi_er}_{_sm_fig_id_er}",
                        )
                        _sel_s_er = _active_series_er[_sel_ui_er]
                        _selected_idx_er = _sel_s_er.series_index
                        _series_name_er = _display_name_er(_selected_idx_er)
                        _orig_mask_path_er = next(s.mask_path for s in _sm_series_items_er if s.series_index == _selected_idx_er)
                        _cur_mask_path_er = _sm_override_masks_er.get(_selected_idx_er, _orig_mask_path_er)
                        _cur_mask_img_er = Image.open(_cur_mask_path_er).convert("L").resize((_disp_w_er, _disp_h_er))

                if _selected_idx_er is not None:
                    _sel_color_er = _SERIES_COLORS_ER[_selected_idx_er % len(_SERIES_COLORS_ER)]
                    _hex_color_er = "#{:02x}{:02x}{:02x}".format(*_sel_color_er)
                    st.markdown(
                        f'**Drawing with color:** <span style="color:{_hex_color_er}; font-size:1.8em; font-weight:bold;">&#9632;</span> {_series_name_er}',
                        unsafe_allow_html=True,
                    )
                    _stroke_rgba_er = f"rgba({_sel_color_er[0]},{_sel_color_er[1]},{_sel_color_er[2]},1.0)"
                    _scale_x_s_er = _w_er / _disp_w_er
                    _scale_y_s_er = _h_er / _disp_h_er

                    _input_mode_er = st.radio(
                        "Input mode", ["Freehand drawing", "Pick exact points"],
                        horizontal=True,
                        key=f"er_input_mode_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}",
                    )
                    _c1_er, _c2_er = st.columns([3, 1])

                    if _input_mode_er == "Freehand drawing":
                        with _c1_er:
                            st.write("Draw on the colored series image below")
                            _brush_er = st.slider("Brush width", 1, 40, 10, key=f"er_brush_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}")
                            _mask_canvas_er = st_canvas(
                                background_image=_composite_disp_er,
                                drawing_mode="freedraw",
                                stroke_width=_brush_er, stroke_color=_stroke_rgba_er,
                                fill_color=_stroke_rgba_er, update_streamlit=True,
                                height=_disp_h_er, width=_disp_w_er,
                                key=f"er_mask_canvas_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}",
                            )
                        if _legend_lines_er:
                            st.markdown(" &nbsp; ".join(_legend_lines_er), unsafe_allow_html=True)

                        if _mask_canvas_er.image_data is not None:
                            _rgba_er = _mask_canvas_er.image_data.astype(np.uint8)
                            _r_er, _g_er, _b_er = _sel_color_er
                            _color_match_er = (
                                (np.abs(_rgba_er[..., 0].astype(int) - _r_er) < 30) &
                                (np.abs(_rgba_er[..., 1].astype(int) - _g_er) < 30) &
                                (np.abs(_rgba_er[..., 2].astype(int) - _b_er) < 30) &
                                (_rgba_er[..., 3] > 0)
                            )
                            _new_mask_small_er = _color_match_er.astype(np.uint8) * 255
                            if np.any(_new_mask_small_er > 0):
                                _new_mask_full_er = cv2.resize(_new_mask_small_er, (_w_er, _h_er), interpolation=cv2.INTER_NEAREST)
                                _new_mask_pil_er = Image.fromarray(_new_mask_full_er).convert("L")
                                _colA_er, _colB_er = st.columns(2)
                                with _colA_er:
                                    _save_label_er = "Create new series" if _is_creating_er else "Save edited mask"
                                    if st.button(_save_label_er, type="primary", key=f"er_save_mask_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}"):
                                        _outp_er = save_mask_override(rr.run_dir, _safe_doi_er, _sm_fig_id_er, _selected_idx_er, _new_mask_pil_er)
                                        if _is_creating_er:
                                            _map_upd_er = load_mapping(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                                            if "series_names" not in _map_upd_er:
                                                _map_upd_er["series_names"] = {}
                                            _map_upd_er["series_names"][str(_selected_idx_er)] = _series_name_er
                                            save_mapping(rr.run_dir, _safe_doi_er, _sm_fig_id_er, _map_upd_er)
                                            _marr3_er = np.array(_new_mask_pil_er)
                                            _yc_er, _xc_er = np.where(_marr3_er > 0)
                                            _cx3_er = float(np.mean(_xc_er)) if len(_xc_er) > 0 else 0.0
                                            _cy3_er = float(np.mean(_yc_er)) if len(_yc_er) > 0 else 0.0
                                            from intervention import SeriesItem as _SI3_er
                                            _sm_series_items_er.append(_SI3_er(series_index=_selected_idx_er, mask_path=_outp_er, centroid=(_cx3_er, _cy3_er)))
                                            st.success(f"Created new series '{_series_name_er}'")
                                        else:
                                            st.success(f"Saved override mask for {_series_name_er}")
                                        _om_upd_er = list_override_masks(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                                        write_series_catalog_override(rr.run_dir, _safe_doi_er, _sm_fig_id_er, _sm_series_items_er, _om_upd_er)
                                        st.rerun()
                                with _colB_er:
                                    if st.button("Reset canvas", key=f"er_reset_mask_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}"):
                                        st.rerun()

                    else:  # Pick exact points
                        _lf_ov_dir_er = override_lineformer_dir(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                        _pts_file_er = _lf_ov_dir_er / f"clicked_points_{_selected_idx_er}.json"
                        _existing_pts_er: list = []
                        if _pts_file_er.exists():
                            try:
                                _existing_pts_er = json.loads(_pts_file_er.read_text())
                            except Exception:
                                pass
                        with _c1_er:
                            _dot_size_er = st.slider(
                                "Point display size", 1, 15, 3,
                                key=f"er_dot_size_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}",
                            )
                            if _existing_pts_er:
                                st.caption(f"Currently saved: **{len(_existing_pts_er)} point(s)**. Placing new points replaces them.")
                            else:
                                st.caption("Click on the figure to place data points.")
                            _point_canvas_er = st_canvas(
                                background_image=_composite_disp_er,
                                drawing_mode="point", point_display_radius=_dot_size_er,
                                stroke_color=_stroke_rgba_er, fill_color=_stroke_rgba_er,
                                update_streamlit=True, height=_disp_h_er, width=_disp_w_er,
                                key=f"er_point_canvas_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}",
                            )
                        if _legend_lines_er:
                            st.markdown(" &nbsp; ".join(_legend_lines_er), unsafe_allow_html=True)

                        _new_pts_er: list = []
                        if _point_canvas_er.json_data and "objects" in _point_canvas_er.json_data:
                            for _obj2_er in _point_canvas_er.json_data["objects"]:
                                if _obj2_er.get("type") == "circle":
                                    _r2_er = float(_obj2_er.get("radius", _dot_size_er))
                                    _cxd_er = float(_obj2_er.get("left", 0)) + _r2_er
                                    _cyd_er = float(_obj2_er.get("top", 0)) + _r2_er
                                    _new_pts_er.append({"x": _cxd_er * _scale_x_s_er, "y": _cyd_er * _scale_y_s_er})
                        _new_pts_er.sort(key=lambda p: p["x"])

                        if _new_pts_er:
                            st.caption(f"Placed **{len(_new_pts_er)} new point(s)** — click Save to apply.")
                            _colA2_er, _colB2_er = st.columns(2)
                            with _colA2_er:
                                _save_label2_er = "Create new series" if _is_creating_er else "Save points"
                                if st.button(_save_label2_er, type="primary", key=f"er_save_pts_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}"):
                                    _lf_ov_dir_er.mkdir(parents=True, exist_ok=True)
                                    _pts_file_er.write_text(json.dumps(_new_pts_er, indent=2))
                                    _dot_r_er = max(3, int(5 * _scale_x_s_er))
                                    _mask_arr_er = np.zeros((_h_er, _w_er), dtype=np.uint8)
                                    for _pt_er in _new_pts_er:
                                        cv2.circle(_mask_arr_er, (int(round(_pt_er["x"])), int(round(_pt_er["y"]))), _dot_r_er, 255, -1)
                                    _mask_pil_er = Image.fromarray(_mask_arr_er).convert("L")
                                    _outp2_er = save_mask_override(rr.run_dir, _safe_doi_er, _sm_fig_id_er, _selected_idx_er, _mask_pil_er)
                                    if _is_creating_er:
                                        _map_upd2_er = load_mapping(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                                        if "series_names" not in _map_upd2_er:
                                            _map_upd2_er["series_names"] = {}
                                        _map_upd2_er["series_names"][str(_selected_idx_er)] = _series_name_er
                                        save_mapping(rr.run_dir, _safe_doi_er, _sm_fig_id_er, _map_upd2_er)
                                        from intervention import SeriesItem as _SI4_er
                                        _cx4_er = float(np.mean([p["x"] for p in _new_pts_er]))
                                        _cy4_er = float(np.mean([p["y"] for p in _new_pts_er]))
                                        _sm_series_items_er.append(_SI4_er(series_index=_selected_idx_er, mask_path=_outp2_er, centroid=(_cx4_er, _cy4_er)))
                                        st.success(f"Created new series '{_series_name_er}' with {len(_new_pts_er)} points")
                                    else:
                                        st.success(f"Saved {len(_new_pts_er)} points for {_series_name_er}")
                                    _om_upd2_er = list_override_masks(rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                                    write_series_catalog_override(rr.run_dir, _safe_doi_er, _sm_fig_id_er, _sm_series_items_er, _om_upd2_er)
                                    st.rerun()
                            with _colB2_er:
                                if st.button("Reset", key=f"er_reset_pts_{_safe_doi_er}_{_sm_fig_id_er}_{_selected_idx_er}_{_mode_er}"):
                                    st.rerun()

                    with _c2_er:
                        if _cur_mask_img_er is not None:
                            st.write("Current mask")
                            st_image_auto(_cur_mask_img_er, full_width=True, use_container_width=True)
                        else:
                            st.info("Drawing new series")

                # Rename / delete section
                st.write("**Rename or delete** each series below.")
                _new_names_er = {}
                _new_deleted_er = set(_sm_deleted_er)
                for _s2_er in sorted(_sm_series_items_er, key=lambda x: x.series_index):
                    _i2_er = _s2_er.series_index
                    if _i2_er in _sm_deleted_er:
                        continue
                    _color2_er = _SERIES_COLORS_ER[_i2_er % len(_SERIES_COLORS_ER)]
                    _hex2_er = "#{:02x}{:02x}{:02x}".format(*_color2_er)
                    _cols_er = st.columns([0.5, 3, 1])
                    with _cols_er[0]:
                        st.markdown(f'<span style="color:{_hex2_er}; font-size:1.4em; font-weight:bold;">&#9632;</span>', unsafe_allow_html=True)
                    with _cols_er[1]:
                        _new_names_er[str(_i2_er)] = st.text_input(
                            f"Name for series {_i2_er}", value=_display_name_er(_i2_er),
                            key=f"er_series_name_{_safe_doi_er}_{_sm_fig_id_er}_{_i2_er}",
                            label_visibility="collapsed",
                        )
                    with _cols_er[2]:
                        _del_flag_er = st.checkbox("Delete", value=False, key=f"er_series_del_{_safe_doi_er}_{_sm_fig_id_er}_{_i2_er}")
                        if _del_flag_er:
                            _new_deleted_er.add(_i2_er)

                if st.button("Save series mapping", type="primary", key=f"er_save_mapping_{_safe_doi_er}_{_sm_fig_id_er}"):
                    _map2_er = {"series_names": _new_names_er, "deleted": sorted(list(_new_deleted_er))}
                    _p_er = save_mapping(rr.run_dir, _safe_doi_er, _sm_fig_id_er, _map2_er)
                    st.success(f"Saved mapping: {_p_er}")
                    st.rerun()

                st.divider()
                # Rerun data extraction from series management
                _is_subfig_sel_er = _sm_fig_id_er != _fig_id_er
                _int_crops_er = sorted(_crops_out_dir_er.glob(f"{_fig_stem_er}_subfig_*.png")) if _crops_out_dir_er.exists() else []
                if _is_subfig_sel_er:
                    st.caption(f"Will run on selected subfigure: **{_sm_fig_id_er}**")
                else:
                    if _int_crops_er:
                        st.caption(f"Will run on **{len(_int_crops_er)} cropped image(s)**.")
                    else:
                        st.caption("Will run on the original figure (no crops found).")

                if st.button("Rerun data extraction", type="primary", key=f"er_rerun_from_series_{_safe_doi_er}_{_sm_fig_id_er}"):
                    try:
                        _rerun_images_er: list = []
                        _rerun_lf_er: dict = {}

                        if _sm_fig_dir_er is not None:
                            with st.spinner("Applying mask overrides..."):
                                apply_overrides_to_plextract(_sm_fig_dir_er, rr.run_dir, _safe_doi_er, _sm_fig_id_er)
                            _lf_dir_rerun_er = get_lineformer_dir(_sm_fig_dir_er)
                        else:
                            # No plextract output exists yet — use the override lineformer dir directly
                            _lf_dir_rerun_er = _sm_lf_override_dir_er

                        if _is_subfig_sel_er:
                            _rerun_images_er = [_sm_source_img_er]
                            if _lf_dir_rerun_er and _lf_dir_rerun_er.exists():
                                _rerun_lf_er[_sm_source_img_er.name] = _lf_dir_rerun_er
                        elif _int_crops_er:
                            _rerun_images_er = list(_int_crops_er)
                            for _crop_er in _int_crops_er:
                                _crop_lf_er = None
                                # Search all workdirs for this crop's lineformer dir
                                for _sm_wd_r_er in _sm_workdirs_to_search_er:
                                    if not _sm_wd_r_er.exists():
                                        continue
                                    _lat2_er = latest_plextract_run_dir(_sm_wd_r_er)
                                    if _lat2_er:
                                        _cand_lf_er = _lat2_er / "output" / _safe_doi_er / _crop_er.name / "lineformer"
                                        if _cand_lf_er.exists():
                                            _crop_lf_er = _cand_lf_er
                                            break
                                _lf_to_use_er = _crop_lf_er or (_lf_dir_rerun_er if _lf_dir_rerun_er and _lf_dir_rerun_er.exists() else None)
                                if _lf_to_use_er:
                                    _rerun_lf_er[_crop_er.name] = _lf_to_use_er
                        else:
                            if _source_img_er and _source_img_er.exists():
                                _rerun_images_er = [_source_img_er]
                                if _lf_dir_rerun_er and _lf_dir_rerun_er.exists():
                                    _rerun_lf_er[_source_img_er.name] = _lf_dir_rerun_er
                        with st.spinner("Re-running data extraction..."):
                            run_plextract_on_crops(
                                scripts_root=scripts_root, run_dir=rr.run_dir,
                                safe_doi=_safe_doi_er, image_paths=_rerun_images_er,
                                override_lineformer_dirs=_rerun_lf_er if _rerun_lf_er else None,
                                figure_id=_fig_id_er,
                            )
                        st.success("Data extraction complete!")
                        st.rerun()
                    except Exception as _e2_er:
                        st.error(f"Failed: {_e2_er}")

        # ========================
        # SECTION: Display metadata
        # ========================
        if f.metadata_path and f.metadata_path.exists():
            with st.expander("Metadata", expanded=False):
                st.json(json.loads(f.metadata_path.read_text()))
                
        # ========================
        # SECTION: Rerun Plextract Data Extraction
        # ========================
        st.markdown("---")
        st.markdown("**Rerun Plextract Data Extraction**")
        st.caption("Reruns ChartDete, LineFormer (or reuses existing masks), OCR, coordinate correction, and series extraction.")
        _rerun_existing_crops_er = sorted(_crops_out_dir_er.glob(f"{_fig_stem_er}_subfig_*.png")) if _crops_out_dir_er.exists() else []
        if _rerun_existing_crops_er:
            st.caption(f"Will run on **{len(_rerun_existing_crops_er)} saved crop(s)**.")
        elif _source_img_er and _source_img_er.exists():
            st.caption("Will run on the original figure image.")

        if st.button("Rerun plextract extraction", type="primary", key=f"er_rerun_plextract_{_safe_doi_er}_{_fig_id_er}"):
            if _rerun_existing_crops_er or (_source_img_er and _source_img_er.exists()):
                try:
                    _images_for_rerun_er = _rerun_existing_crops_er if _rerun_existing_crops_er else [_source_img_er]
                    with st.spinner("Re-running data extraction..."):
                        run_plextract_on_crops(
                            scripts_root=scripts_root, run_dir=rr.run_dir,
                            safe_doi=_safe_doi_er,
                            image_paths=_images_for_rerun_er,
                            figure_id=_fig_id_er,
                        )
                    st.success("Plextract extraction complete!")
                    st.rerun()
                except Exception as _e3_er:
                    st.error(f"Extraction failed: {_e3_er}")
            else:
                st.error("No source image or crops found.")

        # ========================
        # SECTION: Rerun LLM Dataset Generation
        # ========================
        st.markdown("**Rerun LLM Dataset Generation**")
        st.caption("Runs complete_dataset.py to generate normalized dataset rows from the extracted figure data.")

        # Find available figures/subfigures for LLM rerun
        _llm_fig_options_er: list = []
        _llm_fig_dirs_er: list = []
        for _llm_wd_er in [
            rr.run_dir / "interventions" / _safe_doi_er / _fig_id_er / "plextract_interventions",
            rr.run_dir / "interventions" / "interventions_plextract",
            rr.run_dir / "plextract",
        ]:
            if not _llm_wd_er.exists():
                continue
            _llm_lat_er = latest_plextract_run_dir(_llm_wd_er)
            if not _llm_lat_er:
                continue
            _llm_out_base_er = _llm_lat_er / "output" / _safe_doi_er
            if not _llm_out_base_er.exists():
                continue
            for _llm_d_er in sorted(_llm_out_base_er.iterdir()):
                if _llm_d_er.is_dir() and (
                    _llm_d_er.name == _fig_id_er or
                    _llm_d_er.name.startswith(_fig_stem_er + "_subfig_")
                ):
                    if _llm_d_er.name not in _llm_fig_options_er:
                        _llm_fig_options_er.append(_llm_d_er.name)
                        _llm_fig_dirs_er.append(_llm_d_er)

        if not _llm_fig_dirs_er:
            st.info("No plextract output found. Run plextract extraction first.")
        else:
            if len(_llm_fig_dirs_er) > 1:
                _llm_sel_er = st.selectbox(
                    "Select figure/subfigure for LLM rerun",
                    _llm_fig_options_er,
                    key=f"er_llm_sel_{_safe_doi_er}_{_fig_id_er}"
                )
                _llm_sel_dir_er = _llm_fig_dirs_er[_llm_fig_options_er.index(_llm_sel_er)]
            else:
                _llm_sel_er = _llm_fig_options_er[0]
                _llm_sel_dir_er = _llm_fig_dirs_er[0]

            if st.button("Rerun LLM dataset generation", type="primary", key=f"er_rerun_llm_{_safe_doi_er}_{_fig_id_er}"):
                _reparsed_dir_er = rr.run_dir / "reparsed"
                _index_csv_er = _reparsed_dir_er / "index.csv"
                if not _index_csv_er.exists():
                    _err_msg_er = f"**Required file not found:** `{_index_csv_er.relative_to(rr.run_dir)}`\n\n"
                    if not _reparsed_dir_er.exists():
                        _err_msg_er += "The `reparsed/` directory doesn't exist. "
                    else:
                        _err_msg_er += "The `reparsed/` directory exists but `index.csv` is missing. "
                    _err_msg_er += "This file is created during the 'Download DOIs' stage."
                    st.error(_err_msg_er)
                else:
                    try:
                        import tempfile as _tmpmod_er
                        with _tmpmod_er.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as _tmp_er:
                            _tmp_er.write(f"DOI\n{selected_doi}\n")
                            _tmp_path_er = _tmp_er.name

                        _complete_script_er = Path(__file__).parent.parent / "complete_dataset" / "complete_dataset.py"
                        _efr_er = _llm_sel_dir_er.parent.parent  # .../output
                        _cmd_er = [
                            "python", str(_complete_script_er),
                            "--file_path", _tmp_path_er,
                            "--output_dir", str(rr.run_dir / "complete_dataset"),
                            "--index_csv_path", str(_index_csv_er),
                            "--figures_tables_dir", str(rr.run_dir / "figures_tables"),
                            "--extracted_figures_root", str(_efr_er),
                            "--target_figure", _llm_sel_er,
                        ]

                        _env_er = os.environ.copy()
                        _provider_er = st.session_state.get("llm_provider", "Ollama")
                        _model_er = st.session_state.get("llm_model", "")
                        if _provider_er == "Ollama":
                            _env_er["AEM_OCR_LLM_PROVIDER"] = "ollama"
                            _env_er["AEM_OCR_LLM_MODEL"] = _model_er
                            _env_er["OLLAMA_HOST"] = st.session_state.get("ollama_host", "http://127.0.0.1:11434")
                            _om_dir_er = st.session_state.get("ollama_models_dir", "")
                            if _om_dir_er:
                                _env_er["OLLAMA_MODELS"] = _om_dir_er
                        else:
                            _env_er["AEM_OCR_LLM_PROVIDER"] = "openai"
                            _env_er["AEM_OCR_LLM_MODEL"] = _model_er
                            _env_er["OPENAI_BASE_URL"] = st.session_state.get("openai_base_url", "https://api.openai.com/v1")
                            if st.session_state.get("openai_api_key"):
                                _env_er["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
                        if st.session_state.get("use_separate_processing_llm"):
                            _pp_er = st.session_state.get("processing_llm_provider", "Chat GPT")
                            _pm_er = st.session_state.get("processing_llm_model", "")
                            if _pp_er == "Ollama":
                                _env_er["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "ollama"
                                _env_er["AEM_OCR_PROCESSING_LLM_MODEL"] = _pm_er
                            else:
                                _env_er["AEM_OCR_PROCESSING_LLM_PROVIDER"] = "openai"
                                _env_er["AEM_OCR_PROCESSING_LLM_MODEL"] = _pm_er
                                _pb_er = st.session_state.get("processing_openai_base_url", "https://api.openai.com/v1")
                                _pk_er = st.session_state.get("processing_openai_api_key", "")
                                if _pb_er:
                                    _env_er["AEM_OCR_PROCESSING_OPENAI_BASE_URL"] = _pb_er
                                if _pk_er:
                                    _env_er["AEM_OCR_PROCESSING_OPENAI_API_KEY"] = _pk_er
                        else:
                            for _k_er in ["AEM_OCR_PROCESSING_LLM_PROVIDER", "AEM_OCR_PROCESSING_LLM_MODEL",
                                            "AEM_OCR_PROCESSING_OPENAI_BASE_URL", "AEM_OCR_PROCESSING_OPENAI_API_KEY"]:
                                _env_er.pop(_k_er, None)

                        if not _model_er and not st.session_state.get("processing_llm_model"):
                            st.error("No LLM model configured. Configure LLM settings in the sidebar first.")
                            Path(_tmp_path_er).unlink(missing_ok=True)
                        else:
                            _ci_path_er = rr.run_dir.parent / "custom_instructions" / "default.json"
                            if _ci_path_er.exists():
                                _env_er["AEM_OCR_CUSTOM_INSTRUCTIONS"] = str(_ci_path_er)
                            _env_er["AEM_OCR_RUN_ID"] = rr.run_id
                            _env_er["AEM_OCR_RUN_DIR"] = str(rr.run_dir)

                            with st.spinner(f"Running LLM dataset generation for {_llm_sel_er}..."):
                                _result_er = subprocess.run(_cmd_er, capture_output=True, text=True, timeout=300, env=_env_er)
                            Path(_tmp_path_er).unlink(missing_ok=True)
                            if _result_er.returncode == 0:
                                st.success(f"LLM dataset generation complete for {_llm_sel_er}!")
                                st.rerun()
                            else:
                                st.error(f"LLM dataset generation failed:\n```\n{_result_er.stderr}\n```")
                    except subprocess.TimeoutExpired:
                        st.error("LLM dataset generation timed out (>5 minutes)")
                    except Exception as _e4_er:
                        st.error(f"Error: {_e4_er}")

with tab_tables:
    if not doi_item.tables:
        st.warning("No tables found.")
    for idx, t in enumerate(doi_item.tables):
        # Add divider between tables
        if idx > 0:
            st.divider()

        # Table header and basic info (not in expander)
        st.markdown(f"### {t.id}")
        with st.expander(f"### {t.id}"):
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
                st.caption(str(t.table_path))

        # --- Display LLM-generated emit_rows from row_cache ---
        emit_rows_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "table", t.id)
        if emit_rows_path:
            df_emit = load_emit_rows(emit_rows_path)
            if df_emit is not None and not df_emit.empty:
                with st.expander("LLM-Generated Dataset Rows", expanded=False):
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
                with st.expander("LLM-Generated Dataset Rows", expanded=False):
                    st.info(f"No row_cache data found for {t.id}. Cache may not have been generated yet.")

        # Metadata expander
        if t.metadata_path and t.metadata_path.exists():
            with st.expander("Metadata", expanded=False):
                st.json(json.loads(t.metadata_path.read_text()))

with tab_xy:
    st.subheader("Completed dataset")

    # Collect all emit_rows from row_cache (reactive to changes)
    df_combined = collect_all_emit_rows(rr.run_dir, selected_doi, doi_item)

    if not df_combined.empty:
        st.info(f"Dataset contains {len(df_combined)} rows from {len(df_combined.columns)} columns (concatenated from all tables, figures, and subfigures)")
        st.data_editor(df_combined, use_container_width=True, hide_index=True, key=f"dataset_{selected_doi}")

        # Download button
        csv_data = df_combined.to_csv(index=False)
        st.download_button(
            label="Download completed dataset CSV",
            data=csv_data,
            file_name=f"{doi_to_safe(selected_doi)}_dataset.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No dataset rows found in row_cache for this DOI.\n\n"
            "To generate dataset rows: enable **Complete dataset (LLM)** and run the pipeline, "
            "or use the 'Run Data Completion' button on individual figures/tables."
        )

    # Show legacy dataset CSV if it exists (for reference)
    ds_csv = completed_dataset_csv_path(rr.run_dir, selected_doi)
    ds_json = completed_dataset_json_path(rr.run_dir, selected_doi)

    if ds_csv.exists():
        with st.expander("Legacy completed dataset CSV (from complete_dataset.py)", expanded=False):
            try:
                df_legacy = _load_csv(str(ds_csv))
                st.dataframe(df_legacy, use_container_width=True, hide_index=True)
                st.caption(f"CSV path: {ds_csv}")
            except Exception as e:
                st.warning(f"Failed to load legacy dataset CSV: {e}")

with tab_chem:
    st.subheader("Chemical Composition")

    # --- Add src/ to path for chemical_composition imports ---
    _src_for_chem = (Path(__file__).resolve().parent / "..").resolve()
    if str(_src_for_chem) not in sys.path:
        sys.path.insert(0, str(_src_for_chem))

    from chemical_composition.extract_composition import (
        identify_relevant_content as _cc_identify,
        extract_samples_and_compositions as _cc_extract,
        run_chemical_composition_extraction as _cc_run,
    )
    from chemical_composition.drawing_tool import (
        render_jsme_editor_local,
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
        with st.expander("Extracted Samples & Compositions", expanded=True):
            if cc_csv_path and Path(cc_csv_path).exists():
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
                # Fall back to JSON display
                samples = cc_data.get("samples", [])
                if samples:
                    for s in samples:
                        with st.expander(f"Sample: {s.get('sample_name', 'Unknown')}"):
                            if s.get("structure_class"):
                                st.markdown(f"**Structure class:** {s['structure_class']}")
                            if s.get("notes"):
                                st.caption(s["notes"])
                            comps = s.get("components", [])
                            if comps:
                                comp_rows = []
                                for c in comps:
                                    comp_rows.append({
                                        "Component": c.get("name", ""),
                                        "Role": c.get("role", ""),
                                        "Formula": c.get("formula", ""),
                                        "SMILES": c.get("smiles", ""),
                                        "Amount": c.get("amount", ""),
                                    })
                                st.dataframe(
                                    pd.DataFrame(comp_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
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

    # --- Chemical Structure Drawing Tool ---
    with st.expander("Draw Chemical Structures", expanded=False):
        render_jsme_editor_local(height=550, key=f"jsme_{_cc_safe}")

        st.divider()
        st.markdown("#### Save Drawn Structure")

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
    with st.expander("Run Chamical Composition Extraction", expanded=False):
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

                    result = _cc_run(
                        run_dir=rr.run_dir,
                        safe_doi=_cc_safe,
                        parsed_json_path=parsed_json,
                        tables_dir=tables_d if tables_d.exists() else None,
                        figures_dir=figures_d if figures_d.exists() else None,
                    )

                st.success("Extraction complete!")

                # Show results inline
                samples = result.get("samples", [])
                st.markdown(f"**Found {len(samples)} sample(s)**")
                if samples:
                    for s in samples:
                        with st.expander(f"Sample: {s.get('sample_name', 'Unknown')}"):
                            st.json(s)

                relevant = result.get("relevant_content", [])
                st.markdown(f"**Identified {len(relevant)} relevant content section(s)**")

                st.info("Reload the run or switch tabs to see the full results.")

            except Exception as e:
                st.error(f"Chemical composition extraction failed: {e}")

