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
)
from run_plextract_single import run_plextract_single_figure
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
    with st.expander("Run cost summary", expanded=False):
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

# ---------------- Sidebar ----------------
with st.sidebar:

    # Set workspace variables – resolve relative to this file, not cwd
    scripts_root = (Path(__file__).resolve().parent / "..").resolve()
    workspaces_root = Path(str(scripts_root / "ui_runs"))
    workspaces_root.mkdir(parents=True, exist_ok=True)

    st.divider()

    # Choose/create workspace
    st.header("Workspace")

    # List existing workspaces
    existing_workspaces = _list_subdirs(workspaces_root)

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
        run_filter_figures = st.checkbox("Filter figures", value=True)
        run_filter_tables = st.checkbox("Filter tables", value=True)
        run_plextract = st.checkbox("Extract data from figures", value=True)
        run_complete_dataset = st.checkbox("Complete dataset", value=True)
        run_chemical_composition = st.checkbox("Extract chemical composition", value=True)
        exclude_full_text = st.checkbox("Exclude full text from context", value=True)

    st.divider()
    st.header("Select LLM(s)")
    with st.expander("LLM", expanded=False):
        # --- General LLM (used for filtering, figure planning, etc.) ---
        st.subheader("General LLM")
        st.caption("Used for figure/table filtering, figure planning, and other pre-completion steps.")

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
            selected_model = st.selectbox("Model", options=_openai_models, index=0, key="llm_model")

        # --- Dataset Completion LLM ---
        st.subheader("Dataset Completion LLM")
        st.caption("Used for the final dataset completion step. Can differ from the general LLM.")

        use_separate_completion_llm = st.checkbox(
            "Use a different LLM for dataset completion",
            value=False,
            key="use_separate_completion_llm",
        )

        if use_separate_completion_llm:
            completion_provider = st.radio(
                "Completion provider",
                options=["Ollama", "Chat GPT"],
                index=1,
                key="completion_llm_provider",
            )

            if completion_provider == "Ollama":
                models = _ollama_models()
                if models:
                    st.selectbox("Completion model", options=models, index=0, key="completion_llm_model")
                else:
                    st.text_input("Completion model (no models detected)", value="", key="completion_llm_model")
            else:
                # Read completion OpenAI config from environment (not displayed in UI)
                if "completion_openai_base_url" not in st.session_state:
                    st.session_state["completion_openai_base_url"] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                if "completion_openai_api_key" not in st.session_state:
                    st.session_state["completion_openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
                st.selectbox("Completion model", options=_openai_models, index=0, key="completion_llm_model")

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

    with st.expander("Dataset completion rules", expanded=False):
        st.caption(
            "Extra rules applied when the LLM converts tables/figures into "
            "the final normalised dataset rows. E.g. 'before = time 0'."
        )
        st.text_area(
            "Dataset completion extra rules",
            value=ci.get("dataset_completion_rules", ""),
            height=120,
            key="ci_dataset_completion",
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
                _ci_to_save["dataset_completion_rules"] = st.session_state.get("ci_dataset_completion", "")

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
        st.header("Load an existing run")

        # Choose which workspace to browse when loading runs
        load_ws_options = _list_subdirs(workspaces_root)

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

            # --- Apply dataset completion LLM configuration ---
            if st.session_state.get("use_separate_completion_llm"):
                c_provider = st.session_state.get("completion_llm_provider", "Chat GPT")
                c_model = st.session_state.get("completion_llm_model", "")
                if c_provider == "Ollama":
                    os.environ["AEM_OCR_COMPLETION_LLM_PROVIDER"] = "ollama"
                    os.environ["AEM_OCR_COMPLETION_LLM_MODEL"] = c_model
                else:
                    os.environ["AEM_OCR_COMPLETION_LLM_PROVIDER"] = "openai"
                    os.environ["AEM_OCR_COMPLETION_LLM_MODEL"] = c_model
                    c_base = st.session_state.get("completion_openai_base_url", "https://api.openai.com/v1")
                    c_key = st.session_state.get("completion_openai_api_key", "")
                    if c_base:
                        os.environ["AEM_OCR_COMPLETION_OPENAI_BASE_URL"] = c_base
                    if c_key:
                        os.environ["AEM_OCR_COMPLETION_OPENAI_API_KEY"] = c_key
            else:
                # Clear completion-specific env vars so complete_dataset uses the general LLM
                for k in ["AEM_OCR_COMPLETION_LLM_PROVIDER", "AEM_OCR_COMPLETION_LLM_MODEL",
                           "AEM_OCR_COMPLETION_OPENAI_BASE_URL", "AEM_OCR_COMPLETION_OPENAI_API_KEY"]:
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
if "selected_doi" not in st.session_state:
    st.session_state["selected_doi"] = doi_options[0] if doi_options else None

if df_selected_doi:
    st.session_state["selected_doi"] = df_selected_doi

selected_doi = st.selectbox(
    "Select DOI",
    options=doi_options,
    index=doi_options.index(st.session_state["selected_doi"]) if st.session_state["selected_doi"] in doi_options else 0,
)
st.session_state["selected_doi"] = selected_doi

doi_item = next(d for d in rr.dois if d.doi == selected_doi)

# ---------------- Tabs (under DOI dropdown) ----------------
tab_text, tab_figs, tab_tables, tab_xy, tab_chem, tab_intervene = st.tabs(
    ["Parsed text", "Figures", "Tables", "Dataset", "Chemical Composition", "Interventions"]
)

with tab_text:
    if doi_item.parsed_json_path and doi_item.parsed_json_path.exists():
        st.json(json.loads(doi_item.parsed_json_path.read_text()))
    else:
        st.warning("No parsed.json found for this DOI in the run workspace.")

with tab_figs:
    if not doi_item.figures:
        st.warning("No figures found.")
    for f in doi_item.figures:
        with st.expander(f"{f.id}"):
            if f.caption:
                st.markdown(f"**Caption:** {f.caption}")
            if f.figure_path and f.figure_path.exists():
                st_image_auto(str(f.figure_path), full_width=True)
                st.caption(str(f.figure_path))
            
            # --- Extracted dataset table (plextract final_dataset.csv) ---
            final_csv = None
            if isinstance(getattr(f, "extra", None), dict):
                final_csv = f.extra.get("final_dataset_csv")

            if final_csv and Path(final_csv).exists():
                try:
                    df = _load_csv(final_csv)
                    st.markdown("**Extracted Data**")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.caption(final_csv)
                except Exception as e:
                    st.warning(f"Failed to load extracted dataset:\n`{final_csv}`\n\n{e}")
            else:
                st.info("No extracted dataset found for this figure.")

            if f.metadata_path and f.metadata_path.exists():
                st.markdown("**Metadata**")
                st.json(json.loads(f.metadata_path.read_text()))

with tab_tables:
    if not doi_item.tables:
        st.warning("No tables found.")
    for t in doi_item.tables:
        with st.expander(f"{t.id}"):
            if t.caption:
                st.markdown(f"**Caption:** {t.caption}")
            if t.table_path and t.table_path.exists():
                suf = t.table_path.suffix.lower()
                if suf in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                    st_image_auto(str(t.table_path), full_width=True)
                elif suf == ".csv":
                    df = pd.read_csv(t.table_path)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.code(t.table_path.read_text(errors="ignore")[:20000])
                st.caption(str(t.table_path))
            if t.metadata_path and t.metadata_path.exists():
                st.markdown("**Metadata**")
                st.json(json.loads(t.metadata_path.read_text()))

with tab_xy:
    st.subheader("Completed dataset")

    ds_csv = completed_dataset_csv_path(rr.run_dir, selected_doi)
    ds_json = completed_dataset_json_path(rr.run_dir, selected_doi)

    if ds_csv.exists():
        try:
            df_ds = _load_csv(str(ds_csv))
            st.dataframe(df_ds, use_container_width=True, hide_index=True)
            st.caption("CSV path:")
            st.code(str(ds_csv))
            st.download_button(
                label="Download completed dataset CSV",
                data=ds_csv.read_bytes(),
                file_name=ds_csv.name,              # e.g. 10.1039@..._dataset.csv
                mime="text/csv",
            )

        except Exception as e:
            st.warning(f"Failed to load completed dataset CSV:\n`{ds_csv}`\n\n{e}")
    else:
        # If LLM was skipped, they may still have the combined json
        if ds_json.exists():
            print("=====ds_json====", ds_json)
            st.info("No *_dataset.csv found (LLM completion may have been skipped), but *_complete.json exists.")
            with st.expander("Show combined *_complete.json", expanded=False):
                st.json(json.loads(ds_json.read_text()))
            st.caption("CSV path:")
            st.code(str(ds_csv))
        else:
            st.info(
                "No completed dataset found for this DOI in this run.\n\n"
                "To generate it: enable **Complete dataset (LLM)** and re-run the pipeline."
            )

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
        render_ketcher_editor,
        load_drawn_structures,
        save_drawn_structure,
        delete_drawn_structure,
    )

    _cc_safe = doi_to_safe(selected_doi)

    # --- Sub-tabs within Chemical Composition ---
    sub_cc_results, sub_cc_draw, sub_cc_run_tab = st.tabs(
        ["Extracted Results", "Draw Chemical Structure", "Run Extraction"]
    )

    with sub_cc_results:
        # Show previously extracted results if they exist
        cc_json_path = doi_item.extra.get("chemical_composition_json")
        cc_csv_path = doi_item.extra.get("chemical_composition_csv")

        if cc_json_path and Path(cc_json_path).exists():
            cc_data = json.loads(Path(cc_json_path).read_text())

            # --- Relevant content ---
            with st.expander("Relevant content identified", expanded=False):
                relevant = cc_data.get("relevant_content", [])
                if relevant:
                    for i, item_rc in enumerate(relevant):
                        st.markdown(
                            f"**{item_rc.get('source_type', '')}** — "
                            f"{item_rc.get('source_id', '')}"
                        )
                        st.caption(item_rc.get("relevance", ""))
                        if item_rc.get("excerpt"):
                            st.code(item_rc["excerpt"][:500])
                        if i < len(relevant) - 1:
                            st.divider()
                else:
                    st.info("No relevant content identified.")

            # --- Samples table ---
            st.markdown("#### Extracted Samples & Compositions")
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

            # --- Drawn structures ---
            st.markdown("#### User-Drawn Structures")
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
                st.info("No user-drawn structures saved yet. Use the 'Draw Chemical Structure' tab.")

        else:
            st.info(
                "No chemical composition data found for this DOI.\n\n"
                "Enable **Extract chemical composition** in the sidebar and re-run, "
                "or use the **Run Extraction** tab to extract on demand."
            )

    with sub_cc_draw:
        st.markdown("#### Chemical Structure Drawing Tool")
        st.caption(
            "Use the Ketcher editor below to draw a chemical structure. "
            "Click **Get SMILES** to capture the structure, then save it "
            "with a sample name."
        )

        render_ketcher_editor(height=550, key=f"ketcher_{_cc_safe}")

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

    with sub_cc_run_tab:
        st.markdown("#### Run Chemical Composition Extraction")
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

with tab_intervene:
    selected_safe = doi_to_safe(selected_doi)

    # --- Mode: flagged only vs all figures ---
    col_scan, col_mode = st.columns([1, 2])
    with col_scan:
        if st.button("Scan for interventions", type="primary"):
            st.session_state["flagged_all"] = _cached_flagged(str(rr.run_dir))
    with col_mode:
        browse_mode = st.radio(
            "Browse",
            ["Flagged figures only", "All figures"],
            horizontal=True,
            key=f"intervene_mode_{selected_safe}",
        )

    flagged_all = st.session_state.get("flagged_all", [])
    flagged = [x for x in flagged_all if x.safe_doi == selected_safe]

    item = None
    _intervene_ready = True  # gate for rendering the rest of the tab

    if browse_mode == "Flagged figures only":
        st.write(f"Flagged for this DOI: **{len(flagged)}**")
        if not flagged:
            st.info("No flagged figures for this DOI (or scan hasn't been run yet).")
            _intervene_ready = False
        else:
            labels = [f"{x.figure_id} | {x.reason}" for x in flagged]
            idx = st.selectbox(
                "Select flagged figure",
                range(len(labels)),
                format_func=lambda i: labels[i],
                key=f"flagged_sel_{selected_safe}",
            )
            item = flagged[idx]
            st.markdown(f"**Reason:** {item.reason}")
    else:
        # Browse all figures from the DOI
        from intervention import InterventionItem, _try_find_source_image, latest_plextract_run_dir as _lpr
        plextract_workdir = rr.run_dir / "plextract_work"
        latest = _lpr(plextract_workdir)
        filtered_fig_root = rr.run_dir / "figures_tables" / "figures"

        all_items = []
        if latest:
            output_root = latest / "output"
            doi_dir = output_root / selected_safe if output_root.exists() else None
            if doi_dir and doi_dir.exists():
                for fig_dir in sorted(doi_dir.iterdir()):
                    if not fig_dir.is_dir():
                        continue
                    figure_id = fig_dir.name
                    final_dir = fig_dir / "final"
                    source_image = _try_find_source_image(filtered_fig_root, selected_safe, figure_id)
                    # Check if this is flagged
                    reason = ""
                    for f in flagged:
                        if f.figure_id == figure_id:
                            reason = f.reason
                            break
                    all_items.append(InterventionItem(
                        safe_doi=selected_safe,
                        figure_id=figure_id,
                        figure_dir=fig_dir,
                        final_dir=final_dir,
                        source_image=source_image,
                        reason=reason,
                    ))

        if not all_items:
            st.info("No plextract outputs found for this DOI.")
            _intervene_ready = False
        else:
            labels = [f"{x.figure_id}{' | ' + x.reason if x.reason else ''}" for x in all_items]
            idx = st.selectbox(
                "Select figure",
                range(len(labels)),
                format_func=lambda i: labels[i],
                key=f"all_fig_sel_{selected_safe}",
            )
            item = all_items[idx]
            if item.reason:
                st.markdown(f"**Flagged:** {item.reason}")

    if _intervene_ready and item is not None and (item.source_image is None or not item.source_image.exists()):
        st.error("Could not find source image. Check figures_tables/figures/ directory.")
        _intervene_ready = False

    if not _intervene_ready or item is None:
        st.info("Select a figure above to view intervention options.")
    else:
      # Use expanders instead of tabs for better organization
      with st.expander("🔲 Cropping", expanded=False):
        st.subheader("Crop & extract data from sub-regions")

        pil_img = Image.open(item.source_image).convert("RGB")
        orig_w, orig_h = pil_img.size
        display_w = min(orig_w, 1100)
        display_h = int(orig_h * (display_w / orig_w))
        # Resize to display dimensions for canvas (avoids Streamlit media file errors)
        pil_img_disp = pil_img.resize((display_w, display_h))

        img_name = Path(item.figure_id).name
        crops_out_dir = rr.run_dir / "interventions" / item.safe_doi / img_name
        prefix = Path(item.figure_id).stem

        # --- Show existing crops (if any) ---
        existing_crops = sorted(crops_out_dir.glob(f"{prefix}_subfig_*.png")) if crops_out_dir.exists() else []
        if existing_crops:
            st.write(f"**Existing crops:** {len(existing_crops)}")
            crop_cols = st.columns(min(len(existing_crops), 4))
            for ci, cp in enumerate(existing_crops):
                with crop_cols[ci % len(crop_cols)]:
                    st_image_auto(str(cp), full_width=True)
                    st.caption(cp.name)

            col_clear, col_rerun = st.columns(2)
            with col_clear:
                if st.button("Clear all crops"):
                    import shutil
                    for cp in existing_crops:
                        cp.unlink(missing_ok=True)
                    st.success("Cleared existing crops.")
                    st.rerun()
            with col_rerun:
                if st.button("Re-run extraction on existing crops", type="primary"):
                    try:
                        manual_crops_dir = rr.run_dir / "manual_crops" / item.safe_doi
                        manual_crops_dir.mkdir(parents=True, exist_ok=True)
                        import shutil
                        for cp in existing_crops:
                            shutil.copy2(cp, manual_crops_dir / cp.name)
                        with st.spinner("Running plextract on crops..."):
                            run_plextract_on_manual_crops(scripts_root=scripts_root, run_dir=rr.run_dir)
                        st.success("Done. Check manual plextract outputs below.")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
        else:
            st.info("No crops saved yet. Draw rectangles below to create crops.")

        st.divider()

        # --- Drawing canvas ---
        st.write("**Draw crop regions** on the figure below.")
        show_bg = st.checkbox("Show image in crop canvas", value=True, key=f"crop_bg_{item.safe_doi}_{item.figure_id}")

        canvas_kwargs = dict(
            fill_color="rgba(0, 0, 0, 0.0)",
            stroke_width=2,
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key=f"canvas_{item.safe_doi}_{item.figure_id}",
        )

        try:
            if show_bg:
                canvas = st_canvas(background_image=pil_img_disp, **canvas_kwargs)
            else:
                canvas = st_canvas(background_color="rgba(255,255,255,0.0)", **canvas_kwargs)
        except Exception as e:
            st.warning(f"Canvas background failed: {e}")
            with st.expander("Show source image", expanded=False):
                st_image_auto(pil_img_disp, full_width=True)
            canvas = st_canvas(background_color="rgba(255,255,255,0.0)", **canvas_kwargs)

        scale_x = orig_w / display_w
        scale_y = orig_h / display_h

        rects = []
        if canvas.json_data and "objects" in canvas.json_data:
            for obj in canvas.json_data["objects"]:
                if obj.get("type") != "rect":
                    continue
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
                height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
                x1 = int(left * scale_x)
                y1 = int(top * scale_y)
                x2 = int((left + width) * scale_x)
                y2 = int((top + height) * scale_y)
                rects.append((x1, y1, x2, y2))

        st.write(f"Rectangles drawn: **{len(rects)}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save crops", disabled=len(rects) == 0):
                saved = save_crops(item.source_image, rects, crops_out_dir, prefix=prefix)
                st.success(f"Saved {len(saved)} crops to `{crops_out_dir}`")
                for p in saved:
                    st_image_auto(str(p), full_width=True)
                    st.caption(p.name)

        with col2:
            if st.button("Save & run extraction", type="primary", disabled=len(rects) == 0):
                saved = save_crops(item.source_image, rects, crops_out_dir, prefix=prefix)
                if saved:
                    manual_crops_dir = rr.run_dir / "manual_crops" / item.safe_doi
                    manual_crops_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    for cp in saved:
                        shutil.copy2(cp, manual_crops_dir / cp.name)
                    try:
                        with st.spinner("Running plextract on crops..."):
                            run_plextract_on_manual_crops(scripts_root=scripts_root, run_dir=rr.run_dir)
                        st.success("Done. Manual-crops plextract run completed.")
                    except Exception as e:
                        st.error(f"Manual plextract failed: {e}")

        # --- Show manual plextract outputs ---
        st.divider()
        st.subheader("Manual-crops extraction outputs")
        manual_workdir = rr.run_dir / "plextract_work_manual"
        if manual_workdir.exists():
            latest_manual = latest_plextract_run_dir(manual_workdir)
            if latest_manual:
                out_base = latest_manual / "output" / item.safe_doi
                if out_base.exists():
                    found_csvs = list(out_base.glob("**/final/final_dataset.csv"))
                    if found_csvs:
                        for csv_p in found_csvs:
                            fig_folder = csv_p.parent.parent.name
                            with st.expander(f"{fig_folder} - extracted data", expanded=True):
                                try:
                                    df = pd.read_csv(csv_p)
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                except Exception as e:
                                    st.warning(f"Failed to load: {e}")
                                final_img = csv_p.parent / "final_image.png"
                                if final_img.exists():
                                    st_image_auto(str(final_img), full_width=True)
                    else:
                        fig_folders = sorted([p.name for p in out_base.iterdir() if p.is_dir()])
                        st.caption("Figure folders found (no final datasets yet):")
                        st.code("\n".join(fig_folders[:50]) or "(none)")
                else:
                    st.info("No outputs found for this DOI in manual run.")
        else:
            st.info("No manual plextract run found yet. Save crops and run extraction above.")

      with st.expander("📊 Series Management", expanded=False):
        st.subheader("Series management")

        lf_dir = get_lineformer_dir(item.figure_dir)
        if not lf_dir.exists():
            st.warning("No lineformer directory found for this figure.")
        else:
            series_items = load_series_catalog(lf_dir)
            mapping = load_mapping(rr.run_dir, item.safe_doi, item.figure_id)
            names = mapping.get("series_names", {})
            deleted = set(int(x) for x in mapping.get("deleted", []))
            override_masks = list_override_masks(rr.run_dir, item.safe_doi, item.figure_id)

            # --- Helper: get display name for a series index ---
            _SERIES_COLORS = [
                (255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 165, 0),
                (148, 0, 211), (0, 206, 209), (255, 20, 147), (128, 128, 0),
                (0, 255, 127), (100, 149, 237),
            ]

            # Try to load actual series labels from plextract final_dataset.csv
            _plextract_series_labels: dict[int, str] = {}
            _final_csv_path = item.final_dir / "final_dataset.csv"
            if _final_csv_path.exists():
                try:
                    _fdf = pd.read_csv(_final_csv_path)
                    if "series_label" in _fdf.columns:
                        for _sl in _fdf["series_label"].dropna().unique():
                            _sl_str = str(_sl).strip()
                            # series_label is often "Series 0" or "0" or actual name
                            # Try to parse index from numeric labels
                            try:
                                _si = int(_sl_str)
                                _plextract_series_labels[_si] = _sl_str
                            except ValueError:
                                # Non-numeric label - try to match by position
                                _idx = len(_plextract_series_labels)
                                _plextract_series_labels[_idx] = _sl_str
                except Exception:
                    pass

            def _display_name(idx: int) -> str:
                # Priority: user mapping > plextract series label > fallback
                if str(idx) in names and names[str(idx)] != f"Series {idx}":
                    return names[str(idx)]
                if idx in _plextract_series_labels:
                    return _plextract_series_labels[idx]
                return f"Series {idx}"

            # --- Show colored series overlay in its own expander ---
            with st.expander("View colored series masks", expanded=False):
                base_img = Image.open(item.source_image).convert("RGBA")
                overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))

                legend_lines = []
                for s in sorted(series_items, key=lambda x: x.series_index):
                    i = s.series_index
                    if i in deleted:
                        continue
                    color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
                    mask_path = override_masks.get(i, s.mask_path)
                    try:
                        mask = Image.open(mask_path).convert("L").resize(base_img.size, Image.NEAREST)
                        mask_arr = np.array(mask)
                        color_layer = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
                        color_layer[mask_arr > 0] = (*color, 140)  # semi-transparent
                        overlay = Image.alpha_composite(overlay, Image.fromarray(color_layer))
                    except Exception:
                        pass
                    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                    legend_lines.append(f'<span style="color:{hex_color}; font-weight:bold;">&#9632;</span> {_display_name(i)}')

                composite = Image.alpha_composite(base_img, overlay).convert("RGB")
                st_image_auto(composite, full_width=True)
                if legend_lines:
                    st.markdown(" &nbsp; ".join(legend_lines), unsafe_allow_html=True)

            # ---- Rename / delete section ----
            st.write("**Rename or delete** each series below.")

            new_names = {}
            new_deleted = set(deleted)

            for s in sorted(series_items, key=lambda x: x.series_index):
                i = s.series_index
                color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                cols = st.columns([0.5, 3, 1])
                with cols[0]:
                    st.markdown(
                        f'<span style="color:{hex_color}; font-size:1.4em; font-weight:bold;">&#9632;</span>',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    default_name = _display_name(i)
                    new_names[str(i)] = st.text_input(
                        f"Name for series {i}",
                        value=default_name,
                        key=f"series_name_{item.safe_doi}_{item.figure_id}_{i}",
                        label_visibility="collapsed",
                    )
                with cols[2]:
                    del_flag = st.checkbox(
                        "Delete",
                        value=(i in new_deleted),
                        key=f"series_del_{item.safe_doi}_{item.figure_id}_{i}",
                    )
                    if del_flag:
                        new_deleted.add(i)
                    else:
                        new_deleted.discard(i)

            if st.button("Save series mapping", type="primary"):
                mapping2 = {
                    "series_names": new_names,
                    "deleted": sorted(list(new_deleted)),
                }
                p = save_mapping(rr.run_dir, item.safe_doi, item.figure_id, mapping2)
                st.success(f"Saved mapping: {p}")
                st.rerun()

            st.divider()

            # ---- Redraw mask section ----
            st.write("**Redraw series mask**")

            active_series = [s for s in series_items if s.series_index not in deleted]
            if not active_series:
                st.info("No active series to edit.")
            else:
                sel_series_name_options = [f"{_display_name(s.series_index)}" for s in active_series]
                sel_series_ui = st.selectbox(
                    "Select series to edit",
                    range(len(active_series)),
                    format_func=lambda i: sel_series_name_options[i],
                    key=f"mask_edit_sel_{item.safe_doi}_{item.figure_id}",
                )
                sel_s = active_series[sel_series_ui]
                sel_idx = sel_s.series_index

                base_img_rgb = Image.open(item.source_image).convert("RGB")
                w, h = base_img_rgb.size
                disp_w = min(w, 1100)
                disp_h = int(h * (disp_w / w))
                base_img_disp = base_img_rgb.resize((disp_w, disp_h))

                # Load current mask for preview
                orig_mask_path = next(s.mask_path for s in series_items if s.series_index == sel_idx)
                cur_mask_path = override_masks.get(sel_idx, orig_mask_path)
                cur_mask_img = Image.open(cur_mask_path).convert("L").resize((disp_w, disp_h))

                c1, c2 = st.columns([2, 1])

                with c1:
                    st.write("Draw on the image below to redraw the series mask")
                    brush = st.slider("Brush width", 1, 40, 10, key=f"brush_{item.safe_doi}_{item.figure_id}_{sel_idx}")
                    mask_canvas = st_canvas(
                        background_image=base_img_disp,
                        drawing_mode="freedraw",
                        stroke_width=brush,
                        stroke_color="rgba(255,255,255,1.0)",
                        fill_color="rgba(255,255,255,1.0)",
                        update_streamlit=True,
                        height=disp_h,
                        width=disp_w,
                        key=f"mask_canvas_{item.safe_doi}_{item.figure_id}_{sel_idx}",
                    )

                with c2:
                    st.write("Current mask")
                    st_image_auto(cur_mask_img, full_width=True)

                # Build new mask from canvas strokes
                if mask_canvas.image_data is not None:
                    rgba = mask_canvas.image_data.astype(np.uint8)
                    alpha = rgba[..., 3]
                    new_mask_small = (alpha > 0).astype(np.uint8) * 255

                    new_mask_full = cv2.resize(new_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
                    new_mask_pil = Image.fromarray(new_mask_full).convert("L")

                    colA, colB = st.columns(2)
                    with colA:
                        if st.button("Save edited mask", type="primary", key=f"save_mask_{item.safe_doi}_{item.figure_id}_{sel_idx}"):
                            outp = save_mask_override(rr.run_dir, item.safe_doi, item.figure_id, sel_idx, new_mask_pil)
                            st.success(f"Saved override mask for {_display_name(sel_idx)}")
                            override_masks2 = list_override_masks(rr.run_dir, item.safe_doi, item.figure_id)
                            write_series_catalog_override(rr.run_dir, item.safe_doi, item.figure_id, series_items, override_masks2)
                            st.rerun()
                    with colB:
                        if st.button("Reset canvas", key=f"reset_mask_{item.safe_doi}_{item.figure_id}_{sel_idx}"):
                            st.rerun()

                    with st.expander("Preview new mask (full-res)", expanded=False):
                        st_image_auto(new_mask_pil, full_width=True)

            st.divider()

            # ---- Create new series section ----
            st.write("**Create new series/mask**")

            # Find the next available series index
            existing_indices = [s.series_index for s in series_items]
            next_idx = max(existing_indices) + 1 if existing_indices else 0

            new_series_name = st.text_input(
                "Name for new series",
                value=f"Series {next_idx}",
                key=f"new_series_name_{item.safe_doi}_{item.figure_id}",
            )

            base_img_rgb = Image.open(item.source_image).convert("RGB")
            w, h = base_img_rgb.size
            disp_w = min(w, 1100)
            disp_h = int(h * (disp_w / w))
            base_img_disp = base_img_rgb.resize((disp_w, disp_h))

            st.write("Draw the new series mask below")
            new_brush = st.slider("Brush width", 1, 40, 10, key=f"new_brush_{item.safe_doi}_{item.figure_id}")
            new_series_canvas = st_canvas(
                background_image=base_img_disp,
                drawing_mode="freedraw",
                stroke_width=new_brush,
                stroke_color="rgba(255,255,255,1.0)",
                fill_color="rgba(255,255,255,1.0)",
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                key=f"new_series_canvas_{item.safe_doi}_{item.figure_id}",
            )

            if new_series_canvas.image_data is not None:
                rgba = new_series_canvas.image_data.astype(np.uint8)
                alpha = rgba[..., 3]
                new_mask_small = (alpha > 0).astype(np.uint8) * 255

                if np.any(new_mask_small > 0):  # Only show save button if something is drawn
                    new_mask_full = cv2.resize(new_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
                    new_mask_pil = Image.fromarray(new_mask_full).convert("L")

                    if st.button("Create new series", type="primary", key=f"create_series_{item.safe_doi}_{item.figure_id}"):
                        # Save the new mask
                        outp = save_mask_override(rr.run_dir, item.safe_doi, item.figure_id, next_idx, new_mask_pil)

                        # Update mapping with the new series name
                        mapping_updated = load_mapping(rr.run_dir, item.safe_doi, item.figure_id)
                        if "series_names" not in mapping_updated:
                            mapping_updated["series_names"] = {}
                        mapping_updated["series_names"][str(next_idx)] = new_series_name
                        save_mapping(rr.run_dir, item.safe_doi, item.figure_id, mapping_updated)

                        # Calculate centroid for the new series
                        mask_arr = np.array(new_mask_pil)
                        y_coords, x_coords = np.where(mask_arr > 0)
                        if len(x_coords) > 0:
                            centroid_x = float(np.mean(x_coords))
                            centroid_y = float(np.mean(y_coords))
                        else:
                            centroid_x, centroid_y = 0.0, 0.0

                        # Add new series to the catalog
                        from intervention import SeriesItem
                        new_series = SeriesItem(
                            series_index=next_idx,
                            mask_path=outp,
                            centroid=(centroid_x, centroid_y)
                        )
                        series_items.append(new_series)

                        # Update the series catalog override
                        override_masks_updated = list_override_masks(rr.run_dir, item.safe_doi, item.figure_id)
                        write_series_catalog_override(rr.run_dir, item.safe_doi, item.figure_id, series_items, override_masks_updated)

                        st.success(f"Created new series '{new_series_name}' with index {next_idx}")
                        st.rerun()

      with st.expander("🔄 Rerun Figure Data Extraction", expanded=False):
        st.write("This will apply your mask overrides and re-extract data from the figure.")
        st.info("New masks will replace the old respective masks where you have made changes. The extraction will use the updated masks along with the original masks for unchanged series.")

        if st.button("Rerun figure data extraction", type="primary"):
            try:
                # First, apply all overrides to the plextract output
                with st.spinner("Applying mask overrides and series changes..."):
                    apply_overrides_to_plextract(item.figure_dir, rr.run_dir, item.safe_doi, item.figure_id)
                st.success("✓ Applied overrides to plextract output")

                # Then re-run the extraction
                with st.spinner("Re-running data extraction on this figure..."):
                    run_plextract_single_figure(
                        scripts_root=scripts_root,
                        run_dir=rr.run_dir,
                        safe_doi=item.safe_doi,
                        figure_path=item.source_image,
                    )
                st.success("✓ Data extraction complete! Check plextract_work_interventions/ for results.")
            except Exception as e:
                st.error(f"Failed: {e}")



