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
import numpy as np
import cv2
from cropping import save_crops
from run_plextract_manual import run_plextract_on_manual_crops
import os
import subprocess
from typing import List, Optional


def _default_ollama_models_dir() -> Path:
    """
    Determine a sane default for Ollama models directory.
    Priority:
      1) $OLLAMA_MODELS (your log shows you use /data/william/ollama_models)
      2) ~/.ollama/models (common on user installs)
      3) /usr/share/ollama/.ollama/models (common system install)
    """
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env).expanduser()

    candidates = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback to user path even if it doesn't exist yet
    return Path.home() / ".ollama" / "models"

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

st.set_page_config(page_title="AEM_OCR UI", layout="wide")
st.title("AEM_OCR")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Paths")
    scripts_root = Path(st.text_input("AEM_OCR/src", "/home/wschertzer3/AEM_OCR/src")).resolve()
    run_root = Path(st.text_input("UI run workspace", str(scripts_root / "ui_runs"))).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    st.divider()
    st.header("Stages")
    run_download_doi = st.checkbox("Download DOIs", value=True)
    run_extract_figures = st.checkbox("Extract figures", value=True)
    run_extract_tables = st.checkbox("Extract tables", value=True)
    run_filter_figures = st.checkbox("Filter figures", value=True)
    run_filter_tables = st.checkbox("Filter tables", value=True)
    run_plextract = st.checkbox("Extract data from figures", value=True)
    run_complete_dataset = st.checkbox("Complete dataset", value=True)
    exclude_full_text = st.checkbox("Exclude full text from context", value=True)
    
    st.divider()
    st.header("LLM")

    provider = st.radio(
        "Provider",
        options=["Ollama", "Chat GPT"],
        index=0,
        key="llm_provider",
    )

    # Defaults
    default_ollama_dir = _default_ollama_models_dir()
    default_ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    if provider == "Ollama":
        ollama_models_dir = Path(
            st.text_input("Ollama models dir", str(default_ollama_dir), key="ollama_models_dir")
        ).expanduser()

        ollama_host = st.text_input("OLLAMA_HOST", default_ollama_host, key="ollama_host")

        # Try CLI first, then fall back to filesystem scan
        models = _ollama_list_models_via_cli()
        if not models:
            models = _ollama_list_models_from_dir(ollama_models_dir)

        if models:
            selected_model = st.selectbox("Model", options=models, index=0, key="llm_model")
        else:
            selected_model = st.text_input(
                "Model (no models detected — type one, e.g. llama3:8b)",
                value="",
                key="llm_model",
            )
            st.caption("Tip: run `ollama pull <model>` then refresh.")

    else:
        # Keep this flexible: OpenAI model names change; a free-text input avoids hardcoding.
        openai_base_url = st.text_input(
            "OpenAI-compatible base URL",
            value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            key="openai_base_url",
        )

        # Optional: allow env-based key; do NOT print it.
        openai_api_key = st.text_input(
            "API key (stored only in session)",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            key="openai_api_key",
        )

        suggested = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o4-mini"]
        selected_model = st.selectbox("Model", options=suggested, index=0, key="llm_model")

    st.divider()
    st.header("Load an existing run")
    existing = sorted([p.name for p in run_root.iterdir() if p.is_dir()], reverse=True)
    pick = st.selectbox("Run ID", options=[""] + existing)
    if st.button("Load run") and pick:
        rr = discover_run(run_root / pick)
        st.session_state["rr"] = rr

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

            # --- Apply LLM configuration for this run ---
            provider = st.session_state.get("llm_provider", "Ollama")
            model = st.session_state.get("llm_model", "")

            print("======MODEL======", model)

            if provider == "Ollama":
                os.environ["AEM_OCR_LLM_PROVIDER"] = "ollama"
                os.environ["AEM_OCR_LLM_MODEL"] = model
                os.environ["OLLAMA_HOST"] = st.session_state.get("ollama_host", "http://127.0.0.1:11434")
                os.environ["OLLAMA_MODELS"] = st.session_state.get("ollama_models_dir", str(_default_ollama_models_dir()))
            else:
                os.environ["AEM_OCR_LLM_PROVIDER"] = "openai"
                os.environ["AEM_OCR_LLM_MODEL"] = model
                os.environ["OPENAI_BASE_URL"] = st.session_state.get("openai_base_url", "https://api.openai.com/v1")
                # Put key in env for downstream libraries if you use them
                if st.session_state.get("openai_api_key"):
                    os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]


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
                exclude_full_text=exclude_full_text,

            )

        st.session_state["rr"] = rr
        st.success(f"Done. Run ID: {rr.run_id}")

rr: RunResult | None = st.session_state.get("rr")
if not rr:
    st.stop()

st.info(f"Run ID: **{rr.run_id}**")
st.info(f"Workspace: **{rr.run_dir}**")

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
tab_text, tab_figs, tab_tables, tab_xy, tab_intervene = st.tabs(
    ["Parsed text", "Figures", "Tables", "Dataset", "Interventions"]
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

with tab_intervene:
    # Explicit scan (cached)
    if st.button("Scan for interventions", type="primary"):
        st.session_state["flagged_all"] = _cached_flagged(str(rr.run_dir))

    flagged_all = st.session_state.get("flagged_all", [])

    # Restrict to selected DOI
    selected_safe = doi_to_safe(selected_doi)
    flagged = [x for x in flagged_all if x.safe_doi == selected_safe]

    st.write(f"Flagged for this DOI: **{len(flagged)}**")

    item = None

    if not flagged:
        st.info("No flagged figures for this DOI (or scan hasn’t been run yet).")
        st.stop()  # stop ONLY this tab’s rendering
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
        # st.markdown(f"**Plextract dir:** `{item.figure_dir}`")
        # st.code(f"source_image: {item.source_image}")

    # Everything below here is safe: item is guaranteed not None
    if item.source_image is None or not item.source_image.exists():
        st.error("Could not find source image to crop. Fix mapping in _try_find_source_image().")
        st.stop()

    sub_crop, sub_masks, sub_series, sub_apply = st.tabs(["Crop", "Masks", "Series", "Apply / Re-run"])

    with sub_crop:

        # Manual-run outputs (optional)
        st.divider()
        st.subheader("Manual-crops plextract outputs (if run)")
        manual_workdir = rr.run_dir / "plextract_work_manual"
        if manual_workdir.exists():
            latest_manual = latest_plextract_run_dir(manual_workdir)
            if latest_manual:
                out_base = latest_manual / "output" / item.safe_doi
                if out_base.exists():
                    fig_folders = sorted([p.name for p in out_base.iterdir() if p.is_dir()])
                    st.caption("Available figure folders in manual run:")
                    st.code("\n".join(fig_folders[:50]) or "(none)")
        else:
            st.info("No manual plextract run found yet.")

        st.divider()
        st.subheader("Manual crop tool")

        show_bg = st.checkbox("Show image in crop canvas", value=True)

        pil_img = Image.open(item.source_image).convert("RGB")
        orig_w, orig_h = pil_img.size
        display_w = min(orig_w, 1100)
        display_h = int(orig_h * (display_w / orig_w))

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
                canvas = st_canvas(background_image=pil_img, **canvas_kwargs)
            else:
                canvas = st_canvas(background_color="rgba(255,255,255,0.0)", **canvas_kwargs)
        except Exception as e:
            st.warning(f"Canvas background failed in this Streamlit/canvas version. Error: {e}")
            st.info("Fallback: show image in an expander and draw on a blank canvas.")
            with st.expander("Show source image", expanded=False):
                st_image_auto(pil_img, full_width=True)
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

        st.write(f"Rectangles: **{len(rects)}**")

        img_name = Path(item.figure_id).name
        img_dir = rr.run_dir / "interventions" / item.safe_doi / img_name
        crops_out_dir = img_dir
        prefix = Path(item.figure_id).stem

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save crops"):
                saved = save_crops(item.source_image, rects, crops_out_dir, prefix=prefix)
                st.success(f"Saved {len(saved)} crops to {crops_out_dir}")
                for p in saved:
                    st_image_auto(str(p), full_width=True)
                    st.caption(p.name)

        with col2:
            if st.button("Run plextract on manual crops", type="primary"):
                try:
                    with st.spinner("Running plextract on manual crops…"):
                        run_plextract_on_manual_crops(scripts_root=scripts_root, run_dir=rr.run_dir)
                    st.success("Done. Manual-crops plextract run completed.")
                except Exception as e:
                    st.error(f"Manual plextract failed: {e}")

            st.caption("Manual crops input:")
            st.code(str(rr.run_dir / "manual_crops"))
            st.caption("Manual plextract workdir:")
            st.code(str(rr.run_dir / "plextract_work_manual"))
    with sub_masks:
        st.subheader("Redraw / edit masks")

        lf_dir = get_lineformer_dir(item.figure_dir)
        if not lf_dir.exists():
            st.error(f"No lineformer dir found: {lf_dir}")
            st.stop()

        series_items = load_series_catalog(lf_dir)
        series_indices = [s.series_index for s in series_items]

        sel_idx = st.selectbox("Select series_index to edit", series_indices)

        # base figure image to draw on (use source_image)
        base_img = Image.open(item.source_image).convert("RGB")
        w, h = base_img.size
        display_w = min(w, 1100)
        display_h = int(h * (display_w / w))
        base_img_disp = base_img.resize((display_w, display_h))

        # load existing mask (override if exists, else original)
        override_masks = list_override_masks(rr.run_dir, item.safe_doi, item.figure_id)
        orig_mask_path = next(s.mask_path for s in series_items if s.series_index == sel_idx)
        mask_path = override_masks.get(sel_idx, orig_mask_path)
        mask_img = Image.open(mask_path).convert("L").resize((display_w, display_h))

        st.caption(f"Editing series_index={sel_idx}")
        c1, c2 = st.columns([2, 1])

        with c1:
            st.write("Draw white where the mask should be. Use ‘freedraw’ for painting.")
            brush = st.slider("Brush width", 1, 40, 10)

            # canvas returns an RGBA image of strokes; we threshold alpha to a binary mask
            canvas = st_canvas(
                background_image=base_img_disp,
                drawing_mode="freedraw",
                stroke_width=brush,
                stroke_color="rgba(255,255,255,1.0)",
                fill_color="rgba(255,255,255,1.0)",
                update_streamlit=True,
                height=display_h,
                width=display_w,
                key=f"mask_canvas_{item.safe_doi}_{item.figure_id}_{sel_idx}",
            )

        with c2:
            st.write("Current mask preview")
            st_image_auto(mask_img, full_width=True)

        # Build new mask from strokes
        if canvas.image_data is not None:
            rgba = canvas.image_data.astype(np.uint8)
            alpha = rgba[..., 3]
            # binary from alpha (strokes)
            new_mask_small = (alpha > 0).astype(np.uint8) * 255

            # resize back to original size
            new_mask = cv2.resize(new_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            new_mask_pil = Image.fromarray(new_mask).convert("L")

            colA, colB = st.columns(2)
            with colA:
                if st.button("Save edited mask", type="primary"):
                    outp = save_mask_override(rr.run_dir, item.safe_doi, item.figure_id, sel_idx, new_mask_pil)
                    st.success(f"Saved override mask: {outp}")

                    # update series_catalog.override.json too (optional but useful)
                    override_masks2 = list_override_masks(rr.run_dir, item.safe_doi, item.figure_id)
                    write_series_catalog_override(rr.run_dir, item.safe_doi, item.figure_id, series_items, override_masks2)
            with colB:
                if st.button("Reset canvas (rerun)"):
                    st.rerun()

            with st.expander("Preview new mask (full-res)", expanded=False):
                st_image_auto(new_mask_pil, full_width=True)
    with sub_series:
        st.subheader("Remap / rename / delete series")

        lf_dir = get_lineformer_dir(item.figure_dir)
        series_items = load_series_catalog(lf_dir)
        mapping = load_mapping(rr.run_dir, item.safe_doi, item.figure_id)

        # Build default names
        names = mapping.get("series_names", {})
        deleted = set(int(x) for x in mapping.get("deleted", []))

        st.write("Set a human-readable name for each series_index. Mark any for deletion.")
        new_names = {}
        new_deleted = set(deleted)

        for s in sorted(series_items, key=lambda x: x.series_index):
            i = s.series_index
            cols = st.columns([1, 3, 1])
            with cols[0]:
                st.markdown(f"**{i}**")
            with cols[1]:
                default_name = names.get(str(i), f"Series {i}")
                new_names[str(i)] = st.text_input(
                    f"Name for series_index={i}",
                    value=default_name,
                    key=f"series_name_{item.safe_doi}_{item.figure_id}_{i}",
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
                "series_names": new_names,     # series_index -> name
                "deleted": sorted(list(new_deleted)),
            }
            p = save_mapping(rr.run_dir, item.safe_doi, item.figure_id, mapping2)
            st.success(f"Saved mapping: {p}")
    with sub_apply:
        st.subheader("Apply edits to plextract outputs")

        st.warning(
            "This will copy your override masks into the current plextract output folder for this figure "
            "and rewrite lineformer/series_catalog.json (respecting deletions)."
        )

        if st.button("Apply overrides into plextract output folder", type="primary"):
            try:
                apply_overrides_to_plextract(item.figure_dir, rr.run_dir, item.safe_doi, item.figure_id)
                st.success("Applied overrides to plextract output folder.")
            except Exception as e:
                st.error(f"Failed to apply overrides: {e}")

        st.divider()
        st.subheader("Optional: patch final dataset labels (no re-digitization)")

        mapping = load_mapping(rr.run_dir, item.safe_doi, item.figure_id)
        series_names = mapping.get("series_names", {})
        deleted = set(int(x) for x in mapping.get("deleted", []))

        final_csv = item.final_dir / "final_dataset.csv"
        if final_csv.exists():
            st.caption(f"Found: {final_csv}")
            if st.button("Write patched final_dataset_patched.csv"):
                df = pd.read_csv(final_csv)
                # common label columns; adjust if yours differs
                candidate_cols = [c for c in df.columns if c.lower() in {"series", "series_label", "label", "name"}]
                if not candidate_cols:
                    st.error(f"No obvious series label column found in {list(df.columns)}")
                else:
                    col = candidate_cols[0]
                    # if series is numeric index, rename by series_names
                    def _rename(x):
                        try:
                            i = int(x)
                            if i in deleted:
                                return None
                            return series_names.get(str(i), f"Series {i}")
                        except Exception:
                            # if it's already a string label, keep unless you want string-to-string renames
                            return x

                    df[col] = df[col].map(_rename)
                    df = df.dropna(subset=[col])
                    outp = item.final_dir / "final_dataset_patched.csv"
                    df.to_csv(outp, index=False)
                    st.success(f"Wrote: {outp}")
                    st.dataframe(df.head(2000), use_container_width=True)
        else:
            st.info("No final_dataset.csv found in final/ (may depend on your pipeline stage).")
        
        if st.button("Re-run plextract for this figure only"):
            try:
                with st.spinner("Running plextract on this figure only..."):
                    run_plextract_single_figure(
                        scripts_root=scripts_root,
                        run_dir=rr.run_dir,
                        safe_doi=item.safe_doi,
                        figure_path=item.source_image,  # or a crop if you want
                    )
                st.success("Done. Check plextract_work_interventions/results/<n>/output/...")
            except Exception as e:
                st.error(f"Failed: {e}")



