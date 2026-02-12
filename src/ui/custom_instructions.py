# custom_instructions.py
"""
Manages user-customizable LLM instructions for the AEM_OCR pipeline.

Users can override the default prompts/rules for:
  1. Table extraction (merged cell handling, etc.)
  2. Figure filtering (what kinds of figures to keep)
  3. Table filtering (what kinds of tables to keep)
  4. Data processing (extra rules for row emission and chemical extraction)
  5. Output columns and missing-value defaults
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default values — these match the current hardcoded behaviour
# ---------------------------------------------------------------------------

DEFAULT_TABLE_EXTRACTION_RULES = """\
- If a parent header spans multiple columns, and there is a second header row \
that contains labels such as "Before" / "After", "30°C" / "80°C", \
"Cal."/"Exp", "in-plane"/"through-plane", etc., you MUST factor them into \
final column headers."""

DEFAULT_FIGURE_FILTER_CRITERIA = """\
A figure qualifies ONLY if ALL of the following are true:
- It shows conductivity / ionic conductivity / hydroxide conductivity \
OR alkaline stability / retention.
- The x-axis (or time dimension) is clearly time \
(e.g. s, min, h, hr, day, d, time).
- It is not just before/after comparisons, spectra (e.g. FTIR, NMR, XRD), \
SEM/TEM images, photographs, IEC-dependence, RH-dependence, \
temperature-dependence, or other non-time axes."""

DEFAULT_TABLE_FILTER_CRITERIA = """\
Return "YES" if:
(a) the quantity is conductivity / ionic conductivity / hydroxide conductivity \
or alkaline/chemical stability/retention, AND
(b) the table contains a clear time dimension \
(e.g. time, h, hr, hours, days, d, etc.).
Return "NO" otherwise."""

DEFAULT_DATA_PROCESSING_RULES = """\
- Measurements taken before or at the beginning of stability tests \
MUST have time(h) = 0"""

DEFAULT_OUTPUT_COLUMNS = [
    "Sample Name",
    "time(h)",
    "relative humidity(%)",
    "solvent type",
    "solvent molarity (M)",
    "degradation test temp (C)",
    "prop test temp (C)",
    "property",
    "value",
    "DOI",
    "source_kind",
    "source_name",
]

DEFAULT_MISSING_VALUE_DEFAULTS = {
    "prop test temp (C)": "25 Not Found",
    "degradation test temp (C)": "None Not Found",
    "time(h)": "0 Not Found",
    "relative humidity(%)": "100 Not Found",
    "solvent type": "None Not Found",
    "solvent molarity (M)": "None Not Found",
}


def default_instructions() -> Dict[str, Any]:
    """Return the full default custom-instructions dict."""
    return {
        "table_extraction_rules": DEFAULT_TABLE_EXTRACTION_RULES,
        "figure_filter_criteria": DEFAULT_FIGURE_FILTER_CRITERIA,
        "table_filter_criteria": DEFAULT_TABLE_FILTER_CRITERIA,
        "data_processing_rules": DEFAULT_DATA_PROCESSING_RULES,
        "output_columns": list(DEFAULT_OUTPUT_COLUMNS),
        "missing_value_defaults": dict(DEFAULT_MISSING_VALUE_DEFAULTS),
    }


def load_instructions(path: Path) -> Dict[str, Any]:
    """Load custom instructions from a JSON file, falling back to defaults."""
    defaults = default_instructions()
    if not path.exists():
        return defaults
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Merge with defaults so missing keys get filled in
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return defaults


def save_instructions(path: Path, instructions: Dict[str, Any]) -> None:
    """Persist custom instructions to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(instructions, f, indent=2)


def load_instructions_from_env_or_path(
    path_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience loader used by CLI scripts.
    Reads from the path given by --custom_instructions or the
    AEM_OCR_CUSTOM_INSTRUCTIONS environment variable.
    Returns defaults when neither is set.
    """
    import os

    path_str = path_str or os.environ.get("AEM_OCR_CUSTOM_INSTRUCTIONS", "")
    if path_str and path_str.strip():
        return load_instructions(Path(path_str.strip()))
    return default_instructions()


# ---------------------------------------------------------------------------
# Presets — named instruction sets stored under a shared directory
# ---------------------------------------------------------------------------

PRESETS_DIR_NAME = "instruction_presets"


def _presets_dir(workspaces_root: Path) -> Path:
    """Return (and create) the shared presets directory."""
    d = workspaces_root / PRESETS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_presets(workspaces_root: Path) -> List[str]:
    """Return sorted list of preset names (without .json extension)."""
    d = _presets_dir(workspaces_root)
    return sorted(
        p.stem for p in d.glob("*.json") if p.is_file()
    )


def save_preset(workspaces_root: Path, name: str, instructions: Dict[str, Any]) -> Path:
    """Save a named preset. Returns the file path."""
    path = _presets_dir(workspaces_root) / f"{name}.json"
    save_instructions(path, instructions)
    return path


def load_preset(workspaces_root: Path, name: str) -> Dict[str, Any]:
    """Load a named preset."""
    path = _presets_dir(workspaces_root) / f"{name}.json"
    return load_instructions(path)


def delete_preset(workspaces_root: Path, name: str) -> None:
    """Delete a named preset."""
    path = _presets_dir(workspaces_root) / f"{name}.json"
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Discovery — find custom_instructions.json files in runs and workspaces
# ---------------------------------------------------------------------------

def find_run_instructions(run_dir: Path) -> Optional[Path]:
    """Return the path to custom_instructions.json inside a run dir, or None."""
    p = run_dir / "custom_instructions.json"
    return p if p.exists() else None


def list_workspace_runs_with_instructions(
    workspace_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Scan a workspace for runs that have a custom_instructions.json.
    Returns list of {"run_id": str, "path": Path}.
    """
    results: List[Dict[str, Any]] = []
    if not workspace_dir.is_dir():
        return results
    for child in sorted(workspace_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        ci = child / "custom_instructions.json"
        if ci.exists():
            results.append({"run_id": child.name, "path": ci})
    return results
