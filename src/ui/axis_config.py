# AEM_OCR/src/ui/axis_config.py
"""
Manages axis label names, unit aliases, unit dimensions, canonical targets,
and unit conversions.  Persisted as a JSON file per workspace.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default configuration matching what finalize_labels.py ships with
_DEFAULT_CONFIG: Dict[str, Any] = {
    "unit_aliases": {
        "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
        "min": "min", "mins": "min", "minute": "min", "minutes": "min",
        "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
        "days": "day", "day": "day",
        "s/cm": "S/cm", "S/cm": "S/cm",
        "ms/cm": "mS/cm", "mS/cm": "mS/cm",
    },
    "unit_dimensions": {
        "s": "time", "min": "time", "h": "time",
        "S/cm": "conductivity", "mS/cm": "conductivity",
        "S/m": "conductivity", "mS/m": "conductivity",
    },
    "canonical_targets": {
        "time": "h",
        "conductivity": "mS/cm",
    },
    "axis_name_aliases": {
        "x": {
            "time": ["Time", "time", "t"],
            "cycle number": ["Cycle Number", "Cycle number", "cycle number", "Cycles", "cycles", "Cycle", "cycle"],
        },
        "y": {
            "conductivity": ["Conductivity", "conductivity", "Hydroxide Conductivity", "OH- Conductivity", "Ionic Conductivity"],
            "current density": ["Current Density", "current density", "Current density", "J"],
        },
    },
}


def _config_path(workspace_dir: Path) -> Path:
    return workspace_dir / "axis_config.json"


def load_axis_config(workspace_dir: Path) -> Dict[str, Any]:
    p = _config_path(workspace_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy


def save_axis_config(workspace_dir: Path, config: Dict[str, Any]) -> Path:
    p = _config_path(workspace_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2))
    return p


def reset_axis_config(workspace_dir: Path) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(_DEFAULT_CONFIG))
    save_axis_config(workspace_dir, cfg)
    return cfg
