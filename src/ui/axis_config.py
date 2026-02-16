# AEM_OCR/src/ui/axis_config.py
"""
Manages axis label names, unit aliases, unit dimensions, canonical targets,
and unit conversions.  Persisted as a JSON file per workspace.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default configuration: blank for raw data extraction
# Users can add their own axis/unit configurations as needed
_DEFAULT_CONFIG: Dict[str, Any] = {
    "unit_aliases": {},
    "unit_dimensions": {},
    "canonical_targets": {},
    "axis_name_aliases": {
        "x": {},
        "y": {},
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


def _presets_dir(workspace_dir: Path) -> Path:
    """Directory to store axis config presets."""
    return workspace_dir / "axis_presets"


def list_presets(workspace_dir: Path) -> List[str]:
    """List all available preset names."""
    presets_dir = _presets_dir(workspace_dir)
    if not presets_dir.exists():
        return []
    return sorted([p.stem for p in presets_dir.glob("*.json")])


def save_preset(workspace_dir: Path, preset_name: str, config: Dict[str, Any]) -> Path:
    """Save current config as a named preset."""
    presets_dir = _presets_dir(workspace_dir)
    presets_dir.mkdir(parents=True, exist_ok=True)
    preset_path = presets_dir / f"{preset_name}.json"
    preset_path.write_text(json.dumps(config, indent=2))
    return preset_path


def load_preset(workspace_dir: Path, preset_name: str) -> Optional[Dict[str, Any]]:
    """Load a named preset."""
    preset_path = _presets_dir(workspace_dir) / f"{preset_name}.json"
    if not preset_path.exists():
        return None
    try:
        return json.loads(preset_path.read_text())
    except Exception:
        return None


def delete_preset(workspace_dir: Path, preset_name: str) -> bool:
    """Delete a named preset. Returns True if successful."""
    preset_path = _presets_dir(workspace_dir) / f"{preset_name}.json"
    if preset_path.exists():
        preset_path.unlink()
        return True
    return False
