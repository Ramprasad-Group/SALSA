"""
Chemical drawing tool integration for Streamlit.

Embeds the JSME Molecule Editor (open-source, by Peter Ertl / Bruno Bienfait)
as a self-contained JavaScript applet inside a Streamlit HTML component.
Users draw chemical structures and retrieve SMILES / Molfile output directly.

JSME is loaded from a public CDN (jsme-editor on jsDelivr) so no local
installation is required and there are no cross-origin iframe restrictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# JSME-based HTML component
# ---------------------------------------------------------------------------

_JSME_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdn.jsdelivr.net/npm/jsme-editor@2024.8.4/jsme/jsme.nocache.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; font-family: sans-serif; }}
  #editor-wrapper {{ width: 100%; }}
  #jsme_container {{ width: 100%; height: {editor_height}px; border: 1px solid #ccc; }}
  .toolbar {{
    display: flex; gap: 8px; padding: 8px 4px; background: #f5f5f5;
    border-bottom: 1px solid #ddd; align-items: center; flex-wrap: wrap;
  }}
  .toolbar button {{
    padding: 6px 16px; border: 1px solid #999; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 14px;
  }}
  .toolbar button:hover {{ background: #e8e8e8; }}
  .toolbar button.primary {{
    background: #1976d2; color: #fff; border-color: #1565c0;
  }}
  .toolbar button.primary:hover {{ background: #1565c0; }}
  .toolbar .info {{ font-size: 13px; color: #555; margin-left: auto; }}
  #output-area {{
    font-family: monospace; font-size: 13px; padding: 8px;
    background: #fafafa; border-top: 1px solid #ddd;
    min-height: 28px; word-break: break-all; white-space: pre-wrap;
  }}
</style>
</head>
<body>

<div id="editor-wrapper">
  <div class="toolbar">
    <button class="primary" onclick="getSMILES()">Get SMILES</button>
    <button onclick="getMolfile()">Get Molfile</button>
    <button onclick="clearEditor()">Clear</button>
    <span class="info">Draw your chemical structure below</span>
  </div>
  <div id="jsme_container"></div>
  <div id="output-area">SMILES will appear here after clicking "Get SMILES"</div>
</div>

<script>
  var jsmeApp = null;

  // JSME calls this global function when it finishes loading
  function jsmeOnLoad() {{
    // Use the container's actual width for proper sizing
    var container = document.getElementById('jsme_container');
    var w = container ? container.offsetWidth || 700 : 700;
    jsmeApp = new JSApplet.JSME("jsme_container", w + "px", "{editor_height}px", {{
      "options": "query,hydrogens,multipart"
    }});
  }}

  function getSMILES() {{
    if (!jsmeApp) {{
      document.getElementById('output-area').textContent =
        'Editor is still loading — please wait a moment.';
      return;
    }}
    var smiles = jsmeApp.smiles();
    document.getElementById('output-area').textContent = 'SMILES: ' + (smiles || '(empty)');
  }}

  function getMolfile() {{
    if (!jsmeApp) {{
      document.getElementById('output-area').textContent =
        'Editor is still loading — please wait a moment.';
      return;
    }}
    var mol = jsmeApp.molFile();
    document.getElementById('output-area').textContent =
      'Molfile retrieved (' + mol.length + ' chars):\\n' + mol;
  }}

  function clearEditor() {{
    if (jsmeApp) {{
      jsmeApp.reset();
    }}
    document.getElementById('output-area').textContent =
      'SMILES will appear here after clicking "Get SMILES"';
  }}
</script>
</body>
</html>
"""


def render_ketcher_editor(
    height: int = 550,
    key: str = "ketcher_editor",
) -> None:
    """
    Render the JSME chemical structure editor inside Streamlit.

    The editor is loaded as a self-contained JavaScript applet from
    a public CDN (no iframe to a third-party site), so it works
    reliably without cross-origin restrictions.
    """
    editor_height = max(height - 90, 350)
    html = _JSME_HTML.format(
        editor_height=editor_height,
        editor_width="100%",
    )
    components.html(html, height=height, scrolling=True)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_drawn_structures(run_dir: Path, safe_doi: str) -> List[Dict[str, Any]]:
    """Load previously saved drawn structures for a DOI."""
    path = run_dir / "chemical_composition" / safe_doi / "drawn_structures.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    return []


def save_drawn_structure(
    run_dir: Path,
    safe_doi: str,
    sample_name: str,
    smiles: str = "",
    molfile: str = "",
    notes: str = "",
) -> Path:
    """
    Append a drawn structure record to the DOI's drawn_structures.json.
    Returns the path to the JSON file.
    """
    out_dir = run_dir / "chemical_composition" / safe_doi
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "drawn_structures.json"

    existing = load_drawn_structures(run_dir, safe_doi)
    existing.append({
        "sample_name": sample_name,
        "smiles": smiles,
        "molfile": molfile,
        "notes": notes,
    })

    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    return path


def delete_drawn_structure(run_dir: Path, safe_doi: str, index: int) -> None:
    """Delete a drawn structure by index."""
    structures = load_drawn_structures(run_dir, safe_doi)
    if 0 <= index < len(structures):
        structures.pop(index)
        path = run_dir / "chemical_composition" / safe_doi / "drawn_structures.json"
        path.write_text(json.dumps(structures, indent=2, ensure_ascii=False))
