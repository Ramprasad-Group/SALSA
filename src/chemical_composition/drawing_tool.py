"""
Chemical drawing tool integration for Streamlit.

Embeds the Ketcher chemical structure editor (open-source, EPAM)
inside a Streamlit component via an HTML iframe.  Users can draw
structures, and the resulting Molfile / SMILES is captured back
into the Streamlit session for storage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Ketcher HTML component
# ---------------------------------------------------------------------------

# Ketcher is served from a CDN.  The component posts the molfile back
# to the Streamlit parent frame via window.parent.postMessage.

_KETCHER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ margin: 0; padding: 0; overflow: hidden; }}
  #ketcher-container {{ width: 100%; height: {height}px; }}
  iframe {{ width: 100%; height: 100%; border: none; }}
  .toolbar {{
    display: flex; gap: 8px; padding: 8px; background: #f5f5f5;
    border-bottom: 1px solid #ddd; align-items: center;
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
  #smiles-display {{
    font-family: monospace; font-size: 13px; padding: 8px;
    background: #fafafa; border-top: 1px solid #ddd;
    min-height: 24px; word-break: break-all;
  }}
</style>
</head>
<body>
<div class="toolbar">
  <button class="primary" onclick="getSMILES()">Get SMILES</button>
  <button onclick="getMolfile()">Get Molfile</button>
  <button onclick="clearEditor()">Clear</button>
  <span class="info">Draw your chemical structure below</span>
</div>
<div id="ketcher-container">
  <iframe id="ketcher-frame"
          src="https://lifescience.opensource.epam.com/ketcher/demo/"
          allow="clipboard-read; clipboard-write">
  </iframe>
</div>
<div id="smiles-display">SMILES will appear here...</div>
<script>
  // Access the Ketcher instance inside the iframe
  function getKetcher() {{
    var frame = document.getElementById('ketcher-frame');
    if (frame && frame.contentWindow && frame.contentWindow.ketcher) {{
      return frame.contentWindow.ketcher;
    }}
    return null;
  }}

  async function getSMILES() {{
    var ketcher = getKetcher();
    if (!ketcher) {{
      document.getElementById('smiles-display').textContent =
        'Ketcher not ready yet. Please wait for it to load fully.';
      return;
    }}
    try {{
      var smiles = await ketcher.getSmiles();
      document.getElementById('smiles-display').textContent =
        'SMILES: ' + smiles;
      // Post to Streamlit
      window.parent.postMessage({{
        type: 'streamlit:setComponentValue',
        data: {{ smiles: smiles, format: 'smiles' }}
      }}, '*');
    }} catch(e) {{
      document.getElementById('smiles-display').textContent =
        'Error getting SMILES: ' + e.message;
    }}
  }}

  async function getMolfile() {{
    var ketcher = getKetcher();
    if (!ketcher) {{
      document.getElementById('smiles-display').textContent =
        'Ketcher not ready yet.';
      return;
    }}
    try {{
      var mol = await ketcher.getMolfile();
      document.getElementById('smiles-display').textContent =
        'Molfile retrieved (' + mol.length + ' chars)';
      window.parent.postMessage({{
        type: 'streamlit:setComponentValue',
        data: {{ molfile: mol, format: 'molfile' }}
      }}, '*');
    }} catch(e) {{
      document.getElementById('smiles-display').textContent =
        'Error: ' + e.message;
    }}
  }}

  function clearEditor() {{
    var ketcher = getKetcher();
    if (ketcher) {{
      ketcher.setMolecule('');
    }}
    document.getElementById('smiles-display').textContent =
      'SMILES will appear here...';
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
    Render the Ketcher chemical structure editor inside Streamlit.

    The editor is embedded as an HTML iframe component.
    Users draw structures interactively, then click "Get SMILES" or
    "Get Molfile" to capture the output.
    """
    html = _KETCHER_HTML.format(height=height - 80)
    components.html(html, height=height, scrolling=False)


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
