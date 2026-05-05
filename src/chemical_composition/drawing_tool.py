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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

import socket
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components
import streamlit as st
import sys

# ---------------------------------------------------------------------------
# JSME-based HTML component
# ---------------------------------------------------------------------------

_JSME_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<base href="{jsme_base_url}/jsme/"/>
<script src="{jsme_base_url}/jsme/jsme.nocache.js"></script>
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


# ---------- Static server (serves your local JSME folder) ----------

@st.cache_resource(show_spinner=False)
def _start_static_server(static_root: Path, host: str = "127.0.0.1", port: int = 8765) -> str:
    static_root = static_root.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_root), **kwargs)

    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        # Give server a moment to start
        import time
        time.sleep(0.1)

        print(f"Started HTTP server at http://{host}:{port} serving {static_root}", file=sys.stderr)
        return f"http://{host}:{port}"
    except Exception as e:
        print(f"Failed to start HTTP server: {e}", file=sys.stderr)
        raise


def _jsme_bind_host() -> str:
    """Host/interface the helper HTTP server should bind to."""
    return os.environ.get("AEM_OCR_JSME_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _jsme_port() -> int:
    """Port used by the helper HTTP server."""
    raw = os.environ.get("AEM_OCR_JSME_PORT", "8765").strip()
    try:
        return int(raw)
    except ValueError:
        return 8765


def _jsme_public_base_url(port: int) -> str:
    """
    URL the browser should use to reach the helper server.

    Important: when Streamlit is opened from another machine, `127.0.0.1`
    points to the user's browser machine, not the Streamlit server. In that
    case set `AEM_OCR_JSME_PUBLIC_URL` to a browser-reachable URL such as
    `http://<server-host>:8765`.
    """
    env = os.environ.get("AEM_OCR_JSME_PUBLIC_URL", "").strip()
    if env:
        return env.rstrip("/")
    return f"http://127.0.0.1:{port}"



# ---------- Wrapper HTML (lives in-memory, but loaded by iframe URL) ----------

def _write_jsme_wrapper(tmp_dir: Path, *, jsme_base_url: str, height: int) -> Path:
    """
    Create a wrapper HTML file that loads JSME from jsme_base_url and provides buttons.
    We write it to a temp dir *inside the served root* so iframe can load it.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = tmp_dir / "jsme_wrapper.html"

    editor_height = max(height - 90, 350)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>JSME</title>

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
  .toolbar button.primary {{
    background: #1976d2; color: #fff; border-color: #1565c0;
  }}
  .toolbar .info {{ font-size: 13px; color: #555; margin-left: auto; }}
  #output-area {{
    font-family: monospace; font-size: 13px; padding: 8px;
    background: #fafafa; border-top: 1px solid #ddd;
    min-height: 28px; word-break: break-all; white-space: pre-wrap;
  }}
  .loading-step {{
    color: #1976d2;
    font-weight: 500;
  }}
</style>

<script>
  var jsmeApp = null;

  function setStatus(msg) {{
    var out = document.getElementById('output-area');
    if (out) out.textContent = msg;
  }}

  // Loading progress tracking
  document.addEventListener('DOMContentLoaded', function() {{
    setStatus("Step 1/3: DOM ready, preparing container...");
  }});

  function initJSME() {{
    setStatus("Step 2/3: Initializing JSME editor...");
    var container = document.getElementById('jsme_container');
    if (!container) {{
      setStatus("Error: DOM missing #jsme_container");
      return;
    }}
    var w = container.offsetWidth || 700;
    try {{
      jsmeApp = new JSApplet.JSME("jsme_container", w + "px", "{editor_height}px", {{
        "options": "query,hydrogens,multipart"
      }});
      setStatus("JSME ready! Draw your structure, then click Get SMILES.");
    }} catch (e) {{
      setStatus("JSME initialization error: " + (e && e.message ? e.message : e));
    }}
  }}

  function jsmeOnLoad() {{
    setStatus("Step 2/3: JSME library loaded, creating editor...");
    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", initJSME);
    }} else {{
      initJSME();
    }}
  }}

  function getSMILES() {{
    if (!jsmeApp) {{ setStatus("Editor not ready yet, please wait..."); return; }}
    var smiles = jsmeApp.smiles();
    setStatus("SMILES: " + (smiles || "(empty)"));
  }}

  function getMolfile() {{
    if (!jsmeApp) {{ setStatus("Editor not ready yet, please wait..."); return; }}
    var mol = jsmeApp.molFile();
    setStatus("Molfile (" + mol.length + " chars):\\n" + mol);
  }}

  function clearEditor() {{
    if (jsmeApp) jsmeApp.reset();
    setStatus("Cleared. Draw your structure, then click Get SMILES.");
  }}
</script>
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
  <div id="output-area" class="loading-step">Step 1/3: Loading JSME library...</div>
</div>

<!-- IMPORTANT: load local JSME AFTER DOM -->
<script src="{jsme_base_url}/jsme/jsme.nocache.js"></script>
</body>
</html>
"""
    wrapper_path.write_text(html, encoding="utf-8")
    return wrapper_path


def render_jsme_editor_local(
    height: int = 550,
    key: str = "jsme_editor",
) -> None:
    """
    Render a locally vendored JSME distribution located at:
      src/chemical_composition/JSME_2024-04-29

    Uses a local HTTP server + iframe so JSME can load its companion files.
    """
    # This file lives in src/chemical_composition/
    here = Path(__file__).resolve().parent
    jsme_root = here / "JSME_2024-04-29"
    jsme_dir = jsme_root / "jsme"

    # Check if JSME root directory exists
    if not jsme_root.exists():
        st.error(
            f"JSME distribution folder not found at: `{jsme_root}`\n\n"
            "Please download JSME from https://jsme-editor.github.io/ and extract it to:\n"
            f"`{jsme_root}`"
        )
        return

    # Check if jsme subdirectory exists (contains the actual JavaScript files)
    if not jsme_dir.exists():
        st.error(
            f"JSME JavaScript library not found at: `{jsme_dir}`\n\n"
            "The JSME distribution should contain a `jsme` subdirectory with the JavaScript files.\n\n"
            "**To fix this:**\n"
            "1. Download the complete JSME distribution from https://jsme-editor.github.io/\n"
            "2. Extract the archive - you should see:\n"
            "   - `jsme/` directory (contains JavaScript files)\n"
            "   - `JSME_minimal.html` and other example files\n"
            f"3. Place the extracted contents in: `{jsme_root}`\n\n"
            f"**Current contents of `{jsme_root}`:**"
        )
        if jsme_root.exists():
            try:
                contents = sorted([p.name for p in jsme_root.iterdir()])[:20]
                st.code("\n".join(contents), language="text")
                st.caption("(showing first 20 items)")
            except Exception as e:
                st.warning(f"Could not list directory contents: {e}")
        return

    # Check if jsme.nocache.js exists
    jsme_js = jsme_dir / "jsme.nocache.js"
    if not jsme_js.exists():
        st.error(
            f"JSME main JavaScript file not found: `{jsme_js}`\n\n"
            "The `jsme` directory exists but is missing `jsme.nocache.js`.\n"
            "Please re-download and extract the complete JSME distribution."
        )
        return

    bind_host = _jsme_bind_host()
    port = _jsme_port()
    public_base_url = _jsme_public_base_url(port)

    # Serve *the JSME root folder* (which contains the jsme subdirectory)
    _start_static_server(jsme_root, host=bind_host, port=port)
    local_base_url = f"http://127.0.0.1:{port}"

    # Write wrapper HTML into a temp dir INSIDE the served root
    wrapper_dir = jsme_dir / "_streamlit_wrapper"
    wrapper_path = _write_jsme_wrapper(
        wrapper_dir,
        jsme_base_url=public_base_url,
        height=height,
    )

    # iframe URL: the wrapper is at jsme/_streamlit_wrapper/jsme_wrapper.html
    # relative to the served root (jsme_root)
    iframe_url = f"{public_base_url}/jsme/_streamlit_wrapper/{wrapper_path.name}"

    # Test if server is responding
    import urllib.request
    try:
        with urllib.request.urlopen(f"{local_base_url}/jsme/_streamlit_wrapper/{wrapper_path.name}", timeout=2) as response:
            st.write(f"**JSME Server is responding**")
    except Exception as e:
        st.error(f"**Server test failed**: {type(e).__name__}: {e}")
        st.write("**Troubleshooting:**")
        st.write("- The HTTP server may not have started correctly")
        st.write(f"- Port {port} might be in use by another process")
        st.write("- Try restarting the Streamlit app")
        st.write("- If you opened Streamlit from another machine, set `AEM_OCR_JSME_PUBLIC_URL` to a browser-reachable address, for example `http://<server-host>:{port}`")
        st.write("- For remote access, you may also need `AEM_OCR_JSME_BIND_HOST=0.0.0.0`")

    components.iframe(iframe_url, height=height, scrolling=True)
