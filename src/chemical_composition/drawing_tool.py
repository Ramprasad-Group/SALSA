"""
Chemical drawing tool integration for Streamlit.

Embeds the JSME Molecule Editor (open-source, by Peter Ertl / Bruno Bienfait)
as a self-contained JavaScript applet inside a Streamlit HTML component.
Users draw chemical structures and retrieve SMILES / Molfile output directly.

JSME is loaded from a public CDN (jsme-editor on jsDelivr) so no local
installation is required and there are no cross-origin iframe restrictions.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import streamlit as st
import streamlit.components.v1 as components

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover - optional runtime dependency
    Chem = None
    AllChem = None

_JSME_ROOT = Path(__file__).resolve().parent / "JSME_2024-04-29"
_JSME_COMPONENT = components.declare_component(
    "jsme_editor",
    path=str(_JSME_ROOT),
)
_KETCHER_ROOT = Path(__file__).resolve().parent / "standalone"
_KETCHER_COMPONENT = components.declare_component(
    "ketcher_editor",
    path=str(_KETCHER_ROOT),
)


def _smiles_to_rdkit_molfile(smiles: str) -> str:
    """Convert SMILES to a V2000 molfile for JSME using RDKit."""
    if Chem is None or AllChem is None:
        raise RuntimeError(
            "The chemical SMILES converter is not available in the SALSA Python environment."
        )

    clean_smiles = str(smiles or "").strip()
    if not clean_smiles:
        raise ValueError("Paste a SMILES string first.")

    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        raise ValueError("The chemical SMILES converter could not parse the string.")

    AllChem.Compute2DCoords(mol)
    molfile = Chem.MolToMolBlock(mol, kekulize=True)
    if not molfile.strip():
        raise ValueError("The chemical SMILES converter produced an empty structure.")
    return molfile

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
    height: int = 720,
    key: str = "ketcher_editor",
) -> None:
    """
    Render the locally vendored Ketcher chemical structure editor.

    Ketcher is served as a native Streamlit component from:
      src/chemical_composition/standalone
    """
    if not _KETCHER_ROOT.exists():
        st.error(f"Ketcher distribution folder not found at: `{_KETCHER_ROOT}`")
        return

    missing_files = [
        path
        for path in (
            _KETCHER_ROOT / "index.html",
            _KETCHER_ROOT / "static" / "js" / "main.8f1d38b5.js",
            _KETCHER_ROOT / "static" / "css" / "main.bdf7ea17.css",
        )
        if not path.exists()
    ]
    if missing_files:
        st.error(
            "Ketcher distribution is incomplete. Missing:\n\n"
            + "\n".join(f"- `{path}`" for path in missing_files)
        )
        return

    _KETCHER_COMPONENT(
        frame_height=height,
        key=key,
        default=None,
    )


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

class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _probe_host(host: str) -> str:
    """Address the local process should use when checking the helper server."""
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _format_http_host(host: str) -> str:
    """Format hosts safely for http://host:port URLs."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _local_base_url(host: str, port: int) -> str:
    return f"http://{_format_http_host(_probe_host(host))}:{port}"


def _probe_server(static_root: Path, host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Return True only when the existing listener serves the same JSME root.

    This avoids reusing an unrelated process that merely happens to own the
    preferred port.
    """
    probe_url = f"{_local_base_url(host, port)}/jsme/jsme.nocache.js"
    expected_path = static_root / "jsme" / "jsme.nocache.js"
    try:
        expected_body = expected_path.read_bytes()
        with urllib.request.urlopen(probe_url, timeout=timeout) as response:
            return response.read() == expected_body
    except OSError:
        return False


def _start_http_server(static_root: Path, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_root), **kwargs)

    httpd = _ReusableThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # Give the listener a moment to start accepting requests.
    time.sleep(0.1)
    return httpd


@st.cache_resource(show_spinner=False)
def _start_static_server(static_root: Path, host: str = "127.0.0.1", port: int = 8765) -> Dict[str, Any]:
    static_root = static_root.resolve()

    try:
        httpd = _start_http_server(static_root, host, port)
        actual_port = httpd.server_address[1]
        print(
            f"Started HTTP server at {_local_base_url(host, actual_port)} serving {static_root}",
            file=sys.stderr,
        )
        return {"actual_port": actual_port, "server": httpd}
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            if _probe_server(static_root, host, port):
                print(
                    f"Reusing existing HTTP server at {_local_base_url(host, port)} for {static_root}",
                    file=sys.stderr,
                )
                return {"actual_port": port, "server": None}

            httpd = _start_http_server(static_root, host, 0)
            actual_port = httpd.server_address[1]
            print(
                f"Preferred HTTP server port {port} is in use by another process; "
                f"started JSME helper at {_local_base_url(host, actual_port)} instead",
                file=sys.stderr,
            )
            return {"actual_port": actual_port, "server": httpd}
        print(f"Failed to start HTTP server: {e}", file=sys.stderr)
        raise


def _jsme_bind_host() -> str:
    """Host/interface the helper HTTP server should bind to."""
    return os.environ.get("SALSA_JSME_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _jsme_port() -> int:
    """Port used by the helper HTTP server."""
    raw = os.environ.get("SALSA_JSME_PORT", "8765").strip()
    try:
        return int(raw)
    except ValueError:
        return 8765


def _rewrite_url_port(base_url: str, port: int) -> str:
    """Rewrite the port in a full URL when the helper falls back to a free port."""
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.hostname:
        return base_url.rstrip("/")

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"

    host = _format_http_host(parsed.hostname)
    netloc = f"{auth}{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)).rstrip("/")


def _jsme_public_base_url(port: int, requested_port: Optional[int] = None) -> str:
    """
    URL the browser should use to reach the helper server.

    Important: when Streamlit is opened from another machine, `127.0.0.1`
    points to the user's browser machine, not the Streamlit server. In that
    case set `SALSA_JSME_PUBLIC_URL` to a browser-reachable URL such as
    `http://<server-host>:8765`.
    """
    env = os.environ.get("SALSA_JSME_PUBLIC_URL", "").strip()
    if env:
        env = env.rstrip("/")
        if requested_port is not None and port != requested_port:
            adjusted = _rewrite_url_port(env, port)
            if adjusted != env:
                print(
                    f"Adjusted SALSA_JSME_PUBLIC_URL from {env} to {adjusted} "
                    f"because port {requested_port} was unavailable.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Preferred JSME port {requested_port} was unavailable; "
                    f"continuing to use SALSA_JSME_PUBLIC_URL={env}.",
                    file=sys.stderr,
                )
            return adjusted
        return env
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

    JSME is served as a native Streamlit component so all browser requests use
    the same origin and reverse-proxy path as the main app.
    """
    jsme_root = _JSME_ROOT
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

    component_index = jsme_root / "index.html"
    if not component_index.exists():
        st.error(f"JSME component frontend not found: `{component_index}`")
        return

    state_prefix = f"jsme_rdkit_{key}"
    request_state_key = f"{state_prefix}_request_id"
    molfile_state_key = f"{state_prefix}_molfile"
    error_state_key = f"{state_prefix}_error"
    smiles_state_key = f"{state_prefix}_smiles"

    component_value = _JSME_COMPONENT(
        editor_height=max(height - 140, 350),
        frame_height=height,
        load_request_id=st.session_state.get(request_state_key, ""),
        load_molfile=st.session_state.get(molfile_state_key, ""),
        load_error=st.session_state.get(error_state_key, ""),
        load_smiles=st.session_state.get(smiles_state_key, ""),
        key=key,
        default=None,
    )

    if not isinstance(component_value, dict):
        return
    if component_value.get("action") != "load_smiles":
        return

    request_id = str(component_value.get("request_id") or "")
    if not request_id or request_id == st.session_state.get(request_state_key):
        return

    smiles = str(component_value.get("smiles") or "").strip()
    st.session_state[request_state_key] = request_id
    st.session_state[smiles_state_key] = smiles
    try:
        st.session_state[molfile_state_key] = _smiles_to_rdkit_molfile(smiles)
        st.session_state[error_state_key] = ""
    except Exception as exc:
        st.session_state[molfile_state_key] = ""
        st.session_state[error_state_key] = str(exc)

    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun is not None:
        rerun()
