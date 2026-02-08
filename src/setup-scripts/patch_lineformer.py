#!/usr/bin/env python3
"""
Patch LineFormer infer.py so that `from clean_chart import get_clean_input`
resolves correctly when called from outside the LineFormer directory.

Instead of injecting a hardcoded absolute path (which breaks when the repo
moves), we inject a dynamic `os.path.dirname(__file__)` lookup so it always
resolves relative to infer.py itself.
"""
from pathlib import Path
import re

# Resolve repo root relative to this script (src/setup-scripts/patch_lineformer.py)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
infer_path = REPO_ROOT / "extern" / "LineFormer" / "infer.py"

if not infer_path.exists():
    print(f"[setup] {infer_path} not found – skipping LineFormer patch.")
    raise SystemExit(0)

src = infer_path.read_text()

# Already patched?
if "sys.path.insert(0" in src or "from .clean_chart import get_clean_input" in src:
    print("[setup] LineFormer infer.py already patched; skipping.")
    raise SystemExit(0)

# Inject a dynamic sys.path insert based on __file__ (not cwd)
inject = (
    "import sys, os as _os\n"
    "sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n"
    "from clean_chart import get_clean_input\n"
)

new_src, n = re.subn(
    r'^\s*from\s+clean_chart\s+import\s+get_clean_input\s*',
    inject,
    src,
    count=1,
    flags=re.M,
)
if n == 0:
    raise SystemExit(
        "[setup] ERROR: Could not find 'from clean_chart import get_clean_input' in infer.py"
    )

infer_path.write_text(new_src)
print("[setup] Patched LineFormer infer.py with dynamic sys.path insert.")
