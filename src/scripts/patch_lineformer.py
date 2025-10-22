# --- Patch LineFormer infer.py to add sys.path append before clean_chart import ---
from pathlib import Path
import re

infer_path = Path("extern/LineFormer/infer.py")
src = infer_path.read_text()

# Already patched? (either sys.path append or relative import)
if "from .clean_chart import get_clean_input" in src or "sys.path.insert(0" in src:
    print("[setup] LineFormer infer.py already patched; skipping.")
else:
    # Build absolute path to extern/LineFormer at runtime
    lf_dir = (Path.cwd() / "extern" / "LineFormer").resolve()
    inject = (
        "import sys\n"
        f"sys.path.insert(0, r\"{lf_dir}\")\n"
        "from clean_chart import get_clean_input\n"
    )
    # Replace the first occurrence of the target import with our injected block
    new_src, n = re.subn(
        r'^\s*from\s+clean_chart\s+import\s+get_clean_input\s*',
        inject,
        src,
        count=1,
        flags=re.M
    )
    if n == 0:
        raise SystemExit("[setup] ERROR: Could not find 'from clean_chart import get_clean_input' in infer.py")
    infer_path.write_text(new_src)
    print("[setup] Patched LineFormer infer.py to prepend sys.path and import clean_chart.")