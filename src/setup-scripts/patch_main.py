#!/usr/bin/env python3
"""
Patch src/plextract/main.py to use relative paths for run_pipeline.
Replaces:
    run_pipeline(input_dir='/home/.../AEM_OCR/src/input', workdir='/home/.../AEM_OCR/src/')
with:
    run_pipeline(input_dir='src/input', workdir='src/')
"""

from pathlib import Path
import re

main_path = Path("src/plextract/main.py")
if not main_path.exists():
    raise FileNotFoundError(f"Could not find {main_path}")

text = main_path.read_text()

# Replace any run_pipeline(...) call with relative paths
new_text, n = re.subn(
    r"run_pipeline\s*\(\s*input_dir\s*=\s*['\"].*?/src/input['\"]\s*,\s*workdir\s*=\s*['\"].*?/src/['\"]\s*\)",
    "run_pipeline(input_dir=' src/input', workdir='src/')",
    text,
    flags=re.M,
)

if n:
    main_path.write_text(new_text)
    print(f"[patch_main_paths] Patched {n} occurrence(s) in {main_path}")
else:
    print("[patch_main_paths] No matching lines found — nothing changed.")