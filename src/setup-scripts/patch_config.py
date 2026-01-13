#!/usr/bin/env python3
import sys, re, pathlib
cfg = pathlib.Path(sys.argv[1])
print(cfg)
text = cfg.read_text()
if "custom_imports" not in text:
    block = (
        "custom_imports = dict(\n"
        "    imports=['mmdet.models.roi_heads.cascade_roi_head_LGF'],\n"
        "    allow_failed_imports=False\n"
        ")\n\n"
    )
    # Insert after any initial comments/imports
    m = re.search(r"(^[\s\S]*?\n)(?=model\s*=|_base_\s*=)", text, re.MULTILINE)
    if m:
        text = text[:m.end(1)] + block + text[m.end(1):]
    else:
        text = block + text
    cfg.write_text(text)
    print(f"Patched {cfg}")
else:
    print(f"Already has custom_imports: {cfg}")