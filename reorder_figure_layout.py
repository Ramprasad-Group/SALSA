import ast
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parent / "src" / "ui" / "streamlit_app.py"

with open(FILE, "r") as fh:
    original = fh.read()

lines = original.splitlines(keepends=True)

# -----------------------------------------------------------------------
# Helper: find the 1-based line number of the FIRST line that contains
# the given text (exact substring match, leading whitespace included).
# -----------------------------------------------------------------------
def find_line(text, start=1):
    for i, ln in enumerate(lines, 1):
        if i < start:
            continue
        if text in ln:
            return i
    raise ValueError(f"Anchor not found: {text!r}")

# -----------------------------------------------------------------------
# Locate all anchor line numbers (1-based)
# -----------------------------------------------------------------------

# A) render_ocr_dropdown / render_llm_dropdown block ends just before the
#    subfig comment.  We find the first line of the subfig comment.
ln_subfig_comment   = find_line("        # --- Display cropped subfigures from plextract and interventions ---")
ln_safe_doi         = find_line("        safe_doi = selected_doi.replace(\"/\", \"@\").strip()")
ln_fig_stem         = find_line("        fig_stem = Path(f.id).stem")

# B) "if subfig_dirs:" block starts here
ln_if_subfig        = find_line("        if subfig_dirs:")

# C) Intervention Results block
ln_int_start        = find_line("        # ========================\n        # SECTION: Intervention Results", )
# Find it by multi-line match — simpler: find '# SECTION: Intervention Results'
ln_int_start        = find_line("        # SECTION: Intervention Results")
# Back up to the ==== line before it
ln_int_start -= 1   # now points at the first '# ========================' line

# D) Display metadata section
ln_meta_start       = find_line("        # SECTION: Display metadata")
ln_meta_start -= 1  # back up to '# ========================'

# E) render_metadata_dropdown call
ln_meta_call        = find_line("        render_metadata_dropdown(metadata_path=f.metadata_path)")

# F) Rerun Data Extraction section
ln_rerun_ocr_start  = find_line("        # SECTION: Rerun Data Extraction")
ln_rerun_ocr_start -= 1

# G) Rerun LLM section
ln_rerun_llm_start  = find_line("        # SECTION: Rerun LLM Dataset Generation")
ln_rerun_llm_start -= 1

# H) End of Rerun LLM section — the next top-level "with tab_tables:" line
ln_tab_tables       = find_line("\nwith tab_tables:")
# find_line matches within a line; we want the line that starts "with tab_tables:"
# Re-search directly:
for i, ln in enumerate(lines, 1):
    if ln.startswith("with tab_tables:"):
        ln_tab_tables = i
        break

print(f"Anchor lines (1-based):")
print(f"  subfig comment      : {ln_subfig_comment}")
print(f"  safe_doi            : {ln_safe_doi}")
print(f"  fig_stem            : {ln_fig_stem}")
print(f"  if subfig_dirs      : {ln_if_subfig}")
print(f"  int_start           : {ln_int_start}")
print(f"  meta_start          : {ln_meta_start}")
print(f"  meta_call           : {ln_meta_call}")
print(f"  rerun_ocr_start     : {ln_rerun_ocr_start}")
print(f"  rerun_llm_start     : {ln_rerun_llm_start}")
print(f"  tab_tables          : {ln_tab_tables}")

# -----------------------------------------------------------------------
# Convert to 0-based slice indices
# -----------------------------------------------------------------------
def L(n):  # line n (1-based) → 0-based index
    return n - 1

# -----------------------------------------------------------------------
# Extract named blocks (as lists of lines)
# -----------------------------------------------------------------------

# Block 0: everything before the subfig comment (OCR+LLM dropdowns etc.)
block_before = lines[: L(ln_subfig_comment)]

# Block A: safe_doi line (single line)
block_safe_doi = [lines[L(ln_safe_doi)]]

# Block B: subfig comment + collection code (from comment line up to but NOT
#           including "if subfig_dirs:")
#   This range:  ln_subfig_comment .. ln_if_subfig-1
#   Remove the safe_doi line from this block (it moves up).
block_subfig_collection_raw = lines[L(ln_subfig_comment) : L(ln_if_subfig)]
block_subfig_collection = [
    ln for ln in block_subfig_collection_raw
    if "        safe_doi = selected_doi.replace" not in ln
]

# Block C: if subfig_dirs: block
#   From ln_if_subfig up to but not including ln_int_start
block_if_subfig = lines[L(ln_if_subfig) : L(ln_int_start)]

# Block D: Intervention Results block (original, to be REPLACED)
#   From ln_int_start up to but not including ln_meta_start
block_int_original = lines[L(ln_int_start) : L(ln_meta_start)]

# Block E: metadata section (comment + call)
#   From ln_meta_start through ln_meta_call (inclusive), plus trailing blank line
block_meta = lines[L(ln_meta_start) : L(ln_meta_call) + 1]
# Consume any immediately following blank lines that were part of the spacing
meta_end = L(ln_meta_call) + 1
while meta_end < len(lines) and lines[meta_end].strip() == "":
    block_meta.append(lines[meta_end])
    meta_end += 1

# Block F: Rerun Data Extraction section
#   From ln_rerun_ocr_start up to but not including ln_rerun_llm_start
block_rerun_ocr = lines[L(ln_rerun_ocr_start) : L(ln_rerun_llm_start)]

# Block G: Rerun LLM section
#   From ln_rerun_llm_start up to but not including ln_tab_tables
block_rerun_llm = lines[L(ln_rerun_llm_start) : L(ln_tab_tables)]

# Block H: everything from tab_tables onward
block_after = lines[L(ln_tab_tables) :]

# -----------------------------------------------------------------------
# New Intervention Results replacement block
# -----------------------------------------------------------------------
NEW_INT_BLOCK = """\
        # ========================
        # SECTION: Intervention Results
        # Show new OCR and LLM data once a per-figure plextract_interventions
        # run exists (i.e. the user has clicked "Rerun data extraction").
        # ========================
        if _per_fig_plextract and _per_fig_plextract.exists():
            _int_latest_run = latest_plextract_run_dir(_per_fig_plextract)
            _int_out_base = (
                (_int_latest_run / "output" / safe_doi) if _int_latest_run else None
            )
            if _int_out_base and _int_out_base.exists():
                for _idir in sorted(_int_out_base.iterdir()):
                    if _idir.is_dir():
                        render_ocr_dropdown(
                            csv_path=_idir / "final" / "final_dataset.csv",
                            key_prefix=f"int_ocr_{safe_doi}_{f.id}_{_idir.name}",
                            axis_override_dir=_idir,
                            title=f"New OCR-Extracted Data: {_idir.name}",
                        )
                        _int_emit_path = find_emit_rows_for_source(rr.run_dir, selected_doi, "figure", _idir.name)
                        render_llm_dropdown(
                            run_dir=rr.run_dir,
                            emit_rows_path=_int_emit_path,
                            key_prefix=f"int_llm_{safe_doi}_{f.id}_{_idir.name}",
                            file_name_prefix=_idir.name,
                            title=f"New LLM-Generated Dataset: {_idir.name}",
                        )
"""

block_int_new = [line + "\n" for line in NEW_INT_BLOCK.splitlines()]
# Remove duplicated newlines at end — keep exactly one trailing blank line
while block_int_new and block_int_new[-1].strip() == "":
    block_int_new.pop()
block_int_new.append("\n")

# -----------------------------------------------------------------------
# Sanity-check sizes
# -----------------------------------------------------------------------
def preview(label, block):
    text = "".join(block)
    head = text[:200].replace("\n", "\\n")
    tail = text[-100:].replace("\n", "\\n")
    print(f"\n--- {label} ({len(block)} lines) ---")
    print(f"  HEAD: {head}")
    print(f"  TAIL: {tail}")

preview("block_before (last 3 lines)", lines[L(ln_subfig_comment)-3 : L(ln_subfig_comment)])
preview("block_safe_doi", block_safe_doi)
preview("block_subfig_collection (first 5 lines)", block_subfig_collection[:5])
preview("block_if_subfig (first 3 lines)", block_if_subfig[:3])
preview("block_int_original (first 3 lines)", block_int_original[:3])
preview("block_meta", block_meta)
preview("block_rerun_ocr (first 3 lines)", block_rerun_ocr[:3])
preview("block_rerun_llm (first 3 lines)", block_rerun_llm[:3])
preview("block_int_new (first 3 lines)", block_int_new[:3])

# -----------------------------------------------------------------------
# Reassemble in NEW order
# -----------------------------------------------------------------------
# 1. block_before (OCR+LLM dropdowns, etc.)
# 2. block_safe_doi
# 3. block_rerun_ocr   (Rerun Data Extraction)
# 4. block_rerun_llm   (Rerun LLM Dataset Generation)
# 5. block_int_new     (Intervention Results — replaced with render helpers)
# 6. block_meta        (Display metadata)
# 7. block_subfig_collection (without safe_doi line)
# 8. block_if_subfig   (if subfig_dirs: block)
# 9. block_after       (tab_tables onward)

new_lines = (
    block_before
    + block_safe_doi
    + block_rerun_ocr
    + block_rerun_llm
    + block_int_new
    + block_meta
    + block_subfig_collection
    + block_if_subfig
    + block_after
)

new_content = "".join(new_lines)

# -----------------------------------------------------------------------
# Verify: safe_doi line appears exactly once in the figure section
# -----------------------------------------------------------------------
n_safe_doi = new_content.count("        safe_doi = selected_doi.replace")
assert n_safe_doi == 1, f"Expected 1 safe_doi assignment in figure section, got {n_safe_doi}"

# -----------------------------------------------------------------------
# Syntax check
# -----------------------------------------------------------------------
try:
    ast.parse(new_content)
    print("\nast.parse: OK — file is syntactically valid.")
except SyntaxError as e:
    print(f"\nSYNTAX ERROR: {e}")
    # Show the offending context
    err_lines = new_content.splitlines()
    start = max(0, e.lineno - 5)
    end   = min(len(err_lines), e.lineno + 5)
    for i, ln in enumerate(err_lines[start:end], start + 1):
        marker = ">>>" if i == e.lineno else "   "
        print(f"  {marker} {i:4d}: {ln}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Write the result
# -----------------------------------------------------------------------
with open(FILE, "w") as fh:
    fh.write(new_content)

print(f"\nFile written: {FILE}")
print(f"Original lines: {len(lines)}, new lines: {len(new_lines)}")

# -----------------------------------------------------------------------
# Quick post-write verification: grep the new order in the output
# -----------------------------------------------------------------------
with open(FILE, "r") as fh:
    written = fh.read()

wlines = written.splitlines()

pos_safe_doi  = next(i for i, l in enumerate(wlines, 1) if "safe_doi = selected_doi.replace" in l)
pos_rerun_ocr = next(i for i, l in enumerate(wlines, 1) if "# SECTION: Rerun Data Extraction" in l)
pos_rerun_llm = next(i for i, l in enumerate(wlines, 1) if "# SECTION: Rerun LLM Dataset Generation" in l)
pos_int       = next(i for i, l in enumerate(wlines, 1) if "# SECTION: Intervention Results" in l)
pos_meta      = next(i for i, l in enumerate(wlines, 1) if "# SECTION: Display metadata" in l)
pos_subfig_c  = next(i for i, l in enumerate(wlines, 1) if "# --- Display cropped subfigures" in l)
pos_if_subfig = next(i for i, l in enumerate(wlines, 1) if "        if subfig_dirs:" in l)

print("\nNew section order (line numbers in written file):")
print(f"  safe_doi              : {pos_safe_doi}")
print(f"  Rerun Data Extraction : {pos_rerun_ocr}")
print(f"  Rerun LLM Generation  : {pos_rerun_llm}")
print(f"  Intervention Results  : {pos_int}")
print(f"  Display metadata      : {pos_meta}")
print(f"  subfig comment        : {pos_subfig_c}")
print(f"  if subfig_dirs:       : {pos_if_subfig}")

assert pos_safe_doi  < pos_rerun_ocr,  "safe_doi must come before Rerun OCR"
assert pos_rerun_ocr < pos_rerun_llm,  "Rerun OCR must come before Rerun LLM"
assert pos_rerun_llm < pos_int,        "Rerun LLM must come before Intervention Results"
assert pos_int       < pos_meta,       "Intervention Results must come before metadata"
assert pos_meta      < pos_subfig_c,   "metadata must come before subfig comment"
assert pos_subfig_c  < pos_if_subfig,  "subfig comment must come before if subfig_dirs"

print("\nAll ordering assertions PASSED.")
