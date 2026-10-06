# SALSA agent curation guide

You are replacing the human reviewer of a SALSA run for **one paper**. SALSA has
already parsed the paper, filtered figures/tables, run automatic chart
digitization (ChartDete + LineFormer + OCR), drafted table rows, and drafted
chemical compositions. These automatic outputs are often wrong. Do what a careful human
curator does: decide what is relevant, fix the digitization, format the rows,
verify the chemistry, and leave an audit trail.

Domain: anion exchange membranes (AEMs). Target properties are ionic/hydroxide
conductivity, conductivity retention vs time, IEC (vs time), water uptake,
swelling ratio and mechanical properties. Relevant figures show these vs time
(alkaline stability) or vs temperature (incl. Arrhenius plots).
`<run>/custom_instructions.json` holds the run's filter criteria, output
columns, missing-value defaults and processing rules. Read it first.

## Ground rules

- Work **only** inside the run directory you were given. Do not open other run
  directories, `src/agents/data/`, or any other curated dataset. This is a blind
  replication test.
- Never invent data. Every number must come from the figure, a table, or the text.
- Every decision goes through a tool command with a `--reason`, which feeds the
  ledger. Uncertain items go in your summary under "Needs human review".
- Look at every image you rely on (use the Read tool on the PNG paths the tools
  print). Do not skip visual verification.

## Practicalities

- Subagents cannot use the Write tool for new files; write `summary.md` and CSVs with a
  shell heredoc or Python.
- `zoom` on a small box: pass `--scale 8` (or more) for a useful enlargement.
- `rerun` on a brand-new crop runs ChartDete + LineFormer on CPU (1–3 min). Run it
  in the foreground and wait.
- `markers` struggles with markers under ~5 px. Use zooms and place points by
  hand, as a human would.
- Data stated only in the text cannot be emitted yet. List it under "Needs human review".

## Tools

All commands: `cd /Users/william/Developments/SALSA/src && $PY -m agent_curation <cmd> --run <RUN> --doi <DOI> ...`
with `PY=/Users/william/Developments/Users/William/Developments/miniconda3/envs/SALSA/bin/python`.
For `chem`, also export `SALSA_RDKIT_PYTHON=/Users/william/Developments/Users/William/Developments/miniconda3/envs/aem/bin/python`.
Pixel coordinates are always in the **original image** of the figure or subfigure
(the rulers on rendered views show these coordinates, even on zooms).

| Command | Purpose |
|---|---|
| `paper` | Writes `context.json` (figures, candidates, captions, tables, chemistry) and `fulltext.md`. Start here. |
| `inspect --fig ID` | Grid image + check image + OCR ticks, legend, LineFormer series, extracted series. |
| `zoom --fig ID --box x1,y1,x2,y2 [--grid]` | Enlarged region with pixel rulers, for reading tick/marker positions precisely. |
| `crop --fig Figure_N --box x1,y1,x2,y2 [--box ...] [--start k]` | Make subfigure crops when the automatic split is wrong or missing. New ids are `Figure_N_subfig_k`. |
| `calibrate --fig ID --x PX=VAL --x PX=VAL --y PY=VAL --y PY=VAL [--xtype/--ytype log10]` | Axis calibration override. Use tick-mark pixel positions, as far apart as possible. |
| `titles --fig ID --x "Time (h)" --y "..."` | Axis-title override (optional, for clarity). |
| `markers --fig ID --rgb R,G,B or --at X,Y [--tol] [--exclude legend_box] [--fill-holes] [--open r]` | Find marker centroids of one colour. Check the preview. |
| `series --fig ID --index K --points "x,y x,y ..." or last-markers` | Set exact points for series K (new K = new series). |
| `series --fig ID --name K="Sample" --delete K` | Name or delete series. |
| `rerun --fig ID` | Apply overrides and re-run extraction, then render the check image. |
| `draft-rows --fig ID --x-col ... --default-property ... --property K=... --const COL=VAL --sig N` | Draft output rows from the latest extraction. |
| `emit --kind figure/table --source NAME --rows file.csv` | Write the final rows for one source. Replaces automatic rows for that source. |
| `reject --source NAME [--kind table] --reason` | Mark a figure or table as not used (removes the automatic table rows). |
| `verdict --decision include/exclude --reason` | Paper-level decision. **Always call it once at the end.** |
| `chem [--csv candidate.csv]` | RDKit checks plus a rendered structure grid. |
| `chem-apply --csv verified.csv` | Install the verified chemistry (original kept as `samples.salsa.csv`). |

## Workflow

### 1. Understand the paper
Run `paper`, read `context.json`, then read `fulltext.md`, especially the
experimental section. Write down:
- the **samples prepared in this paper** (exact names as used in figures/tables);
- the stability-test conditions (solvent, molarity, temperature, duration);
- the temperature at which each property was measured (often in table footnotes
  or the measurement section).

### 2. Paper verdict
Exclude the paper (`verdict --decision exclude`, no rows needed) if any of these hold:
- it is a review, or the membranes are not anion exchange membranes (e.g. vanadium flow battery PEMs);
- there is no relevant data;
- the polymer structure cannot be specified: no chemical schematic and the
  material is a composite/blend with additives and/or crosslinks (typical for
  chitosan/PVA/inorganic-filler blends), or the degree of functionalization
  needed to define the composition is not reported;
- the data look unreliable or internally inconsistent (explain why).
Otherwise continue. When unsure, include and flag it.

### 3. Figures
For each figure in `context.json`, decide relevance from the caption, the image
and the run's `figure_filter_criteria`. **Apply the criteria literally.** E.g.
mechanical properties vs time are not in the default figure criteria, so they are
rejected, even though mechanical properties are in the *table* criteria. Arrhenius
plots that replot the same data as a conductivity-vs-temperature figure are
rejected as duplicates. Reject irrelevant figures with a reason.
For each relevant panel:

1. **Pick the image.** Use the automatic subfigure candidate if it contains exactly
   one complete plot (axes, tick labels, legend if any). Otherwise `crop` the full
   figure yourself, then `rerun` the new id (this runs full extraction on the crop).
2. **`inspect`**, and Read the grid image and the check image.
3. **Calibration.** In the check image, the green lines must sit exactly on the
   figure's tick marks. If they don't, `zoom` on the axes, read the pixel positions
   of two or more well-separated tick marks per axis, and `calibrate`.
   Confirm with the returned check image. Log axes: use `--ytype log10` and actual values.
4. **Series.** Match legend entries to series by colour and marker shape.
   - Marker plots (the usual case): run `markers` per series colour (sample the
     colour with `--at` on a marker, exclude the legend box). Compare the count
     with the visible markers. Overlapping or occluded markers must be added by
     hand at the visible position (e.g. the shared t=0 point). Drop false hits
     (text, legend, other series). Then `series --index K --points ...`.
   - Line-only plots: LineFormer traces may be acceptable, but check them against
     the curve. For continuous curves, prefer points at regular x intervals that
     capture the shape (about 10–20 per series).
   - Delete series that are not samples prepared in this paper (commercial or
     prior-work reference membranes, e.g. a membrane from the authors' earlier work
     used for comparison), fits/guide lines, and duplicates.
   - Before/after bar charts (e.g. WU/SR before and after the stability test) are
     relevant: emit the "before" bars at `time(h)` = 0 and the "after" bars at the
     test duration.
   - Name each series with the **exact sample name** used in the paper (the
     same name you will use in chemistry).
5. **`rerun`** and Read the check image. Every circle must sit on its marker, with
   no missing or extra points. Cross-check values the text states (e.g. "retained
   64.4% after 192 h", end-point labels on the plot). Iterate until it is right.
6. **Rows.** `draft-rows`, then edit the CSV:
   - x goes in the column that matches its meaning: stability time → `time(h)`
     (days ×24); temperature dependence → `prop test temp (C)`; Arrhenius
     1000/T (K⁻¹) → convert to °C (`--x-transform inverse_temp_1000_K`).
   - Snap x to the nominal values the paper states or the axis makes obvious
     (`--x-snap 24` for 0, 24, 48 … h; `--x-snap 10` for 30, 40 … 80 °C), and note it
     in the reason. `--x-round` means decimal places, not a step.
   - Fill conditions from the text: solvent type and molarity, degradation test
     temperature, property test temperature. Before a degradation test,
     `time(h)` = 0. Keep the "X Not Found" defaults when the paper truly doesn't say.
   - Use a clear property name with units, e.g. `OH- conductivity (mS/cm)`,
     `conductivity retention (%)`, `water uptake (%)`, `swelling ratio (%)`,
     `IEC (mmol/g)`. Split water-uptake vs swelling series correctly.
   - **Retention → absolute (always try).** For a % retention series, multiply
     by the sample's initial absolute value **at the degradation-test temperature**
     (e.g. the 80 °C point of the conductivity-vs-temperature figure, or a table/text
     value), unless the paper states a different measurement temperature. Set
     `prop test temp (C)` to that temperature and use absolute units in the property
     name. Keep % only if no initial value exists anywhere in the paper. Record the
     multiplier in the reason.
   - Round values to the precision readable from the plot (usually 3 significant figures).
   - Then `emit --kind figure --source <ID>`.

### 4. Tables
For each table, compare the automatic rows (`auto_rows_file`) with the table CSV
and caption/footnotes. Then either `emit` a corrected version
(`--kind table --source Table_N`) or `reject --kind table`.
- Keep only measured values of target properties. Drop calculated/theoretical
  values (e.g. theoretical IEC), unrelated properties, and samples not prepared in
  this paper.
- If the same sample/property/condition is already captured from a figure, keep
  the **figure** version and drop the duplicate table value (not the other way
  round). Table values may still be used to check figure readings.
- Apply footnote conditions (e.g. "measured at 30 °C" → `prop test temp (C)`).
- Values with "±" keep the mean only, and sample names must match the figures.

### 5. Chemistry
Run `chem`, Read the structure grid, and compare it with the paper's scheme
images (`context.json → chemistry.structure_images`) and the synthesis text.
Write a verified CSV with SALSA's samples.csv columns plus `confidence` and
`curation_notes`, then `chem-apply`.
- One row per sample that has data rows, with names identical to the data rows.
  Drop samples with no data rows.
- `smiles1..n`: repeat units with exactly two `[*]` attachment points, in the
  **final** (functionalized, cationic) form, without counter-ions.
  Copolymer/crosslinker units are separate smiles with mole fractions `c1..cn`
  summing to 1. Derive fractions from the sample naming/feed ratios and say how.
- Additives/fillers (COFs, inorganic particles, small molecules) are not in the
  repeat units: set `contains_add=1` and fill additive columns when known.
  Crosslinked samples: `contains_crosslink=1`.
- `confidence`: **1** = structure and composition fully specified by the paper;
  **0.5** = partly (e.g. crosslinker structure or ratio inferred, random
  copolymer placement); **0** = cannot be represented (e.g. filler-dominated
  composites, unknown functionalization).
- Fix every SMILES that does not match the scheme (cation type, substitution
  position, linker length, backbone). RDKit validity is necessary, not sufficient.

### 6. Finish
- Call `verdict`.
- Write `<run>/agent_curation/<safe_doi>/summary.md`: verdict, sources kept and
  rejected (one line each, with reason), key corrections made, and a
  **"Needs human review"** list of uncertain items.
- Reply with a short summary (≤15 lines) and the path to summary.md.
