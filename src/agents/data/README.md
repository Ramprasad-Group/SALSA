# Human-curated reference data (AEM_deg)

Final datasets after human review of SALSA output. Used as ground truth when
evaluating agent-driven curation (`src/agent_curation`).

| File | Workspace run | Contents |
|---|---|---|
| `orig_time_batch1.csv` | `batch_1` | Property rows. `batch == "batch_1"` rows (56 DOIs) are the curated batch_1 output. `batch == "original"` rows (61 DOIs) are an older, fully manual dataset and are mostly not in the workspace. In `batch_1` rows the `source kind` column holds a duplicate DOI, so `source name` holds the kind and `note` holds the source name. |
| `orig_time_batch1_chem.csv` | `batch_1` / `batch_1_chem` | Chemistry rows. `batch == "batch 1"` rows are the curated batch_1 output. |
| `batch2.csv` | `redo_batch_2` | Property rows for the first 20 papers (alphabetical) of redo_batch_2. A row with only a DOI means the paper was reviewed and excluded. |
| `batch2_chem.csv` | `redo_batch_2` | Chemistry rows for the same papers. Includes `confidence` and `my notes`. |
| `batch1_skip_notes.csv` | `batch_1` | Review log for the first 24 batch_1 papers. Papers with a note and no rows in the reference were skipped. |

## Curation conventions inferred from the data

These are inferences from diffs against SALSA's output, not stated rules.

- **Paper exclusion** (`my notes`, skip notes): review paper; no chemical
  schematic plus additives/crosslinks; no relevant data / missing data;
  "dont trust"; "functionalization degree" (degree of functionalization not
  reported, so the composition can't be specified); "weird".
- **Chemistry `confidence`**: 1 = structure and composition fully specified;
  0.5 = partly specified (e.g. crosslinker or ratio uncertain);
  0 = not representable (e.g. inorganic/COF-filled composites).
- **Values** are rounded to the precision readable from the figure or table.
- **The x-variable goes into the matching condition column**, e.g. the temperatures
  of an Arrhenius plot go into `prop test temp (C)`.
- **Retention (%) series are converted to absolute values** when the initial
  value is reported.
- **Tables are pruned** to measured (not theoretical) values and to properties
  not already covered by figures. Reference samples from other papers are dropped.
