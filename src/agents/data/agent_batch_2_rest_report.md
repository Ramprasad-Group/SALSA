# Agent curation: redo_batch_2 remainder (intermediate report)

**Status:** stopped by request after 29 of 78 papers. Resume instructions are at the end.
**Run:** `src/ui_runs/users/william_local/workspaces/AEM_deg/agent_batch_2_rest`
**Settings:** custom instructions from the `new_aem_deg_ollama` preset. SALSA's automatic
stage (figure/table filtering, LineFormer/ChartDete, table rows, draft chemistry) was cloned
from `redo_batch_2` (OpenAI models), because `kimi-k2.6:cloud` needs an Ollama Pro plan.
`10.1016/j.electacta.2015.12.181` is out of scope, as you requested.

## Outputs

| File | Contents |
|---|---|
| `agent_batch_2_rest.csv` | 2,069 property rows for the 24 included papers, plus one DOI-only row for each of the 5 excluded papers (same convention as `batch2.csv`). |
| `agent_batch_2_rest_chem.csv` | 124 verified samples with `confidence`, `contains_add`, `contains_crosslink`, `curation_notes`; excluded papers carry the exclusion reason. |
| `agent_batch_2_rest_verdicts.csv` | Include/exclude decision and reason for each paper. |
| `agent_batch_2_rest_progress.json` | Completed, partial and remaining DOIs (for resuming). |
| Run: `agent_curation/<doi>/summary.md` | Per-paper report: sources kept/rejected, corrections, "needs human review". |
| Run: `agent_curation/<doi>/views/` | Every check image the agent looked at (calibration and point overlays, zooms, structure grids). |
| Run: `agent_curation/ledger.jsonl` | Every action with a reason (crop, calibrate, series, emit, reject, chem_apply, verdict). |
| Run: `interventions/` | Figure fixes as standard SALSA interventions, viewable in the SALSA UI. |
| Run: `chemical_composition/<doi>/samples.salsa.csv` | SALSA's original chemistry, for diffing against the verified `samples.csv`. |

SALSA's aggregation keeps only the preset's chemistry columns, which drops `confidence`, the
crosslink/additive flags and the notes. So `agent_batch_2_rest_chem.csv` is built from the
per-paper verified files instead of `aggregated/chemical_compositions.csv`.

## Summary of the 29 completed papers

- **Verdicts:** 24 included, 5 excluded:
  - LDH/chitosan composite (cej.2025.163583);
  - BN-filled QPPO with no degree of functionalization reported (colcom.2020.100265);
  - zirconium-phosphonate/PVA cation exchange membranes (desal.2011.03.048);
  - PBI review (desal.2024.117500);
  - alkali-doped non-ionic poly(oxadiazole) (elecom.2011.06.029).
- **Rows:** 1,655 figure rows from 80 figure/subfigure sources, and 414 table rows from 26 tables.
- **Rejected:** 43 figure sources and 10 tables.

### Figure work

SALSA's automatic digitization was rarely usable as delivered. Across the 29 papers:

| Intervention | Count | What went wrong in the automatic output |
|---|---|---|
| Manual point sets (`series --points`) | 76 figures, 200+ series | LineFormer traced fit/guide lines instead of markers, merged overlapping series, or produced smoothed curves. Most series were re-placed point by point from zooms. The `markers` colour detector failed on markers under about 5 px, which is common in these figures. |
| Series renamed | 76 figures | OCR garbled legend text ("QPAT", "SIO1", "HPST"), and legend↔series matching was often swapped. |
| Axis recalibration | 62 figures | Typical failures: log axes read as linear (compositesb.2018.11.084, electacta.2016.11.137); crops that picked up a neighbouring panel's tick labels; OCR misreads (one y-calibration was off by 14 units); dual-y-axis figures where both series were read on one axis. |
| Crops | 43 new subfigure crops in 21 figures | Missing or wrong automatic panel splits; dual-axis figures cropped twice, once per axis. |
| Series deleted | 12 figures | Commercial/reference membranes (FAA-3, AMX, A201, Alkymer), duplicate traces, fit lines. |

Further corrections the agents made consistently:
- **Retention conversion:** retention % series were converted to absolute conductivity using the starting value at the degradation-test temperature (your rule). The multiplier is recorded in every case.
- **Arrhenius plots:** 1000/T axes were converted to °C. Arrhenius replots of data already digitized as conductivity vs T were rejected as duplicates.
- **Snapping:** x values were snapped to nominal steps (24 h, 10 °C).
- **Values stated in the text:** where the paper gives a number, it replaced the reading (end points, 80 °C values). Readings typically agreed to within 0.1–1 unit before replacement.
- **Tables:** pruned to measured values. Theoretical IEC, λ, Ea and literature-comparison rows were dropped. Where a table duplicates figure data, the figure rows were kept and the table copy dropped. Footnote conditions were applied (e.g. "measured at 30 °C").

### Chemistry work

All 24 included papers got verified chemistry. In **18 of 24 the SMILES were changed**, because SALSA's draft didn't match the paper's scheme. Recurring error types:

- **Wrong substitution positions:**
  - 2,5- instead of 2,6-dimethyl PPO (2 papers);
  - fluorene linked 2,6 instead of 2,7 (2 papers);
  - CH₂NMe₃⁺ next to the isopropylidene instead of ortho to the ether O;
  - BPPO attached ortho instead of para.
- **Missing or extra structure:**
  - units missing entirely (SAF/DF units in cej.2024.158978);
  - duplicated linkers (cej.2025.164169);
  - a missing methyl on the piperidinium N (2 papers);
  - an extra CH₂CH₂ spacer;
  - PDDA missing its backbone CH₂ groups.
- **Composition errors:**
  - mass fractions used as mole fractions;
  - bromination degree per methyl group not converted to per repeat unit;
  - fractions that didn't sum to 1.
- **Housekeeping:** counter-ions in the SMILES were removed. Samples with no data rows were dropped (139 SALSA samples down to 124), and names were aligned with the data rows.

Agents checked compositions against the reported IEC where possible. In about half the papers the
composition reproduced the theoretical IEC exactly, and remaining gaps are listed in the summaries.
Confidence: 24 samples at **1**, 96 at **0.5** (typically inferred fractions or random copolymer placement), and 4 at **0** (filler-dominated composites).
Crosslinking follows your rule: added crosslinkers go in the additive columns with the backbone summing to 1; crosslinking monomers are units. 59 samples are crosslinked, and 37 have additives.

## Needs your decision

1. **Counter-ion forms with no OH⁻ version.** Agents handled this inconsistently:
   - electacta.2016.11.137 dropped Cl⁻/HCO₃⁻ conductivity-vs-temperature data (leaving no σ(T) at all) but kept HCO₃⁻ stability data;
   - electacta.2018.03.117 kept carbonate-form conductivity.

   Proposed rule: drop non-OH⁻ forms only when the same measurement exists in OH⁻ form.
2. **Scope questions:**
   - alkali-doped non-ionic polymers (PBI/POD in KOH/NaOH); excluded so far;
   - crown-ether/Na⁺ complexes as the ionic site (electacta.2017.11.049); included so far.
3. **Swelling vs time.** electacta.2013.07.182 kept it by analogy with before/after bars, though the figure criteria don't list it.
4. **Crosslink representation edge case.** colsurfa.2024.135561 put the crosslinker both as a reacted backbone unit and in the additive columns.
5. **Measurement temperature of stability-test conductivity.** It is often unstated; agents assumed the degradation temperature. This is consistent with your rule, but flagged per paper.
6. **Outside knowledge used, with confidence lowered:**
   - solution densities to convert wt% KOH/NaOH to molarity;
   - the composition of a Sustainion-type additive (cej.2025.162561);
   - a standard PDDA structure.

## Known gaps and caveats

- **Data only in the text or SI can't be emitted** (there is no "text" source type). Tensile values after ageing, IEC values in prose and Ea are listed per paper under "needs human review".
- **SALSA's figure filter drops relevant figures before agents see them.** In electacta.2017.09.126, Figs 3–5 and 7 never reached the run. A pass that shows agents the rejected figures would close this gap.
- **One paper bypassed the SALSA tools.** In cej.2025.161384 (early in the run, before the queue fix) the agent digitized with its own pixel analysis, so its figure fixes are not SALSA interventions. Its rows and reasoning are recorded.
- **Partial papers:** 4 papers were stopped partway, and their partial interventions/emits remain in the run: electacta.2020.135757, electacta.2021.138249, electacta.2021.138919, electacta.2021.138962. They are **not** in the exported CSVs.
- **Tool changes during the run:**
  - `rerun --light` cut new-crop reruns from up to 30 min to about 3 s;
  - full reruns are serialized;
  - non-breaking-space figure ids are handled;
  - `chem-apply` now rejects ragged CSVs, after one misaligned file had to be repaired (electacta.2018.01.064; the fix is logged in the ledger).

## Per-paper results

| DOI | Verdict | Fig rows | Table rows | Crops | Calibrations | Manual series | Chem samples | Confidence |
|---|---|---|---|---|---|---|---|---|
| cej.2024.158978 | include | 72 | 0 | 3 | 4 | 17 | 5 | 0.5, 1 |
| cej.2025.159693 | include | 90 | 0 | 4 | 2 | 12 | 3 | 0.5 |
| cej.2025.161384 | include | 65 | 4 | 4 | 0* | 0* | 4 | 0.5, 1 |
| cej.2025.162561 | include | 44 | 77 | 3 | 3 | 11 | 11 | 0.5, 1 |
| cej.2025.163583 | exclude | – | – | – | – | – | – | – |
| cej.2025.164169 | include | 104 | 4 | 5 | 4 | 20 | 4 | 1 |
| cej.2025.168673 | include | 76 | 0 | 0 | 1 | 16 | 4 | 0.5 |
| cjche.2018.05.015 | include | 78 | 14 | 2 | 2 | 14 | 5 | 1 |
| colcom.2020.100265 | exclude | – | – | – | – | – | – | – |
| colcom.2022.100584 | include | 154 | 50 | 0 | 2 | 19 | 8 | 0.5 |
| colsurfa.2024.135561 | include | 82 | 12 | 0 | 4 | 14 | 4 | 0.5 |
| compositesb.2018.11.084 | include | 90 | 60 | 0 | 2 | 12 | 6 | 0.5 |
| desal.2011.03.048 | exclude | – | – | – | – | – | – | – |
| desal.2023.116600 | include | 48 | 20 | 0 | 0 | 12 | 4 | 1 |
| desal.2024.117500 | exclude | – | – | – | – | – | – | – |
| elecom.2011.06.029 | exclude | – | – | – | – | – | – | – |
| electacta.2007.03.034 | include | 24 | 0 | 0 | 2 | 4 | 2 | 0.5 |
| electacta.2013.07.182 | include | 27 | 20 | 2 | 2 | 3 | 4 | 0.5 |
| electacta.2014.11.043 | include | 220 | 0 | 9 | 10 | 44 | 14 | 0.5, 1 |
| electacta.2016.04.078 | include | 20 | 6 | 2 | 4 | 4 | 1 | 0.5 |
| electacta.2016.04.176 | include | 83 | 50 | 0 | 2 | 13 | 6 | 0.5 |
| electacta.2016.10.041 | include | 30 | 10 | 0 | 1 | 5 | 5 | 0.5 |
| electacta.2016.11.137 | include | 7 | 4 | 0 | 1 | 2 | 4 | 0.5 |
| electacta.2017.07.165 | include | 6 | 0 | 0 | 1 | 2 | 1 | 0.5 |
| electacta.2017.09.126 | include | 61 | 28 | 3 | 4 | 11 | 10 | 0, 0.5 |
| electacta.2017.10.043 | include | 84 | 21 | 0 | 3 | 13 | 7 | 0.5, 1 |
| electacta.2017.11.049 | include | 63 | 15 | 4 | 3 | 10 | 3 | 0.5 |
| electacta.2018.01.064 | include | 52 | 12 | 0 | 2 | 8 | 4 | 0, 1 |
| electacta.2018.03.117 | include | 75 | 7 | 2 | 3 | 13 | 5 | 0.5 |

\* cej.2025.161384 digitized outside the CLI; see caveats.

## Resuming

The 49 remaining DOIs (the 4 partial papers plus 45 not started) are in `agent_batch_2_rest_progress.json` → `remaining_including_partial`.
In Claude Code, ask: *"Resume the agent curation of agent_batch_2_rest from
`src/agents/data/agent_batch_2_rest_progress.json`."* Each paper gets one agent following
`src/agent_curation/GUIDE.md`, with 5 agents in parallel. For the 4 partial papers, the agent
should first read its own DOI's ledger entries and continue from them. At the end, re-run `aggregate`
and the export so that the CSVs above cover all 78 papers.

Approximate cost so far: 50k–160k agent tokens per included paper, and about 45k per excluded paper.
