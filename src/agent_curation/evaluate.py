"""Score an agent-curated run against the human-curated reference datasets.

Run with an env that has pandas + RDKit (e.g. the ``aem`` env)::

    python agent_curation/evaluate.py --agent-run <run_dir> \
        --ref agents/data/batch2.csv --ref-chem agents/data/batch2_chem.csv \
        [--dois 10.1016/... ...] [--out report_dir]

The human reference is not perfect (e.g. occasional calibration drift), so
value disagreements are listed per figure for spot-checking rather than being
treated as agent errors outright.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROP_PATTERNS = [
    ("conductivity_retention", r"(cond|σ|sigma).*(remain|retent|%)|(remain|retent).*(cond|σ)"),
    ("iec_retention", r"iec.*(remain|retent)|(remain|retent).*iec"),
    ("conductivity", r"cond|σ|sigma"),
    ("iec", r"\biec|ion exchange capacity"),
    ("water_uptake", r"water uptake|\bwu\b"),
    ("swelling", r"swell|\bsr\b|srv|\bsd\b"),
    ("tensile_strength", r"tensile|\bts\b"),
    ("elongation", r"elong|\beb\b"),
    ("modulus", r"modulus|\bym\b"),
    ("hydration", r"hydration|λ|lambda"),
]
COND_COLS = ["time(h)", "solvent type", "solvent molarity (M)", "degradation test temp (C)", "prop test temp (C)"]


def prop_cat(p: str) -> str:
    s = str(p).lower()
    for name, pat in PROP_PATTERNS:
        if re.search(pat, s):
            # Retention series are often converted to absolute values by the
            # curator, so score them against the base property.
            return {"conductivity_retention": "conductivity", "iec_retention": "iec"}.get(name, name)
    return "other:" + re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:30]


def norm_sample(s: str) -> str:
    return re.sub(r"[\s\-–_%()·.,]+", "", str(s)).lower().replace("oh", "oh")


def num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if not s or "not found" in s.lower() or s.lower() in ("none", "nan", "n/a"):
        return None
    m = re.match(r"^[-+]?\d*\.?\d+(e[-+]?\d+)?", s.replace("−", "-").replace(",", ""), re.IGNORECASE)
    return float(m.group(0)) if m else None


def cond_val(v):
    # "<default> Not Found" placeholders compare equal to the same explicit value.
    s0 = str(v).strip()
    if s0.lower().endswith("not found"):
        v = s0[: -len("not found")].strip()
    n = num(v)
    if n is not None:
        return round(n, 3)
    s = str(v).strip().lower()
    return None if (not s or "not found" in s or s in ("nan", "none", "n/a")) else s


def parent_source(name: str) -> str:
    n = str(name).replace(" ", "_")
    n = re.sub(r"\.(jpg|png|jpeg).*$", "", n)
    return n.split("_subfig_")[0]


def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if "batch" in df.columns:  # orig_time_batch1.csv: keep the curated batch rows, undo the column shift
        df = df[df["batch"].str.replace(" ", "_") == "batch_1"].copy()
        df["source_kind"] = df["source name"]
        df["source_name"] = df["note"]
    else:
        df["source_kind"] = df["source kind"]
        df["source_name"] = df["source name"]
    df = df.rename(columns={"deg test temp (C)": "degradation test temp (C)", "rh(%)": "relative humidity(%)"})
    return df


def load_agent(run: Path) -> pd.DataFrame:
    p = run / "aggregated" / "tables_figures.csv"
    df = pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame(columns=["DOI"])
    return df


def verdicts(run: Path) -> dict:
    out = {}
    for v in (run / "agent_curation").glob("*/verdict.json"):
        d = json.loads(v.read_text())
        out[d["doi"]] = d
    return out


def match_values(ref: pd.DataFrame, ag: pd.DataFrame, xcol: str):
    """Nearest-x matching within one (doi, sample, property) group."""
    pairs = []
    ax = np.array([num(v) if num(v) is not None else 0.0 for v in ag[xcol]])
    av = np.array([num(v) if num(v) is not None else np.nan for v in ag["value"]])
    span = max(1e-9, (np.nanmax(ax) - np.nanmin(ax)) if len(ax) else 1)
    used = set()
    for _, r in ref.iterrows():
        rx = num(r[xcol]) or 0.0
        rv = num(r["value"])
        if rv is None or not len(ax):
            pairs.append((rx, rv, None, None))
            continue
        d = np.abs(ax - rx)
        order = [i for i in np.argsort(d) if i not in used]
        if order and d[order[0]] <= max(0.06 * span, 1.0 if span > 5 else 0.05 * span + 1e-9):
            i = order[0]
            used.add(i)
            pairs.append((rx, rv, ax[i], av[i]))
        else:
            pairs.append((rx, rv, None, None))
    return pairs, len(ag) - len(used)


def xcol_for(group: pd.DataFrame) -> str:
    for c in ("time(h)", "prop test temp (C)"):
        vals = {cond_val(v) for v in group[c]}
        if len(vals) > 1:
            return c
    return "time(h)"


def eval_rows(ref: pd.DataFrame, ag: pd.DataFrame, dois, verd) -> dict:
    papers = []
    tot = {"ref_points": 0, "matched_points": 0, "agent_extra_points": 0, "rel_err": [],
           "cond_agree": {c: [0, 0] for c in COND_COLS}}
    for doi in dois:
        r = ref[ref.DOI == doi]
        a = ag[ag.DOI == doi] if len(ag) else ag
        r_inc = r["Sample Name"].notna().any()
        a_inc = len(a) > 0 and verd.get(doi, {}).get("decision") != "exclude"
        p = {"doi": doi, "ref_included": bool(r_inc), "agent_included": bool(a_inc),
             "agent_verdict_reason": verd.get(doi, {}).get("reason")}
        r = r[r["Sample Name"].notna()].copy()
        for df in (r, a):
            if len(df):
                df["_src"] = [f"{k}:{parent_source(n)}" for k, n in zip(df["source_kind"], df["source_name"])]
                df["_cat"] = df["property"].map(prop_cat)
                df["_sample"] = df["Sample Name"].map(norm_sample)
        rs = set(r["_src"]) if len(r) else set()
        as_ = set(a["_src"]) if len(a) else set()
        p["sources"] = {"both": sorted(rs & as_), "ref_only": sorted(rs - as_), "agent_only": sorted(as_ - rs)}
        groups = []
        if len(r):
            for (src, cat, smp), g in r.groupby(["_src", "_cat", "_sample"]):
                ga = a[(a["_src"] == src) & (a["_cat"] == cat) & (a["_sample"] == smp)] if len(a) else a
                xcol = xcol_for(g) if len(g) > 1 else "time(h)"
                pairs, extra = match_values(g, ga, xcol)
                errs = [abs(av - rv) / max(abs(rv), 1e-9) for rx, rv, ax_, av in pairs if av is not None and rv is not None and not np.isnan(av)]
                m = sum(1 for q in pairs if q[2] is not None)
                tot["ref_points"] += len(pairs)
                tot["matched_points"] += m
                tot["agent_extra_points"] += extra
                tot["rel_err"] += errs
                if m:
                    ga_idx = ga.reset_index(drop=True)
                    for c in COND_COLS:
                        if c == xcol:
                            continue
                        rv = {cond_val(v) for v in g[c]}
                        av_ = {cond_val(v) for v in ga_idx[c]}
                        tot["cond_agree"][c][1] += 1
                        tot["cond_agree"][c][0] += int(rv == av_)
                groups.append({"source": src, "property": cat, "sample": g["Sample Name"].iloc[0], "x": xcol,
                               "ref_n": len(g), "agent_n": len(ga), "matched": m,
                               "median_rel_err": round(float(np.median(errs)), 4) if errs else None,
                               "worst": sorted(([round(rx, 3), rv, None if ax_ is None else round(float(ax_), 3),
                                                 None if av is None else round(float(av), 4)] for rx, rv, ax_, av in pairs),
                                               key=lambda q: -abs((q[3] or 0) - (q[1] or 0)) / max(abs(q[1] or 1), 1e-9))[:3]})
        if len(a):
            ref_keys = {(g["_src"], g["_cat"], g["_sample"]) for _, g in r.iterrows()} if len(r) else set()
            p["agent_only_groups"] = sorted({f"{s} | {c} | {n}" for s, c, n, smp in zip(a["_src"], a["_cat"], a["Sample Name"], a["_sample"])
                                             if (s, c, smp) not in ref_keys})[:40]
        p["groups"] = groups
        papers.append(p)
    errs = np.array(tot["rel_err"]) if tot["rel_err"] else np.array([np.nan])
    summary = {
        "papers": len(dois),
        "paper_verdict_agreement": sum(p["ref_included"] == p["agent_included"] for p in papers),
        "source_recall": _ratio(sum(len(p["sources"]["both"]) for p in papers),
                                sum(len(p["sources"]["both"]) + len(p["sources"]["ref_only"]) for p in papers)),
        "source_precision": _ratio(sum(len(p["sources"]["both"]) for p in papers),
                                   sum(len(p["sources"]["both"]) + len(p["sources"]["agent_only"]) for p in papers)),
        "point_recall": _ratio(tot["matched_points"], tot["ref_points"]),
        "agent_points_unmatched": tot["agent_extra_points"],
        "value_median_rel_err": None if np.isnan(errs).all() else round(float(np.nanmedian(errs)), 4),
        "values_within_5pct": None if np.isnan(errs).all() else round(float(np.mean(errs <= 0.05)), 3),
        "values_within_10pct": None if np.isnan(errs).all() else round(float(np.mean(errs <= 0.10)), 3),
        "condition_agreement": {c: _ratio(*v) for c, v in tot["cond_agree"].items()},
    }
    return {"summary": summary, "papers": papers}


def _ratio(a, b):
    return {"n": a, "of": b, "rate": round(a / b, 3) if b else None}


def eval_chem(ref_path: Path, run: Path, dois) -> dict:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    ref = pd.read_csv(ref_path, encoding="utf-8-sig", dtype=str)
    if "batch" in ref.columns:
        ref = ref[ref["batch"].str.replace(" ", "_") == "batch_1"]

    def mol(s):
        return Chem.MolFromSmiles(s.replace("*", "[*]").replace("[[*]]", "[*]")) if isinstance(s, str) and s.strip() else None

    def canon(s):
        m = mol(s)
        return Chem.MolToSmiles(m) if m else None

    def sim(s1, s2):
        m1, m2 = mol(s1), mol(s2)
        if not (m1 and m2):
            return None
        f1, f2 = (AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in (m1, m2))
        return DataStructs.TanimotoSimilarity(f1, f2)

    out, agg = [], {"ref_samples": 0, "agent_samples": 0, "matched": 0, "smiles1_exact": 0, "smiles1_sim": [],
                    "all_units_exact": 0, "confidence_agree": 0, "confidence_n": 0}
    for doi in dois:
        r = ref[(ref.DOI == doi) & ref["Sample Name"].notna()]
        p = run / "chemical_composition" / doi.replace("/", "@") / "samples.csv"
        a = pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame(columns=["sample_name"])
        excluded = (run / "agent_curation" / doi.replace("/", "@") / "verdict.json")
        if excluded.exists() and json.loads(excluded.read_text()).get("decision") == "exclude":
            a = a.iloc[0:0]
        am = {norm_sample(n): row for n, row in zip(a["sample_name"], a.to_dict("records"))}
        agg["ref_samples"] += len(r)
        agg["agent_samples"] += len(a)
        rows = []
        for _, rr in r.iterrows():
            ar = am.get(norm_sample(rr["Sample Name"]))
            e = {"sample": rr["Sample Name"], "in_agent": ar is not None}
            if ar is not None:
                agg["matched"] += 1
                s_exact = canon(rr.get("smiles1")) == canon(ar.get("smiles1"))
                agg["smiles1_exact"] += int(s_exact)
                s = sim(rr.get("smiles1"), ar.get("smiles1"))
                if s is not None:
                    agg["smiles1_sim"].append(s)
                ru = sorted(filter(None, (canon(rr.get(f"smiles{i}")) for i in range(1, 5))))
                au = sorted(filter(None, (canon(ar.get(f"smiles{i}")) for i in range(1, 7))))
                agg["all_units_exact"] += int(ru == au)
                e.update({"smiles1_exact": s_exact, "smiles1_tanimoto": None if s is None else round(s, 3),
                          "units_exact": ru == au, "ref_smiles": ru, "agent_smiles": au})
                if rr.get("confidence") is not None and ar.get("confidence") is not None:
                    agg["confidence_n"] += 1
                    agg["confidence_agree"] += int(num(rr.get("confidence")) == num(ar.get("confidence")))
            rows.append(e)
        extra = sorted(set(am) - {norm_sample(n) for n in r["Sample Name"]})
        out.append({"doi": doi, "samples": rows, "agent_only_samples": extra})
    sims = agg.pop("smiles1_sim")
    agg["smiles1_mean_tanimoto"] = round(float(np.mean(sims)), 3) if sims else None
    return {"summary": agg, "papers": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-run", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-chem")
    ap.add_argument("--dois", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    run = Path(a.agent_run)
    ref = load_reference(Path(a.ref))
    dois = a.dois or sorted(ref.DOI.dropna().unique())
    if not a.dois and (run / "agent_curation" / "queue.json").exists():
        dois = json.loads((run / "agent_curation" / "queue.json").read_text())
    res = {"rows": eval_rows(ref, load_agent(run), dois, verdicts(run))}
    if a.ref_chem:
        res["chemistry"] = eval_chem(Path(a.ref_chem), run, dois)
    out = Path(a.out) if a.out else run / "agent_curation" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({"rows": res["rows"]["summary"], "chemistry": res.get("chemistry", {}).get("summary")}, indent=2))
    print(f"\nfull report: {out / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
