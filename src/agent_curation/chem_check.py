"""RDKit sanity checks for a paper's chemical-composition samples.

Runs in an environment with RDKit (SALSA's py3.8 env does not have it);
``python -m agent_curation chem`` locates it via $SALSA_RDKIT_PYTHON.

Prints per-sample findings as JSON and renders all structures to
``<run>/agent_curation/<doi>/views/chem_structures.png`` so an agent can
compare them with the paper's scheme images.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_smiles(s: str) -> dict:
    out = {"smiles": s}
    m = Chem.MolFromSmiles(s)
    if m is None:
        out["error"] = "RDKit cannot parse"
        return out
    frags = Chem.GetMolFrags(m, asMols=True)
    stars = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 0)
    charge = Chem.GetFormalCharge(m)
    out.update({
        "canonical": Chem.MolToSmiles(m),
        "formula": rdMolDescriptors.CalcMolFormula(m),
        "mw_repeat_unit": round(Descriptors.MolWt(m), 2),
        "n_attachment_points": stars,
        "net_formal_charge": charge,
        "n_fragments": len(frags),
        "cationic_groups": sum(1 for a in m.GetAtoms() if a.GetFormalCharge() > 0),
    })
    flags = []
    if stars not in (2,) and stars != 0:
        flags.append(f"{stars} attachment points ([*]); a linear repeat unit has 2 (crosslink/branch units may differ)")
    if stars == 0:
        flags.append("no [*] attachment points: not a repeat unit")
    if charge > 0 and not any(a.GetFormalCharge() < 0 for a in m.GetAtoms()):
        flags.append("cationic with no counter-ion (fine if the convention omits counter-ions)")
    if out["cationic_groups"] == 0:
        flags.append("no cationic group: check whether this unit should carry the ionic group")
    out["flags"] = flags
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--doi", required=True)
    ap.add_argument("--csv")
    a = ap.parse_args()
    run, sdoi = Path(a.run), a.doi.replace("/", "@")
    path = Path(a.csv) if a.csv else run / "chemical_composition" / sdoi / "samples.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    report, mols, legends = [], [], []
    for r in rows:
        name = r.get("sample_name") or r.get("Sample Name") or "?"
        entry = {"sample": name, "units": [], "issues": []}
        cs = []
        for i in range(1, 7):
            s = (r.get(f"smiles{i}") or "").strip()
            c = _f(r.get(f"c{i}"))
            if not s:
                continue
            chk = check_smiles(s)
            chk["c"] = c
            entry["units"].append(chk)
            cs.append(c)
            m = Chem.MolFromSmiles(s)
            if m is not None:
                mols.append(m)
                legends.append(f"{name} | smiles{i} c={c}")
        if not entry["units"]:
            entry["issues"].append("no SMILES given")
        valid_c = [c for c in cs if c is not None]
        if valid_c and abs(sum(valid_c) - 1) > 0.02 and abs(sum(valid_c) - 100) > 2:
            entry["issues"].append(f"composition c1..cn sums to {sum(valid_c):.4g} (expected 1 or 100)")
        if len(valid_c) != len(cs):
            entry["issues"].append("some units have no composition value")
        for i in range(1, 4):
            s = (r.get(f"additive_smiles{i}") or "").strip()
            if s and Chem.MolFromSmiles(s) is None:
                entry["issues"].append(f"additive_smiles{i} unparseable: {s}")
        report.append(entry)
    views = run / "agent_curation" / sdoi / "views"
    views.mkdir(parents=True, exist_ok=True)
    img_path = None
    if mols:
        img = Draw.MolsToGridImage(mols[:24], molsPerRow=3, subImgSize=(420, 320), legends=legends[:24])
        img_path = views / ("chem_structures.png" if not a.csv else f"chem_structures_{Path(a.csv).stem}.png")
        img.save(str(img_path))
    print(json.dumps({"samples_csv": str(path), "n_samples": len(rows), "structures_image": str(img_path) if img_path else None,
                      "samples": report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
