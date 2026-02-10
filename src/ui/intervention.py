from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import shutil
import re
import numpy as np
from PIL import Image



@dataclass
class InterventionItem:
    safe_doi: str
    figure_id: str
    figure_dir: Path              # .../output/<doi>/<figure>/
    final_dir: Path               # .../output/<doi>/<figure>/final/
    source_image: Optional[Path]  # original filtered figure image path (from figures_tables)
    reason: str

def latest_plextract_run_dir(plextract_workdir: Path) -> Optional[Path]:
    results_root = plextract_workdir / "results"
    if not results_root.exists():
        return None
    run_dirs = [p for p in results_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: int(p.name), reverse=True)
    return run_dirs[0]

def find_interventions(run_dir: Path, plextract_workdir_name: str = "plextract_work") -> list[InterventionItem]:
    plextract_workdir = run_dir / plextract_workdir_name
    latest = latest_plextract_run_dir(plextract_workdir)
    """
    run_dir = <workspace>/<run_id> (the UI run workspace)
    expects plextract outputs in: run_dir/plextract_work/results/<n>/output/<safe_doi>/<figure>/final/
    """
    # plextract_workdir = run_dir / "plextract_work"
    # latest = latest_plextract_run_dir(plextract_workdir)
    if latest is None:
        return []

    output_root = latest / "output"
    if not output_root.exists():
        return []

    # Optional: link back to original filtered images (what user will crop)
    filtered_fig_root = run_dir / "figures_tables" / "figures"

    items: list[InterventionItem] = []
    for doi_dir in output_root.iterdir():
        if not doi_dir.is_dir():
            continue
        safe_doi = doi_dir.name

        for fig_dir in doi_dir.iterdir():
            if not fig_dir.is_dir():
                continue
            figure_id = fig_dir.name
            final_dir = fig_dir / "final"
            if not final_dir.exists():
                items.append(InterventionItem(
                    safe_doi=safe_doi,
                    figure_id=figure_id,
                    figure_dir=fig_dir,
                    final_dir=final_dir,
                    source_image=_try_find_source_image(filtered_fig_root, safe_doi, figure_id),
                    reason="Missing final/ directory",
                ))
                continue

            final_image = final_dir / "final_image.png"
            unknown_units = final_dir / "unknown_units.json"

            reasons = []
            if not final_image.exists():
                reasons.append("final_image.png missing")
            if unknown_units.exists():
                reasons.append("unknown_units.json present")

            if reasons:
                items.append(InterventionItem(
                    safe_doi=safe_doi,
                    figure_id=figure_id,
                    figure_dir=fig_dir,
                    final_dir=final_dir,
                    source_image=_try_find_source_image(filtered_fig_root, safe_doi, figure_id),
                    reason="; ".join(reasons),
                ))
    return items

def _try_find_source_image(filtered_fig_root: Path, safe_doi: str, figure_id: str) -> Optional[Path]:
    d = filtered_fig_root / safe_doi
    if not d.exists():
        return None

    # figure_id might be "Figure_10", "Figure_10.jpg", or "Figure_10_subfig_0.png"
    fig_path = Path(figure_id)
    stem = fig_path.stem
    suffix = fig_path.suffix.lower()

    # Also try the parent figure stem when subfigures are present.
    base_stem = re.sub(r"_subfig_\d+$", "", stem)
    candidate_stems = [stem]
    if base_stem != stem:
        candidate_stems.append(base_stem)

    exts = [".png", ".jpg", ".jpeg", ".gif", ".webp"]

    # 1) If figure_id already has a suffix, try exact
    if suffix in exts:
        p = d / figure_id
        if p.exists():
            return p

    # 2) Try by candidate stem + known extensions
    for s in candidate_stems:
        for ext in exts:
            p = d / f"{s}{ext}"
            if p.exists():
                return p

    # 3) Metadata-based match (supports renamed/hashed figure directories)
    for meta in d.glob("*_metadata.json"):
        try:
            row = json.loads(meta.read_text())
        except Exception:
            continue

        clean_label = str(row.get("clean_label", "")).strip()
        if not clean_label:
            continue
        if clean_label not in candidate_stems and all(not s.startswith(clean_label) for s in candidate_stems):
            continue

        p_from_meta = Path(str(row.get("figure_file_path", "")).strip())
        if p_from_meta and p_from_meta.exists():
            return p_from_meta

        for ext in exts:
            p = d / f"{clean_label}{ext}"
            if p.exists():
                return p

    # 4) Fallback: any file whose stem matches one of our candidates
    hits = [
        p
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in exts and p.stem in candidate_stems
    ]
    if hits:
        return hits[0]

    # 5) Last resort: if only one image exists for this DOI folder, use it.
    image_files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if len(image_files) == 1:
        return image_files[0]

    return hits[0] if hits else None

@dataclass
class SeriesItem:
    series_index: int
    mask_path: Path
    centroid: Tuple[float, float]


def load_series_catalog(lineformer_dir: Path) -> List[SeriesItem]:
    cat = lineformer_dir / "series_catalog.json"
    if not cat.exists():
        raise FileNotFoundError(f"Missing: {cat}")
    raw = json.loads(cat.read_text())
    out: List[SeriesItem] = []
    for r in raw:
        out.append(
            SeriesItem(
                series_index=int(r["series_index"]),
                mask_path=Path(r["mask_path"]),
                centroid=(float(r["centroid"][0]), float(r["centroid"][1])),
            )
        )
    return out


def get_lineformer_dir(item_figure_dir: Path) -> Path:
    # item.figure_dir points to .../<safe_doi>/<figure_id>/
    return item_figure_dir / "lineformer"


def intervention_dir(run_dir: Path, safe_doi: str, figure_id: str) -> Path:
    return run_dir / "interventions" / safe_doi / figure_id


def override_lineformer_dir(run_dir: Path, safe_doi: str, figure_id: str) -> Path:
    return intervention_dir(run_dir, safe_doi, figure_id) / "lineformer"


def mapping_json_path(run_dir: Path, safe_doi: str, figure_id: str) -> Path:
    return intervention_dir(run_dir, safe_doi, figure_id) / "mapping.json"


def series_catalog_override_path(run_dir: Path, safe_doi: str, figure_id: str) -> Path:
    return override_lineformer_dir(run_dir, safe_doi, figure_id) / "series_catalog.override.json"


def load_mapping(run_dir: Path, safe_doi: str, figure_id: str) -> Dict[str, Any]:
    p = mapping_json_path(run_dir, safe_doi, figure_id)
    if p.exists():
        return json.loads(p.read_text())
    # default mapping: series_index -> "Series {i}"
    return {"series_names": {}, "deleted": []}


def save_mapping(run_dir: Path, safe_doi: str, figure_id: str, obj: Dict[str, Any]) -> Path:
    d = intervention_dir(run_dir, safe_doi, figure_id)
    d.mkdir(parents=True, exist_ok=True)
    p = mapping_json_path(run_dir, safe_doi, figure_id)
    p.write_text(json.dumps(obj, indent=2))
    return p


def save_mask_override(
    run_dir: Path,
    safe_doi: str,
    figure_id: str,
    series_index: int,
    mask_img: Image.Image,
) -> Path:
    d = override_lineformer_dir(run_dir, safe_doi, figure_id)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"mask_series_{series_index}.png"
    mask_img.save(out)
    return out


def write_series_catalog_override(
    run_dir: Path,
    safe_doi: str,
    figure_id: str,
    series_items: List[SeriesItem],
    override_mask_paths: Dict[int, Path],
) -> Path:
    """
    Writes a series_catalog.override.json where mask_path points to overrides if present,
    otherwise original mask_path.
    """
    d = override_lineformer_dir(run_dir, safe_doi, figure_id)
    d.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in series_items:
        mp = override_mask_paths.get(s.series_index, s.mask_path)
        rows.append(
            {
                "series_index": s.series_index,
                "mask_path": str(mp),
                "centroid": [s.centroid[0], s.centroid[1]],
            }
        )
    p = series_catalog_override_path(run_dir, safe_doi, figure_id)
    p.write_text(json.dumps(rows, indent=2))
    return p


def list_override_masks(run_dir: Path, safe_doi: str, figure_id: str) -> Dict[int, Path]:
    d = override_lineformer_dir(run_dir, safe_doi, figure_id)
    if not d.exists():
        return {}
    out: Dict[int, Path] = {}
    for p in d.glob("mask_series_*.png"):
        try:
            idx = int(p.stem.split("_")[-1])
            out[idx] = p
        except Exception:
            continue
    return out


def apply_overrides_to_plextract(item_figure_dir: Path, run_dir: Path, safe_doi: str, figure_id: str) -> None:
    """
    Copy override masks into <figure_dir>/lineformer/ and replace series_catalog.json
    with the override one (after filtering deleted series if requested).
    Also handles newly created series from interventions.
    """
    lf_dir = get_lineformer_dir(item_figure_dir)
    if not lf_dir.exists():
        raise FileNotFoundError(f"Missing lineformer dir: {lf_dir}")

    # load original catalog (for centroids etc.)
    series_items = load_series_catalog(lf_dir)

    # load mapping (delete list and series names)
    mapping = load_mapping(run_dir, safe_doi, figure_id)
    deleted = set(int(x) for x in mapping.get("deleted", []))

    # load overrides
    override_masks = list_override_masks(run_dir, safe_doi, figure_id)

    # copy masks into plextract lineformer dir
    for idx, p in override_masks.items():
        dest = lf_dir / p.name
        shutil.copy2(p, dest)

    # Track which series indices we've seen from the original catalog
    original_indices = {s.series_index for s in series_items}

    # write new series_catalog.json (mask_path must point to files in lf_dir)
    rows = []

    # First, process existing series from the original catalog
    for s in series_items:
        if s.series_index in deleted:
            continue
        # if overridden, point to lf_dir/mask_series_{i}.png else keep existing
        mp = lf_dir / f"mask_series_{s.series_index}.png"
        if mp.exists():
            rows.append(
                {
                    "series_index": s.series_index,
                    "mask_path": str(mp),
                    "centroid": [s.centroid[0], s.centroid[1]],
                }
            )
        else:
            rows.append(
                {
                    "series_index": s.series_index,
                    "mask_path": str(s.mask_path),
                    "centroid": [s.centroid[0], s.centroid[1]],
                }
            )

    # Now, add newly created series (those with override masks but not in original catalog)
    for idx in override_masks:
        if idx not in original_indices and idx not in deleted:
            # This is a newly created series - calculate centroid from the mask
            mask_path = lf_dir / f"mask_series_{idx}.png"
            if mask_path.exists():
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_arr = np.array(mask_img)
                    y_coords, x_coords = np.where(mask_arr > 0)
                    if len(x_coords) > 0:
                        centroid_x = float(np.mean(x_coords))
                        centroid_y = float(np.mean(y_coords))
                    else:
                        centroid_x, centroid_y = 0.0, 0.0

                    rows.append(
                        {
                            "series_index": idx,
                            "mask_path": str(mask_path),
                            "centroid": [centroid_x, centroid_y],
                        }
                    )
                except Exception:
                    # If we can't process the mask, skip it
                    pass

    (lf_dir / "series_catalog.json").write_text(json.dumps(rows, indent=2))
