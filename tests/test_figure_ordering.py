import sys
from pathlib import Path

UI_SRC = Path(__file__).resolve().parents[1] / "src" / "ui"
sys.path.insert(0, str(UI_SRC))

from discover import _natural_sort_key, discover_run


def test_natural_sort_key_orders_embedded_numbers_numerically():
    names = ["Figure_1", "Figure_12", "Figure_2", "Figure_9", "Figure_3"]
    paths = [Path(n) for n in names]

    ordered = sorted(paths, key=_natural_sort_key)

    assert [p.name for p in ordered] == [
        "Figure_1", "Figure_2", "Figure_3", "Figure_9", "Figure_12",
    ]


def test_discover_run_orders_figures_numerically(tmp_path):
    safe_doi = "10.6028@jres.102.018"
    fig_dir = tmp_path / "figures_tables" / "figures" / safe_doi
    fig_dir.mkdir(parents=True)
    for name in ["Figure_1", "Figure_2", "Figure_3", "Figure_9", "Figure_12"]:
        (fig_dir / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    reparsed = tmp_path / "reparsed" / safe_doi
    reparsed.mkdir(parents=True)
    (reparsed / "parsed.json").write_text("{}")

    rr = discover_run(tmp_path)
    doi_item = next(d for d in rr.dois if d.safe_doi == safe_doi)

    assert [f.id for f in doi_item.figures] == [
        "Figure_1", "Figure_2", "Figure_3", "Figure_9", "Figure_12",
    ]
