import sys
from pathlib import Path


UI_SRC = Path(__file__).resolve().parents[1] / "src" / "ui"
sys.path.insert(0, str(UI_SRC))

from figures_view import _latest_ocr_dataset


def test_latest_ocr_dataset_requires_final_dataset(tmp_path):
    safe_doi = "10.1000@test"
    output = (
        tmp_path
        / "results"
        / "0"
        / "output"
        / safe_doi
        / "Figure_1_subfig_0.png"
    )
    output.mkdir(parents=True)

    assert (
        _latest_ocr_dataset(
            workdir=tmp_path,
            safe_doi=safe_doi,
            image_name="Figure_1_subfig_0.png",
        )
        is None
    )

    final_csv = output / "final" / "final_dataset.csv"
    final_csv.parent.mkdir()
    final_csv.write_text("x,y\n1,2\n")

    assert _latest_ocr_dataset(
        workdir=tmp_path,
        safe_doi=safe_doi,
        image_name="Figure_1_subfig_0.jpg",
    ) == final_csv


def test_latest_ocr_dataset_does_not_accept_stale_older_run(tmp_path):
    safe_doi = "10.1000@test"
    old_csv = (
        tmp_path
        / "results"
        / "0"
        / "output"
        / safe_doi
        / "Figure_1.png"
        / "final"
        / "final_dataset.csv"
    )
    old_csv.parent.mkdir(parents=True)
    old_csv.write_text("x,y\n1,2\n")

    newest = tmp_path / "results" / "1" / "output" / safe_doi / "Figure_1.png"
    newest.mkdir(parents=True)

    assert (
        _latest_ocr_dataset(
            workdir=tmp_path, safe_doi=safe_doi, image_name="Figure_1.png"
        )
        is None
    )
