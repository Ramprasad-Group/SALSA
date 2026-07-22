import csv
import json
import os

from run_aggregation import aggregate_run


def _csv(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_aggregate_run_combines_products_with_provenance(tmp_path):
    run = tmp_path / "run"
    cache = run / "complete_dataset/row_cache"
    cache.mkdir(parents=True)
    (cache / "emit_rows_table_example.json").write_text(json.dumps({"rows": [
        {"DOI": "10.1000/x", "source_kind": "table", "source_name": "Table_1", "sample": "a"}
    ]}))
    (cache / "emit_rows_figure_example.json").write_text(json.dumps({"rows": [
        {"DOI": "10.1000/x", "source_kind": "figure", "source_name": "Figure_2", "value": 3}
    ]}))
    _csv(run / "chemical_composition/10.1000@x/samples.csv", "DOI,sample_name\n10.1000/x,A\n")
    result = aggregate_run(run)
    assert (result.data_rows, result.composition_rows) == (2, 1)
    with result.data_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["source_type"] for row in rows} == {"table", "figure"}
    assert {row["DOI"] for row in rows} == {"10.1000/x"}


def test_newest_figure_dataset_replaces_original_even_with_extension(tmp_path):
    run = tmp_path / "run"
    cache = run / "complete_dataset/row_cache"
    cache.mkdir(parents=True)
    old = cache / "emit_rows_figure_old.json"
    new = cache / "emit_rows_figure_new.json"
    old.write_text(json.dumps({"rows": [{
        "DOI": "10.1000/x", "source_kind": "figure",
        "source_name": "Figure_1", "value": "original",
    }]}))
    new.write_text(json.dumps({"rows": [{
        "DOI": "10.1000/x", "source_kind": "figure",
        "source_name": "Figure_1.jpg", "value": "user-modified",
    }]}))
    os.utime(old, (100, 100))
    os.utime(new, (200, 200))

    result = aggregate_run(run)
    with result.data_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert result.data_rows == 1
    assert rows[0]["value"] == "user-modified"


def test_obsolete_cache_columns_do_not_leak_and_configured_order_is_preserved(tmp_path):
    run = tmp_path / "run"
    cache = run / "complete_dataset/row_cache"
    cache.mkdir(parents=True)
    (run / "custom_instructions.json").write_text(json.dumps({
        "output_columns": ["Sample Name", "property", "value", "DOI", "source_kind", "source_name"]
    }))
    old = cache / "emit_rows_figure_old.json"
    new = cache / "emit_rows_figure_new.json"
    old.write_text(json.dumps({"rows": [{
        "DOI": "10.1000/x", "source_kind": "figure", "source_name": "Figure_1",
        "Sample Name": "A", "dependent value": "obsolete",
    }]}))
    new.write_text(json.dumps({"rows": [{
        "DOI": "10.1000/x", "source_kind": "figure", "source_name": "Figure_1.jpg",
        "Sample Name": "A", "property": "conductivity", "value": "12",
    }]}))
    os.utime(old, (100, 100))
    os.utime(new, (200, 200))

    result = aggregate_run(run)
    with result.data_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames[:6] == [
        "Sample Name", "property", "value", "DOI", "source_kind", "source_name"
    ]
    assert reader.fieldnames == [
        "Sample Name", "property", "value", "DOI", "source_kind", "source_name"
    ]
    assert "dependent value" not in reader.fieldnames
    assert rows[0]["value"] == "12"


def test_configured_composition_columns_are_authoritative(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "custom_instructions.json").write_text(json.dumps({
        "chemical_composition_columns": ["DOI", "sample_name", "formula"]
    }))
    _csv(
        run / "chemical_composition/10.1000@x/samples.csv",
        "DOI,sample_name,formula,notes\n10.1000/x,A,C2H4,omit me\n",
    )

    result = aggregate_run(run)
    with result.composition_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["DOI", "sample_name", "formula"]
    assert rows == [{"DOI": "10.1000/x", "sample_name": "A", "formula": "C2H4"}]
