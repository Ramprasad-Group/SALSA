# Chemistry Text Pipeline (Developer Notes)

This module adds an optional sample-centric pathway:

`literature description -> chemistry schema -> canonical chemistry text -> embedding`

## How it is enabled

The pipeline is invoked from `run_chemical_composition_extraction(...)` when:

- `chemistry_text_config.enable_chemistry_text_pipeline = True`

UI wiring exists in Streamlit under:

- **Chemistry text embedding pipeline** expander.

## Output layout

Artifacts are stored under:

```
<run_dir>/chemistry_text/<safe_doi>/<sample_id>/
  raw_description.json
  chemistry_schema.json
  canonical_chemistry_text.json
  chemistry_embedding.npy
  chemistry_embedding_metadata.json
```

Aggregate outputs:

```
<run_dir>/chemistry_text/<safe_doi>/
  chemistry_schema_all.jsonl
  canonical_text_all.jsonl
  embedding_index.jsonl
```

Embedding cache:

```
<run_dir>/chemistry_text/_cache/
  <cache_key>.npy
  <cache_key>.json
```

## Design choices

1. **Rule-based first**: deterministic baseline extraction works with no API key.
2. **Optional LLM extractor**: available via config (`llm`, `hybrid`).
3. **Conservative chemistry interpretation**: unknown fields stay unknown; no forced exact structures.
4. **Target leakage control**: optional sentence filter removes performance-heavy language before schema extraction.
5. **Deterministic canonical text**: stable sentence ordering for consistent embeddings.

## Embedding backends

- `local`
  - `hash-embedding-v1` (always available deterministic fallback).
  - `sentence-transformers/<model>` if `sentence-transformers` is installed.
- `api`
  - currently falls back to deterministic hash embedding unless an API path is later added.

## Extension points

- Add more domain vocab patterns in `chemistry_text_pipeline.py`.
- Extend canonicalization dictionaries for new abbreviations.
- Replace fallback API branch in `get_text_embedding(...)` with project-preferred embedding endpoint.
