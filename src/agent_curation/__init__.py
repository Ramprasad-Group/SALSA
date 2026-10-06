"""Agent-driven curation toolkit for SALSA runs.

Exposes the same intervention operations a human performs in the Streamlit
UI (crop, axis calibration, series masks/points, series naming/deletion,
re-extraction, dataset row emission) as a CLI that coding agents can drive.
All outputs are written in the UI's on-disk format so agent-curated runs can
be opened in the SALSA UI and aggregated with ``run_aggregation``.
"""
