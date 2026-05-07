# SALSA/src/ui/schemas.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

@dataclass
class FigureItem:
    id: str
    caption: str = ""
    figure_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    filtered: bool = False
    extracted_xy_paths: list[Path] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    

@dataclass
class TableItem:
    id: str
    caption: str = ""
    table_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    filtered: bool = False
    extracted_table_path: Optional[Path] = None
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class DoiResult:
    doi: str
    safe_doi: str
    fulltext_path: Optional[Path] = None
    parsed_json_path: Optional[Path] = None
    figures: list[FigureItem] = field(default_factory=list)
    tables: list[TableItem] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    dois: list[DoiResult]
