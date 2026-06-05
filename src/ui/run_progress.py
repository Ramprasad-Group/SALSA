from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_STATUS_FILE_ENV = "SALSA_RUN_STATUS_FILE"
RUN_STATUS_FILE_NAME = "run_status.json"


def _status_path(status_file: str | Path | None = None) -> Path | None:
    raw = status_file or os.environ.get(RUN_STATUS_FILE_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def read_run_status(status_file: str | Path | None = None) -> dict[str, Any]:
    path = _status_path(status_file)
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_run_status(status_file: str | Path | None = None, **updates: Any) -> None:
    path = _status_path(status_file)
    if path is None:
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = read_run_status(path)
        if "started_at" not in data:
            data["started_at"] = datetime.now().isoformat(timespec="seconds")
        data.update({k: v for k, v in updates.items() if v is not None})
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        data["pid"] = os.getpid()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
    except Exception:
        return


def write_current_doi(doi: str, *, stage: str | None = None) -> None:
    stage = stage or os.environ.get("SALSA_CURRENT_STAGE", "").strip() or None
    write_run_status(state="running", stage=stage, doi=doi)
