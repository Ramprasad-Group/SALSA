from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

try:
    from dateutil import tz as dateutil_tz
except Exception:
    dateutil_tz = None


RUN_STATUS_FILE_ENV = "SALSA_RUN_STATUS_FILE"
RUN_STATUS_FILE_NAME = "run_status.json"
CANCEL_REQUEST_FILE_NAME = "cancel_requested"


def _eastern_tz() -> tzinfo:
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    if dateutil_tz is not None:
        eastern = dateutil_tz.gettz("America/New_York")
        if eastern is not None:
            return eastern
    return timezone(timedelta(hours=-5), "ET")


def _now_eastern_iso() -> str:
    return datetime.now(_eastern_tz()).isoformat(timespec="seconds")


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
            data["started_at"] = _now_eastern_iso()
        explicit_pid = updates.pop("pid", None)
        data.update({k: v for k, v in updates.items() if v is not None})
        data["updated_at"] = _now_eastern_iso()
        if explicit_pid is not None:
            data["pid"] = explicit_pid
        elif "pid" not in data:
            data["pid"] = os.getpid()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
    except Exception:
        return


def append_run_warning(message: str, status_file: str | Path | None = None) -> None:
    """Add a user-visible warning to the current run without duplicating it."""
    path = _status_path(status_file)
    if path is None:
        return
    warning = str(message or "").strip()
    if not warning:
        return
    data = read_run_status(path)
    warnings = data.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    write_run_status(path, warnings=warnings)


def write_current_doi(doi: str, *, stage: str | None = None) -> None:
    stage = stage or os.environ.get("SALSA_CURRENT_STAGE", "").strip() or None
    write_run_status(state="running", stage=stage, doi=doi)


def cancel_request_path(run_dir: Path) -> Path:
    return run_dir / CANCEL_REQUEST_FILE_NAME


def request_run_cancellation(run_dir: Path) -> None:
    path = cancel_request_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_now_eastern_iso(), encoding="utf-8")


def clear_run_cancellation(run_dir: Path) -> None:
    cancel_request_path(run_dir).unlink(missing_ok=True)
