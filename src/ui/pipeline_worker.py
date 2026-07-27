from __future__ import annotations

import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

from auth import send_run_notification_email, write_run_manifest
from pipeline import PipelineCancelled, run_full_pipeline
from run_progress import RUN_STATUS_FILE_NAME, write_run_status
from schemas import RunResult


class DiskCancelEvent:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def is_set(self) -> bool:
        return (self.run_dir / "cancel_requested").exists()


def _notify_run_finished(
    auth_context: dict[str, Any],
    *,
    state: str,
    rr: RunResult | None = None,
    error: str = "",
) -> None:
    if not auth_context.get("notify_on_completion"):
        return
    email = str(auth_context.get("email") or "").strip()
    run_dir = str(getattr(rr, "run_dir", "") or auth_context.get("run_dir") or "")
    status_path = Path(run_dir) / RUN_STATUS_FILE_NAME if run_dir else None
    if not email:
        if status_path is not None:
            write_run_status(
                status_path,
                notification_sent=False,
                notification_error="No email address is available for this user.",
            )
        return

    run_id = str(getattr(rr, "run_id", None) or auth_context.get("run_id") or "")
    workspace = str(
        auth_context.get("workspace")
        or (getattr(rr, "run_dir", Path()).parent.name if rr is not None else "")
    )
    try:
        sent = send_run_notification_email(
            email,
            display_name=str(auth_context.get("display_name") or auth_context.get("username") or ""),
            run_id=run_id,
            workspace=workspace,
            state=state,
            processed_count=len(rr.dois) if rr is not None else None,
            failure_count=len(rr.failures) if rr is not None else None,
            error=error,
            run_dir=run_dir,
        )
    except Exception as exc:
        if status_path is not None:
            write_run_status(
                status_path,
                notification_sent=False,
                notification_error=str(exc),
            )
        print(f"[SALSA email] Run notification failed: {exc}", file=sys.stderr)
        return

    if status_path is not None:
        write_run_status(
            status_path,
            notification_sent=bool(sent),
            notification_error="" if sent else "Email delivery is not configured or no recipient was available.",
        )


def _status_path_for(run_dir: Path) -> Path:
    return Path(run_dir) / RUN_STATUS_FILE_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a SALSA pipeline job outside Streamlit.")
    parser.add_argument("payload", type=Path)
    args = parser.parse_args(argv)

    with args.payload.open("rb") as fh:
        payload = pickle.load(fh)
    try:
        args.payload.unlink()
    except OSError:
        pass

    pipeline_kwargs = dict(payload["pipeline_kwargs"])
    auth_context = dict(payload.get("auth_context") or {})
    run_dir = Path(pipeline_kwargs["run_root"]) / str(pipeline_kwargs["run_id"])
    status_path = _status_path_for(run_dir)
    pipeline_kwargs["cancel_event"] = DiskCancelEvent(run_dir)

    try:
        rr = run_full_pipeline(**pipeline_kwargs)
        if auth_context:
            write_run_manifest(
                rr.run_dir,
                user_id=auth_context.get("user_id"),
                username=str(auth_context.get("username") or ""),
                workspace=str(auth_context.get("workspace") or rr.run_dir.parent.name),
                run_id=rr.run_id,
                query_provider=str(auth_context.get("query_provider") or ""),
                query=str(auth_context.get("query") or ""),
                query_settings=dict(auth_context.get("query_settings") or {}),
            )
        write_run_status(
            status_path,
            state="done",
            stage="Done",
            result_run_id=rr.run_id,
            result_run_dir=str(rr.run_dir),
        )
        _notify_run_finished(auth_context, state="done", rr=rr)
        return 0
    except PipelineCancelled as exc:
        write_run_status(
            status_path,
            state="cancelled",
            error=str(exc),
            stage="Cancelled",
        )
        _notify_run_finished(auth_context, state="cancelled", error=str(exc))
        return 2
    except Exception as exc:
        error = str(exc)
        write_run_status(
            status_path,
            state="error",
            error=error,
            traceback=traceback.format_exc(),
            stage="Failed",
        )
        _notify_run_finished(auth_context, state="error", error=error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
