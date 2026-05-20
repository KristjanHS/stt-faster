"""Run-log inspection commands for stt-faster CLI.

Reads from the append-only JSONL log written by :class:`backend.run_log.JsonlRunLog`
(Stage G.c). The flat top-level record shape lets ``show`` / ``recent`` work
directly off ``dict`` payloads without a SQL row adapter.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, cast

import typer
from rich.console import Console

from backend.run_log import JsonlRunLog

LOGGER = logging.getLogger(__name__)

app = typer.Typer(name="db", help="Run-log inspection commands")
console = Console()

_TZ_OFFSETS = {
    "EET": 2,
    "EEST": 3,
    "UTC": 0,
    "EST": -5,
    "EDT": -4,
}


def _tz_offset(name: str) -> timedelta:
    return timedelta(hours=_TZ_OFFSETS.get(name.upper(), 0))


def _format_recorded_at(value: Any, tz_offset: timedelta) -> tuple[str, str] | None:
    """Render ``recorded_at`` (ISO-8601 ``Z`` suffix) as (utc_str, local_str)."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    utc_str = dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    local_str = (dt.astimezone(UTC) + tz_offset).strftime("%Y-%m-%d %H:%M:%S")
    return utc_str, local_str


def _print_run(record: dict[str, Any], timezone: str, tz_offset: timedelta) -> None:
    """Print the full single-run detail view used by ``db show``."""
    formatted = _format_recorded_at(record.get("recorded_at"), tz_offset)
    utc_str, local_str = formatted if formatted else ("N/A", "N/A")

    preprocess = cast("dict[str, Any]", record.get("preprocess") or {})
    model = cast("dict[str, Any]", record.get("model") or {})
    totals = cast("dict[str, Any]", record.get("totals") or {})

    console.print("=" * 80)
    console.print(f"[bold]RUN ID: {record.get('id', 'N/A')}[/bold]")
    console.print("=" * 80)
    console.print(f"Recorded at (UTC): {utc_str}")
    console.print(f"Recorded at ({timezone}): {local_str}")
    console.print(f"\nInput folder: {record.get('input_folder', 'N/A')}")
    console.print(f"Preset: {record.get('preset', 'N/A')}")
    console.print(f"Language: {record.get('language', 'N/A')}")

    console.print("\n[bold]--- Configuration ---[/bold]")
    console.print(f"Preprocess enabled: {preprocess.get('enabled', False)}")
    if preprocess.get("profile"):
        console.print(f"Preprocess profile: {preprocess['profile']}")
    console.print(f"Model: {model.get('id', 'N/A')}")
    console.print(f"Device: {model.get('device', 'N/A')}")
    console.print(f"Compute type: {model.get('compute_type', 'N/A')}")

    console.print("\n[bold]--- Results ---[/bold]")
    console.print(f"Files found: {totals.get('files_found', 0)}")
    console.print(f"Succeeded: {totals.get('succeeded', 0)}")
    console.print(f"Failed: {totals.get('failed', 0)}")

    console.print("\n[bold]--- Timing ---[/bold]")
    for label, key in (
        ("Total processing time", "processing_time_s"),
        ("Preprocessing time", "preprocess_time_s"),
        ("Transcription time", "transcribe_time_s"),
        ("Total audio duration", "audio_duration_s"),
    ):
        value = totals.get(key)
        if isinstance(value, (int, float)):
            console.print(f"{label}: {value:.2f} seconds ({value / 60:.2f} minutes)")
    speed_ratio = totals.get("speed_ratio")
    if isinstance(speed_ratio, (int, float)):
        console.print(f"Speed ratio: {speed_ratio:.2f}x realtime")
    console.print("=" * 80)


def _print_runs_compact(records: list[dict[str, Any]], tz_offset: timedelta) -> None:
    console.print(f"\n[bold]Recent {len(records)} runs (compact):[/bold]\n")
    for record in records:
        formatted = _format_recorded_at(record.get("recorded_at"), tz_offset)
        if formatted is None:
            continue
        _, local_str = formatted
        totals = cast("dict[str, Any]", record.get("totals") or {})
        console.print(
            f"  {record.get('id', 0):4d} | {local_str} | "
            f"Succeeded: {totals.get('succeeded', 0):3d} | Failed: {totals.get('failed', 0):3d}"
        )


def _print_runs_full(records: list[dict[str, Any]], tz_offset: timedelta, timezone: str) -> None:
    console.print(f"\n[bold]Recent {len(records)} runs:[/bold]\n")
    for record in records:
        formatted = _format_recorded_at(record.get("recorded_at"), tz_offset)
        if formatted is None:
            continue
        utc_str, local_str = formatted
        totals = cast("dict[str, Any]", record.get("totals") or {})
        console.print("=" * 80)
        console.print(f"[bold]Run ID: {record.get('id', 'N/A')}[/bold]")
        console.print(f"Recorded at (UTC): {utc_str}")
        console.print(f"Recorded at ({timezone}): {local_str}")
        console.print(f"Input folder: {record.get('input_folder', 'N/A')}")
        console.print(f"Preset: {record.get('preset', 'N/A')}")
        console.print(f"Succeeded: {totals.get('succeeded', 0)} | Failed: {totals.get('failed', 0)}")
        console.print("")


@app.command()
def show(
    run_id: Annotated[int | None, typer.Argument(help="Run ID to show. If not provided, shows the latest run.")] = None,
    timezone: Annotated[str, typer.Option("--timezone", "-t", help="Timezone name for display")] = "EET",
) -> None:
    """Show run information by ID or latest if no ID provided."""
    tz_offset = _tz_offset(timezone)
    try:
        run_log = JsonlRunLog()
        if run_id is not None:
            record = run_log.get(run_id)
            if record is None:
                console.print(f"[red]Run ID {run_id} not found in run log[/red]")
                raise typer.Exit(1)
        else:
            records = run_log.tail(1)
            if not records:
                console.print("[yellow]No runs found in run log[/yellow]")
                raise typer.Exit(0)
            record = records[0]
        _print_run(record, timezone, tz_offset)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        LOGGER.exception("db show failed")
        raise typer.Exit(1) from e


@app.command()
def recent(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of runs to show")] = 10,
    timezone: Annotated[str, typer.Option("--timezone", "-t", help="Timezone name for display")] = "EET",
    compact: Annotated[bool, typer.Option("--compact", "-c", help="Compact output format")] = False,
) -> None:
    """Show recent runs from the run log (most-recent last)."""
    tz_offset = _tz_offset(timezone)
    try:
        run_log = JsonlRunLog()
        records = run_log.tail(limit)
        if not records:
            console.print("[yellow]No runs found in run log[/yellow]")
            raise typer.Exit(0)
        if compact:
            _print_runs_compact(records, tz_offset)
        else:
            _print_runs_full(records, tz_offset, timezone)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        LOGGER.exception("db recent failed")
        raise typer.Exit(1) from e
