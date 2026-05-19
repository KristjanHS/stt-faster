"""Database inspection commands for stt-faster CLI."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import typer
from rich.console import Console

from backend.database import TranscriptionDatabase

LOGGER = logging.getLogger(__name__)

app = typer.Typer(name="db", help="Database inspection commands")
console = Console()

RunRow = Mapping[str, Any]


def get_timezone_offset(timezone_name: str) -> timedelta:
    """Get timezone offset in hours.

    Args:
        timezone_name: Timezone name like 'EET', 'EST', etc.

    Returns:
        timedelta offset
    """
    offsets = {
        "EET": 2,  # Eastern European Time (UTC+2)
        "EEST": 3,  # Eastern European Summer Time (UTC+3)
        "UTC": 0,
        "EST": -5,  # Eastern Standard Time
        "EDT": -4,  # Eastern Daylight Time
    }
    hours = offsets.get(timezone_name.upper(), 0)
    return timedelta(hours=hours)


def format_timestamp(dt: datetime, tz_offset: timedelta) -> tuple[str, str]:
    """Format timestamp in both UTC and local timezone.

    Args:
        dt: datetime object (assumed UTC if no tzinfo)
        tz_offset: timezone offset

    Returns:
        Tuple of (utc_str, local_str)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    utc_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    local_dt = dt + tz_offset
    local_str = local_dt.strftime("%Y-%m-%d %H:%M:%S")

    return utc_str, local_str


def _resolve_run_timestamps(run: RunRow, tz_offset: timedelta) -> tuple[str, str] | None:
    """Return ``(utc_str, local_str)`` for a run row, or ``None`` if the row carries no datetime."""
    recorded_at = run.get("recorded_at")
    if not isinstance(recorded_at, datetime):
        return None
    return format_timestamp(recorded_at, tz_offset)


def _fetch_run_for_show(run_id: int | None) -> RunRow:
    """Load the run targeted by ``db show``. Raises ``typer.Exit`` on miss / empty history."""
    with TranscriptionDatabase() as db:
        if run_id is not None:
            run = db.get_run_by_id(run_id)
            if not run:
                console.print(f"[red]Run ID {run_id} not found in database[/red]")
                raise typer.Exit(1)
            return run
        runs = db.get_run_history(limit=1)
        if not runs:
            console.print("[yellow]No runs found in database[/yellow]")
            raise typer.Exit(0)
        return runs[0]


def _print_timing_block(run: RunRow) -> None:
    """Print the optional ``--- Timing ---`` rows for a run."""
    console.print("\n[bold]--- Timing ---[/bold]")
    total_time = run.get("total_processing_time")
    if total_time:
        console.print(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    preprocess_time = run.get("total_preprocess_time")
    if preprocess_time:
        console.print(f"Preprocessing time: {preprocess_time:.2f} seconds")
    transcribe_time = run.get("total_transcribe_time")
    if transcribe_time:
        console.print(f"Transcription time: {transcribe_time:.2f} seconds ({transcribe_time / 60:.2f} minutes)")
    audio_duration = run.get("total_audio_duration")
    if audio_duration:
        console.print(f"Total audio duration: {audio_duration:.2f} seconds ({audio_duration / 60:.2f} minutes)")
    speed_ratio = run.get("speed_ratio")
    if speed_ratio:
        console.print(f"Speed ratio: {speed_ratio:.2f}x realtime")


def _print_run_details(run: RunRow, run_id: int | None, timezone: str, tz_offset: timedelta) -> None:
    """Print the full single-run detail view used by ``db show``."""
    timestamps = _resolve_run_timestamps(run, tz_offset)
    if timestamps is None:
        console.print(f"[red]Unexpected timestamp type: {type(run.get('recorded_at'))}[/red]")
        raise typer.Exit(1)
    utc_str, local_str = timestamps

    console.print("=" * 80)
    if run_id is not None:
        console.print(f"[bold]RUN INFORMATION (ID: {run_id})[/bold]")
    else:
        console.print("[bold]LATEST RUN INFORMATION[/bold]")
    console.print("=" * 80)
    console.print(f"\nRun ID: {run['id']}")
    console.print(f"Recorded at (UTC): {utc_str}")
    console.print(f"Recorded at ({timezone}): {local_str}")
    console.print(f"\nInput folder: {run.get('input_folder', 'N/A')}")
    console.print(f"Preset: {run.get('preset', 'N/A')}")
    console.print(f"Language: {run.get('language', 'N/A')}")

    console.print("\n[bold]--- Configuration ---[/bold]")
    console.print(f"Preprocess enabled: {run.get('preprocess_enabled', False)}")
    if run.get("preprocess_profile"):
        console.print(f"Preprocess profile: {run.get('preprocess_profile')}")
    console.print(f"Model: {run.get('model_id', 'N/A')}")
    console.print(f"Device: {run.get('device', 'N/A')}")
    console.print(f"Compute type: {run.get('compute_type', 'N/A')}")

    console.print("\n[bold]--- Results ---[/bold]")
    console.print(f"Files found: {run.get('files_found', 0)}")
    console.print(f"Succeeded: {run.get('succeeded', 0)}")
    console.print(f"Failed: {run.get('failed', 0)}")

    _print_timing_block(run)
    console.print("=" * 80)


def _print_runs_compact(runs: Sequence[RunRow], tz_offset: timedelta) -> None:
    """Render the ``db recent --compact`` listing."""
    console.print(f"\n[bold]Recent {len(runs)} runs (compact):[/bold]\n")
    for run in runs:
        timestamps = _resolve_run_timestamps(run, tz_offset)
        if timestamps is None:
            continue
        _, local_str = timestamps
        console.print(
            f"  {run['id']:4d} | {local_str} | "
            f"Succeeded: {run.get('succeeded', 0):3d} | Failed: {run.get('failed', 0):3d}"
        )


def _print_runs_full(runs: Sequence[RunRow], tz_offset: timedelta, timezone: str) -> None:
    """Render the default per-run summary listing for ``db recent``."""
    console.print(f"\n[bold]Recent {len(runs)} runs:[/bold]\n")
    for run in runs:
        timestamps = _resolve_run_timestamps(run, tz_offset)
        if timestamps is None:
            continue
        utc_str, local_str = timestamps

        console.print("=" * 80)
        console.print(f"[bold]Run ID: {run['id']}[/bold]")
        console.print(f"Recorded at (UTC): {utc_str}")
        console.print(f"Recorded at ({timezone}): {local_str}")
        console.print(f"Input folder: {run.get('input_folder', 'N/A')}")
        console.print(f"Preset: {run.get('preset', 'N/A')}")
        console.print(f"Succeeded: {run.get('succeeded', 0)} | Failed: {run.get('failed', 0)}")
        console.print("")


@app.command()
def show(
    run_id: Annotated[int | None, typer.Argument(help="Run ID to show. If not provided, shows the latest run.")] = None,
    timezone: Annotated[str, typer.Option("--timezone", "-t", help="Timezone name for display")] = "EET",
) -> None:
    """Show run information by ID or latest if no ID provided."""
    tz_offset = get_timezone_offset(timezone)

    try:
        run = _fetch_run_for_show(run_id)
        _print_run_details(run, run_id, timezone, tz_offset)
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
    """Show recent runs from the database."""
    tz_offset = get_timezone_offset(timezone)

    try:
        with TranscriptionDatabase() as db:
            runs = db.get_run_history(limit=limit)

        if not runs:
            console.print("[yellow]No runs found in database[/yellow]")
            raise typer.Exit(0)

        if compact:
            _print_runs_compact(runs, tz_offset)
        else:
            _print_runs_full(runs, tz_offset, timezone)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        LOGGER.exception("db recent failed")
        raise typer.Exit(1) from e


@app.command()
def check_params() -> None:
    """Check parameter completeness in the database."""
    # TODO: Implement check_parameter_completeness functionality
    console.print("[yellow]check-params command not yet implemented[/yellow]")
    raise typer.Exit(0)


@app.command()
def check_orphaned() -> None:
    """Check for orphaned metrics in the database."""
    # TODO: Implement check_orphaned_metrics functionality
    console.print("[yellow]check-orphaned command not yet implemented[/yellow]")
    raise typer.Exit(0)
