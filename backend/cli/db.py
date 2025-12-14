"""Database inspection commands for stt-faster CLI."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from typing import Annotated

import typer
from rich.console import Console

from backend.database import TranscriptionDatabase

app = typer.Typer(name="db", help="Database inspection commands")
console = Console()


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


@app.command()
def show(
    run_id: Annotated[int | None, typer.Argument(help="Run ID to show. If not provided, shows the latest run.")] = None,
    timezone: Annotated[str, typer.Option("--timezone", "-t", help="Timezone name for display")] = "EET",
) -> None:
    """Show run information by ID or latest if no ID provided."""
    tz_offset = get_timezone_offset(timezone)

    try:
        db = TranscriptionDatabase()
        if run_id is not None:
            run = db.get_run_by_id(run_id)
            if not run:
                console.print(f"[red]Run ID {run_id} not found in database[/red]")
                db.close()
                raise typer.Exit(1)
        else:
            runs = db.get_run_history(limit=1)
            if not runs:
                console.print("[yellow]No runs found in database[/yellow]")
                db.close()
                raise typer.Exit(0)
            run = runs[0]
        db.close()
        recorded_at = run.get("recorded_at")

        # Convert to local timezone for display
        if isinstance(recorded_at, datetime):
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=UTC)
            utc_str, local_str = format_timestamp(recorded_at, tz_offset)

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

            console.print("=" * 80)
        else:
            console.print(f"[red]Unexpected timestamp type: {type(recorded_at)}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def recent(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of runs to show")] = 10,
    timezone: Annotated[str, typer.Option("--timezone", "-t", help="Timezone name for display")] = "EET",
    compact: Annotated[bool, typer.Option("--compact", "-c", help="Compact output format")] = False,
) -> None:
    """Show recent runs from the database."""
    tz_offset = get_timezone_offset(timezone)

    try:
        db = TranscriptionDatabase()
        runs = db.get_run_history(limit=limit)
        db.close()

        if not runs:
            console.print("[yellow]No runs found in database[/yellow]")
            raise typer.Exit(0)

        if compact:
            # Compact format: ID, timestamp, succeeded/failed
            console.print(f"\n[bold]Recent {len(runs)} runs (compact):[/bold]\n")
            for run in runs:
                recorded_at = run.get("recorded_at")
                if isinstance(recorded_at, datetime):
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.replace(tzinfo=UTC)
                    _, local_str = format_timestamp(recorded_at, tz_offset)
                    console.print(
                        f"  {run['id']:4d} | {local_str} | "
                        f"Succeeded: {run.get('succeeded', 0):3d} | Failed: {run.get('failed', 0):3d}"
                    )
        else:
            # Full format
            console.print(f"\n[bold]Recent {len(runs)} runs:[/bold]\n")
            for run in runs:
                recorded_at = run.get("recorded_at")
                if isinstance(recorded_at, datetime):
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.replace(tzinfo=UTC)
                    utc_str, local_str = format_timestamp(recorded_at, tz_offset)

                    console.print("=" * 80)
                    console.print(f"[bold]Run ID: {run['id']}[/bold]")
                    console.print(f"Recorded at (UTC): {utc_str}")
                    console.print(f"Recorded at ({timezone}): {local_str}")
                    console.print(f"Input folder: {run.get('input_folder', 'N/A')}")
                    console.print(f"Preset: {run.get('preset', 'N/A')}")
                    console.print(f"Succeeded: {run.get('succeeded', 0)} | Failed: {run.get('failed', 0)}")
                    console.print("")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


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
