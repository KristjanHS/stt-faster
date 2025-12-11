#!/usr/bin/env python3
"""Show run information from the database with timezone conversion.

Usage:
    .venv/bin/python scripts/db/show_run.py                    # Show latest run
    .venv/bin/python scripts/db/show_run.py 21                 # Show run by ID
    .venv/bin/python scripts/db/show_run.py --timezone EET     # Show with custom timezone
"""
# ruff: noqa: T201  # CLI script - print statements are intentional

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.database import TranscriptionDatabase


def get_timezone_offset(timezone_name: str) -> timedelta:
    """Get timezone offset in hours.

    Args:
        timezone_name: Timezone name like 'EET', 'EST', etc.

    Returns:
        timedelta offset
    """
    # Simple mapping for common timezones
    # EET is UTC+2 in winter, UTC+3 in summer (EEST)
    # For simplicity, using UTC+2 for EET
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
        dt = dt.replace(tzinfo=timezone.utc)

    utc_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    local_dt = dt + tz_offset
    local_str = local_dt.strftime("%Y-%m-%d %H:%M:%S")

    return utc_str, local_str


def show_run_info(run_id: int | None = None, tz_name: str = "EET") -> int:
    """Show run information by ID or latest if no ID provided.

    Args:
        run_id: Optional run ID. If None, shows latest run.
        tz_name: Timezone name for display

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    tz_offset = get_timezone_offset(tz_name)

    try:
        db = TranscriptionDatabase()
        if run_id is not None:
            run = db.get_run_by_id(run_id)
            if not run:
                print(f"Run ID {run_id} not found in database")
                db.close()
                return 1
        else:
            runs = db.get_run_history(limit=1)
            if not runs:
                print("No runs found in database")
                db.close()
                return 0
            run = runs[0]
        db.close()
        recorded_at = run.get("recorded_at")

        # Convert to local timezone for display
        if isinstance(recorded_at, datetime):
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=timezone.utc)
            utc_str, local_str = format_timestamp(recorded_at, tz_offset)

            print("=" * 80)
            if run_id is not None:
                print(f"RUN INFORMATION (ID: {run_id})")
            else:
                print("LATEST RUN INFORMATION")
            print("=" * 80)
            print(f"\nRun ID: {run['id']}")
            print(f"Recorded at (UTC): {utc_str}")
            print(f"Recorded at ({tz_name}): {local_str}")
            print(f"\nInput folder: {run.get('input_folder', 'N/A')}")
            print(f"Preset: {run.get('preset', 'N/A')}")
            print(f"Language: {run.get('language', 'N/A')}")

            print("\n--- Configuration ---")
            print(f"Preprocess enabled: {run.get('preprocess_enabled', False)}")
            if run.get("preprocess_profile"):
                print(f"Preprocess profile: {run.get('preprocess_profile')}")
            print(f"Model: {run.get('model_id', 'N/A')}")
            print(f"Device: {run.get('device', 'N/A')}")
            print(f"Compute type: {run.get('compute_type', 'N/A')}")

            print("\n--- Results ---")
            print(f"Files found: {run.get('files_found', 0)}")
            print(f"Succeeded: {run.get('succeeded', 0)}")
            print(f"Failed: {run.get('failed', 0)}")

            print("\n--- Timing ---")
            total_time = run.get("total_processing_time")
            if total_time:
                print(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
            preprocess_time = run.get("total_preprocess_time")
            if preprocess_time:
                print(f"Preprocessing time: {preprocess_time:.2f} seconds")
            transcribe_time = run.get("total_transcribe_time")
            if transcribe_time:
                print(f"Transcription time: {transcribe_time:.2f} seconds ({transcribe_time / 60:.2f} minutes)")
            audio_duration = run.get("total_audio_duration")
            if audio_duration:
                print(f"Total audio duration: {audio_duration:.2f} seconds ({audio_duration / 60:.2f} minutes)")
            speed_ratio = run.get("speed_ratio")
            if speed_ratio:
                print(f"Speed ratio: {speed_ratio:.2f}x realtime")

            print("=" * 80)
            return 0
        else:
            print(f"Unexpected timestamp type: {type(recorded_at)}")
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Show run information from the database. Shows latest run if no ID provided."
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        type=int,
        default=None,
        help="Run ID to show. If not provided, shows the latest run.",
    )
    parser.add_argument(
        "--timezone",
        "-t",
        default="EET",
        help="Timezone name for display (default: EET)",
    )
    args = parser.parse_args()
    return show_run_info(args.run_id, args.timezone)


if __name__ == "__main__":
    sys.exit(main())
