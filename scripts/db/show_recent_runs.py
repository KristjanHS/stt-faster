#!/usr/bin/env python3
"""Show recent runs from the database with timezone conversion.

Usage:
    .venv/bin/python scripts/db/show_recent_runs.py
    .venv/bin/python scripts/db/show_recent_runs.py --limit 10 --timezone EET
    .venv/bin/python scripts/db/show_recent_runs.py --limit 20 --compact
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
    offsets = {
        "EET": 2,
        "EEST": 3,
        "UTC": 0,
        "EST": -5,
        "EDT": -4,
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


def show_recent_runs(limit: int = 10, tz_name: str = "EET", compact: bool = False) -> int:
    """Show recent runs.

    Args:
        limit: Number of runs to show
        tz_name: Timezone name for display
        compact: If True, show in compact single-line format

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    tz_offset = get_timezone_offset(tz_name)

    try:
        db = TranscriptionDatabase()
        runs = db.get_run_history(limit=limit)
        db.close()

        if not runs:
            print("No runs found in database")
            return 0

        if compact:
            # Compact format: single line per run
            header = f"{'ID':<5} {'Time (' + tz_name + ')':<20} {'Files':<15} {'Duration':<12} {'Speed':<10}"
            print(header)
            print("=" * len(header))
            for run in runs:
                recorded_at = run.get("recorded_at")
                run_id = run.get("id")

                if isinstance(recorded_at, datetime):
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
                    _, local_str = format_timestamp(recorded_at, tz_offset)
                    # Shorten timestamp format for compact view (remove year if it's 2025)
                    if local_str.startswith("2025-"):
                        local_str = local_str[5:]  # Remove "2025-"

                    files_found = run.get("files_found", 0)
                    succeeded = run.get("succeeded", 0)
                    failed = run.get("failed", 0)
                    files_str = f"{files_found}/{succeeded}/{failed}"

                    total_time = run.get("total_processing_time")
                    if total_time:
                        duration_str = f"{total_time / 60:.1f}m"
                    else:
                        duration_str = "N/A"

                    speed_ratio = run.get("speed_ratio")
                    if speed_ratio:
                        speed_str = f"{speed_ratio:.1f}x"
                    else:
                        speed_str = "N/A"

                    print(f"{run_id:<5} {local_str:<20} {files_str:<15} {duration_str:<12} {speed_str:<10}")
            print("=" * len(header))
        else:
            # Detailed format (original)
            print(f"Recent runs (showing both UTC and {tz_name}):")
            print("=" * 80)
            for run in runs:
                recorded_at = run.get("recorded_at")
                run_id = run.get("id")

                if isinstance(recorded_at, datetime):
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
                    utc_str, local_str = format_timestamp(recorded_at, tz_offset)

                    print(f"Run ID {run_id}:")
                    print(f"  UTC: {utc_str}")
                    print(f"  {tz_name}: {local_str}")
                    print(
                        f"  Files: {run.get('files_found', 0)} found, "
                        f"{run.get('succeeded', 0)} succeeded, "
                        f"{run.get('failed', 0)} failed"
                    )
                    print()
            print("=" * 80)

        # Check database file modification time (only in detailed mode)
        if not compact:
            import os

            db_path = db.db_path if hasattr(db, "db_path") else None
            if db_path and Path(db_path).exists():
                mtime = os.path.getmtime(db_path)
                mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
                mtime_utc, mtime_local = format_timestamp(mtime_dt, tz_offset)
                print("\nDatabase file last modified:")
                print(f"  UTC: {mtime_utc}")
                print(f"  {tz_name}: {mtime_local}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Show recent runs from the database")
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Number of runs to show (default: 10)",
    )
    parser.add_argument(
        "--timezone",
        "-t",
        default="EET",
        help="Timezone name for display (default: EET)",
    )
    parser.add_argument(
        "--compact",
        "-c",
        action="store_true",
        help="Show runs in compact single-line format",
    )
    args = parser.parse_args()
    return show_recent_runs(args.limit, args.timezone, args.compact)


if __name__ == "__main__":
    sys.exit(main())
