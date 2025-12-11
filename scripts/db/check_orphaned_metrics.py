#!/usr/bin/env python3
"""Check for orphaned file metrics (file metrics without corresponding runs).

This can indicate a run in progress or a failed run save.

Usage:
    .venv/bin/python scripts/db/check_orphaned_metrics.py
"""
# ruff: noqa: T201  # CLI script - print statements are intentional

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.database import TranscriptionDatabase


def check_orphaned_metrics() -> int:
    """Check for orphaned file metrics.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        db = TranscriptionDatabase()
        if not db.conn:
            print("Error: Database connection not initialized", file=sys.stderr)
            return 1

        # Check the highest run_id in file_metrics vs runs
        cursor = db.conn.execute("SELECT MAX(run_id) FROM file_metrics")
        result = cursor.fetchone()
        max_file_metric_run = result[0] if result else None

        cursor = db.conn.execute("SELECT MAX(id) FROM runs")
        result = cursor.fetchone()
        max_run_id = result[0] if result else None

        print(f"Highest run ID in runs table: {max_run_id}")
        print(f"Highest run_id in file_metrics table: {max_file_metric_run}")

        if max_file_metric_run and max_file_metric_run > max_run_id:
            print(f"\n⚠️  Found file_metrics with run_id {max_file_metric_run} but no corresponding run!")
            print("This suggests a run is in progress or failed to save the run record.")

            # Show details
            cursor = db.conn.execute(
                """
                SELECT recorded_at, status, COUNT(*) as count
                FROM file_metrics
                WHERE run_id = ?
                GROUP BY recorded_at, status
            """,
                [max_file_metric_run],
            )
            details = cursor.fetchall()
            for row in details:
                recorded_at = row[0]
                if isinstance(recorded_at, datetime):
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
                    eet = recorded_at + timedelta(hours=2)
                    print(f"  Recorded: {eet.strftime('%Y-%m-%d %H:%M:%S')} EET, Status: {row[1]}, Count: {row[2]}")
            db.close()
            return 1  # Indicate issue found
        elif max_file_metric_run == max_run_id:
            print("\n✓ File metrics match the latest run - no orphaned data")
            db.close()
            return 0
        else:
            print("\nNo file metrics found")
            db.close()
            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    return check_orphaned_metrics()


if __name__ == "__main__":
    sys.exit(main())
