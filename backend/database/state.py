"""File-state tracking (queued / completed / failed) against the legacy `transcriptions` table."""

from __future__ import annotations

import logging
from typing import Any

import duckdb

from backend.exceptions import DatabaseError

LOGGER = logging.getLogger(__name__)


def add_file(conn: duckdb.DuckDBPyConnection, file_path: str, status: str = "pending") -> None:
    """Add a file to track or update its status if it exists.

    Args:
        conn: open DuckDB connection
        file_path: Path to the audio file
        status: Initial status (default: 'pending')
    """
    try:
        conn.execute(
            "INSERT INTO transcriptions (file_path, status) VALUES (?, ?)",
            (file_path, status),
        )
        conn.commit()
        LOGGER.debug("Added file: %s with status: %s", file_path, status)
    except duckdb.ConstraintException:  # DuckDB equivalent of IntegrityError for UNIQUE constraint
        LOGGER.debug("File already exists in database: %s", file_path)
    except Exception as e:
        msg = f"Failed to add file {file_path}: {e}"
        raise DatabaseError(msg) from e


def update_status(
    conn: duckdb.DuckDBPyConnection,
    file_path: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update the status of a file.

    Args:
        conn: open DuckDB connection
        file_path: Path to the audio file
        status: New status ('pending', 'completed', 'failed')
        error_message: Optional error message if status is 'failed'
    """
    try:
        conn.execute(
            "UPDATE transcriptions SET status = ?, error_message = ? WHERE file_path = ?",
            (status, error_message, file_path),
        )
        conn.commit()
        LOGGER.debug("Updated %s to status: %s", file_path, status)
    except Exception as e:
        msg = f"Failed to update status for {file_path}: {e}"
        raise DatabaseError(msg) from e


def get_status(conn: duckdb.DuckDBPyConnection, file_path: str) -> dict[str, Any] | None:
    """Get the status of a specific file.

    Args:
        conn: open DuckDB connection
        file_path: Path to the audio file

    Returns:
        Dictionary with file info or None if not found
    """
    try:
        # DuckDB execute returns self, fetchone returns tuple
        # We need column names to make a dict
        cursor = conn.execute(
            "SELECT * FROM transcriptions WHERE file_path = ?",
            [file_path],
        )
        row = cursor.fetchone()
        if not row:
            return None

        # Map tuple to dict using column descriptions
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    except Exception as e:
        msg = f"Failed to get status for {file_path}: {e}"
        raise DatabaseError(msg) from e


def get_files_by_status(conn: duckdb.DuckDBPyConnection, status: str) -> list[dict[str, Any]]:
    """Get all files with a specific status.

    Args:
        conn: open DuckDB connection
        status: Status to filter by ('pending', 'completed', 'failed')

    Returns:
        List of dictionaries with file info
    """
    try:
        cursor = conn.execute(
            "SELECT * FROM transcriptions WHERE status = ?",
            [status],
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        msg = f"Failed to get files by status {status}: {e}"
        raise DatabaseError(msg) from e


def get_all_files(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Get all files in the database.

    Returns:
        List of dictionaries with file info
    """
    try:
        cursor = conn.execute("SELECT * FROM transcriptions")
        rows = cursor.fetchall()

        if not rows:
            return []

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        msg = f"Failed to get all files: {e}"
        raise DatabaseError(msg) from e


def get_summary(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Get summary statistics of all files.

    Returns:
        Dictionary with counts by status
    """
    try:
        rows = conn.execute("SELECT status, COUNT(*) as count FROM transcriptions GROUP BY status").fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        msg = f"Failed to get summary: {e}"
        raise DatabaseError(msg) from e
