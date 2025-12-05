"""Database operations for transcription state tracking."""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from backend.exceptions import DatabaseError

LOGGER = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    """Get XDG-compliant default database path.

    Uses XDG Base Directory specification:
    - XDG_DATA_HOME if set (e.g., ~/.local/share)
    - Falls back to ~/.local/share if not set

    Returns:
        Path to database file in XDG data directory
    """
    # Get XDG_DATA_HOME or use default
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        data_dir = Path(xdg_data_home)
    else:
        data_dir = Path.home() / ".local" / "share"

    # Create app-specific directory
    app_data_dir = data_dir / "stt-faster"
    app_data_dir.mkdir(parents=True, exist_ok=True)

    return app_data_dir / "transcribe_state.db"


class TranscriptionDatabase:
    """Simple SQLite database for tracking transcription state."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
                     If None, uses XDG-compliant default path.
                     For backward compatibility, can still pass custom path.
        """
        if db_path is None:
            self.db_path = str(get_default_db_path())
        else:
            self.db_path = str(db_path)

        self.conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if it doesn't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            self.conn.commit()
            LOGGER.info("Database initialized at %s", self.db_path)
        except sqlite3.Error as e:
            msg = f"Failed to initialize database at {self.db_path}: {e}"
            raise DatabaseError(msg) from e

    def add_file(self, file_path: str, status: str = "pending") -> None:
        """Add a file to track or update its status if it exists.

        Args:
            file_path: Path to the audio file
            status: Initial status (default: 'pending')
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO transcriptions (file_path, status) VALUES (?, ?)",
                (file_path, status),
            )
            self.conn.commit()
            LOGGER.debug("Added file: %s with status: %s", file_path, status)
        except sqlite3.IntegrityError:
            LOGGER.debug("File already exists in database: %s", file_path)
        except sqlite3.Error as e:
            msg = f"Failed to add file {file_path}: {e}"
            raise DatabaseError(msg) from e

    def update_status(
        self,
        file_path: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a file.

        Args:
            file_path: Path to the audio file
            status: New status ('pending', 'completed', 'failed')
            error_message: Optional error message if status is 'failed'
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE transcriptions SET status = ?, error_message = ? WHERE file_path = ?",
                (status, error_message, file_path),
            )
            self.conn.commit()
            LOGGER.debug("Updated %s to status: %s", file_path, status)
        except sqlite3.Error as e:
            msg = f"Failed to update status for {file_path}: {e}"
            raise DatabaseError(msg) from e

    def get_status(self, file_path: str) -> dict[str, Any] | None:
        """Get the status of a specific file.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with file info or None if not found
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM transcriptions WHERE file_path = ?",
                (file_path,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            msg = f"Failed to get status for {file_path}: {e}"
            raise DatabaseError(msg) from e

    def get_files_by_status(self, status: str) -> list[dict[str, Any]]:
        """Get all files with a specific status.

        Args:
            status: Status to filter by ('pending', 'completed', 'failed')

        Returns:
            List of dictionaries with file info
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM transcriptions WHERE status = ?",
                (status,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            msg = f"Failed to get files by status {status}: {e}"
            raise DatabaseError(msg) from e

    def get_all_files(self) -> list[dict[str, Any]]:
        """Get all files in the database.

        Returns:
            List of dictionaries with file info
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM transcriptions")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            msg = f"Failed to get all files: {e}"
            raise DatabaseError(msg) from e

    def get_summary(self) -> dict[str, int]:
        """Get summary statistics of all files.

        Returns:
            Dictionary with counts by status
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT status, COUNT(*) as count FROM transcriptions GROUP BY status")
            rows = cursor.fetchall()
            return {row["status"]: row["count"] for row in rows}
        except sqlite3.Error as e:
            msg = f"Failed to get summary: {e}"
            raise DatabaseError(msg) from e

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            LOGGER.debug("Database connection closed")

    def __enter__(self) -> "TranscriptionDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
