"""Database operations for transcription state tracking."""

import logging
import sqlite3
from typing import Any

LOGGER = logging.getLogger(__name__)


class TranscriptionDatabase:
    """Simple SQLite database for tracking transcription state."""

    def __init__(self, db_path: str = "transcribe_state.db") -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if it doesn't exist."""
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

    def add_file(self, file_path: str, status: str = "pending") -> None:
        """Add a file to track or update its status if it exists.

        Args:
            file_path: Path to the audio file
            status: Initial status (default: 'pending')
        """
        if not self.conn:
            msg = "Database not initialized"
            raise RuntimeError(msg)

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
            raise RuntimeError(msg)

        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE transcriptions SET status = ?, error_message = ? WHERE file_path = ?",
            (status, error_message, file_path),
        )
        self.conn.commit()
        LOGGER.debug("Updated %s to status: %s", file_path, status)

    def get_status(self, file_path: str) -> dict[str, Any] | None:
        """Get the status of a specific file.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with file info or None if not found
        """
        if not self.conn:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM transcriptions WHERE file_path = ?",
            (file_path,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_files_by_status(self, status: str) -> list[dict[str, Any]]:
        """Get all files with a specific status.

        Args:
            status: Status to filter by ('pending', 'completed', 'failed')

        Returns:
            List of dictionaries with file info
        """
        if not self.conn:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM transcriptions WHERE status = ?",
            (status,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_all_files(self) -> list[dict[str, Any]]:
        """Get all files in the database.

        Returns:
            List of dictionaries with file info
        """
        if not self.conn:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transcriptions")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_summary(self) -> dict[str, int]:
        """Get summary statistics of all files.

        Returns:
            Dictionary with counts by status
        """
        if not self.conn:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = self.conn.cursor()
        cursor.execute("SELECT status, COUNT(*) as count FROM transcriptions GROUP BY status")
        rows = cursor.fetchall()
        return {row["status"]: row["count"] for row in rows}

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
