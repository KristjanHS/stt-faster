"""Database operations for transcription state tracking.

Public surface (``from backend.database import ...``):
    TranscriptionDatabase, RunRecord, FileMetricRecord, MIGRATIONS, get_default_db_path
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

from backend.database import file_metrics as _file_metrics
from backend.database import runs as _runs
from backend.database import state as _state
from backend.database.migrations import MIGRATIONS, validate_migration_ordering
from backend.database.schema import (
    FileMetricRecord,
    RunRecord,
    create_initial_tables,
    get_default_db_path,
)
from backend.exceptions import DatabaseError

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MIGRATIONS",
    "FileMetricRecord",
    "RunRecord",
    "TranscriptionDatabase",
    "get_default_db_path",
]


class TranscriptionDatabase:
    """Coordinator over the schema/runs/file_metrics/state submodules.

    Owns the DuckDB connection lifecycle (open in ``__init__``, close in
    ``close``/``__exit__``) and delegates per-domain operations to free
    functions in ``backend.database.{runs,file_metrics,state}``.
    """

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

        self.conn: duckdb.DuckDBPyConnection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if it doesn't exist."""
        try:
            self.conn = duckdb.connect(self.db_path)
            create_initial_tables(self.conn)
            # Migrate existing tables: add missing columns if they don't exist
            self._migrate_schema()
            LOGGER.debug("DuckDB initialized at %s", self.db_path)
        except Exception as e:
            msg = f"Failed to initialize database at {self.db_path}: {e}"
            raise DatabaseError(msg) from e

    def _migrate_schema(self) -> None:
        """Migrate existing database schema using version-based ordered migrations."""
        if self.conn is None:
            return

        try:
            # Validate migration registry
            validate_migration_ordering()

            # Get current schema version
            current_version = self._get_current_schema_version()

            # Find pending migrations
            pending = [m for m in MIGRATIONS if m.version > current_version]

            if not pending:
                LOGGER.debug("Database schema is up to date (version %d)", current_version)
                return

            LOGGER.info("Running %d pending migration(s)", len(pending))

            # Apply each pending migration
            for migration in pending:
                try:
                    LOGGER.info(
                        "Applying migration %d: %s - %s", migration.version, migration.name, migration.description
                    )
                    migration.migrate(self.conn)
                    self._record_migration(migration.version)
                    self.conn.commit()
                except Exception as e:
                    LOGGER.error("Migration %d failed: %s", migration.version, e)
                    raise DatabaseError(f"Migration {migration.version} failed") from e

        except Exception as e:
            # Log but don't fail - migration errors shouldn't break initialization
            LOGGER.warning("Schema migration encountered an issue: %s", e)

    def _get_current_schema_version(self) -> int:
        """Get current schema version from database."""
        if self.conn is None:
            return 0

        try:
            result = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            return result[0] if result is not None and result[0] is not None else 0
        except Exception:
            # Table doesn't exist yet, no migrations applied
            return 0

    def _record_migration(self, version: int) -> None:
        """Record that a migration was applied."""
        if self.conn is None:
            raise DatabaseError("Database connection not available")

        self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    def get_migration_history(self) -> list[tuple[Any, ...]]:
        """Get migration history for debugging/auditing."""
        if self.conn is None:
            return []

        try:
            return self.conn.execute("""
                SELECT version, applied_at
                FROM schema_version
                ORDER BY version
            """).fetchall()
        except Exception:
            # Table doesn't exist yet
            return []

    def _require_conn(self) -> duckdb.DuckDBPyConnection:
        if self.conn is None:
            raise DatabaseError("Database not initialized")
        return self.conn

    # File-state CRUD ---------------------------------------------------------

    def add_file(self, file_path: str, status: str = "pending") -> None:
        _state.add_file(self._require_conn(), file_path, status)

    def update_status(
        self,
        file_path: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        _state.update_status(self._require_conn(), file_path, status, error_message)

    def get_status(self, file_path: str) -> dict[str, Any] | None:
        return _state.get_status(self._require_conn(), file_path)

    def get_files_by_status(self, status: str) -> list[dict[str, Any]]:
        return _state.get_files_by_status(self._require_conn(), status)

    def get_all_files(self) -> list[dict[str, Any]]:
        return _state.get_all_files(self._require_conn())

    def get_summary(self) -> dict[str, int]:
        return _state.get_summary(self._require_conn())

    # Run + file-metric persistence ------------------------------------------

    def record_file_metric(self, record: FileMetricRecord) -> None:
        _file_metrics.record_file_metric(self._require_conn(), record)

    def record_run(self, record: RunRecord) -> int:
        return _runs.record_run(self._require_conn(), record)

    def get_run_by_id(self, run_id: int) -> dict[str, Any] | None:
        return _runs.get_run_by_id(self._require_conn(), run_id)

    def get_run_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        return _runs.get_run_history(self._require_conn(), limit)

    # Lifecycle --------------------------------------------------------------

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
