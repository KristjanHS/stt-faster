"""Concrete implementation of StateStore using DuckDB."""

from backend.database import FileMetricRecord, RunRecord, TranscriptionDatabase


class DuckDBStateStore:
    """Concrete implementation using DuckDB."""

    def __init__(self, db: TranscriptionDatabase):
        self._db = db

    def update_status(self, file_path: str, status: str, error_message: str | None = None) -> None:
        self._db.update_status(file_path, status, error_message)

    def record_run(self, run_record: RunRecord) -> int:
        return self._db.record_run(run_record)

    def record_file_metric(self, file_record: FileMetricRecord) -> None:
        self._db.record_file_metric(file_record)
