"""Concrete implementation of StateStore using DuckDB."""

from backend.database import FileMetricRecord, RunRecord, TranscriptionDatabase


class DuckDBStateStore:
    """Concrete StateStore (`backend.services.interfaces.StateStore`) backed by `TranscriptionDatabase`.

    The methods here are thin one-liners — by design. They exist to keep
    `TranscriptionDatabase` (DuckDB-flavored) from leaking into every
    caller of the `StateStore` Protocol. Swapping in an in-memory or
    SQLite-backed store for tests or future backends means writing a
    sibling class, not editing every `state_store=` callsite. Removing
    this wrapper would couple the processor/components layer to the
    concrete DB type and erase the Protocol seam.
    """

    def __init__(self, db: TranscriptionDatabase):
        self._db = db

    def update_status(self, file_path: str, status: str, error_message: str | None = None) -> None:
        self._db.update_status(file_path, status, error_message)

    def record_run(self, run_record: RunRecord) -> int:
        return self._db.record_run(run_record)

    def record_file_metric(self, file_record: FileMetricRecord) -> None:
        self._db.record_file_metric(file_record)
