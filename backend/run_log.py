"""Append-only JSONL run log — minimal replacement for the DuckDB state store.

One JSON object per line in ``runs.jsonl``. Top-level fields are flat scalars
(``id``, ``recorded_at``, ``preset`` ...) for trivial ``grep``; logical groups
(``preprocess``, ``model``, ``params``, ``totals``, ``files[]``) live in nested
sub-objects per the Stage G design (``docs/plans/2026-05-20-stage-G-db-to-jsonl.md``
§4.1).

Concurrency note: not designed for concurrent writers — the single-user dev
workflow runs one CLI invocation at a time. ``_next_id`` reads the file then
appends, so a racing writer could repeat an id; an external ``flock`` would
be the proper fix and is out of scope (plan §5 R-3).

Callers MUST convert non-JSON-native types (``datetime``, ``pathlib.Path``)
into strings before passing the record to :meth:`JsonlRunLog.append`. The
encoder has no ``default=`` fallback so unexpected types raise loudly rather
than being silently stringified (plan §8 D-4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from backend.config import get_default_run_log_path

LOGGER = logging.getLogger(__name__)


def _parse_jsonl_record(line: str, source: Path) -> dict[str, Any] | None:
    """Parse a single JSONL line into a dict, or return ``None`` on failure.

    Malformed JSON is logged at WARNING and skipped. Non-object payloads
    (arrays, scalars, null) are also skipped — the log shape is one object
    per line by construction.
    """
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Skipping malformed JSONL line in %s: %s", source, exc)
        return None
    if not isinstance(parsed, dict):
        return None
    return cast("dict[str, Any]", parsed)


class JsonlRunLog:
    """Append-only JSONL log of transcription runs."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else get_default_run_log_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> int:
        """Assign the next ``id`` and append ``record`` as one JSON line.

        Returns the assigned run id. The record is serialised with sorted keys
        and tight separators so byte-identical inputs yield byte-identical
        lines (required by the migration script's idempotency contract).
        """
        run_id = self._next_id()
        record_with_id = {"id": run_id, **record}
        line = json.dumps(record_with_id, separators=(",", ":"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
        return run_id

    def _next_id(self) -> int:
        """Return ``max(id) + 1``; 1 on missing/empty file."""
        if not self.path.exists():
            return 1
        max_id = 0
        with self.path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = _parse_jsonl_record(line, self.path)
                if record is None:
                    continue
                rec_id = record.get("id")
                if isinstance(rec_id, int) and rec_id > max_id:
                    max_id = rec_id
        return max_id + 1

    def tail(self, limit: int) -> list[dict[str, Any]]:
        """Return the last ``limit`` records in stored order (oldest-of-tail first)."""
        if limit <= 0 or not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = _parse_jsonl_record(line, self.path)
                if record is None:
                    continue
                records.append(record)
        return records[-limit:]

    def get(self, run_id: int) -> dict[str, Any] | None:
        """Return the record with the given ``id``, or ``None`` if absent."""
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = _parse_jsonl_record(line, self.path)
                if record is None:
                    continue
                if record.get("id") == run_id:
                    return record
        return None
