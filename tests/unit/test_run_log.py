"""Unit tests for ``backend.run_log.JsonlRunLog``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.run_log import JsonlRunLog


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.jsonl"


@pytest.fixture
def log(log_path: Path) -> JsonlRunLog:
    return JsonlRunLog(log_path)


class TestNextId:
    def test_missing_file_returns_one(self, log: JsonlRunLog) -> None:
        assert log._next_id() == 1

    def test_empty_file_returns_one(self, log: JsonlRunLog, log_path: Path) -> None:
        log_path.touch()
        assert log._next_id() == 1

    def test_existing_records_returns_max_plus_one(self, log: JsonlRunLog, log_path: Path) -> None:
        log_path.write_text('{"id":1}\n{"id":7}\n{"id":3}\n', encoding="utf-8")
        assert log._next_id() == 8


class TestAppend:
    def test_returns_id_and_writes_valid_json_line(self, log: JsonlRunLog, log_path: Path) -> None:
        assigned = log.append({"preset": "et-large"})
        assert assigned == 1
        text = log_path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        line = text.rstrip("\n")
        parsed = json.loads(line)
        assert parsed == {"id": 1, "preset": "et-large"}

    def test_multiple_appends_produce_monotonic_ids(self, log: JsonlRunLog) -> None:
        ids = [log.append({"n": i}) for i in range(3)]
        assert ids == [1, 2, 3]

    def test_append_after_manual_gap_continues_from_max(self, log: JsonlRunLog, log_path: Path) -> None:
        # Pre-seed the log with a gap (id=1, id=5) then append.
        log_path.write_text('{"id":1}\n{"id":5}\n', encoding="utf-8")
        assigned = log.append({"x": "y"})
        # max+1, not "next sequential after 1".
        assert assigned == 6


class TestTail:
    def test_returns_all_when_limit_ge_count(self, log: JsonlRunLog) -> None:
        for i in range(3):
            log.append({"n": i})
        result = log.tail(10)
        assert len(result) == 3
        assert [r["n"] for r in result] == [0, 1, 2]

    def test_returns_last_n_when_limit_lt_count(self, log: JsonlRunLog) -> None:
        for i in range(5):
            log.append({"n": i})
        result = log.tail(2)
        assert [r["n"] for r in result] == [3, 4]

    def test_returns_empty_on_missing_file(self, log_path: Path) -> None:
        # Construct on a path that doesn't exist yet, then never write.
        log = JsonlRunLog(log_path)
        log_path.unlink(missing_ok=True)  # __init__ may have created the dir only
        assert log.tail(10) == []


class TestGet:
    def test_returns_matching_record(self, log: JsonlRunLog) -> None:
        log.append({"label": "a"})
        log.append({"label": "b"})
        result = log.get(2)
        assert result is not None
        assert result["label"] == "b"

    def test_returns_none_for_missing_id(self, log: JsonlRunLog) -> None:
        log.append({"label": "a"})
        assert log.get(999) is None

    def test_returns_none_on_missing_file(self, log_path: Path) -> None:
        log = JsonlRunLog(log_path)
        log_path.unlink(missing_ok=True)
        assert log.get(1) is None


class TestMalformedTolerance:
    def test_skips_malformed_lines(self, log_path: Path) -> None:
        log = JsonlRunLog(log_path)
        log.append({"n": 1})
        log.append({"n": 2})
        # Inject a malformed line between valid ones.
        with log_path.open("a", encoding="utf-8") as f:
            f.write("not json\n")
        log.append({"n": 4})  # noqa: T201 — return value unused intentionally

        # _next_id should be 5 (after the 4th valid append's id=4, since the
        # malformed line was inserted *after* ids 1 and 2 but before the 3rd
        # valid append got id=3 ... wait: the 3rd append happened AFTER the
        # malformed line, and _next_id ignores it, so assigned id was 3).
        # Total valid records on disk: 3. Tail should return 3.
        assert log._next_id() == 4
        records = log.tail(10)
        assert len(records) == 3
        assert [r["n"] for r in records] == [1, 2, 4]


class TestRoundTrip:
    def test_nested_record_round_trips_via_get(self, log: JsonlRunLog) -> None:
        recorded_at = datetime(2026, 5, 20, 12, 34, 56, tzinfo=timezone.utc)
        source_path = Path("/tmp/audio.mp3")
        # Per D-4: caller is responsible for converting non-JSON-native types
        # to strings BEFORE calling append.
        record = {
            "recorded_at": recorded_at.isoformat().replace("+00:00", "Z"),
            "input_folder": str(source_path.parent),
            "preset": "et-large",
            "language": "et",
            "preprocess": {"enabled": True, "profile": "cpu"},
            "model": {"id": "fake-model", "device": "cpu"},
            "params": {"beam_size": 5, "vad": {"filter": True, "threshold": 0.5}},
            "totals": {"files_found": 1, "succeeded": 1},
            "files": [{"path": str(source_path), "status": "completed", "audio_duration_s": 12.5}],
        }
        assigned = log.append(record)
        loaded = log.get(assigned)
        assert loaded is not None
        expected = {"id": assigned, **record}
        assert loaded == expected
