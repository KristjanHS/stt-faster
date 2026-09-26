"""Tests for the GUI progress channel (backend.progress) and its segment-loop wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from backend.progress import MIN_INTERVAL_SECONDS, ProgressEvent, ProgressReporter, parse_progress
from backend.transcribe import _collect_segments  # pyright: ignore[reportPrivateUsage]
from backend.variants.executor import _collect_executor_segments  # pyright: ignore[reportPrivateUsage]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _reporter(clock: FakeClock | None = None) -> tuple[ProgressReporter, list[str]]:
    lines: list[str] = []
    return ProgressReporter(enabled=True, write=lines.append, clock=clock or FakeClock()), lines


def test_advance_is_throttled_but_stage_changes_are_not() -> None:
    clock = FakeClock()
    reporter, lines = _reporter(clock)
    reporter.start_file(2, 3)
    reporter.advance("transcribe", 10.0, 100.0)  # same instant as the stage line: dropped
    clock.now += MIN_INTERVAL_SECONDS
    reporter.advance("transcribe", 50.0, 100.0)
    reporter.stage("diarize")  # forced despite the interval

    events = [parse_progress(line) for line in lines]
    assert events == [
        ProgressEvent(2, 3, "prepare"),
        ProgressEvent(2, 3, "transcribe", 50.0, 100.0),
        ProgressEvent(2, 3, "diarize"),
    ]
    assert events[1] is not None and events[1].overall_fraction == pytest.approx(0.5)  # (1 + 0.5) / 3


def test_disabled_reporter_writes_nothing() -> None:
    lines: list[str] = []
    reporter = ProgressReporter(enabled=False, write=lines.append)
    reporter.start_file(1, 1)
    reporter.advance("transcribe", 1.0, 2.0)
    assert lines == []


@pytest.mark.parametrize(
    "line", ["⌛ Transcription progress: 1.0/2.0 min (50.0%)", "@@progress {not json", '@@progress {"file": 1}']
)
def test_parse_progress_ignores_other_and_malformed_lines(line: str) -> None:
    assert parse_progress(line) is None


def _via_transcribe(segments: list[Any], reporter: ProgressReporter) -> object:
    return _collect_segments(
        segments,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        total_audio_duration=100.0,
        transcribe_start=0.0,
        reporter=reporter,
    )


def _via_executor(segments: list[Any], reporter: ProgressReporter) -> object:
    return _collect_executor_segments(
        segments,
        transcribe_start=0.0,
        total_audio_duration=100.0,
        no_speech_threshold=None,
        logprob_threshold=None,
        reporter=reporter,
    )


@pytest.mark.parametrize("collect", [_via_transcribe, _via_executor])
def test_segment_loops_report_audio_seconds_and_land_the_file_at_its_end(
    collect: Callable[[list[Any], ProgressReporter], object],
) -> None:
    reporter, lines = _reporter()
    reporter.files = reporter.file = 1
    segments = [SimpleNamespace(start=0.0, end=40.0, text="a", words=None)]
    collect(segments, reporter)
    assert [parse_progress(line) for line in lines] == [
        ProgressEvent(1, 1, "transcribe", 40.0, 100.0),
        ProgressEvent(1, 1, "transcribe", 100.0, 100.0),  # forced end point despite the throttle
    ]
