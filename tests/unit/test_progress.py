"""Tests for the GUI progress channel (backend.progress) and its segment-loop wiring."""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from rich import get_console
from rich.console import Console

from backend.progress import (
    ETA_MIN_SECONDS,
    MIN_INTERVAL_SECONDS,
    EtaEstimator,
    ProgressEvent,
    ProgressReporter,
    RichProgressBar,
    format_eta,
    parse_progress,
)
from backend.transcribe import _collect_segments  # pyright: ignore[reportPrivateUsage]
from backend.variants.executor import _collect_executor_segments  # pyright: ignore[reportPrivateUsage]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _reporter(clock: FakeClock | None = None) -> tuple[ProgressReporter, list[str]]:
    lines: list[str] = []
    return ProgressReporter(enabled=True, write=lines.append, clock=clock or FakeClock(), bar=None), lines


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
    assert events[1] is not None and events[1].stage_fraction == pytest.approx(0.5)
    assert events[2] is not None and events[2].stage_fraction is None  # a bare stage has no fraction


def test_substeps_name_each_step_and_force_its_last_point() -> None:
    clock = FakeClock()
    reporter, lines = _reporter(clock)
    reporter.start_file(1, 2)
    report = reporter.substeps("diarize")
    report("speaker_counting", None, None)
    report("speaker_counting", 1, 4)  # same instant as the entry line: dropped
    report("speaker_counting", 4, 4)  # step end: forced

    assert [parse_progress(line) for line in lines][1:] == [
        ProgressEvent(1, 2, "diarize", detail="speaker counting"),
        ProgressEvent(1, 2, "diarize", 4.0, 4.0, "speaker counting"),
    ]


def test_disabled_reporter_writes_nothing() -> None:
    lines: list[str] = []
    reporter = ProgressReporter(enabled=False, write=lines.append, bar=None)
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


class FakeBar:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.closed = False

    def update(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def test_bar_gets_events_without_the_gui_channel_and_is_closed_by_the_reporter() -> None:
    bar, lines, clock = FakeBar(), [], FakeClock()
    reporter = ProgressReporter(enabled=False, write=lines.append, clock=clock, bar=bar)
    assert not reporter.draws_bar  # no file yet: a bare transcribe() keeps its ⌛ lines
    reporter.start_file(1, 2)
    clock.now += MIN_INTERVAL_SECONDS
    reporter.advance("transcribe", 10.0, 100.0)
    assert reporter.draws_bar
    reporter.close()
    reporter.advance("transcribe", 20.0, 100.0, force=True)  # a stray event after the run
    assert (lines, reporter.draws_bar, bar.closed) == ([], False, True)
    assert bar.events == [ProgressEvent(1, 2, "prepare"), ProgressEvent(1, 2, "transcribe", 10.0, 100.0)]


def test_rich_bar_keeps_one_task_per_stage_and_pulses_on_a_bare_one(request: pytest.FixtureRequest) -> None:
    bar = RichProgressBar(Console(file=io.StringIO(), force_terminal=True, width=100))
    request.addfinalizer(bar.close)  # a failed assertion must not leave Live redirecting stdout
    bar.update(ProgressEvent(1, 2, "transcribe", 30.0, 60.0))
    bar.update(ProgressEvent(1, 2, "transcribe", 45.0, 60.0))
    assert [(t.description, t.completed, t.total) for t in bar.progress.tasks] == [
        ("File 1/2 · Transcribing", 45.0, 60.0)
    ]
    bar.update(ProgressEvent(1, 2, "diarize", detail="segmentation"))
    assert [(t.description, t.total) for t in bar.progress.tasks] == [
        ("File 1/2 · Identifying speakers · segmentation", None)
    ]
    bar.close()
    assert (bar.progress.tasks, bar.progress.live.is_started) == ([], False)


@pytest.mark.parametrize("collect", [_via_transcribe, _via_executor])
@pytest.mark.parametrize("with_bar", [False, True])
def test_terminal_bar_replaces_the_hourglass_log_lines(
    collect: Callable[[list[Any], ProgressReporter], object], with_bar: bool, caplog: pytest.LogCaptureFixture
) -> None:
    reporter = ProgressReporter(enabled=False, clock=FakeClock(), bar=FakeBar() if with_bar else None)
    reporter.files = reporter.file = 1
    caplog.set_level(logging.INFO)
    collect([SimpleNamespace(start=0.0, end=40.0, text="a", words=None)], reporter)
    assert any("⌛" in r.getMessage() for r in caplog.records) is not with_bar


def test_eta_extrapolates_the_stage_rate_after_a_warm_up_and_resets_per_stage() -> None:
    eta = EtaEstimator()
    assert eta.seconds_left(ProgressEvent(1, 1, "transcribe", 0.0, 90.0), now=100.0) is None
    assert eta.seconds_left(ProgressEvent(1, 1, "transcribe", 10.0, 90.0), now=100.0 + ETA_MIN_SECONDS - 1) is None
    assert eta.seconds_left(ProgressEvent(1, 1, "transcribe", 30.0, 90.0), now=110.0) == pytest.approx(20.0)
    assert eta.seconds_left(ProgressEvent(1, 1, "diarize", 1.0, 4.0, "embeddings"), now=111.0) is None
    assert eta.seconds_left(ProgressEvent(1, 1, "diarize", 3.0, 4.0, "embeddings"), now=121.0) == pytest.approx(5.0)
    assert eta.seconds_left(ProgressEvent(1, 1, "diarize", 0.0, 4.0, "embeddings"), now=130.0) is None  # restart


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(20.0, "<1 min left"), (100.0, "~2 min left"), (59 * 60.0, "~59 min left"), (80 * 60.0, "~1 h 20 min left")],
)
def test_format_eta(seconds: float, text: str) -> None:
    assert format_eta(seconds) == text


def test_modules_printing_during_a_file_share_the_bars_console() -> None:
    from backend.cli import db, ui
    from backend.variants import executor, preprocess_steps

    assert all(m.console is get_console() for m in (db, ui, executor, preprocess_steps))
