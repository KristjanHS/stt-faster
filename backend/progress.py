"""Progress for the GUI (``@@progress {json}`` lines on stdout) and a Rich bar for a terminal CLI run.

The GUI reads only lines carrying :data:`PROGRESS_PREFIX`; emission is off unless
:data:`PROGRESS_ENV` is ``"1"`` (the GUI sets it). Without it, a TTY stdout gets a
:class:`RichProgressBar` in place of the ``⌛`` progress log lines; piped output is unchanged.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeRemainingColumn

PROGRESS_PREFIX = "@@progress "
PROGRESS_ENV = "STT_PROGRESS"
MIN_INTERVAL_SECONDS = 1.0
ETA_MIN_SECONDS = 5.0  # a stage's rate is too noisy to extrapolate before this much wall time


@dataclass(frozen=True)
class ProgressEvent:
    file: int  # 1-based
    files: int
    stage: str
    done: float | None = None
    total: float | None = None
    detail: str | None = None  # sub-step within the stage, e.g. a pyannote step name

    @property
    def stage_fraction(self) -> float | None:
        """Fraction of the current stage, or None when the event carries no quantities."""
        if self.done is None or not self.total or self.total <= 0:
            return None
        return min(max(self.done / self.total, 0.0), 1.0)


def parse_progress(line: str) -> ProgressEvent | None:
    """The event on ``line``, or None for any other (or malformed) line."""
    _, sep, payload = line.partition(PROGRESS_PREFIX)
    if not sep:
        return None
    try:
        data = json.loads(payload)
        return ProgressEvent(
            file=int(data["file"]),
            files=int(data["files"]),
            stage=str(data["stage"]),
            done=None if data.get("done") is None else float(data["done"]),
            total=None if data.get("total") is None else float(data["total"]),
            detail=None if data.get("detail") is None else str(data["detail"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


STAGE_LABELS = {
    "prepare": "Preparing",
    "preprocess": "Preprocessing",
    "load model": "Loading model",
    "transcribe": "Transcribing",
    "diarize": "Identifying speakers",
}


def describe_progress(event: ProgressEvent) -> str:
    """One status line for ``event``: file position, stage, sub-step."""
    parts = [f"File {event.file}/{event.files}", STAGE_LABELS.get(event.stage, event.stage.capitalize())]
    if event.detail:
        parts.append(event.detail)
    return " · ".join(parts)


@dataclass
class EtaEstimator:
    """Seconds left in the current stage, from its rate since the stage's first quantified event."""

    _key: tuple[int, int, str, str | None] | None = None
    _start: tuple[float, float] | None = None  # (now, done) at the stage's first quantified event

    def seconds_left(self, event: ProgressEvent, now: float) -> float | None:
        key = (event.file, event.files, event.stage, event.detail)
        if event.done is None or not event.total:
            self._key, self._start = key, None
            return None
        if key != self._key or self._start is None or event.done < self._start[1]:
            self._key, self._start = key, (now, event.done)
            return None
        started, done_then = self._start
        elapsed = now - started
        if elapsed < ETA_MIN_SECONDS or event.done <= done_then:
            return None
        rate = (event.done - done_then) / elapsed
        return max(event.total - event.done, 0.0) / rate


def format_eta(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return "<1 min left"
    if minutes < 60:
        return f"~{minutes} min left"
    return f"~{minutes // 60} h {minutes % 60} min left"


class ProgressBar(Protocol):
    def update(self, event: ProgressEvent) -> None: ...

    def close(self) -> None: ...


class RichProgressBar:
    """One transient Rich task per file/stage/sub-step; a stage without quantities pulses."""

    def __init__(self, console: Console | None = None) -> None:
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        self._task: TaskID | None = None
        self._key: tuple[int, int, str, str | None] | None = None

    def update(self, event: ProgressEvent) -> None:
        if self._task is None:
            self.progress.start()
            atexit.register(self.close)  # restores the cursor if the process dies mid-file
        key = (event.file, event.files, event.stage, event.detail)
        if key != self._key or self._task is None:
            if self._task is not None:
                self.progress.remove_task(self._task)
            self._task = self.progress.add_task(describe_progress(event), total=event.total)
            self._key = key
        if event.done is not None:
            self.progress.update(self._task, completed=event.done, total=event.total)

    def close(self) -> None:
        atexit.unregister(self.close)
        if self._task is not None:
            self.progress.stop()
            self.progress.remove_task(self._task)
            self._task = self._key = None


def _default_bar() -> ProgressBar | None:
    if os.environ.get(PROGRESS_ENV) == "1" or not sys.stdout.isatty():
        return None
    return RichProgressBar()


def _write_stdout(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


@dataclass
class ProgressReporter:
    """Tracks the current file and emits throttled :class:`ProgressEvent` lines and/or bar updates."""

    enabled: bool = field(default_factory=lambda: os.environ.get(PROGRESS_ENV) == "1")
    write: Callable[[str], None] = _write_stdout
    bar: ProgressBar | None = field(default_factory=_default_bar)
    clock: Callable[[], float] = time.monotonic
    file: int = 0
    files: int = 0
    _last_emit: float | None = None

    def start_file(self, index: int, total: int) -> None:
        self.file, self.files = index, total
        self.stage("prepare")

    @property
    def draws_bar(self) -> bool:
        """True when a terminal bar stands in for the ``⌛`` progress log lines."""
        return self.bar is not None and self.files > 0

    def close(self) -> None:
        """End the run: close the bar and forget the file, so later events neither redraw nor mute ⌛ lines."""
        self.file = self.files = 0
        self._last_emit = None
        if self.bar is not None:
            self.bar.close()

    def stage(self, name: str, detail: str | None = None) -> None:
        """Announce a stage change; never throttled."""
        self._emit(ProgressEvent(self.file, self.files, name, detail=detail), force=True)

    def advance(
        self, stage: str, done: float, total: float | None, *, detail: str | None = None, force: bool = False
    ) -> None:
        """Report ``done`` of ``total`` within ``stage``; throttled to one line per interval unless ``force``."""
        self._emit(ProgressEvent(self.file, self.files, stage, done, total, detail), force=force)

    def substeps(self, stage: str) -> Callable[[str, int | None, int | None], None]:
        """A ``(step, completed, total)`` callback for sub-steps of ``stage``; ``completed=None`` = step entry."""

        def report(step: str, completed: int | None, total: int | None) -> None:
            detail = step.replace("_", " ")
            if completed is None or not total:
                self.stage(stage, detail)
            else:
                self.advance(stage, completed, total, detail=detail, force=completed >= total)

        return report

    def _emit(self, event: ProgressEvent, *, force: bool) -> None:
        if (not self.enabled and self.bar is None) or self.files <= 0:
            return
        now = self.clock()
        if not force and self._last_emit is not None and now - self._last_emit < MIN_INTERVAL_SECONDS:
            return
        self._last_emit = now
        if self.bar is not None:
            self.bar.update(event)
        if not self.enabled:
            return
        payload: dict[str, object] = {"file": event.file, "files": event.files, "stage": event.stage}
        if event.detail is not None:
            payload["detail"] = event.detail
        if event.done is not None:
            payload["done"] = round(event.done, 1)
            payload["total"] = None if event.total is None else round(event.total, 1)
        self.write(PROGRESS_PREFIX + json.dumps(payload))


REPORTER = ProgressReporter()
"""Process-wide reporter: the file loop sets the file, the segment loops advance it."""
