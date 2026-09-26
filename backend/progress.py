"""Machine-readable progress for the GUI: ``@@progress {json}`` lines on stdout.

Human log lines stay as they are; the GUI reads only lines carrying
:data:`PROGRESS_PREFIX`. Emission is off unless :data:`PROGRESS_ENV` is ``"1"``
(the GUI sets it), so a terminal CLI run prints nothing extra.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

PROGRESS_PREFIX = "@@progress "
PROGRESS_ENV = "STT_PROGRESS"
MIN_INTERVAL_SECONDS = 1.0


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


def _write_stdout(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


@dataclass
class ProgressReporter:
    """Tracks the current file and emits throttled :class:`ProgressEvent` lines."""

    enabled: bool = field(default_factory=lambda: os.environ.get(PROGRESS_ENV) == "1")
    write: Callable[[str], None] = _write_stdout
    clock: Callable[[], float] = time.monotonic
    file: int = 0
    files: int = 0
    _last_emit: float | None = None

    def start_file(self, index: int, total: int) -> None:
        self.file, self.files = index, total
        self.stage("prepare")

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
        if not self.enabled or self.files <= 0:
            return
        now = self.clock()
        if not force and self._last_emit is not None and now - self._last_emit < MIN_INTERVAL_SECONDS:
            return
        self._last_emit = now
        payload: dict[str, object] = {"file": event.file, "files": event.files, "stage": event.stage}
        if event.detail is not None:
            payload["detail"] = event.detail
        if event.done is not None:
            payload["done"] = round(event.done, 1)
            payload["total"] = None if event.total is None else round(event.total, 1)
        self.write(PROGRESS_PREFIX + json.dumps(payload))


REPORTER = ProgressReporter()
"""Process-wide reporter: the file loop sets the file, the segment loops advance it."""
