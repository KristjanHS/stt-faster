"""Diarization pipeline — overlap-assign + chronological speaker relabel.

`annotate(segments, audio_path, *, num_speakers, runner)` is the public entry
point. It calls the pyannote runner, attaches a `speaker` field to each
whisper segment via majority-overlap assignment, then renumbers speaker
labels by chronological appearance so SPEAKER_00 is always the first speaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def overlap_assign(
    segments: list[dict[str, Any]],
    turns: list[SpeakerTurn],
) -> list[dict[str, Any]]:
    """Assign each segment the speaker whose turn covers the most of it.

    Segments with zero overlap to any turn (including zero-length segments —
    start == end) are returned without a `speaker` field. On ties, the
    earlier-starting turn wins because `turns` is expected sorted by start.
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        best_overlap = 0.0
        best_speaker: str | None = None
        for turn in turns:
            overlap = min(seg_end, turn.end) - max(seg_start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        new_seg = dict(seg)
        if best_speaker is not None:
            new_seg["speaker"] = best_speaker
        out.append(new_seg)
    return out


def anchor_speaker_zero(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Renumber speaker labels by chronological appearance.

    The first segment with a `speaker` field defines SPEAKER_00, the next
    unseen speaker becomes SPEAKER_01, etc. Segments without a `speaker`
    field pass through unchanged. Re-runs of the same audio produce stable
    labels regardless of pyannote's internal ordering.
    """
    mapping: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for seg in segments:
        new_seg = dict(seg)
        original = seg.get("speaker")
        if original is not None:
            if original not in mapping:
                mapping[original] = f"SPEAKER_{len(mapping):02d}"
            new_seg["speaker"] = mapping[original]
        out.append(new_seg)
    return out


def annotate(
    segments: list[dict[str, Any]],
    audio_path: str,
    *,
    num_speakers: int = 2,
    runner: Callable[..., list[SpeakerTurn]] | None = None,
    audio_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Attach speaker labels to whisper segments via pyannote diarization.

    Returns segments with a `speaker` field added (or unchanged if the runner
    returned zero turns — e.g. silence/music). Performs full chronological
    relabel via `anchor_speaker_zero` so SPEAKER_00 is deterministic.

    The `runner` parameter is injected for testability; default is the real
    pyannote pipeline (imported lazily to avoid loading torch at import time).
    ``audio_duration`` (in seconds, optional) is forwarded to the runner so it
    can render heartbeat progress against the source-audio minutes.
    """
    if runner is None:
        from backend.diarize.pyannote_runner import run_pyannote

        runner = run_pyannote
    turns = runner(audio_path, num_speakers=num_speakers, audio_duration=audio_duration)
    if not turns:
        return list(segments)
    assigned = overlap_assign(segments, turns)
    return anchor_speaker_zero(assigned)
