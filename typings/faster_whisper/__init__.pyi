from __future__ import annotations

from typing import Any, BinaryIO, Iterable

class Segment:
    start: float | None
    end: float | None
    text: str

class TranscriptionInfo:
    language: str | None
    language_probability: float

class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        *,
        device: str = ...,
        compute_type: str = ...,
    ) -> None: ...
    def transcribe(
        self,
        audio: str | BinaryIO | Any,
        *,
        language: str | None = ...,
        task: str = ...,
        **kwargs: Any,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...
