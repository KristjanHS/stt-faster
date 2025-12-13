"""Service interfaces for transcription system using structural typing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from backend.database import FileMetricRecord, RunRecord
from backend.preprocess.config import PreprocessConfig
from backend.preprocess.orchestrator import PreprocessResult
from backend.transcribe import TranscriptionMetrics


@dataclass
class TranscriptionRequest:
    """Request data for transcription service."""

    audio_path: str
    output_path: str
    preset: str
    language: str | None = None


@dataclass
class TranscriptionResult:
    """Result data from transcription service."""

    metrics: TranscriptionMetrics | None
    payload: dict[str, Any]


@runtime_checkable
class TranscriptionService(Protocol):
    """Service for transcribing audio files."""

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio and return result with metrics."""
        ...


@runtime_checkable
class PreprocessService(Protocol):
    """Service for preprocessing audio files."""

    def preprocess(self, audio_path: str, config: PreprocessConfig) -> PreprocessResult:
        """Preprocess audio file."""
        ...


@runtime_checkable
class StateStore(Protocol):
    """Service for managing transcription state."""

    def update_status(self, file_path: str, status: str, error_message: str | None = None) -> None:
        """Update file processing status."""
        ...

    def record_run(self, run_record: RunRecord) -> int:
        """Record a processing run and return run_id."""
        ...

    def record_file_metric(self, file_record: FileMetricRecord) -> None:
        """Record file-level metrics."""
        ...


@runtime_checkable
class FileMover(Protocol):
    """Service for moving files."""

    def move(self, source: str, destination: str) -> str | None:
        """Move file from source to destination."""
        ...


@runtime_checkable
class OutputWriter(Protocol):
    """Service for writing transcription output."""

    def write(self, output_path: str, payload: dict[str, Any], format: str) -> list[Path]:
        """Write transcription output in specified format. Returns list of created files."""
        ...
