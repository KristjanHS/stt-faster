"""Shared type definitions."""

from dataclasses import dataclass

from backend.transcribe import TranscriptionMetrics


@dataclass(slots=True)
class FileProcessingStats:
    """Results captured for each processed file."""

    file_path: str
    status: str
    error_message: str | None = None
    metrics: TranscriptionMetrics | None = None
