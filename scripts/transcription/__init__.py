"""Transcription processing and database modules.

DEPRECATED: This module is deprecated. Use backend.database and backend.processor instead.
This module is kept for backward compatibility only.
"""

from backend.database import TranscriptionDatabase
from backend.processor import TranscriptionProcessor

__all__ = ["TranscriptionDatabase", "TranscriptionProcessor"]
