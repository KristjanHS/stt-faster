"""Custom exceptions for the transcription backend.

This module defines a hierarchy of exceptions specific to transcription operations,
making error handling more precise and debugging easier.
"""

from __future__ import annotations


class TranscriptionError(Exception):
    """Base exception for all transcription-related errors.

    All custom exceptions in the backend module inherit from this class,
    allowing callers to catch all backend errors with a single except clause.
    """


class ModelLoadError(TranscriptionError):
    """Raised when a model fails to load or initialize.

    This can occur due to:
    - Missing model files
    - Incompatible model format
    - Insufficient GPU memory
    - CUDA/cuDNN library issues
    """


class ModelNotFoundError(TranscriptionError, FileNotFoundError):
    """Raised when a requested model cannot be found.

    Inherits from both TranscriptionError and FileNotFoundError to maintain
    backward compatibility with code catching FileNotFoundError.
    """


class TranscriptionProcessingError(TranscriptionError):
    """Raised when audio transcription processing fails.

    This can occur due to:
    - Corrupted audio files
    - Unsupported audio format
    - Model inference errors
    """
