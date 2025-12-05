"""Custom exceptions for the transcription system.

This module defines a unified hierarchy of exceptions for all transcription operations,
including model loading, audio processing, database operations, and file management.

All exceptions inherit from TranscriptionError base class for consistent error handling.
"""

from __future__ import annotations


class TranscriptionError(Exception):
    """Base exception for all transcription-related errors.

    All custom exceptions in the backend module inherit from this class,
    allowing callers to catch all transcription errors with a single except clause.
    """


# Model-related exceptions


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


# Processing exceptions


class TranscriptionProcessingError(TranscriptionError):
    """Raised when audio transcription processing fails.

    This can occur due to:
    - Corrupted audio files
    - Unsupported audio format
    - Model inference errors
    """


# Database exceptions


class DatabaseError(TranscriptionError, RuntimeError):
    """Raised when database operations fail.

    This can occur due to:
    - Database not initialized
    - Connection failures
    - SQL execution errors
    - Constraint violations

    Inherits from both TranscriptionError and RuntimeError
    to maintain backward compatibility.
    """


# File operations exceptions


class FileProcessingError(TranscriptionError):
    """Raised when file processing operations fail.

    This can occur during:
    - Audio file transcription
    - JSON output generation
    - File validation
    """


class FileMoveError(FileProcessingError, OSError):
    """Raised when file move/copy operations fail.

    This can occur due to:
    - Permission issues
    - Disk space problems
    - File system errors

    Inherits from OSError to maintain compatibility with OS-level
    error handling patterns.
    """
