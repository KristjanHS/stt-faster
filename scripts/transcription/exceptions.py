"""Custom exceptions for the transcription automation system.

This module defines exceptions specific to the file processing and
database operations of the transcription automation workflow.
"""

from __future__ import annotations


class TranscriptionAutomationError(Exception):
    """Base exception for transcription automation errors.

    All custom exceptions in the transcription automation module
    inherit from this class.
    """


class DatabaseError(TranscriptionAutomationError, RuntimeError):
    """Raised when database operations fail.

    This can occur due to:
    - Database not initialized
    - Connection failures
    - SQL execution errors
    - Constraint violations

    Inherits from both TranscriptionAutomationError and RuntimeError
    to maintain backward compatibility.
    """


class FileProcessingError(TranscriptionAutomationError):
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
