from __future__ import annotations


class PreprocessError(Exception):
    """Base class for audio pre-processing failures."""


class StepExecutionError(PreprocessError):
    """Raised when a specific step fails."""

    def __init__(self, step: str, message: str):
        self.step = step
        super().__init__(f"{step} failed: {message}")
