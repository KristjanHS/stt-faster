"""Transcription configuration presets for variants."""

from __future__ import annotations

from typing import Any

from backend.preprocess.config import TranscriptionConfig


def get_project_defaults() -> TranscriptionConfig:
    """Get TranscriptionConfig with project defaults.

    Returns the configuration from environment variables or project defaults.
    This is the standard configuration used by the project.
    """
    return TranscriptionConfig.from_env()


def get_minimal_config() -> TranscriptionConfig:
    """Get TranscriptionConfig with minimal parameters.

    This config provides minimal parameters, allowing faster-whisper
    to use its own internal defaults for most parameters. The executor
    will handle omitting parameters when calling model.transcribe().
    """
    return TranscriptionConfig(
        word_timestamps=False,
        task="transcribe",
    )


def get_transcription_config(preset: str) -> TranscriptionConfig:
    """Get transcription config by preset name.

    Args:
        preset: One of "project" or "minimal"

    Returns:
        TranscriptionConfig instance

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset == "project":
        return get_project_defaults()
    if preset == "minimal":
        return get_minimal_config()
    raise ValueError(f"Unknown transcription preset: {preset}")


def create_minimal_config(
    beam_size: int = 5,
    word_timestamps: bool = False,
    task: str = "transcribe",
    **overrides: Any,
) -> TranscriptionConfig:
    """Create minimal config with only essential params set.

    This creates a TranscriptionConfig with only the minimal parameters set,
    allowing faster-whisper to use its own internal defaults for omitted parameters.

    Args:
        beam_size: Beam search size (default: 5)
        word_timestamps: Enable word-level timestamps (default: False)
        task: Transcription task ("transcribe" or "translate", default: "transcribe")
        **overrides: Any additional TranscriptionConfig parameters to override

    Returns:
        TranscriptionConfig with only specified parameters set
    """
    config = TranscriptionConfig(
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        task=task,
    )
    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_project_config(**overrides: Any) -> TranscriptionConfig:
    """Create config with project defaults, allowing overrides.

    Args:
        **overrides: Any TranscriptionConfig parameters to override

    Returns:
        TranscriptionConfig with project defaults and any overrides applied
    """
    config = TranscriptionConfig.from_env()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
