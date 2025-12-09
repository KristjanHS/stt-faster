"""Core variant dataclass definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass(slots=True)
class PreprocessStep:
    """Definition of a preprocessing step.

    Args:
        name: Human-readable name for the step
        enabled: Whether this step should be executed
        step_type: Type of preprocessing step ("ffmpeg", "denoise", "resample")
        config: Optional configuration dictionary for step-specific parameters
    """

    name: str
    enabled: bool
    step_type: Literal["ffmpeg", "denoise", "resample"]
    config: dict[str, Any] | None = None


@dataclass(slots=True)
class Variant:
    """Definition of a transcription variant.

    A variant combines preprocessing steps with transcription configuration
    to create a complete transcription pipeline configuration.

    Args:
        name: Unique identifier for the variant (e.g., "no_preprocessing")
        number: Display number for the variant (1-15)
        description: Short description for filenames (e.g., "nopre_projdef")
        preprocess_steps: List of preprocessing steps to execute (used if custom_preprocess_runner is None)
        transcription_preset: Preset name ("project", "industry", "minimal")
        transcription_overrides: Optional dict to override specific transcription parameters
        custom_preprocess_runner: Optional custom preprocessing function (takes path, config, **kwargs)
    """

    name: str
    number: int
    description: str
    preprocess_steps: list[PreprocessStep] = field(default_factory=list)  # type: ignore[assignment]
    transcription_preset: str = "project"  # "project", "industry", "minimal"
    transcription_overrides: dict[str, Any] | None = None
    custom_preprocess_runner: Callable[..., Any] | None = None  # type: ignore[type-arg]
