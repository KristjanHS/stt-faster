"""Core variant dataclass definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from backend.preprocess.config import TranscriptionConfig


@dataclass(slots=True)
class PreprocessStep:
    """Definition of a preprocessing step.

    Args:
        name: Human-readable name for the step
        enabled: Whether this step should be executed
        step_type: Type of preprocessing step (
            "ffmpeg", "denoise", "resample", "loudnorm_only",
            "loudnorm_highpass", "dynaudnorm", "denoise_custom",
            "highlow_aform_loudnorm", "highlow_nosampl_loudnorm",
            "aresampl_loudnorm_fixed", "aresampl_loudnorm_fixed2",
            "loudnorm_2pass_linear"
        )
        config: Optional configuration dictionary for step-specific parameters
    """

    name: str
    enabled: bool
    step_type: Literal[
        "ffmpeg",
        "denoise",
        "resample",
        "loudnorm_only",
        "loudnorm_highpass",
        "dynaudnorm",
        "denoise_custom",
        "highlow_aform_loudnorm",
        "highlow_nosampl_loudnorm",
        "aresampl_loudnorm_fixed",
        "aresampl_loudnorm_fixed2",
        "loudnorm_2pass_linear",
    ]
    config: dict[str, Any] | None = None


@dataclass(slots=True)
class Variant:
    """Definition of a transcription variant.

    A variant combines preprocessing steps with transcription configuration
    to create a complete transcription pipeline configuration.

    Args:
        name: Unique identifier for the variant (e.g., "no_preprocessing")
        number: Display number for the variant (1-25)
        preprocess_steps: List of preprocessing steps to execute (used if custom_preprocess_runner is None)
        transcription_config: TranscriptionConfig with all parameters for this variant
        custom_preprocess_runner: Optional custom preprocessing function (takes path, config, **kwargs)
    """

    name: str
    number: int
    preprocess_steps: list[PreprocessStep] = field(default_factory=list)  # type: ignore[assignment]
    transcription_config: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    custom_preprocess_runner: Callable[..., Any] | None = None  # type: ignore[type-arg]
