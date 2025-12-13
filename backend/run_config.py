"""Run configuration with inheritance and validation support."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from backend.preprocess.config import PreprocessConfig, TranscriptionConfig
from backend.variants.variant import Variant

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EffectiveRunConfig:
    """Resolved configuration ready for execution."""

    preprocess: PreprocessConfig
    transcription: TranscriptionConfig

    # Computed fields
    input_folder: Path
    output_format: str
    language: str | None
    model_preset: str


@dataclass(slots=True)
class RunConfig:
    """Validated configuration with inheritance support.

    This class provides a single source of truth for all configuration
    needed to run the transcription pipeline, with proper validation
    and support for environment overrides and variant customizations.
    """

    # Core settings
    input_folder: Path
    output_format: str = "json"
    language: str | None = None
    model_preset: str = "et-large"

    # Component configs with defaults
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)

    # Variant override support
    variant_overrides: Dict[str, Any] = field(default_factory=lambda: {})  # type: ignore[assignment]

    def validate(self) -> None:
        """Validate configuration consistency across all components."""
        # Validate individual component configs
        self._validate_preprocess_config()
        self._validate_transcription_config()

        # Cross-config validation
        self._validate_cross_config_consistency()

        # Variant override validation
        self._validate_variant_overrides()

    def _validate_preprocess_config(self) -> None:
        """Validate preprocess configuration."""
        if self.preprocess.target_sample_rate <= 0:
            raise ValueError(f"target_sample_rate must be positive, got {self.preprocess.target_sample_rate}")

        if self.preprocess.target_channels is not None and self.preprocess.target_channels not in (1, 2):
            raise ValueError(f"target_channels must be 1 or 2, got {self.preprocess.target_channels}")

        if not (0.0 <= self.preprocess.rnnoise_mix <= 1.0):
            raise ValueError(f"rnnoise_mix must be between 0.0 and 1.0, got {self.preprocess.rnnoise_mix}")

    def _validate_transcription_config(self) -> None:
        """Validate transcription configuration."""
        if self.transcription.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.transcription.beam_size}")

        if not (0.0 <= self.transcription.vad_threshold <= 1.0):
            raise ValueError(f"vad_threshold must be between 0.0 and 1.0, got {self.transcription.vad_threshold}")

        if self.transcription.chunk_length <= 0:
            raise ValueError(f"chunk_length must be positive, got {self.transcription.chunk_length}")

        if not (0.0 <= self.transcription.compression_ratio_threshold):
            raise ValueError(
                f"compression_ratio_threshold must be >= 0.0, got {self.transcription.compression_ratio_threshold}"
            )

        if not (0.0 <= self.transcription.logprob_threshold <= 0.0):
            # logprob_threshold is typically negative, but allow 0.0
            if self.transcription.logprob_threshold > 0.0:
                raise ValueError(
                    f"logprob_threshold should typically be negative, got {self.transcription.logprob_threshold}"
                )

        if not (0.0 <= self.transcription.no_speech_threshold <= 1.0):
            raise ValueError(
                f"no_speech_threshold must be between 0.0 and 1.0, got {self.transcription.no_speech_threshold}"
            )

        if self.transcription.best_of < 1:
            raise ValueError(f"best_of must be >= 1, got {self.transcription.best_of}")

        if self.transcription.temperature_increment_on_fallback <= 0.0:
            raise ValueError(
                f"temperature_increment_on_fallback must be positive, got "
                f"{self.transcription.temperature_increment_on_fallback}"
            )

    def _validate_cross_config_consistency(self) -> None:
        """Validate consistency between preprocess and transcription configs."""
        # Warn about potentially problematic combinations
        if self.preprocess.target_sample_rate != 16000 and self.transcription.vad_filter:
            LOGGER.warning(
                "Non-standard sample rate (%d) with VAD enabled may cause issues", self.preprocess.target_sample_rate
            )

        # Check for incompatible settings
        if not self.preprocess.enabled and self.transcription.vad_filter:
            LOGGER.warning("Preprocessing disabled but VAD enabled - VAD may not work optimally on raw audio")

    def _validate_variant_overrides(self) -> None:
        """Validate variant-specific overrides."""
        # Basic validation - could be extended based on variant schema
        if self.variant_overrides:
            LOGGER.debug("Applied variant overrides: %s", list(self.variant_overrides.keys()))

    @classmethod
    def from_env_and_variant(cls, input_folder: Path, variant: Variant | None = None) -> "RunConfig":
        """Build config from environment variables and optional variant overrides.

        Args:
            input_folder: Input folder path
            variant: Optional variant to apply overrides from

        Returns:
            Validated RunConfig instance
        """
        # Start with defaults
        config = cls(input_folder=input_folder)

        # Apply environment overrides
        config.preprocess = PreprocessConfig.from_env()
        config.transcription = TranscriptionConfig.from_env()

        # Apply variant overrides if present
        if variant:
            config.apply_variant_overrides(variant)

        config.validate()
        return config

    def apply_variant_overrides(self, variant: Variant) -> None:
        """Apply variant-specific overrides to the configuration.

        Args:
            variant: Variant containing overrides to apply
        """
        # Override transcription config with variant settings
        variant_kwargs = variant.transcription_config.to_kwargs()
        for key, value in variant_kwargs.items():
            if hasattr(self.transcription, key):
                setattr(self.transcription, key, value)
                self.variant_overrides[f"transcription.{key}"] = value
            else:
                LOGGER.warning("Variant override for unknown transcription field: %s", key)

        # Handle preprocess steps (store for later processing)
        if variant.preprocess_steps:
            # Convert preprocess steps to configuration implications
            self.variant_overrides["preprocess_steps"] = [
                {"name": step.name, "enabled": step.enabled, "step_type": step.step_type}
                for step in variant.preprocess_steps
            ]

            # If any preprocess steps are enabled, ensure preprocessing is enabled
            if any(step.enabled for step in variant.preprocess_steps):
                if not self.preprocess.enabled:
                    LOGGER.info("Enabling preprocessing due to variant requirements")
                    self.preprocess.enabled = True
                    self.variant_overrides["preprocess.enabled"] = True

    def get_effective_config(self) -> EffectiveRunConfig:
        """Return fully resolved configuration for execution.

        Returns:
            EffectiveRunConfig with all settings resolved and ready for use
        """
        return EffectiveRunConfig(
            preprocess=self.preprocess,
            transcription=self.transcription,
            input_folder=self.input_folder,
            output_format=self.output_format,
            language=self.language,
            model_preset=self.model_preset,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization/logging."""
        return {
            "input_folder": str(self.input_folder),
            "output_format": self.output_format,
            "language": self.language,
            "model_preset": self.model_preset,
            "preprocess": {
                "enabled": self.preprocess.enabled,
                "target_sample_rate": self.preprocess.target_sample_rate,
                "target_channels": self.preprocess.target_channels,
                "profile": self.preprocess.profile,
                "loudnorm_preset": self.preprocess.loudnorm_preset,
                "rnnoise_model": self.preprocess.rnnoise_model,
                "rnnoise_mix": self.preprocess.rnnoise_mix,
            },
            "transcription": {
                "beam_size": self.transcription.beam_size,
                "patience": self.transcription.patience,
                "word_timestamps": self.transcription.word_timestamps,
                "task": self.transcription.task,
                "chunk_length": self.transcription.chunk_length,
                "vad_filter": self.transcription.vad_filter,
                "vad_threshold": self.transcription.vad_threshold,
                "best_of": self.transcription.best_of,
                "compression_ratio_threshold": self.transcription.compression_ratio_threshold,
                "logprob_threshold": self.transcription.logprob_threshold,
                "no_speech_threshold": self.transcription.no_speech_threshold,
            },
            "variant_overrides": self.variant_overrides,
        }
