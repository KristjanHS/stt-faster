from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Mapping

_TRUE_VALUES = {"1", "true", "yes", "on"}

_PREPROCESS_ENABLED_ENV: Final = "STT_PREPROCESS_ENABLED"
_PREPROCESS_TMP_ENV: Final = "STT_PREPROCESS_TMP_DIR"
_PREPROCESS_TARGET_SR_ENV: Final = "STT_PREPROCESS_TARGET_SR"
_PREPROCESS_TARGET_CH_ENV: Final = "STT_PREPROCESS_TARGET_CH"
_PREPROCESS_PROFILE_ENV: Final = "STT_PREPROCESS_PROFILE"


def _env_enabled(value: str | None) -> bool:
    if value is None:
        # Preprocessing defaults to enabled to ensure the high-quality signal path
        return True
    return value.strip().lower() in _TRUE_VALUES


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_positive_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _env_profile(value: str | None) -> str:
    """Normalize profile preference with GPU-first default."""
    if value is None:
        return "gpu"

    normalized = value.strip().lower()
    if normalized in {"auto", "gpu", "cpu"}:
        return normalized
    return "auto"


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for the audio pre-processing pipeline."""

    enabled: bool = True
    target_sample_rate: int = 22_050
    target_channels: int | None = None
    temp_dir: str | None = None
    profile: str = "gpu"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "PreprocessConfig":
        """Build configuration from environment variables or a provided mapping."""
        source_env: Mapping[str, str] = env or os.environ

        return cls(
            enabled=_env_enabled(source_env.get(_PREPROCESS_ENABLED_ENV)),
            target_sample_rate=_parse_int(source_env.get(_PREPROCESS_TARGET_SR_ENV), 22_050),
            target_channels=_parse_positive_int(source_env.get(_PREPROCESS_TARGET_CH_ENV)),
            temp_dir=source_env.get(_PREPROCESS_TMP_ENV),
            profile=_env_profile(source_env.get(_PREPROCESS_PROFILE_ENV)),
        )
