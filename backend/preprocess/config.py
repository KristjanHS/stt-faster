from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

_TRUE_VALUES = {"1", "true", "yes", "on"}

_PREPROCESS_ENABLED_ENV: Final = "STT_PREPROCESS_ENABLED"
_PREPROCESS_TMP_ENV: Final = "STT_PREPROCESS_TMP_DIR"
_PREPROCESS_TARGET_SR_ENV: Final = "STT_PREPROCESS_TARGET_SR"
_PREPROCESS_TARGET_CH_ENV: Final = "STT_PREPROCESS_TARGET_CH"


def _env_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for the audio pre-processing pipeline."""

    enabled: bool = False
    target_sample_rate: int = 16_000
    target_channels: int = 1
    temp_dir: str | None = None
    profile: str = "default_cpu"

    @classmethod
    def from_env(cls) -> "PreprocessConfig":
        """Build configuration from environment variables."""
        return cls(
            enabled=_env_enabled(os.getenv(_PREPROCESS_ENABLED_ENV)),
            target_sample_rate=_env_int(_PREPROCESS_TARGET_SR_ENV, 16_000),
            target_channels=_env_int(_PREPROCESS_TARGET_CH_ENV, 1),
            temp_dir=os.getenv(_PREPROCESS_TMP_ENV),
        )
