from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final, Mapping

# Preset configurations for loudnorm
_LOUDNORM_PRESETS: Final[dict[str, dict[str, float]]] = {
    "default": {
        "I": -20.0,
        "TP": -2.0,
        "LRA": 7.0,
    },
    "boost-quiet-voices": {
        "I": -18.0,
        "TP": -2.0,
        "LRA": 7.0,
    },
}

_TRUE_VALUES = {"1", "true", "yes", "on"}

_PREPROCESS_ENABLED_ENV: Final = "STT_PREPROCESS_ENABLED"
_PREPROCESS_TMP_ENV: Final = "STT_PREPROCESS_TMP_DIR"
_PREPROCESS_OUTPUT_DIR_ENV: Final = "STT_PREPROCESS_OUTPUT_DIR"
_PREPROCESS_TARGET_SR_ENV: Final = "STT_PREPROCESS_TARGET_SR"
_PREPROCESS_TARGET_CH_ENV: Final = "STT_PREPROCESS_TARGET_CH"
_PREPROCESS_PROFILE_ENV: Final = "STT_PREPROCESS_PROFILE"
_PREPROCESS_LOUDNORM_PRESET_ENV: Final = "STT_PREPROCESS_LOUDNORM_PRESET"
_PREPROCESS_RNNOISE_MODEL_ENV: Final = "STT_PREPROCESS_RNNOISE_MODEL"
_PREPROCESS_RNNOISE_MIX_ENV: Final = "STT_PREPROCESS_RNNOISE_MIX"
_RNNOISE_MIX_DEFAULT: Final = 0.5
_LOUDNORM_PRESET_DEFAULT: Final = "default"

# Transcription parameter environment variables
_TRANSCRIBE_BEAM_SIZE_ENV: Final = "STT_TRANSCRIBE_BEAM_SIZE"
_TRANSCRIBE_WORD_TIMESTAMPS_ENV: Final = "STT_TRANSCRIBE_WORD_TIMESTAMPS"
_TRANSCRIBE_TASK_ENV: Final = "STT_TRANSCRIBE_TASK"
_TRANSCRIBE_CHUNK_LENGTH_ENV: Final = "STT_TRANSCRIBE_CHUNK_LENGTH"
_TRANSCRIBE_VAD_THRESHOLD_ENV: Final = "STT_TRANSCRIBE_VAD_THRESHOLD"
_TRANSCRIBE_PATIENCE_ENV: Final = "STT_TRANSCRIBE_PATIENCE"
_TRANSCRIBE_TEMPERATURE_ENV: Final = "STT_TRANSCRIBE_TEMPERATURE"
_TRANSCRIBE_TEMPERATURE_INCREMENT_ENV: Final = "STT_TRANSCRIBE_TEMPERATURE_INCREMENT"
_TRANSCRIBE_COMPRESSION_RATIO_THRESHOLD_ENV: Final = "STT_TRANSCRIBE_COMPRESSION_RATIO_THRESHOLD"
_TRANSCRIBE_LOGPROB_THRESHOLD_ENV: Final = "STT_TRANSCRIBE_LOGPROB_THRESHOLD"
_TRANSCRIBE_NO_SPEECH_THRESHOLD_ENV: Final = "STT_TRANSCRIBE_NO_SPEECH_THRESHOLD"
_TRANSCRIBE_BEST_OF_ENV: Final = "STT_TRANSCRIBE_BEST_OF"
_TRANSCRIBE_LENGTH_PENALTY_ENV: Final = "STT_TRANSCRIBE_LENGTH_PENALTY"
_TRANSCRIBE_SUPPRESS_TOKENS_ENV: Final = "STT_TRANSCRIBE_SUPPRESS_TOKENS"
_TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT_ENV: Final = "STT_TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT"
_TRANSCRIBE_INITIAL_PROMPT_ENV: Final = "STT_TRANSCRIBE_INITIAL_PROMPT"
_TRANSCRIBE_VAD_FILTER_ENV: Final = "STT_TRANSCRIBE_VAD_FILTER"
_TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS_ENV: Final = "STT_TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS"
_TRANSCRIBE_VAD_MAX_SPEECH_DURATION_S_ENV: Final = "STT_TRANSCRIBE_VAD_MAX_SPEECH_DURATION_S"
_TRANSCRIBE_VAD_MIN_SILENCE_DURATION_MS_ENV: Final = "STT_TRANSCRIBE_VAD_MIN_SILENCE_DURATION_MS"
_TRANSCRIBE_VAD_SPEECH_PAD_MS_ENV: Final = "STT_TRANSCRIBE_VAD_SPEECH_PAD_MS"
_LOUDNORM_PRESET_ALIASES: Final = {
    _LOUDNORM_PRESET_DEFAULT: _LOUDNORM_PRESET_DEFAULT,
    "boost-quiet-voices": "boost-quiet-voices",
    "boost quiet voices": "boost-quiet-voices",
    "boost_quiet_voices": "boost-quiet-voices",
    "boostquietvoices": "boost-quiet-voices",
}


def _env_enabled(value: str | None) -> bool:
    if value is None:
        # Preprocessing defaults to enabled to ensure the high-quality signal path
        return True
    return value.strip().lower() in _TRUE_VALUES


def _env_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean environment variable with a configurable default."""
    if value is None:
        return default
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


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_profile(value: str | None) -> str:
    """Normalize profile preference."""
    if value is None:
        return "cpu"

    normalized = value.strip().lower()
    if normalized in {"auto", "gpu", "cpu"}:
        return normalized
    return "cpu"


def _normalize_loudnorm_preset(value: str | None) -> str:
    if value is None:
        return _LOUDNORM_PRESET_DEFAULT
    return _LOUDNORM_PRESET_ALIASES.get(value.strip().lower(), _LOUDNORM_PRESET_DEFAULT)


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for the audio pre-processing pipeline."""

    enabled: bool = True
    target_sample_rate: int = 16_000
    target_channels: int | None = None
    temp_dir: str | None = None
    output_dir: str | None = None
    profile: str = "cpu"
    loudnorm_preset: str = _LOUDNORM_PRESET_DEFAULT
    rnnoise_model: str | None = "models/sh.rnnn"
    rnnoise_mix: float = _RNNOISE_MIX_DEFAULT

    def loudnorm_target_i(self) -> float:
        """Get the target loudness (I) value for the current preset."""
        preset_config = _LOUDNORM_PRESETS.get(self.loudnorm_preset, _LOUDNORM_PRESETS[_LOUDNORM_PRESET_DEFAULT])
        return preset_config["I"]

    def loudnorm_target_tp(self) -> float:
        """Get the target true peak (TP) value for the current preset."""
        preset_config = _LOUDNORM_PRESETS.get(self.loudnorm_preset, _LOUDNORM_PRESETS[_LOUDNORM_PRESET_DEFAULT])
        return preset_config.get("TP", -2.0)

    def loudnorm_target_lra(self) -> float:
        """Get the target LRA value for the current preset."""
        preset_config = _LOUDNORM_PRESETS.get(self.loudnorm_preset, _LOUDNORM_PRESETS[_LOUDNORM_PRESET_DEFAULT])
        return preset_config["LRA"]

    @staticmethod
    def get_loudnorm_preset_config(preset: str) -> dict[str, float]:
        """Get the loudnorm preset configuration for a given preset name."""
        return _LOUDNORM_PRESETS.get(preset, _LOUDNORM_PRESETS[_LOUDNORM_PRESET_DEFAULT])

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "PreprocessConfig":
        """Build configuration from environment variables or a provided mapping."""
        source_env: Mapping[str, str] = env or os.environ

        return cls(
            enabled=_env_enabled(source_env.get(_PREPROCESS_ENABLED_ENV)),
            target_sample_rate=_parse_int(source_env.get(_PREPROCESS_TARGET_SR_ENV), 16_000),
            target_channels=_parse_positive_int(source_env.get(_PREPROCESS_TARGET_CH_ENV)),
            temp_dir=source_env.get(_PREPROCESS_TMP_ENV),
            output_dir=source_env.get(_PREPROCESS_OUTPUT_DIR_ENV),
            profile=_env_profile(source_env.get(_PREPROCESS_PROFILE_ENV)),
            loudnorm_preset=_normalize_loudnorm_preset(source_env.get(_PREPROCESS_LOUDNORM_PRESET_ENV)),
            rnnoise_model=source_env.get(_PREPROCESS_RNNOISE_MODEL_ENV) or "models/general.rnnn",
            rnnoise_mix=float(source_env.get(_PREPROCESS_RNNOISE_MIX_ENV) or _RNNOISE_MIX_DEFAULT),
        )


@dataclass(slots=True)
class TranscriptionConfig:
    """Configuration for faster-whisper transcription parameters.

    All default values for faster-whisper transcription are defined here.
    These can be overridden via environment variables.
    """

    # Beam search parameters
    beam_size: int = 5
    patience: float = 1.0

    # Timestamp and task parameters
    word_timestamps: bool = False
    task: str = "transcribe"  # "transcribe" or "translate"

    # Chunk processing
    chunk_length: int = 30  # Length of audio chunks in seconds

    # VAD (Voice Activity Detection) parameters
    vad_filter: bool = True
    vad_threshold: float = 0.5  # Speech probability threshold for VAD
    vad_parameters: dict[str, float | int] = field(
        default_factory=lambda: {
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float("inf"),
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 400,
        }
    )

    # Temperature and sampling parameters
    temperature: float = 0.0
    temperature_increment_on_fallback: float = 0.2
    best_of: int = 5  # Number of candidates for non-zero temperature sampling

    # Quality thresholds
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # Decoding parameters
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"  # Token IDs to suppress (comma-separated or "-1" for default)
    condition_on_previous_text: bool = True

    # Prompting
    initial_prompt: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "TranscriptionConfig":
        """Build configuration from environment variables or a provided mapping."""
        source_env: Mapping[str, str] = env or os.environ

        # Parse VAD parameters
        vad_params = {
            "min_speech_duration_ms": _parse_int(source_env.get(_TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS_ENV), 250),
            "max_speech_duration_s": _parse_float(
                source_env.get(_TRANSCRIBE_VAD_MAX_SPEECH_DURATION_S_ENV), float("inf")
            ),
            "min_silence_duration_ms": _parse_int(source_env.get(_TRANSCRIBE_VAD_MIN_SILENCE_DURATION_MS_ENV), 2000),
            "speech_pad_ms": _parse_int(source_env.get(_TRANSCRIBE_VAD_SPEECH_PAD_MS_ENV), 400),
        }

        return cls(
            beam_size=_parse_int(source_env.get(_TRANSCRIBE_BEAM_SIZE_ENV), 5),
            patience=_parse_float(source_env.get(_TRANSCRIBE_PATIENCE_ENV), 1.0),
            word_timestamps=_env_bool(source_env.get(_TRANSCRIBE_WORD_TIMESTAMPS_ENV), False),
            task=source_env.get(_TRANSCRIBE_TASK_ENV, "transcribe"),
            chunk_length=_parse_int(source_env.get(_TRANSCRIBE_CHUNK_LENGTH_ENV), 30),
            vad_filter=_env_bool(source_env.get(_TRANSCRIBE_VAD_FILTER_ENV), True),
            vad_threshold=_parse_float(source_env.get(_TRANSCRIBE_VAD_THRESHOLD_ENV), 0.5),
            vad_parameters=vad_params,
            temperature=_parse_float(source_env.get(_TRANSCRIBE_TEMPERATURE_ENV), 0.0),
            temperature_increment_on_fallback=_parse_float(source_env.get(_TRANSCRIBE_TEMPERATURE_INCREMENT_ENV), 0.2),
            best_of=_parse_int(source_env.get(_TRANSCRIBE_BEST_OF_ENV), 5),
            compression_ratio_threshold=_parse_float(source_env.get(_TRANSCRIBE_COMPRESSION_RATIO_THRESHOLD_ENV), 2.4),
            logprob_threshold=_parse_float(source_env.get(_TRANSCRIBE_LOGPROB_THRESHOLD_ENV), -1.0),
            no_speech_threshold=_parse_float(source_env.get(_TRANSCRIBE_NO_SPEECH_THRESHOLD_ENV), 0.6),
            length_penalty=_parse_float(source_env.get(_TRANSCRIBE_LENGTH_PENALTY_ENV), 1.0),
            suppress_tokens=source_env.get(_TRANSCRIBE_SUPPRESS_TOKENS_ENV, "-1"),
            condition_on_previous_text=_env_bool(source_env.get(_TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT_ENV), True),
            initial_prompt=source_env.get(_TRANSCRIBE_INITIAL_PROMPT_ENV),
        )
