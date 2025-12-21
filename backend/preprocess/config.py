from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, cast

# Preset configurations for loudnorm
_LOUDNORM_PRESETS: Final[dict[str, dict[str, float]]] = {
    "default": {
        "I": -23.0,
        "TP": -2.0,
        "LRA": 8.0,
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
_RNNOISE_MIX_DEFAULT: Final = 0.4  #
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
_TRANSCRIBE_REPETITION_PENALTY_ENV: Final = "STT_TRANSCRIBE_REPETITION_PENALTY"
_TRANSCRIBE_NO_REPEAT_NGRAM_SIZE_ENV: Final = "STT_TRANSCRIBE_NO_REPEAT_NGRAM_SIZE"
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


def _parse_temperature(raw: str | None, default: float | list[float]) -> float | list[float]:
    """Parse temperature which can be a single float or a comma-separated list of floats."""
    if raw is None:
        return default
    try:
        # Try parsing as a list (comma-separated values)
        values = [float(x.strip()) for x in raw.split(",")]
        if len(values) == 1:
            return values[0]
        return values
    except ValueError:
        # Fallback to single float
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


def _get_dataclass_default(cls: type, field_name: str) -> Any:
    """Get the default value for a dataclass field, handling default_factory if needed."""
    from dataclasses import MISSING, fields  # noqa: PLC0415

    for dataclass_field in fields(cls):
        if dataclass_field.name == field_name:
            if dataclass_field.default is not MISSING:
                return dataclass_field.default
            if dataclass_field.default_factory is not MISSING:
                return dataclass_field.default_factory()
            break
    raise ValueError(f"Field {field_name} not found in {cls.__name__}")


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

    @staticmethod
    def resolve_loudnorm_params(
        target_i: float | None,
        target_tp: float | None,
        target_lra: float | None,
        loudnorm_preset: str,
    ) -> tuple[float, float, float]:
        """Resolve loudnorm parameters from function args or preset config.

        Args:
            target_i: Explicit I parameter, or None to use preset
            target_tp: Explicit TP parameter, or None to use preset
            target_lra: Explicit LRA parameter, or None to use preset
            loudnorm_preset: Preset name to use if parameters are None

        Returns:
            Tuple of (target_i, target_tp, target_lra) as floats.

        Raises:
            ValueError: If required parameters cannot be resolved from preset.
        """
        if target_i is None or target_tp is None or target_lra is None:
            preset = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
            try:
                if target_i is None:
                    target_i = float(preset["I"])
                if target_tp is None:
                    target_tp = float(preset.get("TP", -2.0))
                if target_lra is None:
                    target_lra = float(preset["LRA"])
            except KeyError as e:
                raise ValueError(f"Missing required loudnorm parameter in preset '{loudnorm_preset}': {e}") from e

        # All parameters are guaranteed to be float at this point
        return target_i, target_tp, target_lra

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
            rnnoise_model=source_env.get(_PREPROCESS_RNNOISE_MODEL_ENV) or "models/sh.rnnn",
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
    patience: float = 1.1

    # Timestamp and task parameters
    word_timestamps: bool = False
    task: str = "transcribe"  # "transcribe" or "translate"

    # Chunk processing
    chunk_length: int = 30  # Length of audio chunks in seconds

    # VAD (Voice Activity Detection) parameters
    vad_filter: bool = True
    vad_threshold: float = 0.30  # was 0.35  Speech probability threshold for VAD
    vad_parameters: dict[str, float | int] = field(
        default_factory=lambda: {
            "min_speech_duration_ms": 200,  # was 250
            "max_speech_duration_s": 30,  # was float("inf"),
            "min_silence_duration_ms": 700,  # was 800
            "speech_pad_ms": 500,  # was 300
        }
    )

    # Temperature and sampling parameters
    temperature: float | list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.8])
    temperature_increment_on_fallback: float = 0.2
    best_of: int = 5  # Number of candidates for non-zero temperature sampling

    # Quality thresholds
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.5

    # Decoding parameters
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    suppress_tokens: str = "-1"  # Token IDs to suppress (comma-separated or "-1" for default)
    condition_on_previous_text: bool = True

    # Prompting
    initial_prompt: str | None = None

    # Internal: Track which fields were explicitly set (for baseline configs)
    _explicit_fields: set[str] = field(default_factory=lambda: set[str](), repr=False, init=False)

    def set(self, key: str, value: Any) -> None:
        """Set a field value and mark it as explicitly set.

        Args:
            key: Field name to set
            value: Value to set
        """
        setattr(self, key, value)
        self._explicit_fields.add(key)

    def to_kwargs(self) -> dict[str, Any]:
        """Return only explicitly set fields as kwargs dict.

        Returns:
            Dictionary containing only fields that were explicitly set
        """
        return {k: getattr(self, k) for k in self._explicit_fields}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "TranscriptionConfig":
        """Build configuration from environment variables or a provided mapping."""
        source_env: Mapping[str, str] = env or os.environ

        # Get default VAD parameters from class defaults
        default_vad_params = _get_dataclass_default(cls, "vad_parameters")
        if not isinstance(default_vad_params, dict):
            msg = f"Expected dict for vad_parameters default, got {type(default_vad_params).__name__}"
            raise TypeError(msg)
        default_vad_dict = cast(dict[str, float | int], default_vad_params)

        # Parse VAD parameters, using class defaults
        vad_params = {
            "min_speech_duration_ms": _parse_int(
                source_env.get(_TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS_ENV),
                int(default_vad_dict["min_speech_duration_ms"]),
            ),
            "max_speech_duration_s": _parse_float(
                source_env.get(_TRANSCRIBE_VAD_MAX_SPEECH_DURATION_S_ENV),
                float(default_vad_dict["max_speech_duration_s"]),
            ),
            "min_silence_duration_ms": _parse_int(
                source_env.get(_TRANSCRIBE_VAD_MIN_SILENCE_DURATION_MS_ENV),
                int(default_vad_dict["min_silence_duration_ms"]),
            ),
            "speech_pad_ms": _parse_int(
                source_env.get(_TRANSCRIBE_VAD_SPEECH_PAD_MS_ENV),
                int(default_vad_dict["speech_pad_ms"]),
            ),
        }

        # Get default temperature value (handles list case)
        default_temperature = _get_dataclass_default(cls, "temperature")

        return cls(
            beam_size=_parse_int(
                source_env.get(_TRANSCRIBE_BEAM_SIZE_ENV), cast(int, _get_dataclass_default(cls, "beam_size"))
            ),
            patience=_parse_float(
                source_env.get(_TRANSCRIBE_PATIENCE_ENV), cast(float, _get_dataclass_default(cls, "patience"))
            ),
            word_timestamps=_env_bool(
                source_env.get(_TRANSCRIBE_WORD_TIMESTAMPS_ENV),
                cast(bool, _get_dataclass_default(cls, "word_timestamps")),
            ),
            task=source_env.get(_TRANSCRIBE_TASK_ENV, cast(str, _get_dataclass_default(cls, "task"))),
            chunk_length=_parse_int(
                source_env.get(_TRANSCRIBE_CHUNK_LENGTH_ENV), cast(int, _get_dataclass_default(cls, "chunk_length"))
            ),
            vad_filter=_env_bool(
                source_env.get(_TRANSCRIBE_VAD_FILTER_ENV), cast(bool, _get_dataclass_default(cls, "vad_filter"))
            ),
            vad_threshold=_parse_float(
                source_env.get(_TRANSCRIBE_VAD_THRESHOLD_ENV), cast(float, _get_dataclass_default(cls, "vad_threshold"))
            ),
            vad_parameters=vad_params,
            temperature=_parse_temperature(source_env.get(_TRANSCRIBE_TEMPERATURE_ENV), default_temperature),
            temperature_increment_on_fallback=_parse_float(
                source_env.get(_TRANSCRIBE_TEMPERATURE_INCREMENT_ENV),
                cast(float, _get_dataclass_default(cls, "temperature_increment_on_fallback")),
            ),
            best_of=_parse_int(
                source_env.get(_TRANSCRIBE_BEST_OF_ENV), cast(int, _get_dataclass_default(cls, "best_of"))
            ),
            compression_ratio_threshold=_parse_float(
                source_env.get(_TRANSCRIBE_COMPRESSION_RATIO_THRESHOLD_ENV),
                cast(float, _get_dataclass_default(cls, "compression_ratio_threshold")),
            ),
            logprob_threshold=_parse_float(
                source_env.get(_TRANSCRIBE_LOGPROB_THRESHOLD_ENV),
                cast(float, _get_dataclass_default(cls, "logprob_threshold")),
            ),
            no_speech_threshold=_parse_float(
                source_env.get(_TRANSCRIBE_NO_SPEECH_THRESHOLD_ENV),
                cast(float, _get_dataclass_default(cls, "no_speech_threshold")),
            ),
            length_penalty=_parse_float(
                source_env.get(_TRANSCRIBE_LENGTH_PENALTY_ENV),
                cast(float, _get_dataclass_default(cls, "length_penalty")),
            ),
            repetition_penalty=_parse_float(
                source_env.get(_TRANSCRIBE_REPETITION_PENALTY_ENV),
                cast(float, _get_dataclass_default(cls, "repetition_penalty")),
            ),
            no_repeat_ngram_size=_parse_int(
                source_env.get(_TRANSCRIBE_NO_REPEAT_NGRAM_SIZE_ENV),
                cast(int, _get_dataclass_default(cls, "no_repeat_ngram_size")),
            ),
            suppress_tokens=source_env.get(
                _TRANSCRIBE_SUPPRESS_TOKENS_ENV, cast(str, _get_dataclass_default(cls, "suppress_tokens"))
            ),
            condition_on_previous_text=_env_bool(
                source_env.get(_TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT_ENV),
                cast(bool, _get_dataclass_default(cls, "condition_on_previous_text")),
            ),
            initial_prompt=source_env.get(_TRANSCRIBE_INITIAL_PROMPT_ENV),
        )
