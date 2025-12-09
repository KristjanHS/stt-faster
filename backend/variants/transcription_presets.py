"""Transcription configuration presets for variants."""

from __future__ import annotations

from backend.preprocess.config import TranscriptionConfig


def get_project_defaults() -> TranscriptionConfig:
    """Get TranscriptionConfig with project defaults.

    Returns the configuration from environment variables or project defaults.
    This is the standard configuration used by the project.
    """
    return TranscriptionConfig.from_env()


def get_industry_defaults() -> TranscriptionConfig:
    """Get TranscriptionConfig with industry-standard faster-whisper defaults.

    These are the defaults from the faster-whisper library itself,
    not the project's custom defaults.
    """
    return TranscriptionConfig(
        # Beam search - industry defaults
        beam_size=5,
        patience=1.0,  # Industry default is 1.0, not 1.1
        # Timestamp and task
        word_timestamps=False,
        task="transcribe",
        # Chunk processing - industry default is 30s
        chunk_length=30,  # Industry default is 30, not 20
        # VAD parameters - industry defaults
        vad_filter=True,
        vad_threshold=0.5,  # Industry default is 0.5, not 0.30
        vad_parameters={
            "min_speech_duration_ms": 250,  # Industry default
            "max_speech_duration_s": float("inf"),  # Industry default (unlimited)
            "min_silence_duration_ms": 800,  # Industry default
            "speech_pad_ms": 300,  # Industry default
        },
        # Temperature - industry default is [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] or just 0.0
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        temperature_increment_on_fallback=0.2,
        best_of=5,
        # Quality thresholds - industry defaults
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,  # Industry default is 0.6, not 0.5
        # Decoding parameters
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,  # Industry default is 0 (disabled), not 3
        suppress_tokens="-1",
        condition_on_previous_text=True,
        # Prompting
        initial_prompt=None,
    )


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
        preset: One of "project", "industry", or "minimal"

    Returns:
        TranscriptionConfig instance

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset == "project":
        return get_project_defaults()
    if preset == "industry":
        return get_industry_defaults()
    if preset == "minimal":
        return get_minimal_config()
    raise ValueError(f"Unknown transcription preset: {preset}")
