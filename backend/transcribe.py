"""
MINIMALISTIC solution to convert wav to a structured JSON transcript:

SYSTRAN/faster-whisper — a faster Whisper reimplementation (≈18.4k ⭐ as of Oct 5, 2025).
Why it's popular: up to ~4× faster and lower memory via CTranslate2; easy Python API; great on CPU or modest GPUs.

# Run these commands TO INITIALIZE:
source .venv/bin/activate

# to Install:
pip install -U hf_transfer huggingface_hub faster-whisper
pip install --force-reinstall ctranslate2==4.4.0
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12

# For faster HF transfer and cache:
# keep in your ~/.bashrc
export HF_HOME="$HOME/.cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_HIGH_PERFORMANCE=1

# To run:
python transcribe.py
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Callable, cast

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

from backend.exceptions import ModelNotFoundError
from backend.model_config import ModelConfig, get_preset
from backend.model_loader import ModelLoader
from backend.preprocess import PreprocessConfig, preprocess_audio
from backend.preprocess.orchestrator import PreprocessResult

if TYPE_CHECKING:
    from faster_whisper import Segment, TranscriptionInfo


LOGGER = logging.getLogger(__name__)

# Default transcription parameters
DEFAULT_BEAM_SIZE = 9
DEFAULT_WORD_TIMESTAMPS = False
DEFAULT_TASK = "transcribe"
PROGRESS_LOG_INTERVAL_SECONDS = 60.0

# Float rounding precision for JSON output
FLOAT_PRECISION = 3


@dataclass(slots=True)
class TranscriptionMetrics:
    """Canonical statistics emitted for a transcription run."""

    audio_path: str
    preset: str
    requested_language: str | None
    applied_language: str | None
    detected_language: str | None
    language_probability: float | None
    audio_duration: float | None
    total_processing_time: float
    transcribe_duration: float
    preprocess_duration: float
    preprocess_enabled: bool
    preprocess_profile: str
    target_sample_rate: int
    target_channels: int | None
    preprocess_snr_before: float | None
    preprocess_snr_after: float | None
    preprocess_steps: list[dict[str, Any]]
    speed_ratio: float | None


def _get_estonian_model_path(
    model_id: str,
    *,
    downloader: Callable[..., str] = snapshot_download,
    path_cls: Callable[[str], Path] = Path,
) -> str:
    """Download Estonian model and return path to CT2 subfolder.

    Estonian models from TalTechNLP store CT2 files in a 'ct2' subdirectory.
    This helper downloads only the CT2 files, skipping unnecessary formats.

    Args:
        model_id: HuggingFace model ID (e.g., 'TalTechNLP/whisper-large-v3-turbo-et-verbatim')

    Returns:
        Path to the CT2 model directory

    Raises:
        ModelNotFoundError: If CT2 folder doesn't exist in model
    """
    # Only download ct2 folder, skip transformers/native formats
    model_path = downloader(model_id, allow_patterns=["ct2/*"])  # nosec B615
    ct2_path = path_cls(model_path) / "ct2"
    if not ct2_path.exists():
        raise ModelNotFoundError(f"CT2 folder not found in {model_id}. Expected at: {ct2_path}")
    return str(ct2_path)


def _get_cached_model_path(model_id: str, *, downloader: Callable[..., str] = snapshot_download) -> str:
    """Download/retrieve cached Whisper model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., 'Systran/faster-whisper-large-v3')

    Returns:
        Path to the cached model directory
    """
    return downloader(model_id)  # nosec B615


def _resolve_model_path(
    config: ModelConfig,
    estonian_resolver: Callable[[str], str] = _get_estonian_model_path,
    cached_resolver: Callable[[str], str] = _get_cached_model_path,
) -> str:
    """Resolve the model path based on configuration.

    Args:
        config: Model configuration

    Returns:
        Path to the model (local or cached from HuggingFace)
    """
    if config.is_estonian:
        return estonian_resolver(config.model_id)
    # For non-Estonian models, check if it's a simple preset name or HF model ID
    if "/" in config.model_id:
        # HuggingFace model ID (e.g., "Systran/faster-whisper-large-v3")
        return cached_resolver(config.model_id)
    # Simple model name (e.g., "small") - faster-whisper will download it
    return config.model_id


def pick_model(
    preset: str = "et-large",
    *,
    resolver: Callable[[ModelConfig], str] = _resolve_model_path,
    loader: ModelLoader | None = None,
) -> WhisperModel:
    """Load a Whisper model by preset name.

    This is the main entry point for loading models. It uses configuration-driven
    model selection with automatic device selection and GPU->CPU fallback.

    Available presets: et-large, turbo, distil, large8gb, small

    Args:
        preset: Model preset name (see PRESETS in model_config.py)

    Returns:
        Loaded WhisperModel instance

    Raises:
        KeyError: If preset is not recognized
        ModelLoadError: If model fails to load
        ModelNotFoundError: If Estonian model CT2 folder not found

    Example:
        >>> model = pick_model("et-large")  # Estonian model on GPU (or CPU fallback)
        >>> model = pick_model("small")     # Small model on CPU
    """
    # Get configuration for the preset
    config = get_preset(preset)

    # Resolve the model path (download if needed)
    model_path = resolver(config)

    # Load the model with automatic fallback
    model_loader = loader or ModelLoader()
    return model_loader.load(model_path, config)


def _round_floats(value: Any, places: int = FLOAT_PRECISION) -> Any:
    if isinstance(value, dict):
        return {key: _round_floats(item, places) for key, item in cast(dict[Any, Any], value).items()}
    if isinstance(value, list):
        return [_round_floats(item, places) for item in cast(list[Any], value)]
    if isinstance(value, float):
        return round(value, places)
    return value


def _segment_to_payload(segment: "Segment") -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "id": getattr(segment, "id", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": segment.text.strip(),
    }

    speaker = getattr(segment, "speaker", None)
    if speaker is not None:
        data["speaker"] = speaker

    cleaned = {key: value for key, value in data.items() if value is not None}
    return _round_floats(cleaned)


def _maybe_log_progress(
    processed_seconds: float,
    total_seconds: float | None,
    start_time: float,
    last_log_time: float,
) -> float:
    """Log incremental transcription progress without spamming."""
    now = time.time()
    if now - last_log_time < PROGRESS_LOG_INTERVAL_SECONDS:
        return last_log_time

    elapsed_minutes = (now - start_time) / 60
    processed_minutes = processed_seconds / 60
    if total_seconds and total_seconds > 0:
        percent = min(processed_seconds / total_seconds * 100, 100)
        LOGGER.info(
            "⌛ Transcription progress: %.1f/%.1f min (%.1f%%), elapsed %.1f min",
            processed_minutes,
            total_seconds / 60,
            percent,
            elapsed_minutes,
        )
    else:
        LOGGER.info(
            "⌛ Transcription progress: %.1f min processed, elapsed %.1f min",
            processed_minutes,
            elapsed_minutes,
        )

    return now


def transcribe(
    path: str,
    preset: str = "et-large",
    language: str | None = None,
    *,
    preprocess_config_provider: Callable[[], PreprocessConfig] = PreprocessConfig.from_env,
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult] = preprocess_audio,
    model_picker: Callable[[str], Any] | None = None,
    metrics_collector: Callable[[TranscriptionMetrics], None] | None = None,
) -> Dict[str, Any]:
    LOGGER.info("🎤 Starting transcription of: %s", os.path.basename(path))
    preset_config = get_preset(preset)
    LOGGER.info(
        "🧠 Using preset %s (%s compute, device %s)",
        preset,
        preset_config.compute_type,
        preset_config.device,
    )
    LOGGER.info("⚙️ Beam size: %d", DEFAULT_BEAM_SIZE)
    overall_start = time.time()

    preprocess_config = preprocess_config_provider()
    preprocess_result = preprocess_runner(path, preprocess_config)
    duration_hint = preprocess_result.input_info.duration if preprocess_result.input_info else None
    if duration_hint:
        LOGGER.info("🎧 Input duration: %.1f minutes (from metadata)", duration_hint / 60)

    try:
        model = (model_picker or pick_model)(preset)
        segments: Iterable["Segment"]
        info: "TranscriptionInfo"

        # Auto-detect language based on preset if not explicitly provided
        requested_language = language
        applied_language = requested_language
        if applied_language is None:
            applied_language = "et" if preset.startswith("et-") else None

        # Log language configuration
        if applied_language:
            LOGGER.info("🌐 Language: %s (forced)", applied_language)
        else:
            LOGGER.info("🌐 Language: auto-detect (may be unreliable)")

        LOGGER.info("🔄 Transcribing audio...")
        transcribe_start = time.time()
        segments, info = model.transcribe(
            str(preprocess_result.output_path),
            beam_size=DEFAULT_BEAM_SIZE,
            word_timestamps=DEFAULT_WORD_TIMESTAMPS,
            language=applied_language,
            task=DEFAULT_TASK,
        )

        total_audio_duration = getattr(info, "duration", None) or duration_hint
        if total_audio_duration and not duration_hint:
            LOGGER.info("🎧 Input duration: %.1f minutes", total_audio_duration / 60)

        segment_payloads: list[Dict[str, Any]] = []
        audio_processed = 0.0
        last_progress_log = transcribe_start

        for segment in segments:
            segment_payloads.append(_segment_to_payload(segment))

            end_time = getattr(segment, "end", None)
            if end_time is not None:
                audio_processed = max(audio_processed, float(end_time))

            last_progress_log = _maybe_log_progress(
                processed_seconds=audio_processed,
                total_seconds=total_audio_duration,
                start_time=transcribe_start,
                last_log_time=last_progress_log,
            )

        transcribe_time = time.time() - transcribe_start

        # Log detected/used language from result
        detected_lang = getattr(info, "language", None)
        lang_prob = getattr(info, "language_probability", None)
        if detected_lang:
            if lang_prob is not None:
                LOGGER.info("🗣️  Detected language: %s (confidence: %.2f%%)", detected_lang, lang_prob * 100)
            else:
                LOGGER.info("🗣️  Used language: %s", detected_lang)

        payload: Dict[str, Any] = {
            "audio": os.path.basename(path),
            "language": detected_lang,
            "language_probability": lang_prob,
            "segments": segment_payloads,
        }

        duration = getattr(info, "duration", None)
        if duration is not None:
            payload["duration"] = duration

        overall_time = time.time() - overall_start

        # Log timing summary
        LOGGER.info("⏱️  Transcription completed in %.2f seconds", transcribe_time)
        speed_ratio: float | None = None
        if duration:
            speed_ratio = duration / transcribe_time if transcribe_time > 0 else 0
            LOGGER.info("⚡ Speed: %.2fx realtime (%.1fs audio in %.1fs)", speed_ratio, duration, transcribe_time)
        LOGGER.info("✅ Total processing time: %.2f seconds", overall_time)

        preprocess_steps = [
            {"name": step.name, "backend": step.backend, "duration": step.duration}
            for step in preprocess_result.metrics.steps
        ]
        metrics_payload = TranscriptionMetrics(
            audio_path=path,
            preset=preset,
            requested_language=requested_language,
            applied_language=applied_language,
            detected_language=detected_lang,
            language_probability=lang_prob,
            audio_duration=duration,
            total_processing_time=overall_time,
            transcribe_duration=transcribe_time,
            preprocess_duration=preprocess_result.metrics.total_duration,
            preprocess_enabled=preprocess_config.enabled,
            preprocess_profile=preprocess_result.profile,
            target_sample_rate=preprocess_config.target_sample_rate,
            target_channels=preprocess_config.target_channels,
            preprocess_snr_before=preprocess_result.metrics.snr_before,
            preprocess_snr_after=preprocess_result.metrics.snr_after,
            preprocess_steps=preprocess_steps,
            speed_ratio=speed_ratio,
        )
        if metrics_collector:
            metrics_collector(metrics_payload)

        cleaned = {key: value for key, value in payload.items() if value is not None}
        return _round_floats(cleaned)
    finally:
        preprocess_result.cleanup()


def transcribe_to_json(
    audio_path: str,
    json_path: str,
    preset: str = "et-large",
    language: str | None = None,
    *,
    transcribe_fn: Callable[..., Dict[str, Any]] = transcribe,
    json_dumper: Callable[..., None] = json.dump,
    opener: Callable[..., Any] = open,
) -> None:
    payload = transcribe_fn(audio_path, preset, language=language)
    with opener(json_path, "w", encoding="utf-8") as json_file:
        json_dumper(payload, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    input_audio_path = project_root / "UserData" / "RelMan.wav"
    output_json_path = input_audio_path.with_suffix(".json")

    transcribe_to_json(str(input_audio_path), str(output_json_path), preset="et-large")
    LOGGER.info("Wrote JSON transcript to %s", output_json_path)
