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
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, cast

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

from backend.exceptions import ModelNotFoundError


class _NoOpTqdm:
    """No-op tqdm class to disable progress bars in huggingface_hub downloads.

    This class implements the minimal tqdm interface needed by huggingface_hub
    to suppress progress bar output and avoid duplicate "Download complete" messages.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NoOpTqdm":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __iter__(self) -> "_NoOpTqdm":
        return self

    def __next__(self) -> None:
        raise StopIteration

    def update(self, n: int = 1) -> None:
        """Update progress (no-op)."""
        pass

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        """Set description (no-op)."""
        pass

    def close(self) -> None:
        """Close progress bar (no-op)."""
        pass

    def refresh(self) -> None:
        """Refresh display (no-op)."""
        pass

    @classmethod
    def get_lock(cls) -> threading.Lock:
        """Return a threading lock for thread-safe operations.

        Required by tqdm's thread_map function when using a custom tqdm class.
        """
        if not hasattr(cls, "_lock"):
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock: threading.Lock) -> None:
        """Set the threading lock for thread-safe operations.

        Required by tqdm's thread_map function when using a custom tqdm class.
        """
        cls._lock = lock


from backend.model_config import ModelConfig, get_preset
from backend.model_loader import ModelLoader
from backend.preprocess import PreprocessConfig, preprocess_audio
from backend.preprocess.config import TranscriptionConfig
from backend.preprocess.orchestrator import PreprocessResult

if TYPE_CHECKING:
    from faster_whisper import Segment, TranscriptionInfo


LOGGER = logging.getLogger(__name__)

# Output format and logging constants
DEFAULT_OUTPUT_FORMAT = "txt"
PROGRESS_LOG_INTERVAL_SECONDS = 60.0

# Float rounding precision for JSON output
FLOAT_PRECISION = 3


@dataclass(slots=True)
class TranscriptionMetrics:
    """Canonical statistics emitted for a transcription run."""

    # File and model info
    audio_path: str
    preset: str

    # Language detection
    requested_language: str | None
    applied_language: str | None
    detected_language: str | None
    language_probability: float | None

    # Timing metrics
    audio_duration: float | None
    total_processing_time: float
    transcribe_duration: float
    preprocess_duration: float
    speed_ratio: float | None

    # Preprocessing configuration
    preprocess_enabled: bool
    preprocess_profile: str
    target_sample_rate: int
    target_channels: int | None
    preprocess_snr_before: float | None
    preprocess_snr_after: float | None
    preprocess_steps: list[dict[str, Any]]
    rnnoise_model: str | None = None
    rnnoise_mix: float | None = None

    # Audio inspection (from input_info)
    input_channels: int | None = None
    input_sample_rate: int | None = None
    input_bit_depth: int | None = None
    input_format: str | None = None

    # Downmix/resample parameters
    volume_adjustment_db: float | None = None
    resampler: str | None = None
    sample_format: str | None = None

    # Loudness normalization parameters
    loudnorm_preset: str | None = None
    loudnorm_target_i: float | None = None
    loudnorm_target_tp: float | None = None
    loudnorm_target_lra: float | None = None
    loudnorm_backend: str | None = None

    # Denoise parameters
    denoise_method: str | None = None
    denoise_library: str | None = None

    # SNR estimation
    snr_estimation_method: str | None = None

    # Transcription parameters
    beam_size: int | None = None
    patience: float | None = None
    word_timestamps: bool | None = None
    task: str | None = None
    chunk_length: int | None = None
    vad_filter: bool | None = None
    vad_threshold: float | None = None
    vad_min_speech_duration_ms: int | None = None
    vad_max_speech_duration_s: float | None = None
    vad_min_silence_duration_ms: int | None = None
    vad_speech_pad_ms: int | None = None
    temperature: float | list[float] | None = None
    temperature_increment_on_fallback: float | None = None
    best_of: int | None = None
    compression_ratio_threshold: float | None = None
    logprob_threshold: float | None = None
    no_speech_threshold: float | None = None
    length_penalty: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    suppress_tokens: str | None = None
    condition_on_previous_text: bool | None = None
    initial_prompt: str | None = None

    # Model parameters
    model_id: str | None = None
    device: str | None = None
    compute_type: str | None = None

    # Output parameters
    output_format: str | None = None
    float_precision: int | None = None

    # Segment statistics
    segment_count: int | None = None
    no_speech_skips_count: int | None = None  # Legacy field - use segments_matching_no_speech_rule_count instead
    no_speech_skip_windows: list[dict[str, Any]] | None = None  # Legacy field
    # Note: segments_matching_no_speech_rule_count is NOT a skip count - it counts output segments
    # that match the heuristic. Truly skipped windows produce no segments.
    segments_matching_no_speech_rule_count: int | None = None
    segments_matching_no_speech_rule_windows: list[dict[str, Any]] | None = None


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
        ValueError: If model_id format is invalid or potentially unsafe
    """
    # Validate model_id format to ensure it's a safe HuggingFace model ID
    # Must contain "/" (org/model format) and only alphanumeric, dash, underscore
    if not model_id or "/" not in model_id:
        raise ValueError(f"Invalid HuggingFace model_id format: {model_id}")
    # Ensure model_id only contains safe characters (alphanumeric, dash, underscore, slash)
    if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$", model_id):
        raise ValueError(f"Unsafe model_id format detected: {model_id}")

    # Only download ct2 folder, skip transformers/native formats
    # Disable progress bar to avoid duplicate "Download complete" messages
    # Note: huggingface_hub uses HTTPS by default, ensuring secure downloads
    model_path = downloader(model_id, allow_patterns=["ct2/*"], tqdm_class=_NoOpTqdm)
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

    Raises:
        ValueError: If model_id format is invalid or potentially unsafe
    """
    # Validate model_id format to ensure it's a safe HuggingFace model ID
    # Must contain "/" (org/model format) and only alphanumeric, dash, underscore
    if not model_id or "/" not in model_id:
        raise ValueError(f"Invalid HuggingFace model_id format: {model_id}")
    # Ensure model_id only contains safe characters (alphanumeric, dash, underscore, slash)
    if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$", model_id):
        raise ValueError(f"Unsafe model_id format detected: {model_id}")

    # Disable progress bar to avoid duplicate "Download complete" messages
    # Note: huggingface_hub uses HTTPS by default, ensuring secure downloads
    return downloader(model_id, tqdm_class=_NoOpTqdm)


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


def segment_to_payload(segment: "Segment") -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "id": getattr(segment, "id", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": segment.text.strip(),
    }

    speaker = getattr(segment, "speaker", None)
    if speaker is not None:
        data["speaker"] = speaker

    # Include no_speech_prob and avg_logprob if available (from faster-whisper segments)
    no_speech_prob = getattr(segment, "no_speech_prob", None)
    if no_speech_prob is not None:
        data["no_speech_prob"] = float(no_speech_prob)

    avg_logprob = getattr(segment, "avg_logprob", None)
    if avg_logprob is not None:
        data["avg_logprob"] = float(avg_logprob)

    # Include word timestamps if available
    words = getattr(segment, "words", None)
    if words is not None:
        word_list: list[Dict[str, Any]] = []
        for word in words:
            word_data: Dict[str, Any] = {
                "word": getattr(word, "word", ""),
                "start": getattr(word, "start", None),
                "end": getattr(word, "end", None),
            }
            # Remove None values
            word_data = {k: v for k, v in word_data.items() if v is not None}
            if word_data:
                word_list.append(word_data)
        if word_list:
            data["words"] = word_list

    cleaned = {key: value for key, value in data.items() if value is not None}
    return _round_floats(cleaned)


def maybe_log_progress(
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
    transcription_config_provider: Callable[[], TranscriptionConfig] = TranscriptionConfig.from_env,
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

    preprocess_config = preprocess_config_provider()
    transcription_config = transcription_config_provider()
    LOGGER.info("⚙️ Beam size: %d", transcription_config.beam_size)
    overall_start = time.time()
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
        # Merge vad_threshold into vad_parameters as 'threshold'
        vad_params = dict(transcription_config.vad_parameters)
        vad_params["threshold"] = transcription_config.vad_threshold

        # Parse suppress_tokens from string to list[int] or None
        # faster-whisper expects Optional[List[int]], default is [-1]
        suppress_tokens_list: list[int] | None = None
        if transcription_config.suppress_tokens:
            if transcription_config.suppress_tokens == "-1":
                suppress_tokens_list = [-1]  # Default value
            else:
                # Parse comma-separated token IDs
                try:
                    suppress_tokens_list = [int(t.strip()) for t in transcription_config.suppress_tokens.split(",")]
                except ValueError:
                    LOGGER.warning(
                        "Invalid suppress_tokens format '%s', using default [-1]",
                        transcription_config.suppress_tokens,
                    )
                    suppress_tokens_list = [-1]

        segments, info = model.transcribe(
            str(preprocess_result.output_path),
            beam_size=transcription_config.beam_size,
            patience=transcription_config.patience,
            word_timestamps=transcription_config.word_timestamps,
            language=applied_language,
            task=transcription_config.task,
            chunk_length=transcription_config.chunk_length,
            vad_filter=transcription_config.vad_filter,
            vad_parameters=vad_params,
            temperature=transcription_config.temperature,
            compression_ratio_threshold=transcription_config.compression_ratio_threshold,
            log_prob_threshold=transcription_config.logprob_threshold,
            no_speech_threshold=transcription_config.no_speech_threshold,
            best_of=transcription_config.best_of,
            length_penalty=transcription_config.length_penalty,
            repetition_penalty=transcription_config.repetition_penalty,
            no_repeat_ngram_size=transcription_config.no_repeat_ngram_size,
            suppress_tokens=suppress_tokens_list,
            condition_on_previous_text=transcription_config.condition_on_previous_text,
            initial_prompt=transcription_config.initial_prompt,
        )

        total_audio_duration = getattr(info, "duration", None) or duration_hint
        if total_audio_duration and not duration_hint:
            LOGGER.info("🎧 Input duration: %.1f minutes", total_audio_duration / 60)

        segment_payloads: list[Dict[str, Any]] = []
        audio_processed = 0.0
        last_progress_log = transcribe_start
        no_speech_skips = 0
        no_speech_skip_windows: list[Dict[str, Any]] = []

        # Track segments that would have been skipped due to no_speech_threshold
        no_speech_threshold = transcription_config.no_speech_threshold
        logprob_threshold = transcription_config.logprob_threshold

        for segment in segments:
            segment_payloads.append(segment_to_payload(segment))

            # Track segments that match skip criteria
            # (no_speech_prob > threshold AND avg_logprob <= threshold)
            # Note: These are segments that passed but match skip criteria.
            # Truly skipped windows don't appear in output.
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            avg_logprob = getattr(segment, "avg_logprob", None)
            seg_start = getattr(segment, "start", None)
            seg_end = getattr(segment, "end", None)

            if no_speech_prob is not None and avg_logprob is not None:
                no_speech_val = float(no_speech_prob)
                avg_logprob_val = float(avg_logprob)
                if no_speech_val > no_speech_threshold and avg_logprob_val <= logprob_threshold:
                    no_speech_skips += 1
                    # Add to debug list
                    skip_window = {
                        "start": float(seg_start) if seg_start is not None else None,
                        "end": float(seg_end) if seg_end is not None else None,
                        "no_speech_prob": no_speech_val,
                        "avg_logprob": avg_logprob_val,
                    }
                    # Remove None values
                    skip_window = {k: v for k, v in skip_window.items() if v is not None}
                    if skip_window:
                        no_speech_skip_windows.append(skip_window)

            end_time = getattr(segment, "end", None)
            if end_time is not None:
                audio_processed = max(audio_processed, float(end_time))

            last_progress_log = maybe_log_progress(
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

        # Extract preprocessing step details for new metrics fields
        loudnorm_step = next((s for s in preprocess_result.metrics.steps if s.name == "loudnorm"), None)
        denoise_step = next((s for s in preprocess_result.metrics.steps if s.name == "denoise_light"), None)

        metrics_payload = TranscriptionMetrics(
            # File and model info
            audio_path=path,
            preset=preset,
            # Language detection
            requested_language=requested_language,
            applied_language=applied_language,
            detected_language=detected_lang,
            language_probability=lang_prob,
            # Timing metrics
            audio_duration=duration,
            total_processing_time=overall_time,
            transcribe_duration=transcribe_time,
            preprocess_duration=preprocess_result.metrics.total_duration,
            speed_ratio=speed_ratio,
            # Preprocessing configuration
            preprocess_enabled=preprocess_config.enabled,
            preprocess_profile=preprocess_result.profile,
            target_sample_rate=preprocess_config.target_sample_rate,
            target_channels=preprocess_config.target_channels,
            preprocess_snr_before=preprocess_result.metrics.snr_before,
            preprocess_snr_after=preprocess_result.metrics.snr_after,
            preprocess_steps=preprocess_steps,
            rnnoise_model=preprocess_config.rnnoise_model,
            rnnoise_mix=preprocess_config.rnnoise_mix,
            # Audio inspection (from input_info)
            input_channels=preprocess_result.input_info.channels if preprocess_result.input_info else None,
            input_sample_rate=preprocess_result.input_info.sample_rate if preprocess_result.input_info else None,
            input_bit_depth=None,  # Not available in current AudioInfo
            input_format=preprocess_result.input_info.sample_format if preprocess_result.input_info else None,
            # Downmix/resample parameters
            volume_adjustment_db=-6.0,  # Hardcoded in downmix_and_resample
            resampler="soxr",  # Hardcoded in downmix_and_resample
            sample_format="s16",  # Hardcoded in downmix_and_resample (16-bit signed)
            # Loudness normalization parameters
            loudnorm_preset=preprocess_config.loudnorm_preset,
            loudnorm_target_i=preprocess_config.loudnorm_target_i(),
            loudnorm_target_tp=preprocess_config.loudnorm_target_tp(),
            loudnorm_target_lra=preprocess_config.loudnorm_target_lra(),
            loudnorm_backend=loudnorm_step.backend if loudnorm_step else None,
            # Denoise parameters
            denoise_method="spectral_gate" if denoise_step else None,  # Hardcoded in denoise_light
            denoise_library="noisereduce" if denoise_step else None,  # Hardcoded in denoise_light
            # SNR estimation
            snr_estimation_method="estimate_snr_db",  # Hardcoded method
            # Transcription parameters
            beam_size=transcription_config.beam_size,
            patience=transcription_config.patience,
            word_timestamps=transcription_config.word_timestamps,
            task=transcription_config.task,
            chunk_length=transcription_config.chunk_length,
            vad_filter=transcription_config.vad_filter,
            vad_threshold=transcription_config.vad_threshold,
            vad_min_speech_duration_ms=cast(
                int | None, transcription_config.vad_parameters.get("min_speech_duration_ms")
            ),
            vad_max_speech_duration_s=transcription_config.vad_parameters.get("max_speech_duration_s"),
            vad_min_silence_duration_ms=cast(
                int | None, transcription_config.vad_parameters.get("min_silence_duration_ms")
            ),
            vad_speech_pad_ms=cast(int | None, transcription_config.vad_parameters.get("speech_pad_ms")),
            temperature=transcription_config.temperature,
            temperature_increment_on_fallback=transcription_config.temperature_increment_on_fallback,
            best_of=transcription_config.best_of,
            compression_ratio_threshold=transcription_config.compression_ratio_threshold,
            logprob_threshold=transcription_config.logprob_threshold,
            no_speech_threshold=transcription_config.no_speech_threshold,
            length_penalty=transcription_config.length_penalty,
            repetition_penalty=transcription_config.repetition_penalty,
            no_repeat_ngram_size=transcription_config.no_repeat_ngram_size,
            suppress_tokens=transcription_config.suppress_tokens,
            condition_on_previous_text=transcription_config.condition_on_previous_text,
            initial_prompt=transcription_config.initial_prompt,
            # Model parameters
            model_id=preset_config.model_id,
            device=preset_config.device,
            compute_type=preset_config.compute_type,
            # Output parameters (these will come from processor context in future enhancement)
            output_format=None,  # Not available at this level, will be set by processor
            float_precision=FLOAT_PRECISION,
            # Segment statistics
            segment_count=len(segment_payloads),
            no_speech_skips_count=no_speech_skips,  # Always record, even if 0
            no_speech_skip_windows=no_speech_skip_windows if no_speech_skip_windows else None,
        )
        if metrics_collector:
            metrics_collector(metrics_payload)

        # Add metrics to payload JSON for report generation
        metrics_dict: Dict[str, Any] = {
            "segment_count": metrics_payload.segment_count,
            "no_speech_skips_count": metrics_payload.no_speech_skips_count,
        }
        if metrics_payload.no_speech_skip_windows:
            metrics_dict["no_speech_skip_windows"] = metrics_payload.no_speech_skip_windows
        payload["metrics"] = metrics_dict

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


def transcribe_to_text(
    audio_path: str,
    text_path: str,
    preset: str = "et-large",
    language: str | None = None,
    *,
    transcribe_fn: Callable[..., Dict[str, Any]] = transcribe,
    opener: Callable[..., Any] = open,
) -> None:
    """Transcribe audio and save as plain text.

    Args:
        audio_path: Path to input audio file
        text_path: Path to output text file
        preset: Model preset name
        language: Optional language code
        transcribe_fn: Transcription function to use
        opener: File opener function (for testing)
    """
    payload = transcribe_fn(audio_path, preset, language=language)
    segments = payload.get("segments", [])

    with opener(text_path, "w", encoding="utf-8") as text_file:
        for segment in segments:
            text_file.write(segment["text"])
            text_file.write("\n")


def transcribe_and_save(
    audio_path: str,
    output_path: str,
    preset: str = "et-large",
    language: str | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    *,
    transcribe_fn: Callable[..., Dict[str, Any]] = transcribe,
    json_dumper: Callable[..., None] = json.dump,
    opener: Callable[..., Any] = open,
) -> None:
    """Transcribe audio and save in the specified format.

    Args:
        audio_path: Path to input audio file
        output_path: Path to output file (extension will be adjusted based on format)
        preset: Model preset name
        language: Optional language code
        output_format: Output format - "txt", "json", or "both" (default: "txt")
        transcribe_fn: Transcription function to use
        json_dumper: JSON dump function (for testing)
        opener: File opener function (for testing)

    Raises:
        ValueError: If output_format is not "txt", "json", or "both"
    """
    if output_format == "txt":
        transcribe_to_text(
            audio_path,
            output_path,
            preset,
            language,
            transcribe_fn=transcribe_fn,
            opener=opener,
        )
    elif output_format == "json":
        transcribe_to_json(
            audio_path,
            output_path,
            preset,
            language,
            transcribe_fn=transcribe_fn,
            json_dumper=json_dumper,
            opener=opener,
        )
    elif output_format == "both":
        # Generate both txt and json files
        base_path = Path(output_path)
        txt_path = base_path.with_suffix(".txt")
        json_path = base_path.with_suffix(".json")

        transcribe_to_text(
            audio_path,
            str(txt_path),
            preset,
            language,
            transcribe_fn=transcribe_fn,
            opener=opener,
        )
        transcribe_to_json(
            audio_path,
            str(json_path),
            preset,
            language,
            transcribe_fn=transcribe_fn,
            json_dumper=json_dumper,
            opener=opener,
        )
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'txt', 'json', or 'both'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    input_audio_path = project_root / "UserData" / "RelMan.wav"
    output_text_path = input_audio_path.with_suffix(".txt")

    transcribe_to_text(str(input_audio_path), str(output_text_path), preset="et-large")
    LOGGER.info("Wrote text transcript to %s", output_text_path)
