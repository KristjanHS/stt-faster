"""Variant execution engine."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from backend.model_config import get_preset
from backend.preprocess.config import PreprocessConfig
from backend.preprocess.orchestrator import PreprocessResult
from backend.transcribe import (
    FLOAT_PRECISION,
    TranscriptionMetrics,
    maybe_log_progress,
    pick_model,
    segment_to_payload,
)
from backend.variants.preprocess_steps import create_preprocess_runner
from backend.variants.transcription_presets import get_transcription_config
from backend.variants.variant import Variant

LOGGER = logging.getLogger(__name__)
console = Console()


def create_variant_preprocess_runner(
    variant: Variant,
    preprocess_config: PreprocessConfig,
    *,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> Callable[[str, PreprocessConfig], PreprocessResult]:
    """Create a preprocessing runner function for a variant.

    Args:
        variant: Variant definition
        preprocess_config: PreprocessConfig to use
        base_name: Optional base name for intermediate file naming
        datetime_suffix: Optional datetime suffix for intermediate file naming
        output_dir: Optional output directory for intermediate files
        copy_intermediate: Whether to copy intermediate files to output_dir

    Returns:
        A function that takes (path: str, config: PreprocessConfig) -> PreprocessResult
    """
    if variant.custom_preprocess_runner:
        # Use custom preprocessing runner (for variants 10-16)
        def _custom_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            # Custom runners accept additional keyword arguments
            return variant.custom_preprocess_runner(  # type: ignore[call-arg, misc]
                path,
                cfg,
                variant_number=variant.number,
                variant_description=variant.description,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        return _custom_runner
    else:
        # Use standard step-based preprocessing
        return create_preprocess_runner(
            variant.preprocess_steps,
            preprocess_config,
            variant_number=variant.number,
            variant_description=variant.description,
            base_name=base_name,
            datetime_suffix=datetime_suffix,
            output_dir=output_dir,
            copy_intermediate=copy_intermediate,
        )


def create_variant_transcribe_config(variant: Variant) -> Any:  # TranscriptionConfig
    """Create transcription config for a variant.

    Args:
        variant: Variant definition

    Returns:
        TranscriptionConfig instance
    """
    transcription_config = get_transcription_config(variant.transcription_preset)
    # Apply any overrides
    if variant.transcription_overrides:
        for key, value in variant.transcription_overrides.items():
            if hasattr(transcription_config, key):
                setattr(transcription_config, key, value)
    return transcription_config


def execute_variant(
    variant: Variant,
    audio_path: str,
    preset: str = "et-large",
    language: str | None = None,
    *,
    output_dir: Path | None = None,
    output_base_path: Path | None = None,
    datetime_suffix: str | None = None,
    copy_intermediate: bool = False,
) -> dict[str, Any]:
    """Execute a single variant and return results.

    Args:
        variant: Variant definition to execute
        audio_path: Path to audio file to transcribe
        preset: Model preset to use
        language: Optional language code
        output_dir: Optional output directory for intermediate files
        output_base_path: Optional base path for output files
        datetime_suffix: Optional datetime suffix for filenames
        copy_intermediate: Whether to copy intermediate preprocessing files

    Returns:
        Dictionary with variant results in same format as run_variant()
    """
    # Log variant start with timestamp (major milestone)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.debug("[%s] Running variant: %s", timestamp, variant.name)

    start_time = time.time()

    try:
        # Build preprocessing config
        preprocess_config = PreprocessConfig()  # Use defaults
        if not any(step.enabled for step in variant.preprocess_steps):
            preprocess_config.enabled = False

        # Set output directory only if copy_intermediate is enabled
        if copy_intermediate and output_dir:
            preprocess_config.output_dir = str(output_dir)
        else:
            preprocess_config.output_dir = None

        # Create preprocessing runner from variant steps or use custom runner
        base_name = output_base_path.stem if output_base_path else Path(audio_path).stem
        preprocess_runner = create_variant_preprocess_runner(
            variant,
            preprocess_config,
            base_name=base_name,
            datetime_suffix=datetime_suffix,
            output_dir=output_dir,
            copy_intermediate=copy_intermediate,
        )

        # Get transcription config based on preset
        transcription_config = create_variant_transcribe_config(variant)

        # Run transcription
        if variant.transcription_preset == "minimal":
            # Use minimal params transcription (omits parameters)
            # Pass transcription_config for metrics and transcription_overrides for actual overrides
            result = transcribe_with_minimal_params(
                path=audio_path,
                preset=preset,
                language=language,
                preprocess_config=preprocess_config,
                preprocess_runner=preprocess_runner,
                transcription_config=transcription_config,
                transcription_overrides=variant.transcription_overrides,
            )
        else:
            # Use standard transcription with full config
            result = _transcribe_with_config(
                path=audio_path,
                preset=preset,
                language=language,
                preprocess_config=preprocess_config,
                preprocess_runner=preprocess_runner,
                transcription_config=transcription_config,
            )

        elapsed = time.time() - start_time

        # Save output files if output_base_path is provided
        output_files = {}
        if output_base_path and datetime_suffix:
            base_name = output_base_path.stem
            variant_desc = variant.description
            txt_path = output_base_path.parent / f"{variant.number}_{variant_desc}_{base_name}_{datetime_suffix}.txt"
            json_path = output_base_path.parent / f"{variant.number}_{variant_desc}_{base_name}_{datetime_suffix}.json"

            # Save text file
            segments = result.get("segments", [])
            with txt_path.open("w", encoding="utf-8") as text_file:
                for segment in segments:
                    text_file.write(segment["text"])
                    text_file.write("\n")
            output_files["txt"] = str(txt_path)
            LOGGER.debug("Saved text output: %s", txt_path.name)

            # Save JSON file
            with json_path.open("w", encoding="utf-8") as json_file:
                import json  # noqa: PLC0415

                json.dump(result, json_file, ensure_ascii=False, indent=2)
            output_files["json"] = str(json_path)
            LOGGER.debug("Saved JSON output: %s", json_path.name)

        return {
            "variant": variant.name,
            "variant_number": variant.number,
            "status": "success",
            "elapsed_seconds": elapsed,
            "result": result,
            "output_files": output_files,
        }
    except Exception as exc:
        elapsed = time.time() - start_time
        LOGGER.error("Variant %s failed: %s", variant.name, exc, exc_info=True)
        return {
            "variant": variant.name,
            "variant_number": variant.number,
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(exc),
            "output_files": {},
        }


def _transcribe_with_config(
    path: str,
    preset: str,
    language: str | None,
    preprocess_config: PreprocessConfig,
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
    transcription_config: Any,  # TranscriptionConfig
) -> dict[str, Any]:
    """Transcribe using full transcription config."""
    from backend.transcribe import transcribe  # noqa: PLC0415

    def _config_provider() -> Any:  # TranscriptionConfig
        return transcription_config

    return transcribe(
        path=path,
        preset=preset,
        language=language,
        preprocess_config_provider=lambda: preprocess_config,
        preprocess_runner=preprocess_runner,
        transcription_config_provider=_config_provider,
    )


def transcribe_with_minimal_params(
    path: str,
    preset: str,
    language: str | None,
    preprocess_config: PreprocessConfig,
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
    transcription_config: Any | None = None,  # TranscriptionConfig
    transcription_overrides: dict[str, Any] | None = None,
    metrics_collector: Callable[[TranscriptionMetrics], None] | None = None,
) -> dict[str, Any]:
    """Transcribe with minimal parameters, omitting those that differ between defaults.

    This allows faster-whisper to use its own internal defaults for omitted parameters.
    Only parameters explicitly set in transcription_overrides will be included.

    Args:
        path: Path to audio file
        preset: Model preset to use
        language: Optional language code
        preprocess_config: PreprocessConfig instance
        preprocess_runner: Function to run preprocessing
        transcription_config: Optional TranscriptionConfig (used for metrics)
        transcription_overrides: Optional dict of parameter overrides to include
        metrics_collector: Optional metrics collector callback
    """
    overall_start = time.time()
    preprocess_result = preprocess_runner(path, preprocess_config)
    duration_hint = preprocess_result.input_info.duration if preprocess_result.input_info else None
    if duration_hint:
        console.print(f"[cyan]🎧 Input duration:[/cyan] {duration_hint / 60:.1f} minutes (from metadata)")
        LOGGER.debug("Input duration: %.1f minutes (from metadata)", duration_hint / 60)

    model = pick_model(preset)
    LOGGER.info("✅ Model loaded, ready for transcription")

    # Auto-detect language based on preset if not explicitly provided
    applied_language = language
    if applied_language is None:
        applied_language = "et" if preset.startswith("et-") else None

    # Log language configuration
    if applied_language:
        LOGGER.info("🌐 Language: %s (forced)", applied_language)
    else:
        LOGGER.info("🌐 Language: auto-detect (may be unreliable)")

    # Call model.transcribe with only essential parameters
    # Omit: patience, chunk_length, vad_threshold, vad_parameters,
    # temperature, no_speech_threshold, no_repeat_ngram_size
    # But include any overrides from transcription_overrides
    LOGGER.info("🔄 Starting transcription...")
    transcribe_start = time.time()
    transcribe_kwargs: dict[str, Any] = {
        "beam_size": 5,  # Keep beam_size as it's commonly used
        "word_timestamps": False,
        "language": applied_language,
        "task": "transcribe",
    }
    # Apply overrides from transcription_overrides if provided
    # Only include parameters that were explicitly overridden and are allowed
    # This set defines which parameters can be overridden in minimal preset
    allowed_override_params = {
        "chunk_length",
        "no_speech_threshold",
        "condition_on_previous_text",
    }
    if transcription_overrides:
        for param_name, param_value in transcription_overrides.items():
            if param_name in allowed_override_params:
                transcribe_kwargs[param_name] = param_value
                LOGGER.debug(
                    "Applied override: %s = %s (from transcription_overrides)",
                    param_name,
                    param_value,
                )
    LOGGER.debug("Calling model.transcribe() with kwargs: %s", transcribe_kwargs)
    segments, info = model.transcribe(
        str(preprocess_result.output_path),
        **transcribe_kwargs,
    )

    # Get total audio duration from info if available, fallback to duration_hint
    total_audio_duration = getattr(info, "duration", None) or duration_hint
    if total_audio_duration and not duration_hint:
        console.print(f"[cyan]🎧 Input duration:[/cyan] {total_audio_duration / 60:.1f} minutes")
        LOGGER.debug("Input duration: %.1f minutes", total_audio_duration / 60)

    # Process segments the same way as the main transcribe function
    segment_payloads: list[dict[str, Any]] = []
    audio_processed = 0.0
    last_progress_log = transcribe_start

    for segment in segments:
        segment_payloads.append(segment_to_payload(segment))

        # Track progress for logging
        end_time = getattr(segment, "end", None)
        if end_time is not None:
            audio_processed = max(audio_processed, float(end_time))

        # Log progress every minute (same as main transcribe function)
        last_progress_log = maybe_log_progress(
            processed_seconds=audio_processed,
            total_seconds=total_audio_duration,
            start_time=transcribe_start,
            last_log_time=last_progress_log,
        )

    transcribe_time = time.time() - transcribe_start

    # Build payload similar to main transcribe function
    detected_lang = getattr(info, "language", None)
    lang_prob = getattr(info, "language_probability", None)
    duration = getattr(info, "duration", None)

    payload: dict[str, Any] = {
        "audio": os.path.basename(path),
        "language": detected_lang,
        "language_probability": lang_prob,
        "segments": segment_payloads,
    }
    if duration is not None:
        payload["duration"] = duration

    overall_time = time.time() - overall_start

    # Log timing summary (same format as main transcribe function)
    console.print(f"[green]⏱️  Transcription completed in[/green] {transcribe_time:.2f} seconds")
    speed_ratio: float | None = None
    if duration:
        speed_ratio = duration / transcribe_time if transcribe_time > 0 else 0
        console.print(
            f"[green]⚡ Speed:[/green] {speed_ratio:.2f}x realtime ({duration:.1f}s audio in {transcribe_time:.1f}s)"
        )
    console.print(f"[green]✅ Total processing time:[/green] {overall_time:.2f} seconds")
    LOGGER.debug(
        "Transcription completed in %.2f seconds, speed: %.2fx, total: %.2f",
        transcribe_time,
        speed_ratio or 0,
        overall_time,
    )

    # Create metrics similar to main transcribe function
    preset_config = get_preset(preset)
    preprocess_steps = [
        {"name": step.name, "backend": step.backend, "duration": step.duration}
        for step in preprocess_result.metrics.steps
    ]

    # Extract preprocessing step details for metrics
    loudnorm_step = next((s for s in preprocess_result.metrics.steps if s.name == "loudnorm"), None)
    denoise_step = next((s for s in preprocess_result.metrics.steps if s.name == "denoise_light"), None)

    requested_language = language
    applied_language = requested_language
    if applied_language is None:
        applied_language = "et" if preset.startswith("et-") else None

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
        # Transcription parameters (minimal preset uses defaults/hardcoded values)
        # Use values from transcription_config if provided, otherwise None
        beam_size=5,  # Hardcoded in minimal preset
        patience=getattr(transcription_config, "patience", None) if transcription_config else None,
        word_timestamps=False,  # Hardcoded in minimal preset
        task="transcribe",  # Hardcoded in minimal preset
        chunk_length=(getattr(transcription_config, "chunk_length", None) if transcription_config else None),
        vad_filter=(getattr(transcription_config, "vad_filter", None) if transcription_config else None),
        vad_threshold=(getattr(transcription_config, "vad_threshold", None) if transcription_config else None),
        vad_min_speech_duration_ms=(
            transcription_config.vad_parameters.get("min_speech_duration_ms")
            if transcription_config
            and hasattr(transcription_config, "vad_parameters")
            and transcription_config.vad_parameters
            else None
        ),
        vad_max_speech_duration_s=(
            transcription_config.vad_parameters.get("max_speech_duration_s")
            if transcription_config
            and hasattr(transcription_config, "vad_parameters")
            and transcription_config.vad_parameters
            else None
        ),
        vad_min_silence_duration_ms=(
            transcription_config.vad_parameters.get("min_silence_duration_ms")
            if transcription_config
            and hasattr(transcription_config, "vad_parameters")
            and transcription_config.vad_parameters
            else None
        ),
        vad_speech_pad_ms=(
            transcription_config.vad_parameters.get("speech_pad_ms")
            if transcription_config
            and hasattr(transcription_config, "vad_parameters")
            and transcription_config.vad_parameters
            else None
        ),
        temperature=(getattr(transcription_config, "temperature", None) if transcription_config else None),
        temperature_increment_on_fallback=(
            getattr(transcription_config, "temperature_increment_on_fallback", None) if transcription_config else None
        ),
        best_of=(getattr(transcription_config, "best_of", None) if transcription_config else None),
        compression_ratio_threshold=(
            getattr(transcription_config, "compression_ratio_threshold", None) if transcription_config else None
        ),
        logprob_threshold=(getattr(transcription_config, "logprob_threshold", None) if transcription_config else None),
        no_speech_threshold=(
            getattr(transcription_config, "no_speech_threshold", None) if transcription_config else None
        ),
        length_penalty=(getattr(transcription_config, "length_penalty", None) if transcription_config else None),
        repetition_penalty=(
            getattr(transcription_config, "repetition_penalty", None) if transcription_config else None
        ),
        no_repeat_ngram_size=(
            getattr(transcription_config, "no_repeat_ngram_size", None) if transcription_config else None
        ),
        suppress_tokens=(getattr(transcription_config, "suppress_tokens", None) if transcription_config else None),
        condition_on_previous_text=(
            getattr(transcription_config, "condition_on_previous_text", None) if transcription_config else None
        ),
        initial_prompt=(getattr(transcription_config, "initial_prompt", None) if transcription_config else None),
        # Model parameters
        model_id=preset_config.model_id,
        device=preset_config.device,
        compute_type=preset_config.compute_type,
        # Output parameters
        output_format=None,  # Not available at this level, will be set by processor
        float_precision=FLOAT_PRECISION,
    )
    if metrics_collector:
        metrics_collector(metrics_payload)

    preprocess_result.cleanup()
    return payload
