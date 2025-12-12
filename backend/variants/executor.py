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
from backend.variants.transcription_presets import get_minimal_config
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
                variant_description=variant.name,
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
            variant_description=variant.name,
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
    return variant.transcription_config


def is_baseline_config(config: Any) -> bool:  # TranscriptionConfig
    """Check if a TranscriptionConfig is a true baseline (uses library defaults).

    A baseline config should only have vad_filter set (typically False for raw baseline).
    All other parameters should match TranscriptionConfig defaults, meaning they
    will be filtered out and faster-whisper will use its own defaults.

    Args:
        config: TranscriptionConfig to check

    Returns:
        True if config is a baseline config, False otherwise
    """
    from dataclasses import fields  # noqa: PLC0415

    default_config = type(config)()  # Create default instance
    config_fields = {f.name for f in fields(config)}

    # For baseline, we allow only vad_filter to differ from defaults
    # All other fields must match defaults
    for field_name in config_fields:
        if field_name == "vad_filter":
            continue  # vad_filter is allowed to differ for baseline

        config_value = getattr(config, field_name)
        default_value = getattr(default_config, field_name)

        # Compare values (handle dict and list specially)
        if isinstance(config_value, dict) and isinstance(default_value, dict):
            if config_value != default_value:
                return False
        elif isinstance(config_value, list) and isinstance(default_value, list):
            if config_value != default_value:
                return False
        elif config_value != default_value:
            return False

    return True


def is_minimal_config(config: Any) -> bool:  # TranscriptionConfig
    """Check if a TranscriptionConfig is minimal (only essential params set).

    A minimal config only has beam_size, word_timestamps, task, and maybe a few
    allowed overrides set. All other parameters should use TranscriptionConfig defaults.

    Args:
        config: TranscriptionConfig to check

    Returns:
        True if config is minimal, False otherwise
    """
    # Parameters that are allowed in minimal config
    allowed_minimal_params = {
        "beam_size",
        "word_timestamps",
        "task",
        "chunk_length",
        "no_speech_threshold",
        "condition_on_previous_text",
    }

    # Create a minimal config and a default config for comparison
    minimal_ref = get_minimal_config()
    minimal_ref.beam_size = 5  # Minimal config also sets beam_size
    default_config = type(config)()  # Create default instance

    # Get all field names from the dataclass
    from dataclasses import fields  # noqa: PLC0415

    config_fields = {f.name for f in fields(config)}

    # Check each field - it should either:
    # 1. Be in allowed_minimal_params (can differ from defaults)
    # 2. Match the default value (not explicitly set)
    for field_name in config_fields:
        if field_name in allowed_minimal_params:
            continue  # This field is allowed to differ from defaults

        config_value = getattr(config, field_name)
        default_value = getattr(default_config, field_name)

        # Compare values (handle dict and list specially)
        if isinstance(config_value, dict) and isinstance(default_value, dict):
            if config_value != default_value:
                return False
        elif isinstance(config_value, list) and isinstance(default_value, list):
            if config_value != default_value:
                return False
        elif config_value != default_value:
            return False

    return True


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

        # Get transcription config
        transcription_config = create_variant_transcribe_config(variant)

        # Determine if we should use minimal params (only pass essential params to model.transcribe)
        # A config is "baseline" if it only has vad_filter set (true baseline using library defaults)
        # A config is "minimal" if it only has the essential parameters set (beam_size, word_timestamps, task)
        # plus maybe a few allowed overrides
        is_baseline = is_baseline_config(transcription_config)
        is_minimal = is_minimal_config(transcription_config) if not is_baseline else False

        # Run transcription
        if is_baseline:
            # Use baseline transcription (only passes language and vad_filter, nothing else)
            result = transcribe_with_baseline_params(
                path=audio_path,
                preset=preset,
                language=language,
                preprocess_config=preprocess_config,
                preprocess_runner=preprocess_runner,
                transcription_config=transcription_config,
            )
        elif is_minimal:
            # Use minimal params transcription (omits parameters to let faster-whisper use defaults)
            result = transcribe_with_minimal_params(
                path=audio_path,
                preset=preset,
                language=language,
                preprocess_config=preprocess_config,
                preprocess_runner=preprocess_runner,
                transcription_config=transcription_config,
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
            variant_name = variant.name
            txt_path = output_base_path.parent / f"{variant.number}_{variant_name}_{base_name}_{datetime_suffix}.txt"
            json_path = output_base_path.parent / f"{variant.number}_{variant_name}_{base_name}_{datetime_suffix}.json"

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


def transcribe_with_baseline_params(
    path: str,
    preset: str,
    language: str | None,
    preprocess_config: PreprocessConfig,
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
    transcription_config: Any,  # TranscriptionConfig (required now)
    metrics_collector: Callable[[TranscriptionMetrics], None] | None = None,
) -> dict[str, Any]:
    """Transcribe with baseline parameters, using faster-whisper library defaults.

    This is a true baseline that only passes language (if needed) and vad_filter (if explicitly set).
    All other parameters are omitted, allowing faster-whisper to use its own internal defaults.

    Args:
        path: Path to audio file
        preset: Model preset to use
        language: Optional language code
        preprocess_config: PreprocessConfig instance
        preprocess_runner: Function to run preprocessing
        transcription_config: TranscriptionConfig (only vad_filter will be used if set)
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

    # For baseline, only pass language and vad_filter (if explicitly set)
    # Everything else uses faster-whisper library defaults
    LOGGER.info("🔄 Starting transcription with baseline config (library defaults)...")
    transcribe_start = time.time()

    transcribe_kwargs: dict[str, Any] = {}

    # Only include language if it's set
    if applied_language:
        transcribe_kwargs["language"] = applied_language

    # Only include vad_filter if it's explicitly set in config
    if hasattr(transcription_config, "vad_filter"):
        vad_filter_value = getattr(transcription_config, "vad_filter")
        if vad_filter_value is not None:
            transcribe_kwargs["vad_filter"] = vad_filter_value
            LOGGER.debug("Including vad_filter: %s (from baseline config)", vad_filter_value)

    LOGGER.debug("Calling model.transcribe() with baseline kwargs: %s", transcribe_kwargs)
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
    no_speech_skips = 0
    no_speech_skip_windows: list[dict[str, Any]] = []

    # For baseline, use faster-whisper defaults (we don't know what they are, so we'll use None)
    # These are just for tracking, not for filtering

    for segment in segments:
        segment_payloads.append(segment_to_payload(segment))

        # Track segments that match skip criteria (no_speech_prob > threshold AND avg_logprob <= threshold)
        # Note: These are segments that passed but match skip criteria - truly skipped windows don't appear in output
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        avg_logprob = getattr(segment, "avg_logprob", None)
        seg_start = getattr(segment, "start", None)
        seg_end = getattr(segment, "end", None)

        # For baseline, we can't track skips accurately since we don't know the thresholds used
        # But we'll still collect the data for metrics
        if no_speech_prob is not None and avg_logprob is not None:
            no_speech_val = float(no_speech_prob)
            avg_logprob_val = float(avg_logprob)
            # Use default thresholds for tracking (0.5 and -1.0 are common defaults)
            if no_speech_val > 0.5 and avg_logprob_val <= -1.0:
                no_speech_skips += 1
                skip_window = {
                    "start": float(seg_start) if seg_start is not None else None,
                    "end": float(seg_end) if seg_end is not None else None,
                    "no_speech_prob": no_speech_val,
                    "avg_logprob": avg_logprob_val,
                }
                skip_window = {k: v for k, v in skip_window.items() if v is not None}
                if skip_window:
                    no_speech_skip_windows.append(skip_window)

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

    # For baseline, most transcription parameters are None (using library defaults)
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
        # Transcription parameters (baseline uses library defaults, so most are None)
        beam_size=None,  # Using library default
        patience=None,  # Using library default
        word_timestamps=None,  # Using library default
        task=None,  # Using library default
        chunk_length=None,  # Using library default
        vad_filter=(getattr(transcription_config, "vad_filter", None) if transcription_config else None),
        vad_threshold=None,  # Using library default
        vad_min_speech_duration_ms=None,  # Using library default
        vad_max_speech_duration_s=None,  # Using library default
        vad_min_silence_duration_ms=None,  # Using library default
        vad_speech_pad_ms=None,  # Using library default
        temperature=None,  # Using library default
        temperature_increment_on_fallback=None,  # Using library default
        best_of=None,  # Using library default
        compression_ratio_threshold=None,  # Using library default
        logprob_threshold=None,  # Using library default
        no_speech_threshold=None,  # Using library default
        length_penalty=None,  # Using library default
        repetition_penalty=None,  # Using library default
        no_repeat_ngram_size=None,  # Using library default
        suppress_tokens=None,  # Using library default
        condition_on_previous_text=None,  # Using library default
        initial_prompt=None,  # Using library default
        # Model parameters
        model_id=preset_config.model_id,
        device=preset_config.device,
        compute_type=preset_config.compute_type,
        # Output parameters
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
    metrics_dict: dict[str, Any] = {
        "segment_count": metrics_payload.segment_count,
        "no_speech_skips_count": metrics_payload.no_speech_skips_count,
    }
    if metrics_payload.no_speech_skip_windows:
        metrics_dict["no_speech_skip_windows"] = metrics_payload.no_speech_skip_windows
    payload["metrics"] = metrics_dict

    preprocess_result.cleanup()
    return payload


def transcribe_with_minimal_params(
    path: str,
    preset: str,
    language: str | None,
    preprocess_config: PreprocessConfig,
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
    transcription_config: Any,  # TranscriptionConfig (required now)
    metrics_collector: Callable[[TranscriptionMetrics], None] | None = None,
) -> dict[str, Any]:
    """Transcribe with minimal parameters, omitting those that differ between defaults.

    This allows faster-whisper to use its own internal defaults for omitted parameters.
    Only parameters explicitly set in transcription_config that are in the allowed set
    will be included.

    Args:
        path: Path to audio file
        preset: Model preset to use
        language: Optional language code
        preprocess_config: PreprocessConfig instance
        preprocess_runner: Function to run preprocessing
        transcription_config: TranscriptionConfig with parameters (only allowed ones will be used)
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
    # Only include parameters from config that are in the allowed set
    # Allowed: beam_size, chunk_length, no_speech_threshold, condition_on_previous_text,
    #          patience, vad_filter, vad_parameters (with vad_threshold merged in)
    LOGGER.info("🔄 Starting transcription...")
    transcribe_start = time.time()

    # This set defines which parameters can be included in minimal preset
    allowed_minimal_params = {
        "beam_size",
        "chunk_length",
        "no_speech_threshold",
        "condition_on_previous_text",
        "patience",
        "vad_filter",
    }

    # Start with base minimal parameters
    transcribe_kwargs: dict[str, Any] = {
        "beam_size": getattr(transcription_config, "beam_size", 5),
        "word_timestamps": getattr(transcription_config, "word_timestamps", False),
        "language": applied_language,
        "task": getattr(transcription_config, "task", "transcribe"),
    }

    # Add any allowed parameters from config that differ from defaults
    for param_name in allowed_minimal_params:
        if hasattr(transcription_config, param_name):
            param_value = getattr(transcription_config, param_name)
            # Only include if it's not None and not the default
            if param_value is not None:
                transcribe_kwargs[param_name] = param_value
                LOGGER.debug(
                    "Including minimal param: %s = %s (from transcription_config)",
                    param_name,
                    param_value,
                )

    # Handle VAD parameters specially (vad_threshold is merged into vad_parameters)
    # Only include vad_parameters if vad_filter is True (or not explicitly False)
    if hasattr(transcription_config, "vad_filter") and transcription_config.vad_filter:
        # Merge vad_threshold into vad_parameters dict (like in transcribe.py)
        vad_params = dict(getattr(transcription_config, "vad_parameters", {}))
        if hasattr(transcription_config, "vad_threshold") and transcription_config.vad_threshold is not None:
            vad_params["threshold"] = transcription_config.vad_threshold
        # Include vad_parameters if we have any (even if just from vad_threshold)
        if vad_params:
            transcribe_kwargs["vad_parameters"] = vad_params
            LOGGER.debug(
                "Including VAD parameters: %s (vad_filter=True)",
                vad_params,
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
    no_speech_skips = 0
    no_speech_skip_windows: list[dict[str, Any]] = []

    # Track segments that would have been skipped due to no_speech_threshold
    no_speech_threshold = getattr(transcription_config, "no_speech_threshold", 0.5)  # Default from TranscriptionConfig
    logprob_threshold = getattr(transcription_config, "logprob_threshold", -1.0)  # Default from TranscriptionConfig

    for segment in segments:
        segment_payloads.append(segment_to_payload(segment))

        # Track segments that match skip criteria (no_speech_prob > threshold AND avg_logprob <= threshold)
        # Note: These are segments that passed but match skip criteria - truly skipped windows don't appear in output
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
        # Segment statistics
        segment_count=len(segment_payloads),
        no_speech_skips_count=no_speech_skips,  # Always record, even if 0
        no_speech_skip_windows=no_speech_skip_windows if no_speech_skip_windows else None,
    )
    if metrics_collector:
        metrics_collector(metrics_payload)

    # Add metrics to payload JSON for report generation
    metrics_dict: dict[str, Any] = {
        "segment_count": metrics_payload.segment_count,
        "no_speech_skips_count": metrics_payload.no_speech_skips_count,
    }
    if metrics_payload.no_speech_skip_windows:
        metrics_dict["no_speech_skip_windows"] = metrics_payload.no_speech_skip_windows
    payload["metrics"] = metrics_dict

    preprocess_result.cleanup()
    return payload
