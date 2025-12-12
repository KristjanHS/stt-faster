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

    A baseline config uses _explicit_fields to track which parameters are set.
    For baseline, either no fields are explicitly set (true baseline) or
    only vad_filter is explicitly set (no-VAD baseline).

    Args:
        config: TranscriptionConfig to check

    Returns:
        True if config is a baseline config, False otherwise
    """
    # Check if config has _explicit_fields tracking
    if not hasattr(config, "_explicit_fields"):
        return False

    explicit = config._explicit_fields
    # Baseline: empty explicit fields, or only vad_filter
    return len(explicit) == 0 or (len(explicit) == 1 and "vad_filter" in explicit)


def is_minimal_config(config: Any) -> bool:  # TranscriptionConfig
    """Check if a TranscriptionConfig is a minimal config.

    A minimal config uses _explicit_fields to track which parameters are set.
    For minimal configs, only a small set of allowed parameters should be
    explicitly set (beam_size, word_timestamps, task, plus optional overrides).

    Args:
        config: TranscriptionConfig to check

    Returns:
        True if config is minimal, False otherwise
    """
    # Check if config has _explicit_fields tracking
    if not hasattr(config, "_explicit_fields"):
        return False

    explicit = config._explicit_fields
    # Minimal configs can have beam_size, word_timestamps, task, and various overrides
    # If it's a baseline (empty or only vad_filter), it's not minimal
    if len(explicit) == 0 or (len(explicit) == 1 and "vad_filter" in explicit):
        return False

    # Minimal configs are identified by having explicit fields tracked
    # The executor will use to_kwargs() to get only explicitly set fields
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

    # For baseline, use to_kwargs() to get only explicitly set fields
    # Everything else uses faster-whisper library defaults
    LOGGER.info("🔄 Starting transcription with baseline config (library defaults)...")
    transcribe_start = time.time()

    # Get only explicitly set parameters from config
    transcribe_kwargs = transcription_config.to_kwargs()

    # Guard: baseline should only have vad_filter (or be empty)
    # If it has other keys, log a warning
    baseline_allowed_keys = {"vad_filter"}
    unexpected_keys = set(transcribe_kwargs.keys()) - baseline_allowed_keys
    if unexpected_keys:
        LOGGER.warning(
            "Baseline config has unexpected explicit fields: %s. "
            "Baseline should only have vad_filter or be empty. "
            "This may indicate a misconfigured baseline config.",
            unexpected_keys,
        )

    # Only include language if it's set (language is not part of TranscriptionConfig)
    # Note: Baseline forces language for Estonian presets, so it's not "pure defaults"
    if applied_language:
        transcribe_kwargs["language"] = applied_language

    # Backward/forward compatible mapping: faster-whisper uses log_prob_threshold (underscore)
    if "logprob_threshold" in transcribe_kwargs and "log_prob_threshold" not in transcribe_kwargs:
        transcribe_kwargs["log_prob_threshold"] = transcribe_kwargs.pop("logprob_threshold")

    LOGGER.info("Calling model.transcribe() with baseline kwargs: %s", transcribe_kwargs)
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
    # For baseline, we don't know the actual thresholds used by faster-whisper
    # so we can't accurately count "skips". Set to None to indicate unknown.
    segments_matching_skip_rule_guess = 0
    segments_matching_skip_rule_guess_windows: list[dict[str, Any]] = []

    # For baseline, we don't know the actual thresholds used by the library
    # We could guess (0.5 and -1.0 are common defaults), but that's misleading
    # Instead, we'll collect segment data but not claim it's "skips"

    for segment in segments:
        segment_payloads.append(segment_to_payload(segment))

        # Collect segment metadata for potential analysis, but don't claim these are "skips"
        # since we don't know the actual thresholds used by faster-whisper
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        avg_logprob = getattr(segment, "avg_logprob", None)
        seg_start = getattr(segment, "start", None)
        seg_end = getattr(segment, "end", None)

        # Optional: collect segments matching a guessed rule for analysis
        # but this is NOT accurate skip counting
        if no_speech_prob is not None and avg_logprob is not None:
            no_speech_val = float(no_speech_prob)
            avg_logprob_val = float(avg_logprob)
            # Use guessed thresholds (0.5 and -1.0 are common defaults, but not guaranteed)
            if no_speech_val > 0.5 and avg_logprob_val <= -1.0:
                segments_matching_skip_rule_guess += 1
                guess_window = {
                    "start": float(seg_start) if seg_start is not None else None,
                    "end": float(seg_end) if seg_end is not None else None,
                    "no_speech_prob": no_speech_val,
                    "avg_logprob": avg_logprob_val,
                }
                guess_window = {k: v for k, v in guess_window.items() if v is not None}
                if guess_window:
                    segments_matching_skip_rule_guess_windows.append(guess_window)

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
        # Transcription parameters
        # Only use values from transcribe_kwargs (what was actually passed to model.transcribe)
        # If a parameter is not in transcribe_kwargs, it means faster-whisper is using its own
        # internal defaults, which we don't know. Setting it to None is more honest than
        # using transcription_config defaults that weren't actually used.
        beam_size=transcribe_kwargs.get("beam_size"),
        patience=transcribe_kwargs.get("patience"),
        word_timestamps=transcribe_kwargs.get("word_timestamps"),
        task=transcribe_kwargs.get("task"),
        chunk_length=transcribe_kwargs.get("chunk_length"),
        vad_filter=transcribe_kwargs.get("vad_filter"),
        vad_threshold=transcribe_kwargs.get("vad_threshold"),
        # Extract VAD parameters from vad_parameters dict if present
        vad_min_speech_duration_ms=(
            transcribe_kwargs.get("vad_parameters", {}).get("min_speech_duration_ms")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        vad_max_speech_duration_s=(
            transcribe_kwargs.get("vad_parameters", {}).get("max_speech_duration_s")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        vad_min_silence_duration_ms=(
            transcribe_kwargs.get("vad_parameters", {}).get("min_silence_duration_ms")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        vad_speech_pad_ms=(
            transcribe_kwargs.get("vad_parameters", {}).get("speech_pad_ms")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        temperature=transcribe_kwargs.get("temperature"),
        temperature_increment_on_fallback=transcribe_kwargs.get("temperature_increment_on_fallback"),
        best_of=transcribe_kwargs.get("best_of"),
        compression_ratio_threshold=transcribe_kwargs.get("compression_ratio_threshold"),
        logprob_threshold=transcribe_kwargs.get("logprob_threshold"),
        no_speech_threshold=transcribe_kwargs.get("no_speech_threshold"),
        length_penalty=transcribe_kwargs.get("length_penalty"),
        repetition_penalty=transcribe_kwargs.get("repetition_penalty"),
        no_repeat_ngram_size=transcribe_kwargs.get("no_repeat_ngram_size"),
        suppress_tokens=transcribe_kwargs.get("suppress_tokens"),
        condition_on_previous_text=transcribe_kwargs.get("condition_on_previous_text"),
        initial_prompt=transcribe_kwargs.get("initial_prompt"),
        # Model parameters
        model_id=preset_config.model_id,
        device=preset_config.device,
        compute_type=preset_config.compute_type,
        # Output parameters
        output_format=None,  # Not available at this level, will be set by processor
        float_precision=FLOAT_PRECISION,
        # Segment statistics
        segment_count=len(segment_payloads),
        # For baseline, we don't know actual skip thresholds used by faster-whisper,
        # so we can't accurately count skips. Set to None to indicate unknown.
        no_speech_skips_count=None,
        no_speech_skip_windows=None,
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

    # Add transcribe_kwargs to payload so reports can see what was actually passed
    # This is the source of truth after whitelist + VAD merge + language + key mapping
    # Note: Baseline forces language for Estonian presets, so it's not "pure defaults"
    payload["transcribe_kwargs"] = transcribe_kwargs.copy()
    if applied_language:
        payload["language_forced"] = applied_language

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

    # Use to_kwargs() to get only explicitly set parameters from config
    transcribe_kwargs = transcription_config.to_kwargs()

    # Whitelist: only allow specific parameters for minimal configs
    # This prevents accidentally passing unexpected parameters
    ALLOWED_MINIMAL_KEYS = {
        "beam_size",
        "word_timestamps",
        "task",
        "chunk_length",
        "no_speech_threshold",
        "logprob_threshold",
        "condition_on_previous_text",
        "patience",
        "vad_filter",
        "vad_parameters",
        "temperature",
        "temperature_increment_on_fallback",
        "best_of",
        "compression_ratio_threshold",
        "length_penalty",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "suppress_tokens",
        "initial_prompt",
    }

    # Filter to only allowed keys
    filtered_kwargs: dict[str, Any] = {}
    for key, value in transcribe_kwargs.items():
        if key in ALLOWED_MINIMAL_KEYS:
            filtered_kwargs[key] = value
        else:
            LOGGER.warning(
                "Minimal config has unexpected parameter '%s' (not in ALLOWED_MINIMAL_KEYS). "
                "This parameter will be omitted to prevent contamination.",
                key,
            )

    transcribe_kwargs = filtered_kwargs

    # Language is not part of TranscriptionConfig, add it separately if set
    if applied_language:
        transcribe_kwargs["language"] = applied_language

    # Handle VAD parameters specially (vad_threshold is merged into vad_parameters)
    # Only include vad_parameters if vad_filter is True (or not explicitly False)
    if transcribe_kwargs.get("vad_filter") is True:
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

    # Ensure we never pass vad_threshold directly (it's merged into vad_parameters)
    transcribe_kwargs.pop("vad_threshold", None)

    # Backward/forward compatible mapping: faster-whisper uses log_prob_threshold (underscore)
    if "logprob_threshold" in transcribe_kwargs and "log_prob_threshold" not in transcribe_kwargs:
        transcribe_kwargs["log_prob_threshold"] = transcribe_kwargs.pop("logprob_threshold")

    LOGGER.info("Calling model.transcribe() with minimal kwargs: %s", transcribe_kwargs)
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
    segments_matching_no_speech_rule = 0
    segments_matching_no_speech_rule_windows: list[dict[str, Any]] = []

    # Derive thresholds from what was actually passed to model.transcribe()
    # Note: We check both logprob_threshold and log_prob_threshold for compatibility
    no_speech_threshold = transcribe_kwargs.get("no_speech_threshold")
    logprob_threshold = transcribe_kwargs.get("log_prob_threshold") or transcribe_kwargs.get("logprob_threshold")

    # Only track skip matches if both thresholds were explicitly passed
    can_track_skips = no_speech_threshold is not None and logprob_threshold is not None

    for segment in segments:
        segment_payloads.append(segment_to_payload(segment))

        # Track segments that match the no_speech rule (no_speech_prob > threshold AND avg_logprob <= threshold)
        # Note: These are segments that PASSED but match the rule - truly skipped windows don't appear in output.
        # This is NOT a skip count; it's a count of output segments matching the heuristic.
        # Only do this if we actually passed these thresholds to model.transcribe()
        if can_track_skips:
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            avg_logprob = getattr(segment, "avg_logprob", None)
            seg_start = getattr(segment, "start", None)
            seg_end = getattr(segment, "end", None)

            if no_speech_prob is not None and avg_logprob is not None:
                no_speech_val = float(no_speech_prob)
                avg_logprob_val = float(avg_logprob)
                # Only compare if thresholds are not None (they were explicitly passed)
                if (
                    no_speech_threshold is not None
                    and logprob_threshold is not None
                    and no_speech_val > no_speech_threshold
                    and avg_logprob_val <= logprob_threshold
                ):
                    segments_matching_no_speech_rule += 1
                    # Add to debug list
                    rule_match_window = {
                        "start": float(seg_start) if seg_start is not None else None,
                        "end": float(seg_end) if seg_end is not None else None,
                        "no_speech_prob": no_speech_val,
                        "avg_logprob": avg_logprob_val,
                    }
                    # Remove None values
                    rule_match_window = {k: v for k, v in rule_match_window.items() if v is not None}
                    if rule_match_window:
                        segments_matching_no_speech_rule_windows.append(rule_match_window)

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
        # Transcription parameters
        # Only use values from transcribe_kwargs (what was actually passed to model.transcribe)
        # If a parameter is not in transcribe_kwargs, it means faster-whisper is using its own
        # internal defaults, which we don't know. Setting it to None is more honest than
        # using transcription_config defaults that weren't actually used.
        beam_size=transcribe_kwargs.get("beam_size"),
        patience=transcribe_kwargs.get("patience"),
        word_timestamps=transcribe_kwargs.get("word_timestamps"),
        task=transcribe_kwargs.get("task"),
        chunk_length=transcribe_kwargs.get("chunk_length"),
        vad_filter=transcribe_kwargs.get("vad_filter"),
        vad_threshold=transcribe_kwargs.get("vad_threshold"),
        # Extract VAD parameters from vad_parameters dict if present
        vad_min_speech_duration_ms=(
            transcribe_kwargs.get("vad_parameters", {}).get("min_speech_duration_ms")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        vad_max_speech_duration_s=(
            transcribe_kwargs.get("vad_parameters", {}).get("max_speech_duration_s")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        vad_min_silence_duration_ms=(
            transcribe_kwargs.get("vad_parameters", {}).get("min_silence_duration_ms")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        vad_speech_pad_ms=(
            transcribe_kwargs.get("vad_parameters", {}).get("speech_pad_ms")
            if isinstance(transcribe_kwargs.get("vad_parameters"), dict)
            else None
        ),
        temperature=transcribe_kwargs.get("temperature"),
        temperature_increment_on_fallback=transcribe_kwargs.get("temperature_increment_on_fallback"),
        best_of=transcribe_kwargs.get("best_of"),
        compression_ratio_threshold=transcribe_kwargs.get("compression_ratio_threshold"),
        logprob_threshold=transcribe_kwargs.get("logprob_threshold"),
        no_speech_threshold=transcribe_kwargs.get("no_speech_threshold"),
        length_penalty=transcribe_kwargs.get("length_penalty"),
        repetition_penalty=transcribe_kwargs.get("repetition_penalty"),
        no_repeat_ngram_size=transcribe_kwargs.get("no_repeat_ngram_size"),
        suppress_tokens=transcribe_kwargs.get("suppress_tokens"),
        condition_on_previous_text=transcribe_kwargs.get("condition_on_previous_text"),
        initial_prompt=transcribe_kwargs.get("initial_prompt"),
        # Model parameters
        model_id=preset_config.model_id,
        device=preset_config.device,
        compute_type=preset_config.compute_type,
        # Output parameters
        output_format=None,  # Not available at this level, will be set by processor
        float_precision=FLOAT_PRECISION,
        # Segment statistics
        segment_count=len(segment_payloads),
        # Note: segments_matching_no_speech_rule is NOT a skip count - it counts output segments
        # that match the heuristic. Truly skipped windows produce no segments.
        # Real skip counts would require window-level instrumentation.
        segments_matching_no_speech_rule_count=segments_matching_no_speech_rule if can_track_skips else None,
        segments_matching_no_speech_rule_windows=(
            segments_matching_no_speech_rule_windows
            if (can_track_skips and segments_matching_no_speech_rule_windows)
            else None
        ),
        # Legacy fields for compatibility (set to None since we can't accurately count skips)
        no_speech_skips_count=None,
        no_speech_skip_windows=None,
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

    # Add transcribe_kwargs to payload so reports can see what was actually passed
    # This is the source of truth after whitelist + VAD merge + language + key mapping
    # Note: Baseline forces language for Estonian presets, so it's not "pure defaults"
    payload["transcribe_kwargs"] = transcribe_kwargs.copy()
    if applied_language:
        payload["language_forced"] = applied_language

    preprocess_result.cleanup()
    return payload
