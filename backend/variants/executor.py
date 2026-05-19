"""Variant execution engine."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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

# VariantRunContext is imported lazily inside functions to avoid a circular
# import: backend.run_config imports backend.variants.variant, which triggers
# backend.variants.__init__, which re-exports execute_variant from this module.
if TYPE_CHECKING:
    from backend.run_config import VariantRunContext

LOGGER = logging.getLogger(__name__)
console = Console()


def create_variant_preprocess_runner(
    variant: Variant,
    preprocess_config: PreprocessConfig,
    *,
    base_name: str | None = None,
    context: VariantRunContext | None = None,
) -> Callable[[str, PreprocessConfig], PreprocessResult]:
    """Create a preprocessing runner function for a variant.

    Args:
        variant: Variant definition
        preprocess_config: PreprocessConfig to use
        base_name: Optional base name for intermediate file naming
        context: Optional per-variant runtime context (output_dir / datetime_suffix /
            copy_intermediate). Pass ``None`` for callers that don't write intermediates
            (e.g. service-layer transcription with no output dir).

    Returns:
        A function that takes (path: str, config: PreprocessConfig) -> PreprocessResult
    """
    from backend.run_config import VariantRunContext  # noqa: PLC0415

    ctx = context if context is not None else VariantRunContext()
    datetime_suffix = ctx.datetime_suffix
    output_dir = ctx.output_dir
    copy_intermediate = ctx.copy_intermediate

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
    # Baseline: empty explicit fields, or only vad_filter, or vad_filter + word_timestamps
    return (
        len(explicit) == 0
        or (len(explicit) == 1 and "vad_filter" in explicit)
        or (len(explicit) == 2 and "vad_filter" in explicit and "word_timestamps" in explicit)
    )


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
    context: VariantRunContext | None = None,
) -> dict[str, Any]:
    """Execute a single variant and return results.

    Args:
        variant: Variant definition to execute
        audio_path: Path to audio file to transcribe
        preset: Model preset to use
        language: Optional language code
        context: Per-variant runtime context (output paths, datetime suffix,
            copy_intermediate flag). Defaults to an empty context (no intermediate
            output side effects).

    Returns:
        Dictionary with variant results in same format as run_variant()
    """
    from backend.run_config import VariantRunContext  # noqa: PLC0415

    ctx = context if context is not None else VariantRunContext()

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
        if ctx.copy_intermediate and ctx.output_dir:
            preprocess_config.output_dir = str(ctx.output_dir)
        else:
            preprocess_config.output_dir = None

        # Create preprocessing runner from variant steps or use custom runner
        base_name = ctx.output_base_path.stem if ctx.output_base_path else Path(audio_path).stem
        preprocess_runner = create_variant_preprocess_runner(
            variant,
            preprocess_config,
            base_name=base_name,
            context=ctx,
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
        if ctx.output_base_path and ctx.datetime_suffix:
            base_name = ctx.output_base_path.stem
            variant_name = variant.name
            txt_path = (
                ctx.output_base_path.parent / f"{variant.number}_{variant_name}_{base_name}_{ctx.datetime_suffix}.txt"
            )
            json_path = (
                ctx.output_base_path.parent / f"{variant.number}_{variant_name}_{base_name}_{ctx.datetime_suffix}.json"
            )

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


def _collect_executor_segments(
    segments: Any,
    *,
    transcribe_start: float,
    total_audio_duration: float | None,
    no_speech_threshold: float | None,
    logprob_threshold: float | None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Drain the faster-whisper segment iterator for the baseline/minimal paths.

    Mirrors :func:`backend.transcribe._collect_segments` but accepts optional
    thresholds. When either threshold is ``None`` (baseline / minimal w/o explicit
    thresholds), per-segment heuristic matching is skipped — the returned skip
    counters are ``(0, [])`` so callers can record them as "unknown" in metrics.

    Returns ``(segment_payloads, skip_count, skip_windows)``.
    """
    segment_payloads: list[dict[str, Any]] = []
    skip_count = 0
    skip_windows: list[dict[str, Any]] = []
    audio_processed = 0.0
    last_progress_log = transcribe_start
    can_track = no_speech_threshold is not None and logprob_threshold is not None

    for segment in segments:
        segment_payloads.append(segment_to_payload(segment))

        if can_track:
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            avg_logprob = getattr(segment, "avg_logprob", None)
            seg_start = getattr(segment, "start", None)
            seg_end = getattr(segment, "end", None)

            if no_speech_prob is not None and avg_logprob is not None:
                no_speech_val = float(no_speech_prob)
                avg_logprob_val = float(avg_logprob)
                # `can_track` already implies both thresholds are not None; the
                # explicit checks below make the narrowing visible to pyright.
                if (
                    no_speech_threshold is not None
                    and logprob_threshold is not None
                    and no_speech_val > no_speech_threshold
                    and avg_logprob_val <= logprob_threshold
                ):
                    skip_count += 1
                    skip_window = {
                        "start": float(seg_start) if seg_start is not None else None,
                        "end": float(seg_end) if seg_end is not None else None,
                        "no_speech_prob": no_speech_val,
                        "avg_logprob": avg_logprob_val,
                    }
                    skip_window = {k: v for k, v in skip_window.items() if v is not None}
                    if skip_window:
                        skip_windows.append(skip_window)

        end_time = getattr(segment, "end", None)
        if end_time is not None:
            audio_processed = max(audio_processed, float(end_time))

        last_progress_log = maybe_log_progress(
            processed_seconds=audio_processed,
            total_seconds=total_audio_duration,
            start_time=transcribe_start,
            last_log_time=last_progress_log,
        )

    return segment_payloads, skip_count, skip_windows


def _build_executor_metrics(
    *,
    path: str,
    preset: str,
    requested_language: str | None,
    applied_language: str | None,
    detected_language: str | None,
    language_probability: float | None,
    audio_duration: float | None,
    overall_time: float,
    transcribe_time: float,
    speed_ratio: float | None,
    preprocess_result: PreprocessResult,
    preprocess_config: PreprocessConfig,
    transcribe_kwargs: dict[str, Any],
    segment_payloads: list[dict[str, Any]],
    skip_count: int,
    skip_windows: list[dict[str, Any]],
    can_track_skips: bool,
) -> TranscriptionMetrics:
    """Assemble TranscriptionMetrics for the baseline / minimal executor paths.

    Distinct from :func:`backend.transcribe._build_metrics_payload` because here
    the per-call transcription parameters come from the actual ``transcribe_kwargs``
    that were passed to ``model.transcribe()`` — fields not in that dict are
    recorded as ``None`` to honestly reflect "faster-whisper used its own default".
    The other helper reads from ``transcription_config`` directly (full-config
    path always passes every field).
    """
    preset_config = get_preset(preset)
    preprocess_steps = [
        {"name": step.name, "backend": step.backend, "duration": step.duration}
        for step in preprocess_result.metrics.steps
    ]

    loudnorm_step = next((s for s in preprocess_result.metrics.steps if s.name == "loudnorm"), None)
    denoise_step = next((s for s in preprocess_result.metrics.steps if s.name == "denoise_light"), None)

    vad_params = transcribe_kwargs.get("vad_parameters")
    vad_params_dict: dict[str, Any] = vad_params if isinstance(vad_params, dict) else {}  # type: ignore[reportUnknownVariableType]

    return TranscriptionMetrics(
        # File and model info
        audio_path=path,
        preset=preset,
        # Language detection
        requested_language=requested_language,
        applied_language=applied_language,
        detected_language=detected_language,
        language_probability=language_probability,
        # Timing metrics
        audio_duration=audio_duration,
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
        # Only use values from transcribe_kwargs (what was actually passed to model.transcribe).
        # If a parameter is not in transcribe_kwargs, faster-whisper used its own internal
        # default (which we don't know). Recording None is more honest than copying the
        # transcription_config default that wasn't actually applied.
        beam_size=transcribe_kwargs.get("beam_size"),
        patience=transcribe_kwargs.get("patience"),
        word_timestamps=transcribe_kwargs.get("word_timestamps"),
        task=transcribe_kwargs.get("task"),
        chunk_length=transcribe_kwargs.get("chunk_length"),
        vad_filter=transcribe_kwargs.get("vad_filter"),
        vad_threshold=transcribe_kwargs.get("vad_threshold"),
        vad_min_speech_duration_ms=vad_params_dict.get("min_speech_duration_ms"),
        vad_max_speech_duration_s=vad_params_dict.get("max_speech_duration_s"),
        vad_min_silence_duration_ms=vad_params_dict.get("min_silence_duration_ms"),
        vad_speech_pad_ms=vad_params_dict.get("speech_pad_ms"),
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
        # segments_matching_no_speech_rule_* fields are populated only when the caller
        # passed both thresholds to model.transcribe(); otherwise we can't honestly
        # call any segment a "match" against an unknown library default.
        segments_matching_no_speech_rule_count=skip_count if can_track_skips else None,
        segments_matching_no_speech_rule_windows=(skip_windows if (can_track_skips and skip_windows) else None),
        # Legacy fields preserved for compatibility — baseline/minimal can't claim
        # accurate "skip" counts without window-level instrumentation.
        no_speech_skips_count=None,
        no_speech_skip_windows=None,
    )


def _run_transcription(
    *,
    path: str,
    preset: str,
    requested_language: str | None,
    applied_language: str | None,
    preprocess_config: PreprocessConfig,
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
    transcribe_kwargs: dict[str, Any],
    metrics_collector: Callable[[TranscriptionMetrics], None] | None,
    can_track_skips: bool,
    log_label: str,
) -> dict[str, Any]:
    """Shared scaffolding for the baseline/minimal executor paths.

    Owns: preprocess → model load → language logging → ``model.transcribe(**kwargs)``
    → segment drain → timing logging → metrics assembly → payload return → cleanup.
    Callers are responsible for building ``transcribe_kwargs`` (including the
    ``language`` key and any backward-compatible key remaps) BEFORE invoking this.

    Args:
        path: Original audio path (used for metrics + display name).
        preset: Model preset.
        requested_language: User-supplied language code, or ``None``.
        applied_language: Effective language after preset-aware defaulting.
        preprocess_config: Preprocess config (passed to runner + recorded in metrics).
        preprocess_runner: Function (path, cfg) -> PreprocessResult.
        transcribe_kwargs: The final kwargs dict for ``model.transcribe()``. Skip
            tracking thresholds (``no_speech_threshold`` / ``log_prob_threshold``
            or ``logprob_threshold``) are read from here when ``can_track_skips``.
        metrics_collector: Optional callback to receive the assembled metrics.
        can_track_skips: When True, the segment loop counts segments matching the
            no-speech heuristic using thresholds from ``transcribe_kwargs``.
        log_label: Short human-readable label ("baseline" / "minimal") woven into
            the pre-call DEBUG line.
    """
    overall_start = time.time()
    preprocess_result = preprocess_runner(path, preprocess_config)
    duration_hint = preprocess_result.input_info.duration if preprocess_result.input_info else None
    if duration_hint:
        console.print(f"[cyan]🎧 Input duration:[/cyan] {duration_hint / 60:.1f} minutes (from metadata)")
        LOGGER.debug("Input duration: %.1f minutes (from metadata)", duration_hint / 60)

    model = pick_model(preset)
    LOGGER.info("Model loaded, ready for transcription")

    if applied_language:
        LOGGER.info("Language: %s (forced)", applied_language)
    else:
        LOGGER.info("Language: auto-detect")

    transcribe_start = time.time()

    LOGGER.debug("Calling model.transcribe() with %s kwargs: %s", log_label, transcribe_kwargs)
    segments, info = model.transcribe(
        str(preprocess_result.output_path),
        **transcribe_kwargs,
    )

    total_audio_duration = getattr(info, "duration", None) or duration_hint
    if total_audio_duration and not duration_hint:
        console.print(f"[cyan]🎧 Input duration:[/cyan] {total_audio_duration / 60:.1f} minutes")
        LOGGER.debug("Input duration: %.1f minutes", total_audio_duration / 60)

    no_speech_threshold = transcribe_kwargs.get("no_speech_threshold") if can_track_skips else None
    logprob_threshold = (
        (transcribe_kwargs.get("log_prob_threshold") or transcribe_kwargs.get("logprob_threshold"))
        if can_track_skips
        else None
    )

    segment_payloads, skip_count, skip_windows = _collect_executor_segments(
        segments,
        transcribe_start=transcribe_start,
        total_audio_duration=total_audio_duration,
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
    )

    transcribe_time = time.time() - transcribe_start

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

    metrics_payload = _build_executor_metrics(
        path=path,
        preset=preset,
        requested_language=requested_language,
        applied_language=applied_language,
        detected_language=detected_lang,
        language_probability=lang_prob,
        audio_duration=duration,
        overall_time=overall_time,
        transcribe_time=transcribe_time,
        speed_ratio=speed_ratio,
        preprocess_result=preprocess_result,
        preprocess_config=preprocess_config,
        transcribe_kwargs=transcribe_kwargs,
        segment_payloads=segment_payloads,
        skip_count=skip_count,
        skip_windows=skip_windows,
        can_track_skips=can_track_skips,
    )
    if metrics_collector:
        metrics_collector(metrics_payload)

    metrics_dict: dict[str, Any] = {
        "segment_count": metrics_payload.segment_count,
        "no_speech_skips_count": metrics_payload.no_speech_skips_count,
    }
    if metrics_payload.no_speech_skip_windows:
        metrics_dict["no_speech_skip_windows"] = metrics_payload.no_speech_skip_windows
    payload["metrics"] = metrics_dict

    # transcribe_kwargs is the source of truth for what was actually passed
    # (after whitelist + VAD merge + language + key mapping). Reports rely on this.
    # Note: Baseline forces language for Estonian presets, so the payload is not "pure defaults".
    payload["transcribe_kwargs"] = transcribe_kwargs.copy()
    if applied_language:
        payload["language_forced"] = applied_language

    preprocess_result.cleanup()
    return payload


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

    This is a true baseline that only passes language (if needed) and vad_filter
    (if explicitly set). All other parameters are omitted, allowing faster-whisper
    to use its own internal defaults — so skip counts cannot be tracked (the
    thresholds the library applied are unknown to us).

    Args:
        path: Path to audio file
        preset: Model preset to use
        language: Optional language code
        preprocess_config: PreprocessConfig instance
        preprocess_runner: Function to run preprocessing
        transcription_config: TranscriptionConfig (only vad_filter will be used if set)
        metrics_collector: Optional metrics collector callback
    """
    # Auto-detect language based on preset if not explicitly provided
    applied_language = language if language is not None else ("et" if preset.startswith("et-") else None)

    # For baseline, use to_kwargs() to get only explicitly set fields.
    # Everything else uses faster-whisper library defaults.
    transcribe_kwargs = transcription_config.to_kwargs()

    # Guard: baseline should only have vad_filter (or be empty). Warn on anything else.
    baseline_allowed_keys = {"vad_filter", "word_timestamps"}
    unexpected_keys = set(transcribe_kwargs.keys()) - baseline_allowed_keys
    if unexpected_keys:
        LOGGER.warning(
            "Baseline config has unexpected explicit fields: %s. "
            "Baseline should only have vad_filter or be empty. "
            "This may indicate a misconfigured baseline config.",
            unexpected_keys,
        )

    # Language is not part of TranscriptionConfig; add it separately if set.
    # Note: Baseline forces language for Estonian presets, so it's not "pure defaults".
    if applied_language:
        transcribe_kwargs["language"] = applied_language

    # Backward/forward compatible mapping: faster-whisper uses log_prob_threshold (underscore)
    if "logprob_threshold" in transcribe_kwargs and "log_prob_threshold" not in transcribe_kwargs:
        transcribe_kwargs["log_prob_threshold"] = transcribe_kwargs.pop("logprob_threshold")

    LOGGER.info("Starting transcription with baseline config")

    return _run_transcription(
        path=path,
        preset=preset,
        requested_language=language,
        applied_language=applied_language,
        preprocess_config=preprocess_config,
        preprocess_runner=preprocess_runner,
        transcribe_kwargs=transcribe_kwargs,
        metrics_collector=metrics_collector,
        # Baseline doesn't know what thresholds the library applied — refuse to
        # invent a skip count.
        can_track_skips=False,
        log_label="baseline",
    )


# Whitelist of allowed parameters for minimal configs.
# Anything outside this set is dropped to prevent accidental contamination
# from variant overrides creeping into model.transcribe(**kwargs).
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
    Only parameters explicitly set in transcription_config that are in
    :data:`ALLOWED_MINIMAL_KEYS` will be included.

    Args:
        path: Path to audio file
        preset: Model preset to use
        language: Optional language code
        preprocess_config: PreprocessConfig instance
        preprocess_runner: Function to run preprocessing
        transcription_config: TranscriptionConfig with parameters (only allowed ones will be used)
        metrics_collector: Optional metrics collector callback
    """
    # Auto-detect language based on preset if not explicitly provided
    applied_language = language if language is not None else ("et" if preset.startswith("et-") else None)

    # Use to_kwargs() to get only explicitly set parameters from config
    raw_kwargs = transcription_config.to_kwargs()

    # Whitelist: only allow specific parameters for minimal configs.
    filtered_kwargs: dict[str, Any] = {}
    for key, value in raw_kwargs.items():
        if key in ALLOWED_MINIMAL_KEYS:
            filtered_kwargs[key] = value
        else:
            LOGGER.warning(
                "Minimal config has unexpected parameter '%s' (not in ALLOWED_MINIMAL_KEYS). "
                "This parameter will be omitted to prevent contamination.",
                key,
            )

    transcribe_kwargs = filtered_kwargs

    # Language is not part of TranscriptionConfig; add it separately if set.
    if applied_language:
        transcribe_kwargs["language"] = applied_language

    # Handle VAD parameters specially: vad_threshold merges into vad_parameters,
    # and we only emit vad_parameters if vad_filter is True.
    if transcribe_kwargs.get("vad_filter") is True:
        vad_params = dict(getattr(transcription_config, "vad_parameters", {}))
        # Merge vad_threshold only if it was explicitly set (not just config default).
        vad_threshold_explicit = (
            hasattr(transcription_config, "_explicit_fields")
            and "vad_threshold" in transcription_config._explicit_fields
        )
        if vad_threshold_explicit and transcription_config.vad_threshold is not None:
            vad_params["threshold"] = transcription_config.vad_threshold
        if vad_params:
            transcribe_kwargs["vad_parameters"] = vad_params
            LOGGER.debug(
                "Including VAD parameters: %s (vad_filter=True)",
                vad_params,
            )

    # Never pass vad_threshold directly — it lives inside vad_parameters.
    transcribe_kwargs.pop("vad_threshold", None)

    # Backward/forward compatible mapping: faster-whisper uses log_prob_threshold (underscore)
    if "logprob_threshold" in transcribe_kwargs and "log_prob_threshold" not in transcribe_kwargs:
        transcribe_kwargs["log_prob_threshold"] = transcribe_kwargs.pop("logprob_threshold")

    # Minimal can track skips only when BOTH thresholds were explicitly passed.
    no_speech_threshold = transcribe_kwargs.get("no_speech_threshold")
    logprob_threshold = transcribe_kwargs.get("log_prob_threshold") or transcribe_kwargs.get("logprob_threshold")
    can_track_skips = no_speech_threshold is not None and logprob_threshold is not None

    LOGGER.info("Starting transcription")

    return _run_transcription(
        path=path,
        preset=preset,
        requested_language=language,
        applied_language=applied_language,
        preprocess_config=preprocess_config,
        preprocess_runner=preprocess_runner,
        transcribe_kwargs=transcribe_kwargs,
        metrics_collector=metrics_collector,
        can_track_skips=can_track_skips,
        log_label="minimal",
    )
