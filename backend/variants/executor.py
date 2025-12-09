"""Variant execution engine."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.orchestrator import PreprocessResult
from backend.transcribe import (
    maybe_log_progress,
    pick_model,
    segment_to_payload,
)
from backend.variants.preprocess_steps import create_preprocess_runner
from backend.variants.transcription_presets import get_transcription_config
from backend.variants.variant import Variant

LOGGER = logging.getLogger(__name__)


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
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("[%s] Running variant: %s", timestamp, variant.name)

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
            result = transcribe_with_minimal_params(
                path=audio_path,
                preset=preset,
                language=language,
                preprocess_config=preprocess_config,
                preprocess_runner=preprocess_runner,
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
            LOGGER.info("Saved text output: %s", txt_path.name)

            # Save JSON file
            with json_path.open("w", encoding="utf-8") as json_file:
                import json  # noqa: PLC0415

                json.dump(result, json_file, ensure_ascii=False, indent=2)
            output_files["json"] = str(json_path)
            LOGGER.info("Saved JSON output: %s", json_path.name)

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
) -> dict[str, Any]:
    """Transcribe with minimal parameters, omitting those that differ between defaults.

    This allows faster-whisper to use its own internal defaults for omitted parameters.
    """
    overall_start = time.time()
    preprocess_result = preprocess_runner(path, preprocess_config)
    duration_hint = preprocess_result.input_info.duration if preprocess_result.input_info else None
    if duration_hint:
        LOGGER.info("🎧 Input duration: %.1f minutes (from metadata)", duration_hint / 60)

    model = pick_model(preset)

    # Auto-detect language based on preset if not explicitly provided
    applied_language = language
    if applied_language is None:
        applied_language = "et" if preset.startswith("et-") else None

    # Call model.transcribe with only essential parameters
    # Omit: patience, chunk_length, vad_threshold, vad_parameters,
    # temperature, no_speech_threshold, no_repeat_ngram_size
    transcribe_start = time.time()
    segments, info = model.transcribe(
        str(preprocess_result.output_path),
        beam_size=5,  # Keep beam_size as it's commonly used
        word_timestamps=False,
        language=applied_language,
        task="transcribe",
        # Omit: patience, chunk_length, vad_threshold, vad_parameters,
        # temperature, no_speech_threshold, no_repeat_ngram_size
        # Let faster-whisper use its internal defaults for these
    )

    # Get total audio duration from info if available, fallback to duration_hint
    total_audio_duration = getattr(info, "duration", None) or duration_hint
    if total_audio_duration and not duration_hint:
        LOGGER.info("🎧 Input duration: %.1f minutes", total_audio_duration / 60)

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
    LOGGER.info("⏱️  Transcription completed in %.2f seconds", transcribe_time)
    if duration:
        speed_ratio = duration / transcribe_time if transcribe_time > 0 else 0
        LOGGER.info("⚡ Speed: %.2fx realtime (%.1fs audio in %.1fs)", speed_ratio, duration, transcribe_time)
    LOGGER.info("✅ Total processing time: %.2f seconds", overall_time)

    preprocess_result.cleanup()
    return payload
