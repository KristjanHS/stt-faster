#!/usr/bin/env python3
"""Compare different transcription pipeline configurations on the same file.

This script runs 12 different transcription variants (see the variants list in main() below for details).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from backend.preprocess.config import PreprocessConfig, TranscriptionConfig
from backend.preprocess.errors import PreprocessError, StepExecutionError
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.metrics import PreprocessMetrics, StepMetrics
from backend.preprocess.orchestrator import PreprocessResult, _copy_stage_output
from backend.preprocess.steps.denoise_light import apply_light_denoise
from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline
from backend.transcribe import pick_model, segment_to_payload, transcribe

LOGGER = logging.getLogger(__name__)


def _noop_denoise(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
) -> StepMetrics:
    """No-op denoise function that just copies the input to output."""
    start = time.time()
    shutil.copy2(input_path, output_path)
    duration = time.time() - start
    return StepMetrics(name="noop_denoise", backend="copy", duration=duration)


def _simple_resample(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Simple resampling function using ffmpeg (no loudnorm, no rnnoise)."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("simple_resample", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="simple_resample", backend="ffmpeg", duration=duration)


def _loudnorm_only(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
) -> StepMetrics:
    """Lightweight ffmpeg step that only does resampling and loudness normalization (no highpass, no RNNoise)."""
    import ffmpeg  # type: ignore[import-untyped]

    from backend.preprocess.config import PreprocessConfig

    start = time.time()
    try:
        preset_config = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
        target_i = preset_config["I"]
        target_tp = preset_config.get("TP", -2.0)
        target_lra = preset_config["LRA"]

        # Build lightweight filter graph: only resample + loudnorm (no highpass, no RNNoise)
        filter_graph = (
            f"aresample=resampler=soxr:osr={target_sample_rate},loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        )

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("loudnorm_only", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="loudnorm_only", backend="ffmpeg", duration=duration)


def _loudnorm_with_highpass(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
) -> StepMetrics:
    """Lightweight ffmpeg step with highpass filter, resampling, and loudness normalization (no RNNoise)."""
    import ffmpeg  # type: ignore[import-untyped]

    from backend.preprocess.config import PreprocessConfig

    start = time.time()
    try:
        preset_config = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
        target_i = preset_config["I"]
        target_tp = preset_config.get("TP", -2.0)
        target_lra = preset_config["LRA"]

        # Build lightweight filter graph: highpass + resample + loudnorm (no RNNoise)
        filter_graph = (
            f"highpass=f=60,"
            f"aresample=resampler=soxr:osr={target_sample_rate},"
            f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        )

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("loudnorm_with_highpass", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="loudnorm_with_highpass", backend="ffmpeg", duration=duration)


def _dynaudnorm_only(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Lightweight ffmpeg step that only does resampling and dynamic audio normalization (no highpass, no RNNoise)."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Build lightweight filter graph: only resample + dynaudnorm (no highpass, no RNNoise)
        filter_graph = f"aresample=resampler=soxr:osr={target_sample_rate},dynaudnorm"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("dynaudnorm_only", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="dynaudnorm_only", backend="ffmpeg", duration=duration)


def _highlow_aform_loudnorm(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
) -> StepMetrics:
    """Lightweight ffmpeg step with highpass, lowpass, aformat, and loudness normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    from backend.preprocess.config import PreprocessConfig

    start = time.time()
    try:
        preset_config = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
        target_i = preset_config["I"]
        target_tp = preset_config.get("TP", -2.0)
        target_lra = preset_config["LRA"]

        # Build filter graph: highpass + lowpass + aformat + loudnorm
        filter_graph = (
            f"highpass=f=60,"
            f"lowpass=f=8000,"
            f"aformat=sample_rates=16000,"
            f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        )

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("highlow_aform_loudnorm", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="highlow_aform_loudnorm", backend="ffmpeg", duration=duration)


def _highlow_nosampl_loudnorm(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
) -> StepMetrics:
    """Lightweight ffmpeg step with highpass, lowpass, and loudness normalization (no aformat/sampling)."""
    import ffmpeg  # type: ignore[import-untyped]

    from backend.preprocess.config import PreprocessConfig

    start = time.time()
    try:
        preset_config = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
        target_i = preset_config["I"]
        target_tp = preset_config.get("TP", -2.0)
        target_lra = preset_config["LRA"]

        # Build filter graph: highpass + lowpass + loudnorm (no aformat/sampling)
        filter_graph = f"highpass=f=60,lowpass=f=8000,loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("highlow_nosampl_loudnorm", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="highlow_nosampl_loudnorm", backend="ffmpeg", duration=duration)


def _aresampl_loudnorm_fixed(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Lightweight ffmpeg step with aresample to 16kHz and loudness normalization with fixed parameters."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Build filter graph: aresample to 16kHz + loudnorm with fixed parameters
        filter_graph = "aresample=resampler=soxr:osr=16000,loudnorm=I=-23:TP=-2:LRA=11"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("aresampl_loudnorm_fixed", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="aresampl_loudnorm_fixed", backend="ffmpeg", duration=duration)


def _aresampl_loudnorm_fixed2(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Lightweight ffmpeg step with aresample to 16kHz and loudness normalization with fixed params (I=-24, LRA=15)."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Build filter graph: aresample to 16kHz + loudnorm with fixed parameters
        filter_graph = "aresample=resampler=soxr:osr=16000,loudnorm=I=-24:TP=-2:LRA=15"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("aresampl_loudnorm_fixed2", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="aresampl_loudnorm_fixed2", backend="ffmpeg", duration=duration)


def _get_variant_description(variant_name: str) -> str:
    """Get short description for a variant name.

    Args:
        variant_name: Name of the variant

    Returns:
        Short description string for use in filenames
    """
    descriptions = {
        "no_preprocessing": "nopre_projdef",
        "industry_defaults": "nopre_inddef",
        "ffmpeg_only": "ffmpegonly_projdef",
        "denoise_only": "onlyden_projdef",
        "ffmpeg_industry_defaults": "ffmpegonly_inddef",
        "full_industry_defaults": "fullpre_inddef",
        "noprep_noparamtrans": "nopre_noparam",
        "normonly_noparamtrans": "normonly_noparam",
        "norm_highp_noparamtrans": "norm_highp_noparam",
        "norm_dynaud_noparamtrans": "norm_dynaud_noparam",
        "lnorm_highlow_aform_noparamtrans": "lnorm_highlow_aform_noparam",
        "lnorm_highlow_nosampl_noparamtrans": "lnorm_highlow_nosampl_noparam",
        "lnorm2_aresampl_noparamtrans": "lnorm2_aresampl_noparam",
        "lnorm3_aresampl_noparamtrans": "lnorm3_aresampl_noparam",
        "onlyden_noparamtrans": "onlyden_noparam",
        "onlyden_noparamtrans_custom": "den2_noparam",
    }
    return descriptions.get(variant_name, variant_name)


def _copy_stage_output_with_variant(
    source: Path,
    output_dir: Path,
    variant_number: int,
    variant_description: str,
    stage_prefix: str,
    base_name: str,
    datetime_suffix: str,
) -> None:
    """Copy a preprocessing stage output with variant number, description, and datetime suffix.

    Args:
        source: Path to the stage output file
        output_dir: Directory to copy the file to
        variant_number: Variant number (1-6)
        variant_description: Short description of the variant
        stage_prefix: Prefix for the output filename (e.g., "01_ffmpeg_processed")
        base_name: Base name of the input file (without extension)
        datetime_suffix: Datetime suffix for the filename
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_file = (
            output_dir / f"{variant_number}_{variant_description}_{stage_prefix}_{base_name}_{datetime_suffix}.wav"
        )
        shutil.copy2(source, dest_file)
        LOGGER.info("Saved stage output: %s", dest_file.name)
    except Exception as exc:
        LOGGER.warning("Failed to copy stage output to %s: %s", output_dir, exc)


def preprocess_only_ffmpeg(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using only the ffmpeg pipeline (no denoise_light)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run only ffmpeg pipeline
        step_metric = run_ffmpeg_pipeline(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
            rnnoise_mix=cfg.rnnoise_mix,
            loudnorm_preset=cfg.loudnorm_preset,
            rnnoise_model=cfg.rnnoise_model,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_ffmpeg_only",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_ffmpeg_only", original_filename)

        # Skip denoise_light - just use the ffmpeg output
        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_loudnorm_only(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using only loudness normalization (lightweight ffmpeg step, no highpass, no RNNoise)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run only loudness normalization (lightweight step)
        step_metric = _loudnorm_only(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
            loudnorm_preset=cfg.loudnorm_preset,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_loudnorm_only",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_loudnorm_only", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_loudnorm_with_highpass(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using loudness normalization with highpass filter (lightweight ffmpeg step, no RNNoise)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run loudness normalization with highpass filter (lightweight step)
        step_metric = _loudnorm_with_highpass(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
            loudnorm_preset=cfg.loudnorm_preset,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_loudnorm_highpass",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_loudnorm_highpass", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_dynaudnorm_only(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using only dynamic audio normalization (lightweight ffmpeg step, no highpass, no RNNoise)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run only dynamic audio normalization (lightweight step)
        step_metric = _dynaudnorm_only(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_dynaudnorm_only",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_dynaudnorm_only", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_highlow_aform_loudnorm(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using highpass, lowpass, aformat, and loudness normalization (lightweight ffmpeg step)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run highpass, lowpass, aformat, and loudness normalization
        step_metric = _highlow_aform_loudnorm(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
            loudnorm_preset=cfg.loudnorm_preset,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_highlow_aform_loudnorm",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_highlow_aform_loudnorm", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_highlow_nosampl_loudnorm(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using highpass, lowpass, and loudness normalization (lightweight ffmpeg step, no aformat)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run highpass, lowpass, and loudness normalization (no aformat/sampling)
        step_metric = _highlow_nosampl_loudnorm(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
            loudnorm_preset=cfg.loudnorm_preset,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_highlow_nosampl_loudnorm",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_highlow_nosampl_loudnorm", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_aresampl_loudnorm_fixed(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using aresample to 16kHz and loudness normalization with fixed parameters."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run aresample to 16kHz and loudness normalization with fixed parameters
        step_metric = _aresampl_loudnorm_fixed(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_aresampl_loudnorm_fixed",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_aresampl_loudnorm_fixed", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_aresampl_loudnorm_fixed2(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using aresample to 16kHz and loudness normalization with fixed parameters (I=-24, LRA=15)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # Run aresample to 16kHz and loudness normalization with fixed parameters
        step_metric = _aresampl_loudnorm_fixed2(
            input_path=source,
            output_path=processed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_aresampl_loudnorm_fixed2",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_aresampl_loudnorm_fixed2", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_only_denoise(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> PreprocessResult:
    """Preprocess audio using only denoise_light (with simple resampling first)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    resampled_path = Path(temp_dir.name) / "resampled.wav"
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # First, resample to target format (simple resample, no loudnorm/rnnoise)
        step_metric = _simple_resample(
            input_path=source,
            output_path=resampled_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    resampled_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "00_resample",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(resampled_path, cfg.output_dir, "00_resample", original_filename)

        # Then apply denoise_light
        step_metric = apply_light_denoise(
            input_path=resampled_path,
            output_path=processed_path,
            sample_rate=cfg.target_sample_rate,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_denoise_only",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_denoise_only", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def preprocess_only_denoise_custom(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
    noise_clip_duration_s: float = 5.0,
    n_std_thresh_stationary: float = 0.75,
    prop_decrease: float = 0.25,
) -> PreprocessResult:
    """Preprocess audio using only denoise_light with custom parameters (with simple resampling first)."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
            cleanup=lambda: None,
        )

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1

    temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    resampled_path = Path(temp_dir.name) / "resampled.wav"
    processed_path = Path(temp_dir.name) / "preprocessed.wav"
    original_filename = source.stem

    step_metrics: list[StepMetrics] = []
    overall_start = time.time()

    try:
        # First, resample to target format (simple resample, no loudnorm/rnnoise)
        step_metric = _simple_resample(
            input_path=source,
            output_path=resampled_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    resampled_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "00_resample",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(resampled_path, cfg.output_dir, "00_resample", original_filename)

        # Then apply denoise_light with custom parameters
        step_metric = apply_light_denoise(
            input_path=resampled_path,
            output_path=processed_path,
            sample_rate=cfg.target_sample_rate,
            noise_clip_duration_s=noise_clip_duration_s,
            n_std_thresh_stationary=n_std_thresh_stationary,
            prop_decrease=prop_decrease,
        )
        step_metrics.append(step_metric)
        # Save with variant number and datetime if provided
        if copy_intermediate:
            if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
                _copy_stage_output_with_variant(
                    processed_path,
                    output_dir,
                    variant_number,
                    variant_description,
                    "01_denoise_only",
                    base_name,
                    datetime_suffix,
                )
            elif cfg.output_dir:
                _copy_stage_output(processed_path, cfg.output_dir, "01_denoise_only", original_filename)

        metrics = PreprocessMetrics(total_duration=time.time() - overall_start, steps=step_metrics)
    except Exception as exc:
        temp_dir.cleanup()
        raise PreprocessError(f"Preprocessing failure: {exc}") from exc

    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile="cpu",
        cleanup=temp_dir.cleanup,
    )


def get_industry_default_transcription_config() -> TranscriptionConfig:
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


def get_minimal_transcription_config() -> TranscriptionConfig:
    """Get TranscriptionConfig with minimal parameters.

    This config removes all parameters that differ between industry and project defaults.
    However, since TranscriptionConfig is a dataclass with defaults, we still get project defaults.
    The actual removal of these parameters happens in the transcribe wrapper.
    """
    # Create a minimal config - the wrapper will handle removing parameters
    return TranscriptionConfig(
        word_timestamps=False,
        task="transcribe",
    )


def transcribe_with_minimal_params(
    path: str,
    preset: str = "et-large",
    language: str | None = None,
    *,
    preprocess_config_provider: Callable[[], PreprocessConfig] = PreprocessConfig.from_env,
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult] | None = None,
    model_picker: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Transcribe with minimal parameters, omitting those that differ between defaults.

    This wrapper calls transcribe but removes specific parameters from the model.transcribe call
    to let faster-whisper use its own internal defaults.
    """
    from backend.preprocess.orchestrator import preprocess_audio

    overall_start = time.time()
    preprocess_config = preprocess_config_provider()
    preprocess_result = (preprocess_runner or preprocess_audio)(path, preprocess_config)

    model = (model_picker or pick_model)(preset)

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

    # Process segments the same way as the main transcribe function
    segment_payloads: list[dict[str, Any]] = []
    for segment in segments:
        segment_payloads.append(segment_to_payload(segment))

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


def run_variant(
    variant_name: str,
    variant_number: int,
    audio_path: str,
    preset: str = "et-large",
    language: str | None = None,
    output_dir: Path | None = None,
    output_base_path: Path | None = None,
    datetime_suffix: str | None = None,
    copy_intermediate: bool = False,
) -> dict[str, Any]:
    """Run a single transcription variant and return results."""
    # Log variant start with timestamp (major milestone)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("[%s] Running variant: %s", timestamp, variant_name)

    # Configure preprocessing and transcription based on variant
    if variant_name == "no_preprocessing":
        preprocess_config = PreprocessConfig(enabled=False)
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult] | None = None
        transcription_config_provider: Callable[[], TranscriptionConfig] | None = None
    elif variant_name == "industry_defaults":
        # No preprocessing + industry defaults for transcription
        preprocess_config = PreprocessConfig(enabled=False)
        preprocess_runner = None
        transcription_config_provider = get_industry_default_transcription_config
    elif variant_name == "ffmpeg_only":
        preprocess_config = PreprocessConfig()  # Use defaults for ffmpeg params

        # Create a wrapper that passes variant info to preprocess function
        def _ffmpeg_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_only_ffmpeg(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _ffmpeg_runner
        transcription_config_provider = None
    elif variant_name == "denoise_only":
        preprocess_config = PreprocessConfig()  # Use defaults

        # Create a wrapper that passes variant info to preprocess function
        def _denoise_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_only_denoise(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _denoise_runner
        transcription_config_provider = None
    elif variant_name == "ffmpeg_industry_defaults":
        # Only ffmpeg pipeline + industry defaults for transcription
        preprocess_config = PreprocessConfig()  # Use defaults for ffmpeg params

        # Create a wrapper that passes variant info to preprocess function
        def _ffmpeg_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_only_ffmpeg(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _ffmpeg_runner
        transcription_config_provider = get_industry_default_transcription_config
    elif variant_name == "full_industry_defaults":
        # Full preprocessing (ffmpeg + denoise_light) + industry defaults for transcription
        preprocess_config = PreprocessConfig()  # Use defaults (includes ffmpeg + denoise)

        # Create a wrapper that uses default preprocessing but saves intermediate files with variant info
        def _full_preprocess_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            # Use default preprocessing pipeline
            from backend.preprocess.orchestrator import preprocess_audio as default_preprocess

            result = default_preprocess(path, cfg)
            # After preprocessing, copy intermediate files with variant number and datetime
            if copy_intermediate and output_dir and datetime_suffix:
                # The default pipeline saves files to cfg.output_dir, but we need to rename them
                # Check if intermediate files exist and copy them with variant prefix
                if cfg.output_dir:
                    output_path = Path(cfg.output_dir)
                    # Look for intermediate files and copy with variant prefix
                    for stage_file in output_path.glob(f"*_{base_name}.wav"):
                        if stage_file.exists():
                            # Extract stage prefix (e.g., "01_ffmpeg_processed")
                            stage_prefix = stage_file.stem.replace(f"_{base_name}", "")
                            variant_file = (
                                output_dir / f"{variant_number}_{stage_prefix}_{base_name}_{datetime_suffix}.wav"
                            )
                            shutil.copy2(stage_file, variant_file)
                            LOGGER.info("Copied intermediate file: %s -> %s", stage_file, variant_file)
            return result

        preprocess_runner = _full_preprocess_runner
        transcription_config_provider = get_industry_default_transcription_config
    elif variant_name == "noprep_noparamtrans":
        # No preprocessing + minimal transcription parameters (let faster-whisper use defaults)
        preprocess_config = PreprocessConfig(enabled=False)
        preprocess_runner = None
        transcription_config_provider = None  # Use custom wrapper that omits parameters
    elif variant_name == "normonly_noparamtrans":
        # Only loudness normalization (lightweight ffmpeg step) + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults for loudnorm params

        # Create a wrapper that passes variant info to preprocess function
        def _loudnorm_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_loudnorm_only(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _loudnorm_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "norm_highp_noparamtrans":
        # Loudness normalization with highpass filter (lightweight ffmpeg step) + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults for loudnorm params

        # Create a wrapper that passes variant info to preprocess function
        def _loudnorm_highpass_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_loudnorm_with_highpass(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _loudnorm_highpass_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "norm_dynaud_noparamtrans":
        # Dynamic audio normalization (lightweight ffmpeg step) + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults

        # Create a wrapper that passes variant info to preprocess function
        def _dynaudnorm_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_dynaudnorm_only(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _dynaudnorm_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "lnorm_highlow_aform_noparamtrans":
        # Highpass, lowpass, aformat, and loudness normalization (lightweight ffmpeg step)
        # + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults for loudnorm params

        # Create a wrapper that passes variant info to preprocess function
        def _highlow_aform_loudnorm_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_highlow_aform_loudnorm(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _highlow_aform_loudnorm_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "lnorm_highlow_nosampl_noparamtrans":
        # Highpass, lowpass, and loudness normalization (lightweight ffmpeg step, no aformat)
        # + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults for loudnorm params

        # Create a wrapper that passes variant info to preprocess function
        def _highlow_nosampl_loudnorm_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_highlow_nosampl_loudnorm(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _highlow_nosampl_loudnorm_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "lnorm2_aresampl_noparamtrans":
        # Aresample to 16kHz and loudness normalization with fixed parameters + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults (loudnorm preset not used with fixed params)

        # Create a wrapper that passes variant info to preprocess function
        def _aresampl_loudnorm_fixed_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_aresampl_loudnorm_fixed(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _aresampl_loudnorm_fixed_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "lnorm3_aresampl_noparamtrans":
        # Aresample to 16kHz and loudness normalization with fixed parameters (I=-24, LRA=15)
        # + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults (loudnorm preset not used with fixed params)

        # Create a wrapper that passes variant info to preprocess function
        def _aresampl_loudnorm_fixed2_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_aresampl_loudnorm_fixed2(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _aresampl_loudnorm_fixed2_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "onlyden_noparamtrans":
        # Only denoise_light + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults

        # Create a wrapper that passes variant info to preprocess function
        def _denoise_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_only_denoise(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
            )

        preprocess_runner = _denoise_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    elif variant_name == "onlyden_noparamtrans_custom":
        # Only denoise_light with custom parameters + minimal transcription parameters
        preprocess_config = PreprocessConfig()  # Use defaults

        # Create a wrapper that passes variant info to preprocess function with custom denoise params
        def _denoise_custom_runner(path: str, cfg: PreprocessConfig) -> PreprocessResult:
            base_name = output_base_path.stem if output_base_path else Path(path).stem
            variant_desc = _get_variant_description(variant_name)
            return preprocess_only_denoise_custom(
                path,
                cfg,
                variant_number=variant_number,
                variant_description=variant_desc,
                base_name=base_name,
                datetime_suffix=datetime_suffix,
                output_dir=output_dir,
                copy_intermediate=copy_intermediate,
                noise_clip_duration_s=5.0,
                n_std_thresh_stationary=0.75,
                prop_decrease=0.25,
            )

        preprocess_runner = _denoise_custom_runner
        # Use custom transcribe wrapper that omits parameters
        transcription_config_provider = None  # Not used with custom wrapper
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    # Set output directory only if copy_intermediate is enabled
    # This prevents intermediate files from being written to the target folder by default
    if copy_intermediate and output_dir:
        preprocess_config.output_dir = str(output_dir)
    else:
        # Explicitly clear output_dir to prevent default preprocessing from writing intermediate files
        preprocess_config.output_dir = None

    # Run transcription
    start_time = time.time()
    try:
        # Check if this variant uses minimal parameters (no transcription_config_provider)
        if transcription_config_provider is None and variant_name.endswith("_noparamtrans"):
            # Use custom transcribe wrapper that omits parameters
            transcribe_kwargs: dict[str, Any] = {
                "path": audio_path,
                "preset": preset,
                "language": language,
                "preprocess_config_provider": lambda: preprocess_config,
            }
            if preprocess_runner:
                transcribe_kwargs["preprocess_runner"] = preprocess_runner

            result = transcribe_with_minimal_params(**transcribe_kwargs)
        else:
            # Use standard transcribe function
            transcribe_kwargs = {
                "path": audio_path,
                "preset": preset,
                "language": language,
                "preprocess_config_provider": lambda: preprocess_config,
            }
            if preprocess_runner:
                transcribe_kwargs["preprocess_runner"] = preprocess_runner
            if transcription_config_provider:
                transcribe_kwargs["transcription_config_provider"] = transcription_config_provider

            result = transcribe(**transcribe_kwargs)
        elapsed = time.time() - start_time

        # Save output files if output_base_path is provided
        output_files = {}
        if output_base_path and datetime_suffix:
            # Create filenames: {variant_number}_{variant_description}_{base_name}_{datetime}.{ext}
            base_name = output_base_path.stem
            variant_desc = _get_variant_description(variant_name)
            txt_path = output_base_path.parent / f"{variant_number}_{variant_desc}_{base_name}_{datetime_suffix}.txt"
            json_path = output_base_path.parent / f"{variant_number}_{variant_desc}_{base_name}_{datetime_suffix}.json"

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
                json.dump(result, json_file, ensure_ascii=False, indent=2)
            output_files["json"] = str(json_path)
            LOGGER.info("Saved JSON output: %s", json_path.name)

        return {
            "variant": variant_name,
            "variant_number": variant_number,
            "status": "success",
            "elapsed_seconds": elapsed,
            "result": result,
            "output_files": output_files,
        }
    except Exception as exc:
        elapsed = time.time() - start_time
        LOGGER.error("Variant %s failed: %s", variant_name, exc, exc_info=True)
        return {
            "variant": variant_name,
            "variant_number": variant_number,
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(exc),
            "output_files": {},
        }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare different transcription pipeline configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file to transcribe (Windows paths like C:\\Users\\... are supported)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="et-large",
        help="Model preset to use (default: et-large). All variants will use this preset.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'et', 'en'). If not specified, will auto-detect based on preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=("Directory to save outputs (optional, defaults to Alt_[datetime]/ relative to input file)"),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save comparison results as JSON (optional, defaults to {input_stem}_comparison_{datetime}.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--intermcopy",
        action="store_true",
        default=True,
        help="Copy intermediate preprocessing WAV files to output directory",
    )
    parser.add_argument(
        "--use-new-variants",
        action="store_true",
        help="Use new variant system (default: False, uses legacy system)",
    )
    parser.add_argument(
        "--skip-variants",
        type=int,
        nargs="+",
        default=[],
        help="Variant numbers to skip (e.g., --skip-variants 1 2 3)",
    )

    args = parser.parse_args()

    # Setup logging first, before any log messages
    # Default to INFO level if --verbose is not used
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Suppress third-party library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Use format without logger names (cleaner output)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s - %(message)s",
    )

    # Ensure Estonian model, GPU, and 16-bit are used for all variants
    # Set STT_DEVICE environment variable to force GPU with float16 (16-bit)
    if "STT_DEVICE" not in os.environ:
        os.environ["STT_DEVICE"] = "cuda/float16"
        LOGGER.info("Set STT_DEVICE=cuda/float16 to ensure GPU and 16-bit precision")

    # Ensure et-large preset is used (Estonian model)
    if args.preset != "et-large":
        LOGGER.warning(
            "Preset is set to '%s', but Estonian model (et-large) is recommended. Continuing with '%s'...",
            args.preset,
            args.preset,
        )
    else:
        LOGGER.info("Using Estonian model preset: et-large (GPU, 16-bit float16)")

    # Normalize Windows paths - convert Windows paths to WSL paths if needed
    audio_file_path = args.audio_file
    # Convert Windows paths (C:\Users\...) to WSL paths (/mnt/c/Users/...)
    if audio_file_path.startswith("C:\\") or audio_file_path.startswith("C:/"):
        # Replace C:\ or C:/ with /mnt/c/
        audio_file_path = audio_file_path.replace("C:\\", "/mnt/c/").replace("C:/", "/mnt/c/")
        # Replace backslashes with forward slashes
        audio_file_path = audio_file_path.replace("\\", "/")
        LOGGER.info("Converted Windows path to WSL path: %s", audio_file_path)
    elif audio_file_path.startswith(("D:\\", "D:/", "E:\\", "E:/")):
        # Handle other drives (D:, E:, etc.)
        drive_letter = audio_file_path[0].lower()
        audio_file_path = audio_file_path.replace(f"{drive_letter}:\\", f"/mnt/{drive_letter}/").replace(
            f"{drive_letter}:/", f"/mnt/{drive_letter}/"
        )
        audio_file_path = audio_file_path.replace("\\", "/")
        LOGGER.info("Converted Windows path to WSL path: %s", audio_file_path)

    audio_path = Path(audio_file_path).resolve()
    if not audio_path.exists():
        LOGGER.error("Audio file not found: %s", audio_path)
        LOGGER.error("Original path: %s", args.audio_file)
        LOGGER.error("If using Windows path, ensure WSL can access it via /mnt/c/...")
        return 1

    # Generate a single datetime suffix for all variants (they run in sequence)
    datetime_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory: Alt_[datetime]/ relative to input file location
    # If --output-dir is specified, use it; otherwise create Alt_[datetime]/ folder
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Using specified output directory: %s", output_dir)
    else:
        output_dir_name = f"Alt_{datetime_suffix}"
        output_dir = audio_path.parent / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Created output directory: %s", output_dir)

    # Generate base output path from input file name
    audio_stem = audio_path.stem
    output_base_path = output_dir / audio_stem

    # Determine comparison summary JSON path
    if args.output_json:
        comparison_json_path = Path(args.output_json).resolve()
    else:
        # Default: {input_stem}_comparison_{datetime}.json in the Alt_[datetime]/ folder
        comparison_json_path = output_dir / f"{audio_stem}_comparison_{datetime_suffix}.json"

    # Run all variants
    results: list[dict[str, Any]] = []
    summary_variants: list[tuple[str, int]] = []

    if args.use_new_variants:
        # Use new variant system
        from backend.variants import execute_variant, get_builtin_variants

        builtin_variants = get_builtin_variants()
        # Filter out skipped variants
        if args.skip_variants:
            skip_set = set(args.skip_variants)
            builtin_variants = [v for v in builtin_variants if v.number not in skip_set]
            if skip_set:
                LOGGER.info("Skipping variants: %s", sorted(skip_set))
        summary_variants = [(v.name, v.number) for v in builtin_variants]
        for variant in builtin_variants:
            result = execute_variant(
                variant=variant,
                audio_path=str(audio_path),
                preset=args.preset,
                language=args.language,
                output_dir=output_dir,
                output_base_path=output_base_path,
                datetime_suffix=datetime_suffix,
                copy_intermediate=args.intermcopy,
            )
            results.append(result)

            # Show statistics after each variant in table-like format
            LOGGER.info("")
            LOGGER.info("-" * 80)
            LOGGER.info("Variant %d: %s - Statistics", variant.number, variant.name)
            LOGGER.info("-" * 80)
            if result["status"] == "success":
                transcribe_result = result.get("result", {})
                duration = transcribe_result.get("duration")
                segments = transcribe_result.get("segments", [])
                language = transcribe_result.get("language")
                lang_prob = transcribe_result.get("language_probability")
                elapsed = result["elapsed_seconds"]

                LOGGER.info("Status:         ✅ Success")
                LOGGER.info("Processing:     %.2f seconds", elapsed)
                if duration:
                    speed_ratio = duration / elapsed if elapsed > 0 else 0
                    LOGGER.info("Audio duration: %.2f seconds (%.2f minutes)", duration, duration / 60)
                    LOGGER.info("Speed:          %.2fx realtime", speed_ratio)
                LOGGER.info("Segments:       %d", len(segments))
                if language:
                    if lang_prob is not None:
                        LOGGER.info("Language:       %s (confidence: %.1f%%)", language, lang_prob * 100)
                    else:
                        LOGGER.info("Language:       %s", language)
                if result.get("output_files"):
                    LOGGER.info("Output files:")
                    for file_type, file_path in result["output_files"].items():
                        # Show only filename, not full path
                        file_name = Path(file_path).name
                        LOGGER.info("  %-6s: %s", file_type.upper(), file_name)
            else:
                LOGGER.error("Status:         ❌ Failed")
                LOGGER.error("Processing:     %.2f seconds", result["elapsed_seconds"])
                LOGGER.error("Error:          %s", result.get("error", "Unknown error"))
    else:
        # Use legacy variant system
        variants = [
            # ("no_preprocessing", 1),  # No preprocessing + project defaults - TEMPORARILY DISABLED
            # ("industry_defaults", 2),  # No preprocessing + industry defaults - TEMPORARILY DISABLED
            # ("ffmpeg_only", 3),  # Only ffmpeg pipeline + transcription - TEMPORARILY DISABLED
            # ("denoise_only", 4),  # Only denoise_light + transcription - TEMPORARILY DISABLED
            # ("ffmpeg_industry_defaults", 5),  # Only ffmpeg pipeline + industry defaults
            # ("full_industry_defaults", 6),  # Full preprocessing (ffmpeg + denoise) + industry defaults
            ("noprep_noparamtrans", 7),  # No preprocessing + minimal transcription parameters
            # ("normonly_noparamtrans", 8),  # Only loudness normalization (lightweight) + minimal params
            # TEMPORARILY DISABLED
            # ("onlyden_noparamtrans", 9),  # Only denoise + minimal transcription parameters
            ("norm_highp_noparamtrans", 10),  # Loudness normalization + highpass + minimal params
            ("norm_dynaud_noparamtrans", 11),  # Dynamic audio normalization + minimal params
            (
                "lnorm_highlow_aform_noparamtrans",
                12,
            ),  # Highpass + lowpass + aformat + loudness normalization + minimal params
            (
                "lnorm_highlow_nosampl_noparamtrans",
                13,
            ),  # Highpass + lowpass + loudness normalization (no aformat) + minimal params
            (
                "lnorm2_aresampl_noparamtrans",
                14,
            ),  # Aresample to 16kHz + loudness normalization (fixed params) + minimal params
            (
                "lnorm3_aresampl_noparamtrans",
                15,
            ),  # Aresample to 16kHz + loudness normalization (I=-24, LRA=15) + minimal params
            # ("onlyden_noparamtrans_custom", 16),  # Variant 9a: Only denoise with custom params + minimal params
        ]
        # Filter out skipped variants
        if args.skip_variants:
            skip_set = set(args.skip_variants)
            variants = [(name, num) for name, num in variants if num not in skip_set]
            if skip_set:
                LOGGER.info("Skipping variants: %s", sorted(skip_set))
        summary_variants = variants

        for variant_name, variant_number in variants:
            result = run_variant(
                variant_name=variant_name,
                variant_number=variant_number,
                audio_path=str(audio_path),
                preset=args.preset,
                language=args.language,
                output_dir=output_dir,
                output_base_path=output_base_path,
                datetime_suffix=datetime_suffix,
                copy_intermediate=args.intermcopy,
            )
            results.append(result)

            # Show statistics after each variant in table-like format
            LOGGER.info("")
            LOGGER.info("-" * 80)
            LOGGER.info("Variant %d: %s - Statistics", variant_number, variant_name)
            LOGGER.info("-" * 80)
            if result["status"] == "success":
                transcribe_result = result.get("result", {})
                duration = transcribe_result.get("duration")
                segments = transcribe_result.get("segments", [])
                language = transcribe_result.get("language")
                lang_prob = transcribe_result.get("language_probability")
                elapsed = result["elapsed_seconds"]

                LOGGER.info("Status:         ✅ Success")
                LOGGER.info("Processing:     %.2f seconds", elapsed)
                if duration:
                    speed_ratio = duration / elapsed if elapsed > 0 else 0
                    LOGGER.info("Audio duration: %.2f seconds (%.2f minutes)", duration, duration / 60)
                    LOGGER.info("Speed:          %.2fx realtime", speed_ratio)
                LOGGER.info("Segments:       %d", len(segments))
                if language:
                    if lang_prob is not None:
                        LOGGER.info("Language:       %s (confidence: %.1f%%)", language, lang_prob * 100)
                    else:
                        LOGGER.info("Language:       %s", language)
                if result.get("output_files"):
                    LOGGER.info("Output files:")
                    for file_type, file_path in result["output_files"].items():
                        # Show only filename, not full path
                        file_name = Path(file_path).name
                        LOGGER.info("  %-6s: %s", file_type.upper(), file_name)
            else:
                LOGGER.error("Status:         ❌ Failed")
                LOGGER.error("Processing:     %.2f seconds", result["elapsed_seconds"])
                LOGGER.error("Error:          %s", result.get("error", "Unknown error"))

    # Print final summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("FINAL SUMMARY - All Variants")
    LOGGER.info("=" * 80)
    for variant_name, variant_number in summary_variants:
        # Find result for this variant
        result = next((r for r in results if r["variant_number"] == variant_number), None)
        if result:
            status_icon = "✅" if result["status"] == "success" else "❌"
            transcribe_result = result.get("result", {})
            duration = transcribe_result.get("duration")
            segments = transcribe_result.get("segments", [])
            elapsed = result["elapsed_seconds"]

            # Build summary line
            summary_parts = [
                f"{status_icon} Variant {variant_number}: {variant_name}",
                f"Status: {result['status']}",
                f"Time: {elapsed:.1f}s",
            ]
            if duration:
                speed_ratio = duration / elapsed if elapsed > 0 else 0
                summary_parts.append(f"Speed: {speed_ratio:.2f}x")
            summary_parts.append(f"Segments: {len(segments)}")

            LOGGER.info(" | ".join(summary_parts))
            if result["status"] == "error":
                LOGGER.error("   Error: %s", result.get("error", "Unknown error"))
        else:
            LOGGER.warning("⚠️  Variant %d: %s - No result found", variant_number, variant_name)

    # Always save comparison summary JSON
    with comparison_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    LOGGER.info("")
    LOGGER.info("Comparison summary saved to: %s", comparison_json_path.name)

    # Return non-zero if any variant failed
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
