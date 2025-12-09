from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import PreprocessError, StepExecutionError
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.metrics import PreprocessMetrics, StepMetrics, estimate_snr_db
from backend.preprocess.steps.denoise_light import apply_light_denoise
from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessResult:
    output_path: Path
    input_info: AudioInfo | None
    metrics: PreprocessMetrics
    profile: str
    cleanup: Callable[[], None] = field(default=lambda: None)


def _copy_stage_output(
    source: Path,
    output_dir: str | None,
    stage_prefix: str,
    original_filename: str,
) -> None:
    """Copy a preprocessing stage output to the configured output directory.

    Args:
        source: Path to the stage output file
        output_dir: Directory to copy the file to (None to skip)
        stage_prefix: Prefix for the output filename (e.g., "01_downmix")
        original_filename: Original input filename (without extension)
    """
    if not output_dir:
        return

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dest_file = output_path / f"{stage_prefix}_{original_filename}.wav"
        shutil.copy2(source, dest_file)
        LOGGER.info("Saved stage output: %s", dest_file)
    except Exception as exc:
        LOGGER.warning("Failed to copy stage output to %s: %s", output_dir, exc)


def preprocess_audio(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    inspector: Callable[[Path], AudioInfo] = inspect_audio,
    ffmpeg_fn: Callable[..., StepMetrics] = run_ffmpeg_pipeline,
    downmix_fn: Callable[..., StepMetrics] | None = None,
    loudnorm_fn: Callable[..., StepMetrics] | None = None,
    denoise_fn: Callable[..., StepMetrics] = apply_light_denoise,
    snr_estimator: Callable[[Path, int, int], float | None] = estimate_snr_db,
    temp_dir_factory: Callable[..., TemporaryDirectory[str]] = TemporaryDirectory,
) -> PreprocessResult:
    """Run the light audio pre-processing pipeline."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspector(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
        )

    profile = "cpu"

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1
    LOGGER.info(
        "Audio preprocessing enabled (profile=%s, target=%dkHz, channels=%s -> using %d)",
        profile,
        cfg.target_sample_rate,
        str(cfg.target_channels or "auto"),
        resolved_channels,
    )

    temp_dir: TemporaryDirectory[str] = temp_dir_factory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"

    # Extract original filename (without extension) for stage output naming
    original_filename = source.stem

    step_metrics: List[StepMetrics] = []
    overall_start = time.time()
    try:
        metrics = _run_pipeline(
            source=source,
            destination=processed_path,
            cfg=cfg,
            step_metrics=step_metrics,
            ffmpeg_fn=ffmpeg_fn,
            downmix_fn=downmix_fn,
            loudnorm_fn=loudnorm_fn,
            denoise_fn=denoise_fn,
            resolved_channels=resolved_channels,
            original_filename=original_filename,
            wrap_errors=True,
        )
    except PreprocessError:
        temp_dir.cleanup()
        raise
    except Exception as exc:  # pragma: no cover - defensive
        temp_dir.cleanup()
        raise PreprocessError(f"Unexpected preprocessing failure: {exc}") from exc

    metrics.total_duration = time.time() - overall_start
    metrics.snr_before = snr_estimator(source, cfg.target_sample_rate, resolved_channels)
    metrics.snr_after = snr_estimator(processed_path, cfg.target_sample_rate, resolved_channels)
    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    if metrics.snr_delta is not None:
        LOGGER.info(
            "SNR delta: %.2f dB -> %.2f dB (Δ %.2f dB)", metrics.snr_before, metrics.snr_after, metrics.snr_delta
        )
    else:
        LOGGER.info("SNR metrics unavailable; see warnings above for decoding issues.")
    for metric in metrics.steps:
        lra_info = ""
        if metric.metadata and "lra_used" in metric.metadata:
            lra_info = f" LRA={metric.metadata['lra_used']:.1f}"
        LOGGER.info(" - Step %s (%s): %.2fs%s", metric.name, metric.backend, metric.duration, lra_info)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile=profile,
        cleanup=temp_dir.cleanup,
    )


def _run_pipeline(
    source: Path,
    destination: Path,
    cfg: PreprocessConfig,
    step_metrics: List[StepMetrics],
    *,
    resolved_channels: int,
    ffmpeg_fn: Callable[..., StepMetrics],
    downmix_fn: Callable[..., StepMetrics] | None,
    loudnorm_fn: Callable[..., StepMetrics] | None,
    denoise_fn: Callable[..., StepMetrics],
    original_filename: str,
    wrap_errors: bool = True,
) -> PreprocessMetrics:
    # Option A: separate steps (downmix -> loudnorm -> denoise)
    try:
        if downmix_fn is not None and loudnorm_fn is not None:
            downmix_path = destination.parent / "downmixed.wav"
            step_metric = downmix_fn(
                input_path=source,
                output_path=downmix_path,
                sample_rate=cfg.target_sample_rate,
                channels=resolved_channels,
            )
            step_metrics.append(step_metric)
            _copy_stage_output(downmix_path, cfg.output_dir, "01_downmix", original_filename)

            loudnorm_path = destination.parent / "loudnorm.wav"
            step_metric = loudnorm_fn(
                input_path=downmix_path, output_path=loudnorm_path, sample_rate=cfg.target_sample_rate
            )
            step_metrics.append(step_metric)
            _copy_stage_output(loudnorm_path, cfg.output_dir, "02_loudnorm", original_filename)

            intermediate_path = loudnorm_path
        else:
            # Fallback: combined ffmpeg pipeline
            intermediate_path = destination.parent / "intermediate.wav"
            step_metric = ffmpeg_fn(
                input_path=source,
                output_path=intermediate_path,
                target_sample_rate=cfg.target_sample_rate,
                target_channels=resolved_channels,
                rnnoise_mix=cfg.rnnoise_mix,
                loudnorm_preset=cfg.loudnorm_preset,
                rnnoise_model=cfg.rnnoise_model,
            )
            step_metrics.append(step_metric)
            _copy_stage_output(intermediate_path, cfg.output_dir, "01_ffmpeg_processed", original_filename)
    except StepExecutionError as exc:
        if wrap_errors:
            raise PreprocessError(str(exc)) from exc
        raise

    try:
        step_metric = denoise_fn(
            input_path=intermediate_path,
            output_path=destination,
            sample_rate=cfg.target_sample_rate,
        )
        step_metrics.append(step_metric)
        _copy_stage_output(destination, cfg.output_dir, "03_denoise", original_filename)
    except StepExecutionError as exc:
        if wrap_errors:
            raise PreprocessError(str(exc)) from exc
        raise

    return PreprocessMetrics(total_duration=0.0, steps=step_metrics)
