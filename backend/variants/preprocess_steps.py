"""Composable preprocessing step system for variants."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import PreprocessError, StepExecutionError
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.metrics import PreprocessMetrics, StepMetrics
from backend.preprocess.orchestrator import PreprocessResult
from backend.preprocess.steps.denoise_light import apply_light_denoise
from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline

LOGGER = logging.getLogger(__name__)


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
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
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


def create_preprocess_runner(
    steps: list[Any],  # list[PreprocessStep] but avoiding circular import
    base_config: PreprocessConfig,
    *,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    copy_intermediate: bool = False,
) -> Callable[[str, PreprocessConfig], PreprocessResult]:
    """Create a preprocessing runner function from step definitions.

    Args:
        steps: List of PreprocessStep objects defining the pipeline
        base_config: Base PreprocessConfig to use
        variant_number: Optional variant number for intermediate file naming
        variant_description: Optional variant description for intermediate file naming
        base_name: Optional base name for intermediate file naming
        datetime_suffix: Optional datetime suffix for intermediate file naming
        output_dir: Optional output directory for intermediate files
        copy_intermediate: Whether to copy intermediate files to output_dir

    Returns:
        A function that takes (path: str, config: PreprocessConfig) -> PreprocessResult
    """
    from backend.variants.variant import PreprocessStep  # noqa: PLC0415

    # Validate steps are PreprocessStep instances
    validated_steps = [s for s in steps if isinstance(s, PreprocessStep)]

    def _runner(path: str, config: PreprocessConfig) -> PreprocessResult:
        """Run preprocessing pipeline based on step definitions."""
        source = Path(path)
        input_info: AudioInfo | None = None

        try:
            input_info = inspect_audio(source)
        except PreprocessError as exc:
            LOGGER.warning("Skipping metadata inspection: %s", exc)

        # Merge base_config with provided config
        merged_config = PreprocessConfig(
            enabled=config.enabled,
            target_sample_rate=config.target_sample_rate or base_config.target_sample_rate,
            target_channels=config.target_channels or base_config.target_channels,
            temp_dir=config.temp_dir or base_config.temp_dir,
            output_dir=config.output_dir or base_config.output_dir,
            profile=config.profile or base_config.profile,
            loudnorm_preset=config.loudnorm_preset or base_config.loudnorm_preset,
            rnnoise_model=config.rnnoise_model or base_config.rnnoise_model,
            rnnoise_mix=config.rnnoise_mix
            if config.rnnoise_mix != base_config.rnnoise_mix
            else base_config.rnnoise_mix,
        )

        if not merged_config.enabled:
            LOGGER.info("Audio preprocessing disabled; using input as-is.")
            return PreprocessResult(
                output_path=source,
                input_info=input_info,
                metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
                profile="disabled",
                cleanup=lambda: None,
            )

        input_channels = input_info.channels if input_info else None
        resolved_channels = merged_config.target_channels or input_channels or 1

        temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=merged_config.temp_dir)
        processed_path = Path(temp_dir.name) / "preprocessed.wav"
        original_filename = source.stem

        step_metrics: list[StepMetrics] = []
        overall_start = time.time()

        try:
            current_path = source
            step_index = 0

            # Filter to only enabled steps
            enabled_steps = [s for s in validated_steps if s.enabled]

            for step in enabled_steps:
                if step.step_type == "resample":
                    # Simple resample step (used before denoise when ffmpeg is not used)
                    resampled_path = Path(temp_dir.name) / f"resampled_{step_index}.wav"
                    step_metric = _simple_resample(
                        input_path=current_path,
                        output_path=resampled_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                    )
                    step_metrics.append(step_metric)
                    current_path = resampled_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            resampled_path,
                            "00_resample",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "ffmpeg":
                    # FFmpeg pipeline step (includes resample, rnnoise, loudnorm)
                    intermediate_path = Path(temp_dir.name) / f"ffmpeg_{step_index}.wav"
                    step_metric = run_ffmpeg_pipeline(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        rnnoise_mix=merged_config.rnnoise_mix,
                        loudnorm_preset=merged_config.loudnorm_preset,
                        rnnoise_model=merged_config.rnnoise_model,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_ffmpeg_only"
                            if "denoise" not in [s.step_type for s in enabled_steps[step_index + 1 :]]
                            else "01_ffmpeg_processed",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "denoise":
                    # Denoise step (light denoising)
                    denoised_path = Path(temp_dir.name) / f"denoise_{step_index}.wav"
                    step_metric = apply_light_denoise(
                        input_path=current_path,
                        output_path=denoised_path,
                        sample_rate=merged_config.target_sample_rate,
                    )
                    step_metrics.append(step_metric)
                    current_path = denoised_path

                    if copy_intermediate:
                        stage_name = "01_denoise_only" if step_index == 0 else "02_denoise"
                        _copy_intermediate_file(
                            denoised_path,
                            stage_name,
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                step_index += 1

            # Final output is the last processed file
            if current_path != processed_path:
                shutil.copy2(current_path, processed_path)

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

    return _runner


def _copy_intermediate_file(
    source: Path,
    stage_prefix: str,
    variant_number: int | None,
    variant_description: str | None,
    base_name: str | None,
    datetime_suffix: str | None,
    output_dir: Path | None,
    original_filename: str,
    config_output_dir: str | None,
) -> None:
    """Copy intermediate file with variant naming if available, otherwise use standard naming."""
    import shutil  # noqa: PLC0415

    if variant_number is not None and variant_description and base_name and datetime_suffix and output_dir:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            dest_file = (
                output_dir / f"{variant_number}_{variant_description}_{stage_prefix}_{base_name}_{datetime_suffix}.wav"
            )
            shutil.copy2(source, dest_file)
            LOGGER.info("Saved stage output: %s", dest_file.name)
        except Exception as exc:
            LOGGER.warning("Failed to copy stage output to %s: %s", output_dir, exc)
    elif config_output_dir:
        try:
            output_path = Path(config_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            dest_file = output_path / f"{stage_prefix}_{original_filename}.wav"
            shutil.copy2(source, dest_file)
            LOGGER.info("Saved stage output: %s", dest_file)
        except Exception as exc:
            LOGGER.warning("Failed to copy stage output to %s: %s", config_output_dir, exc)
