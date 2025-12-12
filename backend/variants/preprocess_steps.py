"""Composable preprocessing step system for variants."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from rich.console import Console

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import PreprocessError, StepExecutionError
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.metrics import PreprocessMetrics, StepMetrics
from backend.preprocess.orchestrator import PreprocessResult
from backend.preprocess.steps.denoise_light import apply_light_denoise
from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline

LOGGER = logging.getLogger(__name__)
console = Console()


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
            ac=1,  # Force mono for Whisper/faster-whisper
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

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
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
    *,
    target_i: float | None = None,
    target_tp: float | None = None,
    target_lra: float | None = None,
) -> StepMetrics:
    """Lightweight ffmpeg step with highpass filter, resampling, and loudness normalization (no RNNoise)."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Require explicit values (no defaults)
        if target_i is None or target_tp is None or target_lra is None:
            raise ValueError(
                "loudnorm parameters (I, TP, LRA) must be provided via step config. "
                "All variants using loudnorm must specify these values explicitly."
            )

        # Build lightweight filter graph: highpass + resample + loudnorm (no RNNoise)
        filter_graph = (
            f"highpass=f=60,"
            f"aresample=resampler=soxr:osr={target_sample_rate},"
            f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        )

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
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

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
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
    *,
    target_i: float | None = None,
    target_tp: float | None = None,
    target_lra: float | None = None,
) -> StepMetrics:
    """Lightweight ffmpeg step with highpass, lowpass, aformat, and loudness normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Require explicit values (no defaults)
        if target_i is None or target_tp is None or target_lra is None:
            raise ValueError(
                "loudnorm parameters (I, TP, LRA) must be provided via step config. "
                "All variants using loudnorm must specify these values explicitly."
            )

        # Build filter graph: highpass + lowpass + aformat + loudnorm
        filter_graph = (
            f"highpass=f=60,"
            f"lowpass=f=8000,"
            f"aformat=sample_rates=16000,"
            f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        )

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
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
    *,
    target_i: float | None = None,
    target_tp: float | None = None,
    target_lra: float | None = None,
) -> StepMetrics:
    """Lightweight ffmpeg step with highpass, lowpass, and loudness normalization (no aformat/sampling)."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Require explicit values (no defaults)
        if target_i is None or target_tp is None or target_lra is None:
            raise ValueError(
                "loudnorm parameters (I, TP, LRA) must be provided via step config. "
                "All variants using loudnorm must specify these values explicitly."
            )

        # Build filter graph: highpass + lowpass + loudnorm (no aformat/sampling)
        filter_graph = f"highpass=f=60,lowpass=f=8000,loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
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
    *,
    target_i: float | None = None,
    target_tp: float | None = None,
    target_lra: float | None = None,
) -> StepMetrics:
    """Lightweight ffmpeg step with aresample to 16kHz and loudness normalization with fixed parameters."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Require explicit values (no defaults)
        if target_i is None or target_tp is None or target_lra is None:
            raise ValueError(
                "loudnorm parameters (I, TP, LRA) must be provided via step config. "
                "All variants using loudnorm must specify these values explicitly."
            )

        # Build filter graph: aresample to 16kHz + loudnorm with fixed parameters
        filter_graph = f"aresample=resampler=soxr:osr=16000,loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
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
    *,
    target_i: float | None = None,
    target_tp: float | None = None,
    target_lra: float | None = None,
) -> StepMetrics:
    """Lightweight ffmpeg step with aresample to 16kHz and loudness normalization with fixed params (I=-24, LRA=15)."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Require explicit values (no defaults)
        if target_i is None or target_tp is None or target_lra is None:
            raise ValueError(
                "loudnorm parameters (I, TP, LRA) must be provided via step config. "
                "All variants using loudnorm must specify these values explicitly."
            )

        # Build filter graph: aresample to 16kHz + loudnorm with fixed parameters
        filter_graph = f"aresample=resampler=soxr:osr=16000,loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
            af=filter_graph,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("aresampl_loudnorm_fixed2", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="aresampl_loudnorm_fixed2", backend="ffmpeg", duration=duration)


def _to_float(value: Any, default: float) -> float:
    """Best-effort float conversion with a safe default."""
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _resolve_loudnorm_params(
    target_i: float | None,
    target_tp: float | None,
    target_lra: float | None,
    loudnorm_preset: str,
) -> tuple[float, float, float]:
    """Resolve loudnorm parameters from function args or preset config.

    Args:
        target_i: Explicit I parameter, or None to use preset
        target_tp: Explicit TP parameter, or None to use preset
        target_lra: Explicit LRA parameter, or None to use preset
        loudnorm_preset: Preset name to use if parameters are None

    Returns:
        Tuple of (target_i, target_tp, target_lra) as floats.

    Raises:
        ValueError: If required parameters cannot be resolved from preset.
    """
    from backend.preprocess.config import PreprocessConfig

    if target_i is None or target_tp is None or target_lra is None:
        preset = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
        try:
            if target_i is None:
                target_i = float(preset["I"])
            if target_tp is None:
                target_tp = float(preset.get("TP", -2.0))
            if target_lra is None:
                target_lra = float(preset["LRA"])
        except KeyError as e:
            raise ValueError(f"Missing required loudnorm parameter in preset '{loudnorm_preset}': {e}") from e

    # All parameters are guaranteed to be float at this point
    # Type narrowing happens naturally - no explicit checks needed
    return target_i, target_tp, target_lra


def _loudnorm_2pass_linear(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
    *,
    target_i: float | None = None,
    target_tp: float | None = None,
    target_lra: float | None = None,
) -> StepMetrics:
    """2-pass loudnorm in linear mode: measure, then apply file-wide gain.

    First pass: measure I/LRA/TP/threshold via loudnorm=print_format=json.
    Second pass: reuse measured_* with linear=true to avoid dynamic AGC.
    """
    import json

    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()

    try:
        # Resolve parameters - returns non-optional floats
        # Type narrowing happens naturally, runtime safety via exception handling in resolver
        target_i, target_tp, target_lra = _resolve_loudnorm_params(target_i, target_tp, target_lra, loudnorm_preset)

        # ----- 1. First pass: measure loudness stats (no actual output) -----
        first_pass_filter = f"aresample=resampler=soxr:osr={target_sample_rate},loudnorm=print_format=json"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "null",
            af=first_pass_filter,
            f="null",
        )
        _, stderr = ffmpeg.run(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            overwrite_output=True,
            quiet=True,
            capture_stdout=True,
            capture_stderr=True,
        )

        # ----- 2. Extract JSON from stderr -----
        stderr_str = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr  # type: ignore[reportUnnecessaryIsInstance]
        json_start = stderr_str.rfind("{")
        if json_start == -1:
            LOGGER.warning(
                "First pass stderr (first 500 chars): %s",
                stderr_str[:500],
            )
            raise StepExecutionError(
                "loudnorm_2pass_linear",
                "Could not find JSON output in first pass stderr",
            )

        brace_count = 0
        json_end = json_start
        for i, ch in enumerate(stderr_str[json_start:], start=json_start):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if brace_count != 0:
            LOGGER.warning(
                "First pass stderr (first 500 chars): %s",
                stderr_str[:500],
            )
            raise StepExecutionError(
                "loudnorm_2pass_linear",
                "Could not find complete JSON object in first pass stderr",
            )

        measurements = json.loads(stderr_str[json_start:json_end])

        measured_i = measurements.get("input_i", target_i)
        measured_lra_raw = measurements.get("input_lra", target_lra)
        measured_tp = measurements.get("input_tp", target_tp)
        measured_thresh = measurements.get("input_thresh", -70.0)
        measured_offset = measurements.get("target_offset")

        # ----- 3. LRA handling (simplified) -----

        # 3.1 Measured LRA: from JSON, then fall back to target_lra, then 0.0
        measured_lra_float = _to_float(measured_lra_raw, _to_float(target_lra, 0.0))

        # 3.2 Special case: if loudnorm reports LRA == 0, use a safe default
        if measured_lra_float == 0.0:
            measured_lra_float = 15.0

        # 3.3 Clamp measured LRA to valid range
        measured_lra = max(11.0, min(measured_lra_float, 15.0))

        # 3.4 Target LRA for second pass:
        #     - at least measured_lra (required for linear mode)
        #     - within [measured_lra, MY MAX]
        target_lra_float = _to_float(target_lra, measured_lra)
        target_lra_clamped = max(measured_lra, min(target_lra_float, 15.0))

        # ----- 4. Second pass: apply linear loudnorm with measured stats -----
        second_pass_filter = (
            f"aresample=resampler=soxr:osr={target_sample_rate},"
            f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra_clamped}:"
            f"measured_I={measured_i}:measured_LRA={measured_lra}:"
            f"measured_TP={measured_tp}:measured_thresh={measured_thresh}:"
            "linear=true"
        )
        if measured_offset is not None:
            second_pass_filter += f":offset={measured_offset}"

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
            af=second_pass_filter,
            ar=target_sample_rate,
            sample_fmt="s16",
        )
        ffmpeg.run(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            overwrite_output=True,
            quiet=True,
            capture_stderr=True,
        )

    except ffmpeg.Error as exc:  # type: ignore[misc]
        stderr = exc.stderr.decode("utf-8") if exc.stderr else "unknown error"  # type: ignore[union-attr]
        raise StepExecutionError("loudnorm_2pass_linear", f"ffmpeg failed: {stderr}") from exc
    except json.JSONDecodeError as exc:
        raise StepExecutionError("loudnorm_2pass_linear", f"Failed to parse JSON from first pass: {exc}") from exc
    except Exception as exc:
        raise StepExecutionError("loudnorm_2pass_linear", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(
        name="loudnorm_2pass_linear",
        backend="ffmpeg",
        duration=duration,
        metadata={"lra_used": target_lra_clamped},
    )


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
            LOGGER.debug("Audio preprocessing disabled; using input as-is.")
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

                elif step.step_type == "denoise_custom":
                    # Denoise step with custom parameters
                    denoised_path = Path(temp_dir.name) / f"denoise_custom_{step_index}.wav"
                    # Extract custom parameters from step config
                    noise_clip_duration_s = step.config.get("noise_clip_duration_s", 5.0) if step.config else 5.0
                    n_std_thresh_stationary = step.config.get("n_std_thresh_stationary", 0.75) if step.config else 0.75
                    prop_decrease = step.config.get("prop_decrease", 0.25) if step.config else 0.25
                    step_metric = apply_light_denoise(
                        input_path=current_path,
                        output_path=denoised_path,
                        sample_rate=merged_config.target_sample_rate,
                        noise_clip_duration_s=noise_clip_duration_s,
                        n_std_thresh_stationary=n_std_thresh_stationary,
                        prop_decrease=prop_decrease,
                    )
                    step_metrics.append(step_metric)
                    current_path = denoised_path

                    if copy_intermediate:
                        stage_name = "01_denoise_custom" if step_index == 0 else "02_denoise_custom"
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

                elif step.step_type == "loudnorm_only":
                    # Lightweight loudnorm step (resample + loudnorm only)
                    intermediate_path = Path(temp_dir.name) / f"loudnorm_only_{step_index}.wav"
                    step_metric = _loudnorm_only(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        loudnorm_preset=merged_config.loudnorm_preset,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_loudnorm_only",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "loudnorm_highpass":
                    # Lightweight loudnorm step with highpass filter
                    intermediate_path = Path(temp_dir.name) / f"loudnorm_highpass_{step_index}.wav"
                    # Get custom loudnorm values from step config if provided
                    custom_i = step.config.get("I") if step.config else None
                    custom_tp = step.config.get("TP") if step.config else None
                    custom_lra = step.config.get("LRA") if step.config else None
                    step_metric = _loudnorm_with_highpass(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        loudnorm_preset=merged_config.loudnorm_preset,
                        target_i=custom_i,
                        target_tp=custom_tp,
                        target_lra=custom_lra,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_loudnorm_highpass",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "dynaudnorm":
                    # Dynamic audio normalization step
                    intermediate_path = Path(temp_dir.name) / f"dynaudnorm_{step_index}.wav"
                    step_metric = _dynaudnorm_only(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_dynaudnorm",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "highlow_aform_loudnorm":
                    # Highpass + lowpass + aformat + loudnorm step
                    intermediate_path = Path(temp_dir.name) / f"highlow_aform_loudnorm_{step_index}.wav"
                    # Get custom loudnorm values from step config if provided
                    custom_i = step.config.get("I") if step.config else None
                    custom_tp = step.config.get("TP") if step.config else None
                    custom_lra = step.config.get("LRA") if step.config else None
                    step_metric = _highlow_aform_loudnorm(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        loudnorm_preset=merged_config.loudnorm_preset,
                        target_i=custom_i,
                        target_tp=custom_tp,
                        target_lra=custom_lra,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_highlow_aform_loudnorm",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "highlow_nosampl_loudnorm":
                    # Highpass + lowpass + loudnorm (no aformat) step
                    intermediate_path = Path(temp_dir.name) / f"highlow_nosampl_loudnorm_{step_index}.wav"
                    # Get custom loudnorm values from step config if provided
                    custom_i = step.config.get("I") if step.config else None
                    custom_tp = step.config.get("TP") if step.config else None
                    custom_lra = step.config.get("LRA") if step.config else None
                    step_metric = _highlow_nosampl_loudnorm(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        loudnorm_preset=merged_config.loudnorm_preset,
                        target_i=custom_i,
                        target_tp=custom_tp,
                        target_lra=custom_lra,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_highlow_nosampl_loudnorm",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "aresampl_loudnorm_fixed":
                    # Aresample to 16kHz + loudnorm with fixed parameters step
                    intermediate_path = Path(temp_dir.name) / f"aresampl_loudnorm_fixed_{step_index}.wav"
                    # Get custom loudnorm values from step config if provided
                    custom_i = step.config.get("I") if step.config else None
                    custom_tp = step.config.get("TP") if step.config else None
                    custom_lra = step.config.get("LRA") if step.config else None
                    step_metric = _aresampl_loudnorm_fixed(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        target_i=custom_i,
                        target_tp=custom_tp,
                        target_lra=custom_lra,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_aresampl_loudnorm_fixed",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "aresampl_loudnorm_fixed2":
                    # Aresample to 16kHz + loudnorm with fixed params (I=-24, LRA=15) step
                    intermediate_path = Path(temp_dir.name) / f"aresampl_loudnorm_fixed2_{step_index}.wav"
                    # Get custom loudnorm values from step config if provided
                    custom_i = step.config.get("I") if step.config else None
                    custom_tp = step.config.get("TP") if step.config else None
                    custom_lra = step.config.get("LRA") if step.config else None
                    step_metric = _aresampl_loudnorm_fixed2(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        target_i=custom_i,
                        target_tp=custom_tp,
                        target_lra=custom_lra,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_aresampl_loudnorm_fixed2",
                            variant_number,
                            variant_description,
                            base_name,
                            datetime_suffix,
                            output_dir,
                            original_filename,
                            merged_config.output_dir,
                        )

                elif step.step_type == "loudnorm_2pass_linear":
                    # 2-pass loudnorm in linear mode step
                    intermediate_path = Path(temp_dir.name) / f"loudnorm_2pass_linear_{step_index}.wav"
                    # Get custom loudnorm values from step config if provided
                    custom_i = step.config.get("I") if step.config else None
                    custom_tp = step.config.get("TP") if step.config else None
                    custom_lra = step.config.get("LRA") if step.config else None
                    step_metric = _loudnorm_2pass_linear(
                        input_path=current_path,
                        output_path=intermediate_path,
                        target_sample_rate=merged_config.target_sample_rate,
                        target_channels=resolved_channels,
                        loudnorm_preset=merged_config.loudnorm_preset,
                        target_i=custom_i,
                        target_tp=custom_tp,
                        target_lra=custom_lra,
                    )
                    step_metrics.append(step_metric)
                    current_path = intermediate_path

                    if copy_intermediate:
                        _copy_intermediate_file(
                            intermediate_path,
                            "01_loudnorm_2pass_linear",
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

        # Display preprocessing summary with Rich
        step_details: list[str] = []
        for metric in metrics.steps:
            lra_info = ""
            if metric.metadata and "lra_used" in metric.metadata:
                lra_info = f" LRA={metric.metadata['lra_used']:.1f}"
            step_details.append(f"  └─ {metric.name} ({metric.backend}): {metric.duration:.2f}s{lra_info}")
            LOGGER.debug("Preprocessing step %s (%s): %.2fs%s", metric.name, metric.backend, metric.duration, lra_info)

        console.print(f"[cyan]⏳ Pre-processing completed in[/cyan] {metrics.total_duration:.2f}s")
        for detail in step_details:
            console.print(detail)
        LOGGER.debug("Pre-processing completed in %.2fs", metrics.total_duration)

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
