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
from backend.preprocess.errors import StepExecutionError
from backend.preprocess.io import inspect_audio
from backend.preprocess.metrics import PreprocessMetrics, StepMetrics
from backend.preprocess.orchestrator import PreprocessResult
from backend.variants.steps import StepRegistry

LOGGER = logging.getLogger(__name__)
console = Console()


def _execute_step_with_registry(
    step: Any,  # PreprocessStep - avoiding circular import
    current_path: Path,
    temp_dir: TemporaryDirectory[str],
    merged_config: PreprocessConfig,
    step_index: int,
    *,
    copy_intermediate: bool = False,
    variant_number: int | None = None,
    variant_description: str | None = None,
    base_name: str | None = None,
    datetime_suffix: str | None = None,
    output_dir: Path | None = None,
    original_filename: str = "",
) -> tuple[Path, StepMetrics]:
    """Execute a preprocessing step using the registry system.

    Returns:
        Tuple of (output_path, step_metrics)
    """
    # Determine output path for this step
    output_path = Path(temp_dir.name) / f"{step.step_type}_{step_index}.wav"

    # Log step execution with parameters
    LOGGER.info("Running: %s (%s)", step.name, step.step_type)
    if step.config:
        # Typed config - show dataclass fields
        config_dict = step.config.__dict__
        params_str = ", ".join(f"{k}={v}" for k, v in config_dict.items() if not k.startswith("_"))
        LOGGER.info("  Config params: %s", params_str)
    LOGGER.info(
        "  Global params: sample_rate=%d, channels=%d",
        merged_config.target_sample_rate,
        merged_config.target_channels or 1,
    )

    # Create step instance and execute
    step_instance = StepRegistry.create_step(step.step_type, step.config)
    metrics = step_instance.execute(
        input_path=current_path,
        output_path=output_path,
        global_config=merged_config,
        step_index=step_index,
    )

    # Handle intermediate file copying if requested
    if copy_intermediate:
        stage_name = f"{step_index:02d}_{step.step_type}"
        _copy_intermediate_file(
            output_path,
            stage_name,
            variant_number,
            variant_description,
            base_name,
            datetime_suffix,
            output_dir,
            original_filename,
            merged_config.output_dir,
        )

    return output_path, metrics


def loudnorm_only(
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
        # Get loudnorm params based on preset
        loudnorm_params = resolve_loudnorm_params(loudnorm_preset)

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
            **loudnorm_params,  # type: ignore[reportUnknownArgumentType]
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,  # Force mono for Whisper/faster-whisper
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("loudnorm_only", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name=f"loudnorm_only_{loudnorm_preset}", backend="ffmpeg", duration=duration)


def volume_with_limiter(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    volume_db: float,
) -> StepMetrics:
    """Apply volume adjustment with limiter."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        # Apply volume adjustment
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "volume",
            volume=f"{volume_db}dB",
        )
        # Apply limiter to prevent clipping
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "alimiter",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("volume_with_limiter", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name=f"volume_with_limiter_{volume_db}dB", backend="ffmpeg", duration=duration)


def peak_normalize_2pass(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    target_db: float,
    max_gain_db: float,
) -> StepMetrics:
    """Apply two-pass peak normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Use loudnorm in two-pass mode for peak normalization
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
            i=f"{target_db:.1f}",
            tp=f"{target_db:.1f}",
            lra=max_gain_db,
            dual_mono="true",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("peak_normalize_2pass", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name=f"peak_normalize_2pass_{target_db}dB_{max_gain_db}dB", backend="ffmpeg", duration=duration)


def loudnorm_with_highpass(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
) -> StepMetrics:
    """Apply loudness normalization with highpass filter."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        # Get loudnorm params based on preset
        loudnorm_params = resolve_loudnorm_params(loudnorm_preset)

        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        # Apply highpass filter first
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "highpass",
            f="80",
        )
        # Then apply loudnorm
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
            **loudnorm_params,  # type: ignore[reportUnknownArgumentType]
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("loudnorm_with_highpass", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name=f"loudnorm_with_highpass_{loudnorm_preset}", backend="ffmpeg", duration=duration)


def dynaudnorm_only(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Apply dynamic audio normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "dynaudnorm",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("dynaudnorm_only", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="dynaudnorm_only", backend="ffmpeg", duration=duration)


def highlow_aform_loudnorm(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Apply high-low audio format loudness normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        # Apply highpass and lowpass filtering before loudnorm
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "highpass",
            f="80",
        )
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "lowpass",
            f="8000",
        )
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("highlow_aform_loudnorm", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="highlow_aform_loudnorm", backend="ffmpeg", duration=duration)


def highlow_nosampl_loudnorm(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Apply high-low no sample loudness normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        # Apply highpass and lowpass filtering before loudnorm
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "highpass",
            f="80",
        )
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "lowpass",
            f="8000",
        )
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
            dual_mono="true",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("highlow_nosampl_loudnorm", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="highlow_nosampl_loudnorm", backend="ffmpeg", duration=duration)


def aresampl_loudnorm_fixed(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> StepMetrics:
    """Apply aresample with fixed loudness normalization parameters."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "aresample",
            ar=target_sample_rate,
        )
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
            i=f"{target_i:.1f}",
            tp=f"{target_tp:.1f}",
            lra=f"{target_lra:.1f}",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("aresampl_loudnorm_fixed", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(
        name=f"aresampl_loudnorm_fixed_{target_i}dB_{target_tp}dB_{target_lra}dB",
        backend="ffmpeg",
        duration=duration,
    )


def aresampl_loudnorm_fixed2(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> StepMetrics:
    """Apply aresample with fixed loudness normalization parameters (variant 2)."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "aresample",
            ar=target_sample_rate,
            precision="24",
        )
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
            i=f"{target_i:.1f}",
            tp=f"{target_tp:.1f}",
            lra=f"{target_lra:.1f}",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("aresampl_loudnorm_fixed2", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(
        name=f"aresampl_loudnorm_fixed2_{target_i}dB_{target_tp}dB_{target_lra}dB",
        backend="ffmpeg",
        duration=duration,
    )


def loudnorm_2pass_linear(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Apply two-pass linear loudness normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "loudnorm",
            linear="true",
            dual_mono="true",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("loudnorm_2pass_linear", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="loudnorm_2pass_linear", backend="ffmpeg", duration=duration)


def limiter_only(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Apply audio limiter only."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "alimiter",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("limiter_only", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="limiter_only", backend="ffmpeg", duration=duration)


def sox_peak_normalize(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    target_db: float,
) -> StepMetrics:
    """Apply SoX-style peak normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        # Use volume filter to normalize to target dB
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "volume",
            volume=f"{target_db:.1f}dB",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("sox_peak_normalize", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name=f"sox_peak_normalize_{target_db}dB", backend="ffmpeg", duration=duration)


def compressor_with_limiter(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Apply compression with limiter."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        # Apply compressor
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "compand",
            attacks="0.3,1",
            decays="0.8,0.3",
            points="-70/-60|-30/-15|0/-3",
            gain="-3",
            volume="-3",
        )
        # Apply limiter
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "alimiter",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("compressor_with_limiter", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="compressor_with_limiter", backend="ffmpeg", duration=duration)


def dynaudnorm_conservative(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Apply conservative dynamic audio normalization."""
    import ffmpeg  # type: ignore[import-untyped]

    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
        stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            "dynaudnorm",
            f="150",
            g="15",
            p="0.95",
        )
        stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=target_channels,
            ar=target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception as exc:
        raise StepExecutionError("dynaudnorm_conservative", f"ffmpeg error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="dynaudnorm_conservative", backend="ffmpeg", duration=duration)


def resolve_loudnorm_params(preset: str) -> dict[str, Any]:
    """Resolve loudnorm parameters based on preset."""
    presets = {
        "default": {"I": -24, "TP": -2, "LRA": 15},
        "speech": {"I": -20, "TP": -1, "LRA": 20},
        "music": {"I": -16, "TP": -1, "LRA": 11},
    }
    return presets.get(preset, presets["default"])


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
    """Create a preprocessing runner function from step definitions."""

    def preprocess_audio(input_path: str, config: PreprocessConfig) -> PreprocessResult:
        """Preprocess a single audio file."""
        start_time = time.time()
        step_metrics: list[StepMetrics] = []

        # Get audio info
        input_info = inspect_audio(Path(input_path))

        # Determine output path - use system temp dir as fallback if temp_dir is None
        import tempfile

        temp_dir_path = config.temp_dir or tempfile.gettempdir()
        output_path = Path(temp_dir_path) / f"preprocessed_{Path(input_path).stem}.wav"

        # Log preprocessing start
        LOGGER.info("Starting preprocessing: %s", Path(input_path).name)
        if steps:
            LOGGER.info("Steps: %s", [step.name for step in steps])
        else:
            LOGGER.info("No preprocessing steps configured")

        # Enable intermediate file copying if requested
        merged_config = config

        # Process steps
        current_path = Path(input_path)
        step_index = 0
        temp_dir = None  # Initialize to avoid unbound variable warning

        enabled_steps = [step for step in steps if step.enabled]

        if enabled_steps:
            temp_dir = TemporaryDirectory(prefix="stt-preprocess_", dir=merged_config.temp_dir)

            for step in enabled_steps:
                current_path, step_metric = _execute_step_with_registry(
                    step=step,
                    current_path=current_path,
                    temp_dir=temp_dir,
                    merged_config=merged_config,
                    step_index=step_index,
                    copy_intermediate=copy_intermediate,
                    variant_number=variant_number,
                    variant_description=variant_description,
                    base_name=base_name,
                    datetime_suffix=datetime_suffix,
                    output_dir=output_dir,
                    original_filename=Path(input_path).name,
                )
                step_metrics.append(step_metric)
                step_index += 1

            # Copy final result to output path
            shutil.copy2(str(current_path), str(output_path))
        else:
            # No preprocessing, just copy input to output
            shutil.copy2(str(input_path), str(output_path))

        # Calculate metrics
        total_duration = time.time() - start_time
        metrics = PreprocessMetrics(
            total_duration=total_duration,
            steps=step_metrics,
        )

        # Log completion
        LOGGER.info("Preprocessing completed in %.2fs", total_duration)
        if console.is_interactive:
            console.print(f"[green]✓[/green] Preprocessing completed in {total_duration:.2f}s")

        return PreprocessResult(
            output_path=output_path,
            input_info=input_info,
            metrics=metrics,
            profile="cpu",
            cleanup=temp_dir.cleanup if temp_dir else (lambda: None),
        )

    return preprocess_audio


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
            LOGGER.debug("Copied intermediate file to: %s", dest_file)
        except Exception as exc:
            LOGGER.warning("Failed to copy intermediate file %s: %s", source, exc)
    else:
        # Fallback: copy to config output dir if specified
        if config_output_dir:
            try:
                config_output_dir_path = Path(config_output_dir)
                config_output_dir_path.mkdir(parents=True, exist_ok=True)
                dest_file = config_output_dir_path / f"{stage_prefix}_{Path(original_filename).stem}.wav"
                shutil.copy2(source, dest_file)
                LOGGER.debug("Copied intermediate file to fallback: %s", dest_file)
            except Exception as exc:
                LOGGER.warning("Failed to copy intermediate file to fallback %s: %s", source, exc)
