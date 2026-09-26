from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import ffmpeg  # type: ignore[import-untyped, unused-ignore]

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

LOGGER = logging.getLogger(__name__)

RNNOISE_NOT_INSTALLED = (
    "RNNoise model not installed at {path} — run `make rnnoise-model` (dev) or run setup again to repair (Windows)"
)


def _ensure_rnnoise_model(rnnoise_model: str | None) -> str | None:
    """The model is prefetched at install time; transcription never downloads it."""
    if not rnnoise_model:
        return None
    if not Path(rnnoise_model).is_file():
        raise StepExecutionError("ffmpeg_pipeline", RNNOISE_NOT_INSTALLED.format(path=rnnoise_model))
    return rnnoise_model


def _build_filter_graph(
    target_sample_rate: int,
    target_i: float,
    target_tp: float,
    target_lra: float,
    rnnoise_mix: float,
    rnnoise_model: str | None = None,
) -> str:
    """Build the ffmpeg filter graph string.

    Filter order: highpass -> resample -> RNNoise (if enabled) -> loudnorm
    This implements the 3-phase preprocessing:
    1. FFmpeg decode + resample (first)
    2. Strong denoising (RNNoise) before loudnorm
    3. Loudnorm (last)
    """
    filters = [
        "highpass=f=80:poles=2",
        f"aresample=resampler=soxr:osr={target_sample_rate}",
    ]

    if rnnoise_model:
        filters.append(f"arnndn=m={rnnoise_model}:mix={rnnoise_mix}")

    filters.append(f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}")

    return ",".join(filters)


def _process_with_ffmpeg_python(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    filter_graph: str,
    *,
    ffmpeg_module: Any = ffmpeg,
) -> None:
    """Process audio using ffmpeg-python library.

    The -ar parameter is set after the filter graph to ensure the final output
    is definitely at the target sample rate (16 kHz) even after loudnorm's
    internal upsampling operations.
    """
    try:
        stream = ffmpeg_module.input(str(input_path))  # type: ignore[assignment, no-untyped-call]
        # Note: ar (sample rate) is set after af (filter) to ensure final output
        # is at target_sample_rate even if loudnorm does internal resampling
        stream = ffmpeg_module.output(  # type: ignore[assignment, no-untyped-call]
            stream,  # type: ignore[arg-type]
            str(output_path),
            ac=target_channels,
            af=filter_graph,  # Filter graph applied first
            ar=target_sample_rate,  # -ar 16000 applied after filters to guarantee final sample rate
            sample_fmt="s16",
        )
        ffmpeg_module.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[no-untyped-call]
    except ffmpeg_module.Error as exc:  # type: ignore[misc]
        stderr = exc.stderr.decode() if exc.stderr else "unknown error"  # type: ignore[union-attr]
        raise StepExecutionError("ffmpeg_pipeline", f"ffmpeg failed: {stderr}") from exc
    except Exception as exc:
        raise StepExecutionError("ffmpeg_pipeline", f"ffmpeg error: {exc}") from exc


def run_ffmpeg_pipeline(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    rnnoise_mix: float,
    loudnorm_preset: str = "default",
    rnnoise_model: str | None = None,
    run_cmd: Callable[..., object] | None = None,  # Deprecated, kept for compatibility
    *,
    ffmpeg_module: Any = ffmpeg,
) -> StepMetrics:
    """Run the ffmpeg pipeline (decode/resample -> RNNoise -> loudnorm).

    Implements 3-phase preprocessing:
    1. FFmpeg decode + resample (first)
    2. Strong denoising (RNNoise) before loudnorm
    3. Loudnorm (last)

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        target_sample_rate: Target sample rate in Hz
        target_channels: Target number of channels
        loudnorm_preset: Loudness normalization preset name
        rnnoise_model: Optional path to RNNoise model file
        rnnoise_mix: RNNoise mix level (0.0 to 1.0)
        run_cmd: Deprecated parameter, kept for backward compatibility
        ffmpeg_module: The ffmpeg-python module (injectable for tests)

    Returns:
        StepMetrics with processing duration
    """
    if run_cmd is not None:
        LOGGER.warning("run_cmd parameter is deprecated; using ffmpeg-python library")

    # Resolve loudnorm parameters from preset
    target_i, target_tp, target_lra = PreprocessConfig.resolve_loudnorm_params(None, None, None, loudnorm_preset)

    # Ensure RNNoise model exists if provided
    resolved_rnnoise_model = _ensure_rnnoise_model(rnnoise_model)

    # Build filter graph
    filter_graph = _build_filter_graph(
        target_sample_rate=target_sample_rate,
        target_i=target_i,
        target_tp=target_tp,
        target_lra=target_lra,
        rnnoise_mix=rnnoise_mix,
        rnnoise_model=resolved_rnnoise_model,
    )

    LOGGER.info("Running ffmpeg pipeline with filter: %s", filter_graph)

    start = time.time()
    try:
        _process_with_ffmpeg_python(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            filter_graph=filter_graph,
            ffmpeg_module=ffmpeg_module,
        )
    except StepExecutionError:
        raise
    except Exception as exc:
        raise StepExecutionError("ffmpeg_pipeline", f"unexpected error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="ffmpeg_pipeline", backend="ffmpeg-python", duration=duration)
