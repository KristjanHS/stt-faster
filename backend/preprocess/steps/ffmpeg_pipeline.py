from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import ffmpeg  # type: ignore[import-untyped, unused-ignore]
import httpx

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

LOGGER = logging.getLogger(__name__)

# Default RNNoise model URL (fallback used when RNNOISE_MODEL_URL env not set)
DEFAULT_RNNOISE_MODEL_URL = (
    "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn"
)


def _ensure_rnnoise_model(rnnoise_model: str | None) -> str | None:
    """Ensure the RNNoise model file exists locally, downloading if necessary."""
    if not rnnoise_model:
        return None

    model_path = Path(rnnoise_model)
    if model_path.exists():
        return rnnoise_model

    # Download the model if it doesn't exist
    url = os.environ.get("RNNOISE_MODEL_URL") or DEFAULT_RNNOISE_MODEL_URL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("RNNoise model not found at %s, downloading from %s", model_path, url)
    try:
        # Validate URL scheme - only allow http/https for security
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsafe URL scheme '{parsed.scheme}' in URL: {url}. Only http/https allowed.")
        # Prefer HTTPS for secure downloads
        if parsed.scheme == "http":
            LOGGER.warning("Using HTTP instead of HTTPS for URL: %s", url)

        # Use httpx with SSL verification enabled by default for secure downloads
        with httpx.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(model_path, "wb") as out:
                for chunk in resp.iter_bytes(chunk_size=16_384):
                    out.write(chunk)
    except Exception as exc:  # pragma: no cover - network I/O
        raise StepExecutionError("ffmpeg_pipeline", f"failed to download rnnoise model: {exc}") from exc

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
) -> None:
    """Process audio using ffmpeg-python library.

    The -ar parameter is set after the filter graph to ensure the final output
    is definitely at the target sample rate (16 kHz) even after loudnorm's
    internal upsampling operations.
    """
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[assignment, no-untyped-call]
        # Note: ar (sample rate) is set after af (filter) to ensure final output
        # is at target_sample_rate even if loudnorm does internal resampling
        stream = ffmpeg.output(  # type: ignore[assignment, no-untyped-call]
            stream,  # type: ignore[arg-type]
            str(output_path),
            ac=target_channels,
            af=filter_graph,  # Filter graph applied first
            ar=target_sample_rate,  # -ar 16000 applied after filters to guarantee final sample rate
            sample_fmt="s16",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[no-untyped-call]
    except ffmpeg.Error as exc:  # type: ignore[misc]
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
        )
    except StepExecutionError:
        raise
    except Exception as exc:
        raise StepExecutionError("ffmpeg_pipeline", f"unexpected error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name="ffmpeg_pipeline", backend="ffmpeg-python", duration=duration)
