from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Final, Tuple, cast

import numpy as np
import pyloudnorm as pyln  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

LOGGER = logging.getLogger(__name__)

# Preset configurations for loudnorm
_LOUDNORM_PRESETS: Final[Dict[str, Dict[str, float]]] = {
    "default": {
        "I": -20.0,
        "TP": -2.0,
    },
    "boost-quiet-voices": {
        "I": -18.0,
        "TP": -2.0,
    },
}


def apply_loudnorm(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    *,
    preset: str = "default",
    measure: Callable[[Path], Dict[str, float]] | None = None,
    normalize: Callable[[Path, Path, Dict[str, float] | None, int], float] | None = None,
) -> StepMetrics:
    """Normalize loudness using pyloudnorm. Falls back to ffmpeg is NOT supported."""
    preset_config = _LOUDNORM_PRESETS.get(preset, _LOUDNORM_PRESETS["default"])
    target_loudness = preset_config["I"]
    target_peak = preset_config.get("TP", -2.0)

    duration = _run_pyloudnorm(
        input_path, output_path, sample_rate, target_loudness=target_loudness, target_peak=target_peak
    )
    return StepMetrics(name="loudnorm", backend="pyloudnorm", duration=duration)


def _run_pyloudnorm(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    *,
    target_loudness: float = -20.0,
    target_peak: float | None = None,
) -> float:
    start = time.time()
    samples, detected_sr = cast(
        Tuple[np.ndarray, int],
        sf.read(input_path, always_2d=False),  # pyright: ignore[reportUnknownMemberType]
    )

    if detected_sr != sample_rate:
        raise StepExecutionError("loudnorm", f"unexpected sample rate {detected_sr}; expected {sample_rate}")

    waveform = np.asarray(samples, dtype=np.float32)

    try:
        meter = _create_meter(sample_rate)
        loudness = meter.integrated_loudness(waveform)  # pyright: ignore[reportUnknownMemberType]
        normalized = pyln.normalize.loudness(waveform, loudness, target_loudness)  # pyright: ignore[reportUnknownMemberType]

        # Apply True Peak limiting if needed
        # pyloudnorm's peak normalization simply scales the audio so the peak is at target_peak
        # Standard EBU R128 compliance usually implies adhering to the TP limit.
        if target_peak is not None:
            normalized = pyln.normalize.peak(normalized, target_peak)  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:  # noqa: BLE001 - propagate wrapped failure
        raise StepExecutionError("loudnorm", f"pyloudnorm processing failed: {exc}") from exc

    writer = cast(Callable[..., None], sf.write)  # pyright: ignore[reportUnknownMemberType]
    try:
        writer(output_path, normalized, sample_rate, subtype="PCM_16")  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:  # noqa: BLE001
        raise StepExecutionError("loudnorm", f"failed to write audio: {exc}") from exc

    return time.time() - start


def _create_meter(sample_rate: int) -> Any:
    """Instantiate pyloudnorm Meter while tolerating removed keyword args."""
    meter_kwargs = {"block_size": 0.400}
    try:
        return pyln.Meter(sample_rate, filter_true=True, **meter_kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]
    except TypeError as exc:  # pragma: no cover - fallback depends on pyloudnorm version
        if "filter_true" not in str(exc):
            raise
        return pyln.Meter(sample_rate, **meter_kwargs)  # pyright: ignore[reportArgumentType]
