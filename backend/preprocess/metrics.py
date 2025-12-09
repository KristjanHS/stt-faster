from __future__ import annotations

import logging
import math
import subprocess  # nosec B404 - ffmpeg invocation with fixed arguments
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
MIN_RMS = 1e-9


@dataclass(slots=True)
class StepMetrics:
    name: str
    backend: str
    duration: float
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class PreprocessMetrics:
    total_duration: float
    steps: list[StepMetrics] = field(
        default_factory=lambda: cast(list[StepMetrics], []),
    )
    snr_before: float | None = None
    snr_after: float | None = None

    @property
    def snr_delta(self) -> float | None:
        if self.snr_before is None or self.snr_after is None:
            return None
        return self.snr_after - self.snr_before


def estimate_snr_db(path: Path, sample_rate: int, channels: int) -> float | None:
    """Estimate SNR in dB using a simple noise floor percentile."""
    samples = _decode_audio(path, sample_rate, channels)
    if samples is None or samples.size == 0:
        return None

    frame_size = min(4096, max(512, len(samples)))
    frame_size = min(frame_size, len(samples))
    frame_count = len(samples) // frame_size
    if frame_count == 0:
        return None

    trimmed: NDArray[np.float64] = (
        samples[: frame_count * frame_size].reshape(frame_count, frame_size).astype(np.float64, copy=False)
    )
    rms: NDArray[np.float64] = np.sqrt(np.mean(trimmed**2, axis=1, dtype=np.float64))
    noise_rms = float(np.percentile(rms, 10))
    signal_rms = float(np.sqrt(np.mean(samples**2)))

    if signal_rms <= MIN_RMS:
        LOGGER.warning("SNR estimation failed: near-silent signal (rms=%.3e)", signal_rms)
        return None

    if noise_rms <= MIN_RMS:
        LOGGER.info("Noise floor too low (%.3e); clamping to minimum RMS for SNR estimation", noise_rms)
        noise_rms = MIN_RMS

    return 20.0 * math.log10(signal_rms / noise_rms)


def _decode_audio(path: Path, sample_rate: int, channels: int) -> NDArray[np.float32] | None:
    """Decode audio to mono float32 samples via ffmpeg."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-map",
        "0:a:0",
        "-",
    ]

    try:
        proc = subprocess.run(  # nosec B603 - fixed command list, no shell
            command,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        LOGGER.warning("ffmpeg is required for SNR estimation but not installed: %s", exc)
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="ignore").strip()
        LOGGER.warning("ffmpeg failed to decode audio for SNR estimation: %s", stderr or exc)
        return None

    if not proc.stdout:
        return None

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels > 1:
        frame_count = samples.size // channels
        if frame_count == 0:
            return None
        samples = samples[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)

    return samples.astype(np.float32, copy=False)
