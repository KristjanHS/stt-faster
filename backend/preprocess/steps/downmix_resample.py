from __future__ import annotations

import subprocess  # nosec B404 - ffmpeg invocation with fixed arguments
import time
from pathlib import Path

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics


def downmix_and_resample(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
) -> StepMetrics:
    """Downmix to mono and resample using ffmpeg."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        str(target_channels),
        "-ar",
        str(target_sample_rate),
        "-af",
        "volume=-6dB,aresample=resampler=soxr",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    start = time.time()
    try:
        subprocess.run(  # nosec B603 - fixed command list, no shell
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise StepExecutionError("downmix_resample", "ffmpeg is required but not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise StepExecutionError("downmix_resample", exc.stderr.strip() or "ffmpeg failed") from exc

    duration = time.time() - start
    return StepMetrics(name="downmix_resample", backend="ffmpeg", duration=duration)
