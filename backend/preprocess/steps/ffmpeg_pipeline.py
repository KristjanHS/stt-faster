from __future__ import annotations

import logging
import subprocess  # nosec B404 - ffmpeg invocation with fixed arguments
import time
from pathlib import Path
from typing import Dict, Final

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

LOGGER = logging.getLogger(__name__)

# Preset configurations for loudnorm
_LOUDNORM_PRESETS: Final[Dict[str, Dict[str, float]]] = {
    "default": {
        "I": -20.0,
        "TP": -2.0,
        "LRA": 7.0,
    },
    "boost-quiet-voices": {
        "I": -18.0,
        "TP": -2.0,
        "LRA": 7.0,
    },
}


def run_ffmpeg_pipeline(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
    rnnoise_model: str | None = None,
    rnnoise_mix: float = 1.0,
) -> StepMetrics:
    """Run the ffmpeg pipeline (downmix, resample, loudnorm)."""
    preset_config = _LOUDNORM_PRESETS.get(loudnorm_preset, _LOUDNORM_PRESETS["default"])
    target_i = preset_config["I"]
    target_tp = preset_config.get("TP", -2.0)
    target_lra = preset_config.get("LRA", 7.0)

    # Construct filter chain
    # 1. highpass: remove low-frequency rumble
    # 2. arnndn: RNNoise denoiser (model path optional)
    # 3. aresample: Resample to target rate
    # 4. loudnorm: Loudness normalization
    rnnoise_filter = ""
    if rnnoise_model:
        rnnoise_filter = f"arnndn=m={rnnoise_model}:mix={rnnoise_mix},"

    filters = [
        "highpass=f=80:order=2",
        # arnndn filter is only added if a model path was provided
        rnnoise_filter.rstrip(","),
        f"aresample=resampler=soxr:osr={target_sample_rate}",
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}",
    ]

    # Remove any empty entries and join
    filter_graph = ",".join([f for f in filters if f])

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
        filter_graph,
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    LOGGER.info("Running ffmpeg pipeline: %s", " ".join(command))

    start = time.time()
    try:
        subprocess.run(  # nosec B603 - fixed command list, no shell
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise StepExecutionError("ffmpeg_pipeline", "ffmpeg is required but not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise StepExecutionError("ffmpeg_pipeline", exc.stderr.strip() or "ffmpeg failed") from exc

    duration = time.time() - start
    return StepMetrics(name="ffmpeg_pipeline", backend="ffmpeg", duration=duration)
