from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import noisereduce as nr  # type: ignore[import-untyped]

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

_STEP_NAME = "denoise_light"


def apply_light_denoise(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    *,
    noise_reducer: Callable[..., np.ndarray] | None = None,
    writer: Callable[..., object] | None = None,
) -> StepMetrics:
    """Apply a light denoise using noisereduce.spectral_gate."""
    start = time.time()
    reader = cast(Callable[..., tuple[np.ndarray, int]], sf.read)  # pyright: ignore[reportUnknownMemberType]
    try:
        samples, sr = reader(input_path, always_2d=False)
    except (OSError, ValueError) as exc:
        raise StepExecutionError(_STEP_NAME, f"failed to read audio: {exc}") from exc

    if sr != sample_rate:
        raise StepExecutionError(_STEP_NAME, f"unexpected sample rate {sr}; expected {sample_rate}")

    mono = np.asarray(samples, dtype=np.float32)
    if mono.size == 0:
        raise StepExecutionError(_STEP_NAME, "input audio is empty")

    if mono.ndim > 1:
        mono = mono.mean(axis=1)

    default_reducer = cast(Callable[..., np.ndarray], nr.reduce_noise)  # pyright: ignore[reportUnknownMemberType]
    reducer = noise_reducer or default_reducer
    try:
        denoised = reducer(
            y=mono,
            sr=sample_rate,
            stationary=True,
            use_tqdm=False,
        )
    except Exception as exc:  # noqa: BLE001 - propagate wrapped failure
        raise StepExecutionError(_STEP_NAME, f"noisereduce failed: {exc}") from exc

    default_writer = cast(Callable[..., object], sf.write)  # pyright: ignore[reportUnknownMemberType]
    write_audio = writer or default_writer
    try:
        write_audio(output_path, denoised, sample_rate, subtype="PCM_16")
    except (OSError, ValueError) as exc:
        raise StepExecutionError(_STEP_NAME, f"failed to write audio: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name=_STEP_NAME, backend="noisereduce", duration=duration)
