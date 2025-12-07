from __future__ import annotations

# pyright: reportUnknownVariableType=false
import re
import time
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import noisereduce as nr  # type: ignore[import-untyped]

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

_SPECTRAL_GATE_N_FFT = 512
_SPECTRAL_GATE_WINDOW_SIZE = 512
# 256-hop keeps updates frequent while keeping overlap for speech.
_SPECTRAL_GATE_HOP_LENGTH = 256

_STEP_NAME = "denoise_light"
_UNEXPECTED_KWARG_PATTERN = re.compile(r"unexpected keyword argument '([^']+)'")


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

    reducer_candidate = getattr(nr, "spectral_gate", None)
    if reducer_candidate is not None:
        default_reducer = cast(Callable[..., np.ndarray], reducer_candidate)  # pyright: ignore[reportUnknownVariableType]
    else:
        default_reducer = nr.reduce_noise  # pyright: ignore[reportUnknownMemberType]
    reducer: Callable[..., np.ndarray] = noise_reducer or default_reducer  # type: ignore[assignment]
    noise_clip = mono[: min(len(mono), sample_rate // 2)]  # use up to first 0.5s as noise profile
    reducer_kwargs = {
        "y": mono,
        "sr": sample_rate,
        "noise_clip": noise_clip,
        "n_std_thresh_stationary": 1.5,
        "prop_decrease": 0.85,
        "use_tensor": False,
        "stationary": True,
        "use_tqdm": False,
        "n_fft": _SPECTRAL_GATE_N_FFT,
        "win_length": _SPECTRAL_GATE_WINDOW_SIZE,
        "hop_length": _SPECTRAL_GATE_HOP_LENGTH,
    }
    try:
        denoised = _call_reducer(reducer, reducer_kwargs)
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


def _call_reducer(reducer: Callable[..., np.ndarray], kwargs: dict[str, Any]) -> np.ndarray:
    """Invoke the denoise reducer while dropping unsupported args."""
    remaining_kwargs = dict(kwargs)
    while True:
        try:
            return reducer(**remaining_kwargs)  # pyright: ignore[reportUnknownMemberType]
        except TypeError as exc:
            match = _UNEXPECTED_KWARG_PATTERN.search(str(exc))
            if not match:
                raise
            key = match.group(1)
            if key not in remaining_kwargs:
                raise
            remaining_kwargs.pop(key)
