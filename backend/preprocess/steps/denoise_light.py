from __future__ import annotations

# pyright: reportUnknownVariableType=false
import re
import time
from pathlib import Path
from typing import Any, Callable, cast

import noisereduce as nr  # type: ignore[import-untyped]
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

_SPECTRAL_GATE_N_FFT = 512
_SPECTRAL_GATE_WINDOW_SIZE = 400
# 200-hop keeps updates frequent while keeping overlap for speech.
_SPECTRAL_GATE_HOP_LENGTH = 200
_SPECTRAL_GATE_PROP_DECREASE = 0.5
_SPECTRAL_GATE_N_STD_THRESH_STATIONARY = 1.25
_SPECTRAL_GATE_NOISE_CLIP_DURATION_S = 5.0

_STEP_NAME = "denoise_light"
_UNEXPECTED_KWARG_PATTERN = re.compile(r"unexpected keyword argument '([^']+)'")


def apply_light_denoise(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    *,
    noise_reducer: Callable[..., np.ndarray] | None = None,
    writer: Callable[..., object] | None = None,
    noise_clip_duration_s: float | None = None,
    n_std_thresh_stationary: float | None = None,
    prop_decrease: float | None = None,
) -> StepMetrics:
    """Apply a light denoise using noisereduce.spectral_gate.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        sample_rate: Target sample rate
        noise_reducer: Optional custom noise reducer function
        writer: Optional custom audio writer function
        noise_clip_duration_s: Duration in seconds for noise clip (default: 0.8)
        n_std_thresh_stationary: Standard deviation threshold for stationary noise (default: 1.25)
        prop_decrease: Proportion decrease for noise reduction (default: 0.5)
    """
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

    # Use custom parameters if provided, otherwise use defaults
    noise_clip_duration = (
        noise_clip_duration_s if noise_clip_duration_s is not None else _SPECTRAL_GATE_NOISE_CLIP_DURATION_S
    )
    n_std_thresh = (
        n_std_thresh_stationary if n_std_thresh_stationary is not None else _SPECTRAL_GATE_N_STD_THRESH_STATIONARY
    )
    prop_dec = prop_decrease if prop_decrease is not None else _SPECTRAL_GATE_PROP_DECREASE

    noise_clip = mono[: min(len(mono), int(sample_rate * noise_clip_duration))]
    reducer_kwargs = {
        "y": mono,
        "sr": sample_rate,
        "noise_clip": noise_clip,
        "n_std_thresh_stationary": n_std_thresh,
        "prop_decrease": prop_dec,
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
