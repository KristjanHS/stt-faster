from __future__ import annotations

import json
import logging
import re
import subprocess  # nosec B404 - ffmpeg invocation with fixed arguments
import time
from pathlib import Path
from typing import Any, Callable, Dict, Final, Tuple, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

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
        "LRA": 6.0,
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
    """Normalize loudness using pyloudnorm if available; otherwise fall back to ffmpeg."""
    preset_config = _LOUDNORM_PRESETS.get(preset, _LOUDNORM_PRESETS["default"])
    target_loudness = preset_config["I"]

    pyln = _import_pyloudnorm()
    use_pyloudnorm = pyln is not None and measure is None and normalize is None
    if use_pyloudnorm:
        duration = _run_pyloudnorm(pyln, input_path, output_path, sample_rate, target_loudness=target_loudness)
        return StepMetrics(name="loudnorm", backend="pyloudnorm", duration=duration)

    def _default_measure(input_path: Path) -> Dict[str, float]:
        return _measure_loudness(input_path, sample_rate=sample_rate, preset=preset)

    def _default_normalize(
        input_path: Path, output_path: Path, measured: Dict[str, float] | None, sample_rate: int
    ) -> float:
        return _normalize_with_ffmpeg(input_path, output_path, measured, sample_rate, preset=preset)

    measure_fn: Callable[[Path], Dict[str, float]] = measure or _default_measure
    normalize_fn: Callable[[Path, Path, Dict[str, float] | None, int], float] = normalize or _default_normalize

    measured: Dict[str, float] | None = None
    try:
        measured = measure_fn(input_path)
    except StepExecutionError:
        measured = None

    try:
        duration = normalize_fn(input_path, output_path, measured, sample_rate)
    except StepExecutionError:
        if measured is None:
            raise
        duration = normalize_fn(input_path, output_path, None, sample_rate)

    return StepMetrics(name="loudnorm", backend="ffmpeg", duration=duration)


def _import_pyloudnorm() -> Any | None:
    try:
        import pyloudnorm as pyln  # type: ignore[import-untyped]
    except ImportError:
        LOGGER.debug("pyloudnorm unavailable; falling back to ffmpeg loudnorm")
        return None
    return pyln


def _run_pyloudnorm(
    pyln: Any, input_path: Path, output_path: Path, sample_rate: int, *, target_loudness: float = -20.0
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
        meter = _create_meter(pyln, sample_rate)
        loudness = meter.integrated_loudness(waveform)
        normalized = pyln.normalize.loudness(waveform, loudness, target_loudness)
    except Exception as exc:  # noqa: BLE001 - propagate wrapped failure
        raise StepExecutionError("loudnorm", f"pyloudnorm processing failed: {exc}") from exc

    writer = cast(Callable[..., None], sf.write)  # pyright: ignore[reportUnknownMemberType]
    try:
        writer(output_path, normalized, sample_rate, subtype="PCM_16")  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:  # noqa: BLE001
        raise StepExecutionError("loudnorm", f"failed to write audio: {exc}") from exc

    return time.time() - start


def _run_ffmpeg_loudnorm(input_path: Path, output_path: Path, sample_rate: int, *, preset: str = "default") -> float:
    start = time.time()
    filter_str = _build_loudnorm_filter(preset=preset)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_str,
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    try:
        subprocess.run(  # nosec B603 - fixed command list, no shell
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise StepExecutionError("loudnorm", "ffmpeg is required but not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise StepExecutionError("loudnorm", exc.stderr.strip() or "ffmpeg loudnorm failed") from exc

    return time.time() - start


def _parse_loudnorm_stats(payload: str) -> Dict[str, float]:
    match = re.search(r"\{[^{}]*\}", payload, re.S)
    if not match:
        raise StepExecutionError("loudnorm", "no loudnorm stats found")
    try:
        raw = match.group(0)
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StepExecutionError("loudnorm", f"failed to parse loudnorm stats: {exc}") from exc
    parsed: Dict[str, float] = {}
    for key, value in data.items():
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise StepExecutionError("loudnorm", f"invalid loudnorm stat for {key}: {exc}") from exc
    return parsed


def _measure_loudness(
    input_path: Path,
    *,
    sample_rate: int | None = None,
    preset: str = "default",
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, float]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",
        "-y",
        "-i",
        str(input_path),
        "-af",
        _build_loudnorm_filter(preset=preset),
        "-f",
        "null",
        "-",
    ]
    if sample_rate is not None:
        index = command.index("-f")
        command[index:index] = ["-ar", str(sample_rate)]
    try:
        result = runner(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise StepExecutionError("loudnorm", "ffmpeg is required but not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise StepExecutionError("loudnorm", exc.stderr.strip() or "ffmpeg loudnorm measurement failed") from exc

    stderr = getattr(result, "stderr", "") or ""
    return _parse_loudnorm_stats(stderr)


def _create_meter(pyln: Any, sample_rate: int) -> Any:
    """Instantiate pyloudnorm Meter while tolerating removed keyword args."""
    meter_kwargs = {"block_size": 0.400}
    try:
        return pyln.Meter(sample_rate, filter_true=True, **meter_kwargs)
    except TypeError as exc:  # pragma: no cover - fallback depends on pyloudnorm version
        if "filter_true" not in str(exc):
            raise
        return pyln.Meter(sample_rate, **meter_kwargs)


def _normalize_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    measured: Dict[str, float] | None,
    sample_rate: int,
    *,
    preset: str = "default",
) -> float:
    return _run_ffmpeg_loudnorm(input_path, output_path, sample_rate, preset=preset)


def _build_loudnorm_filter(*, preset: str = "default") -> str:
    """Build ffmpeg loudnorm filter string from preset configuration."""
    preset_config = _LOUDNORM_PRESETS.get(preset, _LOUDNORM_PRESETS["default"])
    parts = [
        f"I={preset_config['I']}",
        f"TP={preset_config['TP']}",
        f"LRA={preset_config['LRA']}",
        "print_format=summary",
    ]
    return "loudnorm=" + ":".join(parts)
