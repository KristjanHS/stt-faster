from __future__ import annotations

import json
import logging
import subprocess  # nosec B404 - ffmpeg invocation with fixed arguments
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any, Dict, cast

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

LOGGER = logging.getLogger(__name__)

_TARGET_I = -18.0
_TARGET_TP = -1.5
_TARGET_LRA = 11.0


def apply_loudnorm(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    *,
    measure: Callable[..., Dict[str, float]] | None = None,
    normalize: Callable[..., None] | None = None,
) -> StepMetrics:
    """Normalize loudness using ffmpeg loudnorm (two-pass when possible)."""
    start = time.time()
    measure_loudness = cast(Callable[..., Dict[str, float]], measure or _measure_loudness)
    normalize_audio = cast(Callable[..., None], normalize or _normalize_audio)
    measured: Dict[str, float] | None = None

    try:
        measured = measure_loudness(input_path)
    except StepExecutionError as exc:
        LOGGER.warning("Loudnorm analysis failed; falling back to single-pass: %s", exc)

    try:
        normalize_audio(
            input_path=input_path,
            output_path=output_path,
            measured=measured,
            sample_rate=sample_rate,
        )
    except StepExecutionError as exc:
        if measured:
            LOGGER.warning("Two-pass loudnorm failed; retrying single-pass. Reason: %s", exc)
            normalize_audio(
                input_path=input_path,
                output_path=output_path,
                measured=None,
                sample_rate=sample_rate,
            )
        else:
            raise

    duration = time.time() - start
    return StepMetrics(name="loudnorm", backend="ffmpeg", duration=duration)


def _measure_loudness(
    input_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> Dict[str, float]:
    """Run loudnorm measurement pass and parse the emitted JSON stats."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",
        "-i",
        str(input_path),
        "-af",
        _build_loudnorm_filter(print_format="json"),
        "-f",
        "null",
        "-",
    ]

    run_command = runner or subprocess.run

    try:
        proc = run_command(  # nosec B603 - fixed command list, no shell
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise StepExecutionError("loudnorm", "ffmpeg is required but not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise StepExecutionError("loudnorm", exc.stderr.strip() or "ffmpeg loudnorm analysis failed") from exc

    stats = _parse_loudnorm_stats(proc.stderr)
    if stats is None:
        raise StepExecutionError("loudnorm", "ffmpeg loudnorm analysis returned no stats")
    return stats


def _normalize_audio(input_path: Path, output_path: Path, measured: Dict[str, float] | None, sample_rate: int) -> None:
    filter_str = _build_loudnorm_filter(measured=measured)
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


def _build_loudnorm_filter(print_format: str = "summary", measured: Dict[str, float] | None = None) -> str:
    parts = [
        f"I={_TARGET_I}",
        f"TP={_TARGET_TP}",
        f"LRA={_TARGET_LRA}",
        f"print_format={print_format}",
    ]

    if measured:
        parts.extend(
            [
                f"measured_I={measured['input_i']}",
                f"measured_TP={measured['input_tp']}",
                f"measured_LRA={measured['input_lra']}",
                f"measured_thresh={measured['input_thresh']}",
                f"offset={measured['target_offset']}",
            ]
        )

    return "loudnorm=" + ":".join(parts)


def _parse_loudnorm_stats(raw: str) -> Dict[str, float] | None:
    """Extract loudnorm measurement JSON block."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload: Dict[str, Any] = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None

    required_keys = ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset"]
    if not all(key in payload for key in required_keys):
        return None

    try:
        return {key: float(payload[key]) for key in required_keys}
    except (TypeError, ValueError):
        return None
