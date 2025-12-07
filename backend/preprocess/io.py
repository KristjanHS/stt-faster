from __future__ import annotations

import json
import subprocess  # nosec B404 - using fixed ffprobe command
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from backend.preprocess.errors import PreprocessError


@dataclass(slots=True)
class AudioInfo:
    channels: int | None
    sample_rate: int | None
    duration: float | None
    sample_format: str | None


def inspect_audio(path: Path) -> AudioInfo:
    """Inspect the input audio using ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,bit_rate,sample_fmt,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        output = subprocess.check_output(  # nosec B603 - fixed args, no untrusted input
            command,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PreprocessError("ffprobe is required but not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise PreprocessError(f"ffprobe failed: {exc.output.strip()}") from exc

    try:
        payload: Dict[str, Any] = json.loads(output)
        streams: list[Dict[str, Any]] = payload.get("streams") or []
        stream: Dict[str, Any] = streams[0] if streams else {}
    except (json.JSONDecodeError, IndexError) as exc:
        raise PreprocessError("ffprobe returned invalid metadata") from exc

    sample_rate = _safe_int(stream.get("sample_rate"))
    channels = _safe_int(stream.get("channels"))
    duration = _safe_float(stream.get("duration"))
    sample_fmt: str | None = stream.get("sample_fmt")

    return AudioInfo(
        channels=channels,
        sample_rate=sample_rate,
        duration=duration,
        sample_format=sample_fmt,
    )


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
