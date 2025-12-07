from __future__ import annotations

import json
import subprocess  # nosec B404 - using fixed ffprobe command
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from backend.preprocess.errors import PreprocessError


@dataclass(slots=True)
class AudioInfo:
    channels: int | None
    sample_rate: int | None
    duration: float | None
    sample_format: str | None


def inspect_audio(path: Path, *, run_command: Callable[..., str] | None = None) -> AudioInfo:
    """Inspect the input audio using ffprobe."""
    try:
        if path.stat().st_size == 0:
            raise PreprocessError("audio file is empty")
    except FileNotFoundError as exc:
        raise PreprocessError(f"audio file not found: {path}") from exc

    runner = run_command or subprocess.check_output
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
        output = runner(command, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as exc:
        raise PreprocessError("ffprobe is required but not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        message = _format_ffprobe_error(exc.output)
        raise PreprocessError(f"ffprobe failed: {message}") from exc

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


def _format_ffprobe_error(raw_output: str | bytes | None) -> str:
    """Flatten ffprobe stderr/stdout into a concise single-line message."""
    if raw_output is None:
        return "no diagnostic output"

    if isinstance(raw_output, bytes):
        raw_output = raw_output.decode(errors="replace")

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return "no diagnostic output"

    def _keep(line: str) -> bool:
        # Drop brace-only artifacts from ffprobe output
        return any(ch.isalnum() for ch in line)

    filtered = [line for line in lines if _keep(line)]
    if not filtered:
        return "no diagnostic output"

    # Preserve order while dropping duplicates and collapsing whitespace/newlines
    seen: dict[str, None] = {}
    for line in filtered:
        seen.setdefault(line, None)

    return "; ".join(seen.keys())
