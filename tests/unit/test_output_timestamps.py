"""`--timestamps/--no-timestamps`: TXT formatter, output writer, and CLI plumbing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from backend.cli.transcription_commands import create_app
from backend.services.json_output_writer import JsonOutputWriter
from backend.transcribe import format_segments_as_text

SEGMENTS: list[dict[str, Any]] = [
    {"start": 0.0, "end": 4.2, "text": " Tere, alustame."},
    {"start": 4.2, "end": 9.8, "text": "Esimene punkt.", "speaker": "SPEAKER_01"},
]


def test_timestamps_on_keeps_existing_line_format() -> None:
    assert format_segments_as_text(SEGMENTS) == (
        "[00:00:00.00 --> 00:00:04.20] Tere, alustame.\n[00:00:04.20 --> 00:00:09.80] SPEAKER_01: Esimene punkt.\n"
    )


def test_timestamps_off_is_one_line_per_segment() -> None:
    assert format_segments_as_text(SEGMENTS, include_timestamps=False) == (
        "Tere, alustame.\nSPEAKER_01: Esimene punkt.\n"
    )


def test_writer_honours_include_timestamps(tmp_path: Path) -> None:
    [txt] = JsonOutputWriter(include_timestamps=False).write(str(tmp_path / "a.txt"), {"segments": SEGMENTS}, "txt")
    assert txt.read_text(encoding="utf-8") == "Tere, alustame.\nSPEAKER_01: Esimene punkt.\n"


def _invoke_process(tmp_path: Path, *flags: str) -> argparse.Namespace:
    captured: list[argparse.Namespace] = []

    def fake_process(a: argparse.Namespace) -> int:
        captured.append(a)
        return 0

    result = CliRunner().invoke(create_app(process_fn=fake_process), [str(tmp_path), *flags])
    assert result.exit_code == 0, result.output
    return captured[0]


def test_cli_timestamps_default_on(tmp_path: Path) -> None:
    assert _invoke_process(tmp_path).timestamps is True


def test_cli_no_timestamps_flag(tmp_path: Path) -> None:
    assert _invoke_process(tmp_path, "--no-timestamps").timestamps is False
