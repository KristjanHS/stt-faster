"""E2E validation of CLI with real tiny model and real audio."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@pytest.mark.slow
@pytest.mark.network
def test_process_command_with_cached_model_e2e(tmp_path: Path, tiny_whisper_model: "WhisperModel | None") -> None:
    """Run the process command end-to-end with the real tiny model."""
    if tiny_whisper_model is None:
        pytest.skip("tiny model unavailable (USE_CACHED_MODEL=false or offline)")

    source_mp3 = Path("tests/test.mp3")
    if not source_mp3.exists():
        pytest.skip("tests/test.mp3 not available for E2E CLI real-model test")

    workdir = tmp_path / "real_audio"
    workdir.mkdir()
    dest_mp3 = workdir / "test.mp3"
    shutil.copy2(source_mp3, dest_mp3)

    scripts_path = os.path.join(os.getcwd(), "scripts")
    sys.path.insert(0, scripts_path)

    try:
        from transcribe_manager import cmd_process, create_parser

        parser = create_parser()
        args = parser.parse_args(["--db-path", str(tmp_path / "e2e.db"), "process", str(workdir), "--preset", "turbo"])

        result = cmd_process(args)

        assert result == 0
    finally:
        sys.path.remove(scripts_path)
