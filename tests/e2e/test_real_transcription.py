"""E2E tests for real transcription with actual models.

These tests perform actual transcription using real audio files and real models.
They are slower and may require network access for model downloads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.run_log import JsonlRunLog


@pytest.mark.slow
@pytest.mark.network
class TestRealTranscription:
    """E2E tests that perform actual transcription using real audio files.

    These tests require transcription models to be available and may take longer to run.
    They are marked with @pytest.mark.slow to allow selective execution.
    """

    def test_transcribe_real_mp3_file(self, e2e_test_folder: Path, tmp_path: Path) -> None:
        """E2E: Transcribe test.mp3 file using Estonian model.

        Note: test.mp3 is Estonian audio, so we use the et-large preset.
        """
        # Skip if test.mp3 doesn't exist
        test_mp3 = e2e_test_folder / "test.mp3"
        if not test_mp3.exists():
            pytest.fail("test.mp3 not found in test folder; add tests/test.mp3 for E2E transcription")

        # Isolate the JSONL run log to tmp_path. The subprocess inherits
        # XDG_DATA_HOME so ``get_default_run_log_path`` resolves to
        # ``<tmp_path>/stt-faster/runs.jsonl`` instead of the user's real log.
        xdg_data_home = tmp_path / "xdg_data"
        xdg_data_home.mkdir()
        env = {**os.environ, "XDG_DATA_HOME": str(xdg_data_home)}

        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "process",
                str(e2e_test_folder),
                "--preset",
                "et-large",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout for actual transcription
            env=env,
        )

        output = result.stdout + result.stderr

        # Check if transcription succeeded
        if result.returncode == 0:
            # Verify output files were created
            processed_folder = e2e_test_folder / "processed"
            assert processed_folder.exists(), "Processed folder should be created"

            # Check for JSON output
            json_file = processed_folder / "test.json"
            if json_file.exists():
                # Verify JSON content is valid and contains expected Estonian text
                with json_file.open() as f:
                    transcription_data = json.load(f)
                    assert "segments" in transcription_data, "JSON should contain segments"

                    language = transcription_data.get("language")
                    if language:
                        language = language.lower()
                        assert language in {"et", "est"} or language.startswith("et"), (
                            f"Expected Estonian language code, got: {language}"
                        )

                    segments = transcription_data["segments"]
                    assert isinstance(segments, list), "Segments should be a list"
                    assert len(segments) > 0, "Should have at least one segment"

                    full_text = " ".join(seg.get("text", "") for seg in segments).strip()
                    assert full_text, "Transcription text should be non-empty"

            # Stage G.b: assert against the JSONL run log (DB write path retired).
            from typing import Any, cast  # noqa: PLC0415

            run_log = JsonlRunLog(xdg_data_home / "stt-faster" / "runs.jsonl")
            records = run_log.tail(10)
            matching_file_statuses: list[str] = []
            for r in records:
                files = cast("list[dict[str, Any]]", r.get("files", []) or [])
                for f in files:
                    if f.get("path") == str(test_mp3):
                        status = f.get("status")
                        if isinstance(status, str):
                            matching_file_statuses.append(status)
            if matching_file_statuses:
                assert matching_file_statuses[-1] in ("completed", "pending"), (
                    f"Unexpected status: {matching_file_statuses[-1]}"
                )
        else:
            # If it failed, check if it's because models are not available
            if "model" in output.lower() or "not found" in output.lower():
                pytest.fail(f"Transcription models not available: {output}")
            else:
                pytest.fail(f"Transcription failed with unexpected error: {output}")
