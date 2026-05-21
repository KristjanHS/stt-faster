"""Integration tests for CLI commands in transcribe_manager.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Note: Fixtures cli_test_folder, real_audio_test_folder are defined in
# tests/integration/conftest.py for reuse across integration tests.


class TestProcessCommand:
    """Tests for the 'process' CLI command."""

    def test_process_command_rejects_nonexistent_folder(self) -> None:
        """Test that process command fails gracefully on nonexistent folder."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "process",
                "/nonexistent/folder",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        assert "does not exist" in result.stderr or "does not exist" in result.stdout

    def test_process_command_rejects_file_as_folder(self, tmp_path: Path) -> None:
        """Test that process command fails when given a file instead of directory."""
        test_file = tmp_path / "not_a_folder.txt"
        test_file.touch()

        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "process",
                str(test_file),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        assert "not a directory" in result.stderr or "not a directory" in result.stdout

    def test_process_command_accepts_valid_folder(
        self, cli_test_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration test: CLI → Processor → JSONL run log with mocked model loading.

        This test validates the integration between CLI and processor layers
        by mocking only the external dependency (model loading from HuggingFace).
        Isolates the JSONL run log to ``tmp_path`` via ``XDG_DATA_HOME`` so the
        run doesn't pollute the user's real ``~/.local/share/stt-faster/runs.jsonl``.
        """
        import os
        import sys
        from unittest.mock import MagicMock, patch

        xdg_data_home = tmp_path / "xdg_data"
        xdg_data_home.mkdir()
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data_home))

        scripts_path = os.path.join(os.getcwd(), "scripts")
        sys.path.insert(0, scripts_path)

        try:
            from transcribe_manager import cmd_process, create_parser

            parser = create_parser()
            args = parser.parse_args(
                ["process", str(cli_test_folder), "--preset", "turbo", "--no-diarize"],
            )

            with (
                patch("backend.variants.executor.pick_model") as mock_pick_model_executor,
                patch("backend.transcribe.pick_model") as mock_pick_model_transcribe,
            ):
                mock_model = MagicMock()
                mock_model.transcribe.return_value = (
                    [MagicMock(id=1, start=0.0, end=1.0, text="test", speaker=None)],
                    MagicMock(language="en", language_probability=0.99, duration=1.0),
                )
                mock_pick_model_executor.return_value = mock_model
                mock_pick_model_transcribe.return_value = mock_model

                result = cmd_process(args)

                assert result == 0

                # The variant executor calls pick_model once per file
                total_calls = mock_pick_model_executor.call_count + mock_pick_model_transcribe.call_count
                assert total_calls == 3, (
                    f"Expected 3 calls, got {total_calls} "
                    f"(executor: {mock_pick_model_executor.call_count}, "
                    f"transcribe: {mock_pick_model_transcribe.call_count})"
                )

                # I-2 review fix: assert the JSONL append actually happened.
                # Without this, a regression in run_log.append (bad path, IO
                # failure swallowed) would not be caught here.
                from backend.run_log import JsonlRunLog

                run_log = JsonlRunLog(xdg_data_home / "stt-faster" / "runs.jsonl")
                appended = run_log.tail(1)
                assert len(appended) == 1
                assert appended[0]["preset"] == "turbo"
        finally:
            sys.path.remove(scripts_path)
