"""Integration tests for CLI commands in transcribe_manager.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Note: Fixtures cli_test_folder, real_audio_test_folder, and cli_test_db
# are now defined in tests/integration/conftest.py for reuse across integration tests


class TestProcessCommand:
    """Tests for the 'process' CLI command."""

    def test_process_command_rejects_nonexistent_folder(self, cli_test_db: Path) -> None:
        """Test that process command fails gracefully on nonexistent folder."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(cli_test_db),
                "process",
                "/nonexistent/folder",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        assert "does not exist" in result.stderr or "does not exist" in result.stdout

    def test_process_command_rejects_file_as_folder(self, cli_test_db: Path, tmp_path: Path) -> None:
        """Test that process command fails when given a file instead of directory."""
        test_file = tmp_path / "not_a_folder.txt"
        test_file.touch()

        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(cli_test_db),
                "process",
                str(test_file),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        assert "not a directory" in result.stderr or "not a directory" in result.stdout

    def test_process_command_accepts_valid_folder(self, cli_test_folder: Path, cli_test_db: Path) -> None:
        """Integration test: CLI → Processor → Database with mocked model loading.

        This test validates the integration between CLI, processor, and database layers
        by mocking only the external dependency (model loading from HuggingFace).

        What's tested:
        - CLI argument parsing
        - Folder validation and scanning
        - Processor orchestration
        - Database operations
        - File moving logic
        - Transcription logic flow

        What's mocked:
        - Model loading (pick_model) to avoid HuggingFace downloads
        """
        # Import and test directly with mocking
        import os
        import sys
        from unittest.mock import MagicMock, patch

        # Add scripts directory to path for import
        scripts_path = os.path.join(os.getcwd(), "scripts")
        sys.path.insert(0, scripts_path)

        try:
            from transcribe_manager import cmd_process, create_parser

            # Use the actual parser to create args - ensures all attributes are present
            parser = create_parser()
            args = parser.parse_args(
                ["--db-path", str(cli_test_db), "process", str(cli_test_folder), "--preset", "turbo"]
            )

            # Mock only the MODEL LOADING, not the transcription logic
            # Patch both locations where pick_model is used (transcribe and variants.executor)
            with (
                patch("backend.variants.executor.pick_model") as mock_pick_model_executor,
                patch("backend.transcribe.pick_model") as mock_pick_model_transcribe,
            ):
                # Return a fake model that does minimal work
                mock_model = MagicMock()
                mock_model.transcribe.return_value = (
                    # Fake segments
                    [MagicMock(id=1, start=0.0, end=1.0, text="test", speaker=None)],
                    # Fake info
                    MagicMock(language="en", language_probability=0.99, duration=1.0),
                )
                mock_pick_model_executor.return_value = mock_model
                mock_pick_model_transcribe.return_value = mock_model

                result = cmd_process(args)

                # Should succeed with mocked model
                assert result == 0

                # Verify model was loaded (variant executor path is used for minimal preset)
                # The variant executor calls pick_model once per file
                total_calls = mock_pick_model_executor.call_count + mock_pick_model_transcribe.call_count
                assert total_calls == 3, (
                    f"Expected 3 calls, got {total_calls} "
                    f"(executor: {mock_pick_model_executor.call_count}, "
                    f"transcribe: {mock_pick_model_transcribe.call_count})"
                )
        finally:
            sys.path.remove(scripts_path)


class TestStatusCommand:
    """Tests for the 'status' CLI command."""

    def test_status_command_on_empty_database(self, cli_test_db: Path) -> None:
        """Test status command on empty database."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(cli_test_db),
                "status",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "No files tracked" in output or "Total files: 0" in output

    def test_status_command_verbose_flag(self, cli_test_db: Path) -> None:
        """Test status command with verbose flag produces output."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(cli_test_db),
                "status",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        # Verify that verbose flag produces output (even if database is empty)
        assert len(output.strip()) > 0, "Verbose flag should produce output"
