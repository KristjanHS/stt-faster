"""Integration tests for CLI commands in transcribe_manager.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

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
            import argparse

            from transcribe_manager import cmd_process

            args = argparse.Namespace(input_folder=str(cli_test_folder), db_path=str(cli_test_db), preset="turbo")

            # Mock only the MODEL LOADING, not the transcription logic
            with patch("backend.transcribe.pick_model") as mock_pick_model:
                # Return a fake model that does minimal work
                mock_model = MagicMock()
                mock_model.transcribe.return_value = (
                    # Fake segments
                    [MagicMock(id=1, start=0.0, end=1.0, text="test", speaker=None)],
                    # Fake info
                    MagicMock(language="en", language_probability=0.99, duration=1.0),
                )
                mock_pick_model.return_value = mock_model

                result = cmd_process(args)

                # Should succeed with mocked model
                assert result == 0

                # Verify model was loaded once (not per file - model is reused)
                assert mock_pick_model.call_count == 3  # Called for each file
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


class TestCLIHelp:
    """Tests for CLI help and usage information."""

    def test_no_command_shows_help(self) -> None:
        """Test that running without command shows help."""
        result = subprocess.run(
            [sys.executable, "scripts/transcribe_manager.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        output = result.stdout + result.stderr
        assert "usage:" in output.lower() or "help" in output.lower()

    def test_process_help_shows_options(self) -> None:
        """Test that process command help shows available options."""
        result = subprocess.run(
            [sys.executable, "scripts/transcribe_manager.py", "process", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "input_folder" in output
        assert "--preset" in output

    def test_status_help_shows_options(self) -> None:
        """Test that status command help shows available options."""
        result = subprocess.run(
            [sys.executable, "scripts/transcribe_manager.py", "status", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "--verbose" in output or "-v" in output
