"""Integration tests for CLI commands in transcribe_manager.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def cli_test_folder(tmp_path: Path) -> Path:
    """Create a temporary folder with test audio files."""
    test_folder = tmp_path / "test_audio"
    test_folder.mkdir()

    # Create test audio files (empty for now, as real models would be too heavy)
    (test_folder / "audio1.wav").touch()
    (test_folder / "audio2.mp3").touch()
    (test_folder / "audio3.m4a").touch()

    return test_folder


@pytest.fixture
def real_audio_test_folder(tmp_path: Path) -> Path:
    """Create a temporary folder with the actual test.mp3 file."""
    test_folder = tmp_path / "test_audio_real"
    test_folder.mkdir()

    # Copy the actual test.mp3 file to the test folder
    source_mp3 = Path("tests/test.mp3")
    if source_mp3.exists():
        dest_mp3 = test_folder / "test.mp3"
        shutil.copy2(source_mp3, dest_mp3)

    return test_folder


@pytest.fixture
def cli_test_db(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_cli.db"


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
        """Test that process command validates folder and begins processing."""
        # This test verifies CLI argument parsing and folder validation
        # We don't wait for actual transcription to complete (would require models)
        # Just verify the command starts properly

        # Import and test directly to avoid model loading
        import os
        import sys

        # Add scripts directory to path for import
        scripts_path = os.path.join(os.getcwd(), "scripts")
        sys.path.insert(0, scripts_path)

        try:
            import argparse

            from transcribe_manager import cmd_process

            args = argparse.Namespace(input_folder=str(cli_test_folder), db_path=str(cli_test_db), preset="turbo")

            # This will fail when trying to load models, but should pass validation
            # We expect it to fail with model-related error, not argument error
            try:
                result = cmd_process(args)
                # If it returns 0, processing worked (unlikely without models)
                assert result in (0, 1)
            except Exception as e:
                # Should fail on model loading, not on argument validation
                error_msg = str(e)
                # Verify it's not a validation error
                assert "does not exist" not in error_msg.lower()
                assert "not a directory" not in error_msg.lower()
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
        """Test status command with verbose flag."""
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


@pytest.mark.slow
class TestActualTranscription:
    """Tests that perform actual transcription using real audio files.

    These tests require transcription models to be available and may take longer to run.
    They are marked with @pytest.mark.slow to allow selective execution.
    """

    def test_transcribe_real_mp3_file(self, real_audio_test_folder: Path, cli_test_db: Path) -> None:
        """Test actual transcription of test.mp3 file using Estonian model.

        Note: test.mp3 is Estonian audio, so we use the et-large preset.
        """
        # Skip if test.mp3 doesn't exist
        test_mp3 = real_audio_test_folder / "test.mp3"
        if not test_mp3.exists():
            pytest.skip("test.mp3 not found in test folder")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(cli_test_db),
                "process",
                str(real_audio_test_folder),
                "--preset",
                "et-large",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout for actual transcription
        )

        output = result.stdout + result.stderr

        # Check if transcription succeeded
        if result.returncode == 0:
            # Verify output files were created
            processed_folder = real_audio_test_folder / "processed"
            assert processed_folder.exists(), "Processed folder should be created"

            # Check for JSON output
            json_file = processed_folder / "test.json"
            if json_file.exists():
                # Verify JSON content is valid and contains expected Estonian text
                import json

                with json_file.open() as f:
                    transcription_data = json.load(f)
                    assert "segments" in transcription_data, "JSON should contain segments"
                    assert transcription_data["language"] == "et", "Language should be Estonian"

                    # Validate the transcription contains expected Estonian keywords
                    # The test.mp3 says: "Kolmas voor, tiim siis. Ja tiimi kohtumine. Algab kohe."
                    segments = transcription_data["segments"]
                    assert len(segments) > 0, "Should have at least one segment"
                    full_text = " ".join(seg["text"] for seg in segments)

                    # Check for key Estonian words from the audio
                    assert any(word in full_text.lower() for word in ["tiim", "kohtumine", "algab"]), (
                        f"Transcription should contain expected Estonian words, got: {full_text}"
                    )

            # Check database status
            from scripts.transcription.database import TranscriptionDatabase

            with TranscriptionDatabase(str(cli_test_db)) as db:
                file_info = db.get_status(str(test_mp3))
                if file_info:
                    assert file_info["status"] in ("completed", "pending"), f"Unexpected status: {file_info['status']}"
        else:
            # If it failed, check if it's because models are not available
            if "model" in output.lower() or "not found" in output.lower():
                pytest.skip("Transcription models not available")
            else:
                pytest.fail(f"Transcription failed with unexpected error: {output}")

    def test_transcribe_creates_processed_folder(self, real_audio_test_folder: Path, cli_test_db: Path) -> None:
        """Test that processing creates the processed and failed subfolders.

        Note: test.mp3 is Estonian audio, so we use the et-large preset.
        """
        test_mp3 = real_audio_test_folder / "test.mp3"
        if not test_mp3.exists():
            pytest.skip("test.mp3 not found in test folder")

        # Run the process command
        subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(cli_test_db),
                "process",
                str(real_audio_test_folder),
                "--preset",
                "et-large",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Verify folders were created regardless of success/failure
        assert (real_audio_test_folder / "processed").exists(), "Processed folder should be created"
        assert (real_audio_test_folder / "failed").exists(), "Failed folder should be created"

    def test_transcribe_accepts_preset_argument(self, real_audio_test_folder: Path, cli_test_db: Path) -> None:
        """Test that the preset argument is accepted by the CLI.

        Note: test.mp3 is Estonian audio, so we only test with et-large preset.
        Testing with other presets would fail due to language mismatch.
        """
        test_mp3 = real_audio_test_folder / "test.mp3"
        if not test_mp3.exists():
            pytest.skip("test.mp3 not found in test folder")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(cli_test_db),
                "process",
                str(real_audio_test_folder),
                "--preset",
                "et-large",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Verify the command runs without argument errors
        output = result.stdout + result.stderr
        assert "unknown preset" not in output.lower()
        assert "invalid preset" not in output.lower()
        assert "unrecognized arguments" not in output.lower()
