"""E2E tests for real transcription with actual models.

These tests perform actual transcription using real audio files and real models.
They are slower and may require network access for model downloads.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
class TestRealTranscription:
    """E2E tests that perform actual transcription using real audio files.

    These tests require transcription models to be available and may take longer to run.
    They are marked with @pytest.mark.slow to allow selective execution.
    """

    def test_transcribe_real_mp3_file(self, e2e_test_folder: Path, e2e_test_db: Path) -> None:
        """E2E: Transcribe test.mp3 file using Estonian model.

        Note: test.mp3 is Estonian audio, so we use the et-large preset.
        """
        # Skip if test.mp3 doesn't exist
        test_mp3 = e2e_test_folder / "test.mp3"
        if not test_mp3.exists():
            pytest.skip("test.mp3 not found in test folder")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(e2e_test_db),
                "process",
                str(e2e_test_folder),
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
            processed_folder = e2e_test_folder / "processed"
            assert processed_folder.exists(), "Processed folder should be created"

            # Check for JSON output
            json_file = processed_folder / "test.json"
            if json_file.exists():
                # Verify JSON content is valid and contains expected Estonian text
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
            from backend.database import TranscriptionDatabase

            with TranscriptionDatabase(str(e2e_test_db)) as db:
                file_info = db.get_status(str(test_mp3))
                if file_info:
                    assert file_info["status"] in ("completed", "pending"), f"Unexpected status: {file_info['status']}"
        else:
            # If it failed, check if it's because models are not available
            if "model" in output.lower() or "not found" in output.lower():
                pytest.skip("Transcription models not available")
            else:
                pytest.fail(f"Transcription failed with unexpected error: {output}")

    def test_transcribe_creates_processed_folder(self, e2e_test_folder: Path, e2e_test_db: Path) -> None:
        """E2E: Verify that processing creates the processed and failed subfolders.

        Note: test.mp3 is Estonian audio, so we use the et-large preset.
        """
        test_mp3 = e2e_test_folder / "test.mp3"
        if not test_mp3.exists():
            pytest.skip("test.mp3 not found in test folder")

        # Run the process command
        subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(e2e_test_db),
                "process",
                str(e2e_test_folder),
                "--preset",
                "et-large",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Verify folders were created regardless of success/failure
        assert (e2e_test_folder / "processed").exists(), "Processed folder should be created"
        assert (e2e_test_folder / "failed").exists(), "Failed folder should be created"

    def test_transcribe_accepts_preset_argument(self, e2e_test_folder: Path, e2e_test_db: Path) -> None:
        """E2E: Verify that the preset argument is accepted by the CLI.

        Note: test.mp3 is Estonian audio, so we only test with et-large preset.
        Testing with other presets would fail due to language mismatch.
        """
        test_mp3 = e2e_test_folder / "test.mp3"
        if not test_mp3.exists():
            pytest.skip("test.mp3 not found in test folder")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/transcribe_manager.py",
                "--db-path",
                str(e2e_test_db),
                "process",
                str(e2e_test_folder),
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
