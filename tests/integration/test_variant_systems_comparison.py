"""Integration tests comparing old and new variant systems.

These tests ensure that the new variant system produces identical outputs
to the legacy system, validating backward compatibility.

The legacy system remains untouched and functional.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
class TestVariantSystemsComparison:
    """Integration tests comparing old and new variant system outputs.

    These tests require actual audio files and model execution.
    They are marked as slow and integration tests.
    """

    @pytest.fixture
    def test_audio_file(self, project_root: Path) -> Path:
        """Provide a test audio file for variant comparison."""
        # Use the test audio file if available
        test_audio = project_root / "tests" / "test_short.mp3"
        if not test_audio.exists():
            test_audio = project_root / "tests" / "test.mp3"
        if not test_audio.exists():
            pytest.skip("No test audio file found (test_short.mp3 or test.mp3)")
        return test_audio

    @pytest.fixture
    def output_dir_old(self, tmp_path: Path) -> Path:
        """Output directory for old system results."""
        return tmp_path / "old_system"

    @pytest.fixture
    def output_dir_new(self, tmp_path: Path) -> Path:
        """Output directory for new system results."""
        return tmp_path / "new_system"

    def test_variant_7_outputs_match(self, test_audio_file: Path, output_dir_old: Path, output_dir_new: Path) -> None:
        """Test that variant 7 produces identical outputs in both systems.

        Variant 7 (noprep_noparamtrans) is the simplest variant and should
        produce identical results in both systems. This test validates that
        the new system produces the same transcription results as the legacy system.
        """
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "compare_transcription_variants.py"

        # Run old system (legacy, no --use-new-variants flag) with variant 7 only
        result_old = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--skip-variants",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "--output-dir",
                str(output_dir_old),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Verify old system ran successfully
        assert result_old.returncode == 0, f"Old system failed: {result_old.stderr}\n{result_old.stdout}"

        # Run new system with variant 7 only
        result_new = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--use-new-variants",
                "--skip-variants",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "--output-dir",
                str(output_dir_new),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Verify new system ran successfully
        assert result_new.returncode == 0, f"New system failed: {result_new.stderr}\n{result_new.stdout}"

        # Load results from both systems
        old_json_files = list(output_dir_old.glob("**/*_comparison_*.json"))
        new_json_files = list(output_dir_new.glob("**/*_comparison_*.json"))

        assert len(old_json_files) > 0, "Old system should create comparison JSON files"
        assert len(new_json_files) > 0, "New system should create comparison JSON files"

        with old_json_files[0].open() as f:
            old_results = json.load(f)
        with new_json_files[0].open() as f:
            new_results = json.load(f)

        # Verify structure matches expected format
        assert isinstance(old_results, list), "Old results should be a list"
        assert isinstance(new_results, list), "New results should be a list"
        assert len(old_results) == 1, "Should have exactly one variant result from old system"
        assert len(new_results) == 1, "Should have exactly one variant result from new system"

        old_result = old_results[0]
        new_result = new_results[0]

        # Verify both are variant 7
        assert old_result["variant_number"] == 7, "Old system should be variant 7"
        assert new_result["variant_number"] == 7, "New system should be variant 7"
        assert old_result["variant"] == "noprep_noparamtrans", "Old system should be noprep_noparamtrans"
        assert new_result["variant"] == "noprep_noparamtrans", "New system should be noprep_noparamtrans"

        # Verify both succeeded
        assert old_result["status"] == "success", (
            f"Old system should succeed, got: {old_result.get('error', 'unknown')}"
        )
        assert new_result["status"] == "success", (
            f"New system should succeed, got: {new_result.get('error', 'unknown')}"
        )

        # Compare transcription results
        old_transcription = old_result.get("result", {})
        new_transcription = new_result.get("result", {})

        # Compare key transcription fields
        old_text = old_transcription.get("text", "")
        new_text = new_transcription.get("text", "")
        assert old_text == new_text, (
            f"Transcription text should match.\nOld: {old_text[:100]}...\nNew: {new_text[:100]}..."
        )

        # Compare segments (allow for minor timing differences, but text should match)
        old_segments = old_transcription.get("segments", [])
        new_segments = new_transcription.get("segments", [])

        assert len(old_segments) == len(new_segments), (
            f"Segment count should match. Old: {len(old_segments)}, New: {len(new_segments)}"
        )

        for i, (old_seg, new_seg) in enumerate(zip(old_segments, new_segments)):
            old_seg_text = old_seg.get("text", "")
            new_seg_text = new_seg.get("text", "")
            assert old_seg_text == new_seg_text, (
                f"Segment {i} text should match.\nOld: {old_seg_text}\nNew: {new_seg_text}"
            )

        # Compare language detection
        old_lang = old_transcription.get("language")
        new_lang = new_transcription.get("language")
        assert old_lang == new_lang, f"Language should match. Old: {old_lang}, New: {new_lang}"

        # Compare duration (should be very close, allow small floating point differences)
        old_duration = old_transcription.get("duration")
        new_duration = new_transcription.get("duration")
        if old_duration is not None and new_duration is not None:
            assert abs(old_duration - new_duration) < 0.01, (
                f"Duration should be very close. Old: {old_duration}, New: {new_duration}"
            )

    def test_all_active_variants_execute_successfully(self, test_audio_file: Path, output_dir_new: Path) -> None:
        """Test that all active variants (7, 10-15) execute successfully in new system."""
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "compare_transcription_variants.py"

        # Run new system with active variants only (skip 1-6, 8, 9)
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--use-new-variants",
                "--skip-variants",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "8",
                "9",
                "--output-dir",
                str(output_dir_new),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for multiple variants
        )

        # Verify system ran successfully
        assert result.returncode == 0, f"New system failed: {result.stderr}\n{result.stdout}"

        # Verify output files were created
        json_files = list(output_dir_new.glob("**/*_comparison_*.json"))
        assert len(json_files) > 0, "New system should create comparison JSON files"

        # Load and verify results
        with json_files[0].open() as f:
            results = json.load(f)

        # Should have 7 variants: 7, 10, 11, 12, 13, 14, 15
        assert len(results) == 7, f"Should have 7 variant results, got {len(results)}"

        # Verify all variants succeeded
        variant_numbers = {r["variant_number"] for r in results}
        expected_numbers = {7, 10, 11, 12, 13, 14, 15}
        assert variant_numbers == expected_numbers, f"Expected variants {expected_numbers}, got {variant_numbers}"

        # Verify all have success status
        for result_item in results:
            assert result_item["status"] == "success", (
                f"Variant {result_item['variant_number']} should succeed, "
                f"got: {result_item.get('error', 'unknown error')}"
            )

    def test_variant_filtering_works_correctly(self, test_audio_file: Path, output_dir_new: Path) -> None:
        """Test that --skip-variants filtering works correctly."""
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "compare_transcription_variants.py"

        # Run with only variant 7
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--use-new-variants",
                "--skip-variants",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "--output-dir",
                str(output_dir_new),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Filtering failed: {result.stderr}"

        # Verify only variant 7 was executed
        json_files = list(output_dir_new.glob("**/*_comparison_*.json"))
        assert len(json_files) > 0

        with json_files[0].open() as f:
            results = json.load(f)

        assert len(results) == 1, "Should have exactly one variant result"
        assert results[0]["variant_number"] == 7, "Should be variant 7 only"
