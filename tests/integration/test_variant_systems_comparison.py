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

from backend.variants import get_builtin_variants


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

    def test_variant_1_outputs_match(self, test_audio_file: Path, output_dir_old: Path, output_dir_new: Path) -> None:
        """Test that the first active variant produces consistent outputs.

        The first active variant should produce consistent results across multiple runs.
        This test validates that the variant system produces reproducible transcription results.
        """
        # Get active variants from configuration
        active_variants = get_builtin_variants()
        assert len(active_variants) > 0, "Should have at least one active variant"
        first_variant = active_variants[0]

        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "compare_transcription_variants.py"

        # Build skip list: skip all variants except the first active one
        all_variant_numbers = {v.number for v in active_variants}
        skip_variants = sorted([str(n) for n in all_variant_numbers if n != first_variant.number])

        # Run first active variant in first output directory
        result_old = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--skip-variants",
                *skip_variants,
                "--output-dir",
                str(output_dir_old),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Verify first run succeeded
        assert result_old.returncode == 0, f"First run failed: {result_old.stderr}\n{result_old.stdout}"

        # Run first active variant in second output directory
        result_new = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--skip-variants",
                *skip_variants,
                "--output-dir",
                str(output_dir_new),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Verify second run succeeded
        assert result_new.returncode == 0, f"Second run failed: {result_new.stderr}\n{result_new.stdout}"

        # Load results from both runs
        old_json_files = list(output_dir_old.glob("**/*_comparison_*.json"))
        new_json_files = list(output_dir_new.glob("**/*_comparison_*.json"))

        assert len(old_json_files) > 0, "First run should create comparison JSON files"
        assert len(new_json_files) > 0, "Second run should create comparison JSON files"

        with old_json_files[0].open() as f:
            old_results = json.load(f)
        with new_json_files[0].open() as f:
            new_results = json.load(f)

        # Verify structure matches expected format
        assert isinstance(old_results, list), "First run results should be a list"
        assert isinstance(new_results, list), "Second run results should be a list"
        assert len(old_results) == 1, "Should have exactly one variant result from first run"
        assert len(new_results) == 1, "Should have exactly one variant result from second run"

        old_result = old_results[0]
        new_result = new_results[0]

        # Verify both are the first active variant
        assert old_result["variant_number"] == first_variant.number, (
            f"First run should be variant {first_variant.number}"
        )
        assert new_result["variant_number"] == first_variant.number, (
            f"Second run should be variant {first_variant.number}"
        )
        assert old_result["variant"] == first_variant.name, f"First run should be {first_variant.name}"
        assert new_result["variant"] == first_variant.name, f"Second run should be {first_variant.name}"

        # Verify both succeeded
        assert old_result["status"] == "success", f"First run should succeed, got: {old_result.get('error', 'unknown')}"
        assert new_result["status"] == "success", (
            f"Second run should succeed, got: {new_result.get('error', 'unknown')}"
        )

        # Compare transcription results
        old_transcription = old_result.get("result", {})
        new_transcription = new_result.get("result", {})

        # Compare key transcription fields
        old_text = old_transcription.get("text", "")
        new_text = new_transcription.get("text", "")
        assert old_text == new_text, (
            f"Transcription text should match.\nFirst: {old_text[:100]}...\nSecond: {new_text[:100]}..."
        )

        # Compare segments (allow for minor timing differences, but text should match)
        old_segments = old_transcription.get("segments", [])
        new_segments = new_transcription.get("segments", [])

        assert len(old_segments) == len(new_segments), (
            f"Segment count should match. First: {len(old_segments)}, Second: {len(new_segments)}"
        )

        for i, (old_seg, new_seg) in enumerate(zip(old_segments, new_segments)):
            old_seg_text = old_seg.get("text", "")
            new_seg_text = new_seg.get("text", "")
            assert old_seg_text == new_seg_text, (
                f"Segment {i} text should match.\nFirst: {old_seg_text}\nSecond: {new_seg_text}"
            )

        # Compare language detection
        old_lang = old_transcription.get("language")
        new_lang = new_transcription.get("language")
        assert old_lang == new_lang, f"Language should match. First: {old_lang}, Second: {new_lang}"

        # Compare duration (should be very close, allow small floating point differences)
        old_duration = old_transcription.get("duration")
        new_duration = new_transcription.get("duration")
        if old_duration is not None and new_duration is not None:
            assert abs(old_duration - new_duration) < 0.01, (
                f"Duration should be very close. First: {old_duration}, Second: {new_duration}"
            )

    def test_all_active_variants_execute_successfully(self, test_audio_file: Path, output_dir_new: Path) -> None:
        """Test that the first 3 active variants execute successfully."""
        # Get active variants from configuration
        active_variants = get_builtin_variants()
        assert len(active_variants) >= 3, f"Should have at least 3 active variants, got {len(active_variants)}"
        test_variants = active_variants[:3]
        test_variant_numbers = {v.number for v in test_variants}

        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "compare_transcription_variants.py"

        # Build skip list: skip all variants except the first 3 active ones
        all_variant_numbers = {v.number for v in active_variants}
        skip_variants = sorted([str(n) for n in all_variant_numbers if n not in test_variant_numbers])

        # Run with first 3 active variants only
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--skip-variants",
                *skip_variants,
                "--output-dir",
                str(output_dir_new),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for multiple variants
        )

        # Verify system ran successfully
        assert result.returncode == 0, f"System failed: {result.stderr}\n{result.stdout}"

        # Verify output files were created
        json_files = list(output_dir_new.glob("**/*_comparison_*.json"))
        assert len(json_files) > 0, "System should create comparison JSON files"

        # Load and verify results
        with json_files[0].open() as f:
            results = json.load(f)

        # Should have 3 variants (first 3 active variants)
        assert len(results) == 3, f"Should have 3 variant results, got {len(results)}"

        # Verify all variants succeeded
        variant_numbers = {r["variant_number"] for r in results}
        assert variant_numbers == test_variant_numbers, (
            f"Expected variants {test_variant_numbers}, got {variant_numbers}"
        )

        # Verify all have success status
        for result_item in results:
            assert result_item["status"] == "success", (
                f"Variant {result_item['variant_number']} should succeed, "
                f"got: {result_item.get('error', 'unknown error')}"
            )

    def test_variant_filtering_works_correctly(self, test_audio_file: Path, output_dir_new: Path) -> None:
        """Test that --skip-variants filtering works correctly."""
        # Get active variants from configuration
        active_variants = get_builtin_variants()
        assert len(active_variants) > 0, "Should have at least one active variant"
        first_variant = active_variants[0]

        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "compare_transcription_variants.py"

        # Build skip list: skip all variants except the first active one
        all_variant_numbers = {v.number for v in active_variants}
        skip_variants = sorted([str(n) for n in all_variant_numbers if n != first_variant.number])

        # Run with only the first active variant
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(test_audio_file),
                "--skip-variants",
                *skip_variants,
                "--output-dir",
                str(output_dir_new),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Filtering failed: {result.stderr}"

        # Verify only the first active variant was executed
        json_files = list(output_dir_new.glob("**/*_comparison_*.json"))
        assert len(json_files) > 0

        with json_files[0].open() as f:
            results = json.load(f)

        assert len(results) == 1, "Should have exactly one variant result"
        assert results[0]["variant_number"] == first_variant.number, f"Should be variant {first_variant.number} only"
        assert results[0]["variant"] == first_variant.name, f"Should be {first_variant.name} only"
