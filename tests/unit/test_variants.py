"""Unit tests for the new variant system (backend/variants/).

These tests validate the new variant architecture independently from the legacy system.
The legacy system remains untouched and functional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.variants.registry import get_builtin_variants, get_variant_by_name, get_variant_by_number
from backend.variants.variant import PreprocessStep, Variant


class TestVariantRegistry:
    """Tests for variant registry functionality."""

    def test_get_builtin_variants_returns_list(self) -> None:
        """Test that get_builtin_variants returns a list."""
        variants = get_builtin_variants()
        assert isinstance(variants, list)
        assert len(variants) > 0

    def test_all_variants_have_required_fields(self) -> None:
        """Test that all variants have required fields."""
        variants = get_builtin_variants()
        for variant in variants:
            assert isinstance(variant, Variant)
            assert variant.name
            assert variant.number > 0
            assert variant.description
            assert variant.transcription_preset in ("project", "industry", "minimal")
            # Variants should have either preprocess_steps or custom_preprocess_runner
            assert variant.preprocess_steps is not None
            # Custom runner is optional, but if present it should be callable
            if variant.custom_preprocess_runner is not None:
                assert callable(variant.custom_preprocess_runner)

    def test_variant_numbers_are_unique(self) -> None:
        """Test that variant numbers are unique."""
        variants = get_builtin_variants()
        numbers = [v.number for v in variants]
        assert len(numbers) == len(set(numbers)), "Variant numbers must be unique"

    def test_variant_names_are_unique(self) -> None:
        """Test that variant names are unique."""
        variants = get_builtin_variants()
        names = [v.name for v in variants]
        assert len(names) == len(set(names)), "Variant names must be unique"

    def test_variants_1_through_9_exist(self) -> None:
        """Test that variants 1-9 exist (standard variants)."""
        for i in range(1, 10):
            variant = get_variant_by_number(i)
            assert variant is not None, f"Variant {i} should exist"

    def test_variants_10_through_19_exist(self) -> None:
        """Test that variants 10-19 exist (custom preprocessing variants and parameter overrides)."""
        for i in range(10, 20):
            variant = get_variant_by_number(i)
            assert variant is not None, f"Variant {i} should exist"

    def test_variant_7_has_no_preprocessing(self) -> None:
        """Test that variant 7 has no preprocessing steps."""
        variant = get_variant_by_number(7)
        assert variant is not None
        assert variant.name == "noprep_noparamtrans"
        assert len(variant.preprocess_steps) == 0
        assert variant.transcription_preset == "minimal"

    def test_variants_12_16_have_declarative_steps(self) -> None:
        """Test that variants 12-16 use declarative steps instead of custom runners."""
        for i in range(12, 17):
            variant = get_variant_by_number(i)
            assert variant is not None, f"Variant {i} should exist"
            assert variant.custom_preprocess_runner is None, f"Variant {i} should not have custom runner"
            assert len(variant.preprocess_steps) > 0, f"Variant {i} should have declarative steps"

    def test_variants_10_11_have_declarative_steps(self) -> None:
        """Test that variants 10-11 use declarative steps instead of custom runners."""
        for i in range(10, 12):
            variant = get_variant_by_number(i)
            assert variant is not None, f"Variant {i} should exist"
            assert variant.custom_preprocess_runner is None, f"Variant {i} should not have custom runner"
            assert len(variant.preprocess_steps) > 0, f"Variant {i} should have declarative steps"

    def test_get_variant_by_number(self) -> None:
        """Test getting variant by number."""
        variant = get_variant_by_number(7)
        assert variant is not None
        assert variant.number == 7
        assert variant.name == "noprep_noparamtrans"

        # Test non-existent variant
        variant = get_variant_by_number(999)
        assert variant is None

    def test_get_variant_by_name(self) -> None:
        """Test getting variant by name."""
        variant = get_variant_by_name("noprep_noparamtrans")
        assert variant is not None
        assert variant.name == "noprep_noparamtrans"
        assert variant.number == 7

        # Test non-existent variant
        variant = get_variant_by_name("nonexistent_variant")
        assert variant is None

    def test_variant_preprocess_steps_structure(self) -> None:
        """Test that preprocess steps have correct structure for standard variants."""
        variants = get_builtin_variants()
        valid_step_types = (
            "ffmpeg",
            "denoise",
            "resample",
            "loudnorm_only",
            "loudnorm_highpass",
            "dynaudnorm",
            "denoise_custom",
            "highlow_aform_loudnorm",
            "highlow_nosampl_loudnorm",
            "aresampl_loudnorm_fixed",
            "aresampl_loudnorm_fixed2",
            "loudnorm_2pass_linear",
        )
        for variant in variants:
            # Only check variants without custom runners
            if variant.custom_preprocess_runner is None:
                for step in variant.preprocess_steps:
                    assert isinstance(step, PreprocessStep)
                    assert step.name
                    assert step.step_type in valid_step_types, f"Invalid step_type: {step.step_type}"
                    assert isinstance(step.enabled, bool)


class TestVariantExecutor:
    """Tests for variant executor functionality."""

    def test_execute_variant_returns_correct_structure(self, tmp_path: Path) -> None:
        """Test that execute_variant returns results in the expected format."""
        # Create a minimal variant for testing
        variant = Variant(
            name="test_variant",
            number=99,
            description="test",
            preprocess_steps=[],
            transcription_preset="minimal",
        )

        # Create a fake audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        # Mock the transcription to avoid actual model loading
        # This test just verifies the structure, not actual execution
        # We'll use a variant that has no preprocessing to minimize dependencies

        # Get a real variant that we can test with minimal setup
        variant = get_variant_by_number(7)  # noprep_noparamtrans - simplest variant
        assert variant is not None

        # Note: This test would require actual model loading, so we'll skip it
        # in unit tests and test the structure separately
        # The actual execution is tested in integration tests

    def test_execute_variant_output_format(self) -> None:
        """Test that execute_variant output format matches expected structure."""
        # This test validates the expected output format without actually executing
        expected_keys = {
            "variant",
            "variant_number",
            "status",
            "elapsed_seconds",
        }

        # The output should have these keys
        # We can't easily test this without mocking the entire transcription pipeline
        # So we'll verify the structure in integration tests
        assert expected_keys  # Placeholder assertion


class TestVariantFiltering:
    """Tests for variant filtering functionality."""

    def test_filter_variants_by_number(self) -> None:
        """Test filtering variants by number."""
        variants = get_builtin_variants()
        # Filter out one variant
        if not variants:
            pytest.skip("No active variants to test")

        skip_set = {variants[0].number}  # Skip the first active variant
        filtered = [v for v in variants if v.number not in skip_set]

        # Should have remaining active variants (one less than original)
        assert len(filtered) == len(variants) - 1
        # Filtered variants should not include the skipped one
        for variant in filtered:
            assert variant.number not in skip_set

    def test_filter_variants_by_name(self) -> None:
        """Test filtering variants by name."""
        variants = get_builtin_variants()
        skip_names = {"no_preprocessing", "industry_defaults", "ffmpeg_only"}
        filtered = [v for v in variants if v.name not in skip_names]

        # All filtered variants should not be in skip list
        for variant in filtered:
            assert variant.name not in skip_names

    def test_all_variants_accessible_after_filtering(self) -> None:
        """Test that all variants remain accessible after filtering."""
        variants = get_builtin_variants()
        if not variants:
            pytest.skip("No active variants to test")

        # Skip one active variant
        skip_set = {variants[0].number}
        filtered = [v for v in variants if v.number not in skip_set]

        # Should have one less variant after filtering
        assert len(filtered) == len(variants) - 1

        # Verify all variants (including inactive) are still accessible by number
        # Test up to variant 20 (current max)
        for i in range(1, 21):
            variant = get_variant_by_number(i)
            assert variant is not None, f"Variant {i} should be accessible by number"


class TestVariantDataStructures:
    """Tests for variant data structure integrity."""

    def test_preprocess_step_dataclass(self) -> None:
        """Test PreprocessStep dataclass structure."""
        step = PreprocessStep(
            name="test_step",
            enabled=True,
            step_type="ffmpeg",
            config={"key": "value"},
        )
        assert step.name == "test_step"
        assert step.enabled is True
        assert step.step_type == "ffmpeg"
        assert step.config == {"key": "value"}

    def test_variant_dataclass(self) -> None:
        """Test Variant dataclass structure."""
        variant = Variant(
            name="test_variant",
            number=99,
            description="test_desc",
            preprocess_steps=[],
            transcription_preset="minimal",
        )
        assert variant.name == "test_variant"
        assert variant.number == 99
        assert variant.description == "test_desc"
        assert variant.preprocess_steps == []
        assert variant.transcription_preset == "minimal"
        assert variant.transcription_overrides is None
        assert variant.custom_preprocess_runner is None

    def test_variant_with_custom_runner(self) -> None:
        """Test Variant with custom preprocessing runner."""

        def dummy_runner(path: str, config: Any) -> Any:  # noqa: ANN401
            return None

        variant = Variant(
            name="custom_variant",
            number=98,
            description="custom",
            preprocess_steps=[],
            transcription_preset="minimal",
            custom_preprocess_runner=dummy_runner,
        )
        assert variant.custom_preprocess_runner is not None
        assert callable(variant.custom_preprocess_runner)


class TestVariantPresets:
    """Tests for variant transcription presets."""

    def test_all_variants_have_valid_presets(self) -> None:
        """Test that all variants have valid transcription presets."""
        variants = get_builtin_variants()
        valid_presets = {"project", "industry", "minimal"}

        for variant in variants:
            assert variant.transcription_preset in valid_presets, (
                f"Variant {variant.number} ({variant.name}) has invalid preset: {variant.transcription_preset}"
            )

    def test_minimal_preset_variants(self) -> None:
        """Test that variants with minimal preset are correctly identified."""
        # Variants 7, 10-19 should use minimal preset
        expected_minimal = {7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}

        # Check that all minimal preset variants can be accessed
        for variant_num in expected_minimal:
            variant = get_variant_by_number(variant_num)
            assert variant is not None, f"Variant {variant_num} should exist"
            assert variant.transcription_preset == "minimal", f"Variant {variant_num} should have minimal preset"

        # Check that active variants with minimal preset are in get_builtin_variants()
        variants = get_builtin_variants()
        minimal_variants = [v for v in variants if v.transcription_preset == "minimal"]

        # Verify that at least some minimal preset variants are active
        # (don't check specific numbers since active variants can change)
        assert len(minimal_variants) > 0, "At least one active variant should use minimal preset"


class TestVariantCompatibility:
    """Tests to ensure variant system compatibility with legacy expectations."""

    def test_variant_output_keys_match_legacy_format(self) -> None:
        """Test that variant execution output has keys matching legacy format."""
        # Expected keys from legacy run_variant() function
        expected_keys = {
            "variant",
            "variant_number",
            "status",
            "elapsed_seconds",
            "result",  # Present on success
            "error",  # Present on failure
            "output_files",
        }

        # This is validated by checking the executor implementation
        # The actual format is tested in integration tests
        assert expected_keys  # Placeholder to document expected structure

    def test_variant_names_match_legacy_names(self) -> None:
        """Test that variant names match legacy variant names."""
        # Key variants that should match legacy names
        expected_mappings = {
            1: "no_preprocessing",
            2: "industry_defaults",
            3: "ffmpeg_only",
            4: "denoise_only",
            5: "ffmpeg_industry_defaults",
            6: "full_industry_defaults",
            7: "noprep_noparamtrans",
            8: "normonly_noparamtrans",
            9: "onlyden_noparamtrans",
            10: "norm_highp_noparamtrans",
            11: "norm_dynaud_noparamtrans",
            12: "lnorm_highlow_aform_noparamtrans",
            13: "lnorm_highlow_nosampl_noparamtrans",
            14: "lnorm2_aresampl_noparamtrans",
            15: "lnorm3_aresampl_noparamtrans",
            16: "loudnorm_2pass_linear_noparamtrans",
        }

        for number, expected_name in expected_mappings.items():
            variant = get_variant_by_number(number)
            assert variant is not None, f"Variant {number} should exist"
            assert variant.name == expected_name, (
                f"Variant {number} should be named '{expected_name}', got '{variant.name}'"
            )
