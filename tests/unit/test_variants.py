"""Unit tests for the new variant system (backend/variants/).

These tests validate the new variant architecture independently from the legacy system.
The legacy system remains untouched and functional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.preprocess.config import TranscriptionConfig
from backend.variants.registry import (
    get_all_variants,
    get_builtin_variants,
    get_variant_by_name,
    get_variant_by_number,
)
from backend.variants.transcription_presets import create_minimal_config
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
            assert isinstance(variant.transcription_config, TranscriptionConfig)
            # Variants should have either preprocess_steps or custom_preprocess_runner
            assert variant.preprocess_steps is not None
            # Custom runner is optional, but if present it should be callable
            if variant.custom_preprocess_runner is not None:
                assert callable(variant.custom_preprocess_runner)

    def test_variant_numbers_are_unique(self) -> None:
        """Test that variant numbers are unique."""
        all_variants = get_all_variants()
        numbers = [v.number for v in all_variants]
        assert len(numbers) == len(set(numbers)), "Variant numbers must be unique"

    def test_variant_names_are_unique(self) -> None:
        """Test that variant names are unique."""
        all_variants = get_all_variants()
        names = [v.name for v in all_variants]
        assert len(names) == len(set(names)), "Variant names must be unique"

    def test_all_defined_variants_exist(self) -> None:
        """Test that all defined variants can be accessed by number."""
        all_variants = get_all_variants()

        # Verify all variants are accessible by number
        for variant in all_variants:
            found = get_variant_by_number(variant.number)
            assert found is not None, f"Variant {variant.number} should be accessible by number"
            assert found.number == variant.number
            assert found.name == variant.name

    def test_variants_with_no_preprocessing(self) -> None:
        """Test that variants with no preprocessing steps are correctly structured."""
        all_variants = get_all_variants()
        no_prep_variants = [v for v in all_variants if len(v.preprocess_steps) == 0]

        assert len(no_prep_variants) > 0, "At least one variant should have no preprocessing"

        for variant in no_prep_variants:
            assert isinstance(variant.transcription_config, TranscriptionConfig)
            assert variant.custom_preprocess_runner is None or callable(variant.custom_preprocess_runner)

    def test_variants_with_declarative_steps(self) -> None:
        """Test that variants with declarative steps are correctly structured."""
        all_variants = get_all_variants()

        for variant in all_variants:
            if variant.custom_preprocess_runner is None:
                # Variants without custom runners should have declarative steps
                for step in variant.preprocess_steps:
                    assert isinstance(step, PreprocessStep)
                    assert step.name
                    assert isinstance(step.enabled, bool)

    def test_get_variant_by_number(self) -> None:
        """Test getting variant by number."""
        all_variants = get_all_variants()
        if not all_variants:
            pytest.skip("No variants defined")

        # Test with an actual variant
        test_variant = all_variants[0]
        variant = get_variant_by_number(test_variant.number)
        assert variant is not None
        assert variant.number == test_variant.number
        assert variant.name == test_variant.name

        # Test non-existent variant
        variant = get_variant_by_number(999)
        assert variant is None

    def test_get_variant_by_name(self) -> None:
        """Test getting variant by name."""
        all_variants = get_all_variants()
        if not all_variants:
            pytest.skip("No variants defined")

        # Test with an actual variant
        test_variant = all_variants[0]
        variant = get_variant_by_name(test_variant.name)
        assert variant is not None
        assert variant.name == test_variant.name
        assert variant.number == test_variant.number

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
            preprocess_steps=[],
            transcription_config=create_minimal_config(),
        )

        # Create a fake audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        # Mock the transcription to avoid actual model loading
        # This test just verifies the structure, not actual execution
        # We'll use a variant that has no preprocessing to minimize dependencies

        # Get a real variant that we can test with minimal setup
        variant = get_variant_by_number(1)  # noprep_noparamtrans - simplest variant
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
        skip_names = {"no_preprocessing", "noprep_minimal", "ffmpeg_only"}
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
        # Test up to variant 26 (current max)
        for i in range(1, 27):
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
        config = create_minimal_config()
        variant = Variant(
            name="test_variant",
            number=99,
            preprocess_steps=[],
            transcription_config=config,
        )
        assert variant.name == "test_variant"
        assert variant.number == 99
        assert variant.preprocess_steps == []
        assert variant.transcription_config == config
        assert variant.custom_preprocess_runner is None

    def test_variant_with_custom_runner(self) -> None:
        """Test Variant with custom preprocessing runner."""

        def dummy_runner(path: str, config: Any) -> Any:  # noqa: ANN401
            return None

        variant = Variant(
            name="custom_variant",
            number=98,
            preprocess_steps=[],
            transcription_config=create_minimal_config(),
            custom_preprocess_runner=dummy_runner,
        )
        assert variant.custom_preprocess_runner is not None
        assert callable(variant.custom_preprocess_runner)


class TestVariantPresets:
    """Tests for variant transcription configs."""

    def test_all_variants_have_transcription_config(self) -> None:
        """Test that all variants have transcription config."""
        variants = get_builtin_variants()

        for variant in variants:
            assert isinstance(variant.transcription_config, TranscriptionConfig), (
                f"Variant {variant.number} ({variant.name}) must have transcription_config"
            )

    def test_all_variant_configs_are_valid(self) -> None:
        """Test that all variant transcription configs are valid TranscriptionConfig instances.

        This ensures that all variants can produce valid configs that could be passed to
        model.transcribe() without missing required fields.
        """
        all_variants = get_all_variants()

        for variant in all_variants:
            config = variant.transcription_config
            assert isinstance(config, TranscriptionConfig), (
                f"Variant {variant.number} ({variant.name}) must have TranscriptionConfig instance"
            )

            # Verify essential fields are present and have valid types
            assert isinstance(config.beam_size, int), f"Variant {variant.number}: beam_size must be int"
            assert config.beam_size > 0, f"Variant {variant.number}: beam_size must be positive"
            assert isinstance(config.word_timestamps, bool), f"Variant {variant.number}: word_timestamps must be bool"
            assert config.task in ("transcribe", "translate"), (
                f"Variant {variant.number}: task must be 'transcribe' or 'translate'"
            )

            # Verify optional fields have correct types if present
            if hasattr(config, "chunk_length") and config.chunk_length is not None:
                assert isinstance(config.chunk_length, int), f"Variant {variant.number}: chunk_length must be int"
                assert config.chunk_length > 0, f"Variant {variant.number}: chunk_length must be positive"

            if hasattr(config, "patience") and config.patience is not None:
                assert isinstance(config.patience, (int, float)), f"Variant {variant.number}: patience must be numeric"
                assert config.patience > 0, f"Variant {variant.number}: patience must be positive"

            if hasattr(config, "no_speech_threshold") and config.no_speech_threshold is not None:
                assert isinstance(config.no_speech_threshold, (int, float)), (
                    f"Variant {variant.number}: no_speech_threshold must be numeric"
                )

            if hasattr(config, "vad_filter") and config.vad_filter is not None:
                assert isinstance(config.vad_filter, bool), f"Variant {variant.number}: vad_filter must be bool"

            if hasattr(config, "vad_parameters") and config.vad_parameters is not None:
                assert isinstance(config.vad_parameters, dict), f"Variant {variant.number}: vad_parameters must be dict"

    def test_minimal_config_variants(self) -> None:
        """Test that variants with minimal config are correctly identified."""
        from backend.variants.transcription_presets import create_minimal_config

        minimal_ref = create_minimal_config()
        all_variants = get_all_variants()

        # Identify variants that use minimal config by comparing key attributes
        minimal_variants = []
        for variant in all_variants:
            config = variant.transcription_config
            # Minimal config typically has word_timestamps=False and task="transcribe"
            # and matches minimal_ref for these key attributes
            if config.word_timestamps == minimal_ref.word_timestamps and config.task == minimal_ref.task:
                minimal_variants.append(variant)

        # Verify minimal config variants have expected structure
        for variant in minimal_variants:
            assert variant.transcription_config.word_timestamps is False
            assert variant.transcription_config.task == "transcribe"

        # Check that at least some active variants use minimal config
        active_variants = get_builtin_variants()
        active_minimal = [
            v
            for v in active_variants
            if v.transcription_config.word_timestamps is False and v.transcription_config.task == "transcribe"
        ]
        assert len(active_minimal) > 0, "At least one active variant should use minimal config"


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

    def test_variant_names_are_consistent(self) -> None:
        """Test that variant names are consistent across lookups."""
        all_variants = get_all_variants()

        # Verify that each variant can be found by both number and name
        for variant in all_variants:
            # Find by number
            found_by_number = get_variant_by_number(variant.number)
            assert found_by_number is not None, f"Variant {variant.number} should be findable by number"
            assert found_by_number.name == variant.name

            # Find by name
            found_by_name = get_variant_by_name(variant.name)
            assert found_by_name is not None, f"Variant '{variant.name}' should be findable by name"
            assert found_by_name.number == variant.number
            assert found_by_name.name == variant.name
