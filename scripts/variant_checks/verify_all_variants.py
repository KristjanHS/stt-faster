#!/usr/bin/env python3
"""Verify transcription parameters for all enabled variants.

This script actually executes each enabled variant (with mocked model) to verify
what parameters are passed to model.transcribe().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.orchestrator import PreprocessResult
from backend.variants.executor import create_variant_transcribe_config, transcribe_with_minimal_params
from backend.variants.registry import get_builtin_variants, get_variant_by_number


class RecordingModel:
    """Mock model that records all transcribe() calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._segments = []
        self._info = MagicMock(language="et", language_probability=0.95, duration=10.0)

    def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list, Any]:
        """Record the call and return fake results."""
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._segments, self._info


def verify_variant(variant_number: int) -> bool:
    """Verify a single variant's parameters by actually executing it."""
    variant = get_variant_by_number(variant_number)
    if variant is None:
        print(f"❌ Variant {variant_number} not found!")  # noqa: T201
        return False

    print(f"\n{'=' * 80}")  # noqa: T201
    print(f"Variant {variant_number}: {variant.name}")  # noqa: T201
    print("=" * 80)  # noqa: T201
    config = variant.transcription_config
    print(f"  Config: TranscriptionConfig with {len([f for f in dir(config) if not f.startswith('_')])} parameters")  # noqa: T201
    print()  # noqa: T201

    # Create a fake audio file
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "test.wav"
        audio_path.write_bytes(b"fake audio data")

        # Create fake preprocess result
        from backend.preprocess.metrics import PreprocessMetrics

        preprocess_result = PreprocessResult(
            output_path=audio_path,
            input_info=None,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
        )

        def fake_preprocess_runner(path: str, config: PreprocessConfig) -> PreprocessResult:
            return preprocess_result

        # Create recording model
        recording_model = RecordingModel()

        # Mock pick_model to return our recording model
        with patch("backend.variants.executor.pick_model", return_value=recording_model):
            # Get transcription config
            transcription_config = create_variant_transcribe_config(variant)

            # Actually call transcribe_with_minimal_params
            try:
                transcribe_with_minimal_params(
                    path=str(audio_path),
                    preset="et-large",
                    language="et",
                    preprocess_config=PreprocessConfig(enabled=False),
                    preprocess_runner=fake_preprocess_runner,
                    transcription_config=transcription_config,
                )
            except Exception as e:  # noqa: B110
                # We expect it to fail since we're using fake audio, but we should have captured the call
                # The call should have been made before any error occurs
                if not recording_model.calls:
                    # If no call was captured, this is a real error
                    print(f"  ❌ Error: Transcription failed before model.transcribe() was called: {e}")  # noqa: T201
                    return False

        # Verify we captured a call
        if not recording_model.calls:
            print("  ❌ Error: model.transcribe() was never called!")  # noqa: T201
            return False

        # Get the actual kwargs and args that were passed
        actual_kwargs = recording_model.calls[0]["kwargs"]
        actual_args = recording_model.calls[0]["args"]

        print("  ACTUAL PARAMETERS PASSED TO model.transcribe():")  # noqa: T201
        print(f"    Args: {actual_args}")  # noqa: T201
        print("    Kwargs:")  # noqa: T201
        for key, value in sorted(actual_kwargs.items()):
            print(f"      - {key}: {value}")  # noqa: T201
        print()  # noqa: T201

        # Derive expected parameters from the variant's transcription config
        # Base parameters that are always included
        expected_base = {
            "beam_size": getattr(transcription_config, "beam_size", 5),
            "word_timestamps": getattr(transcription_config, "word_timestamps", False),
            "task": getattr(transcription_config, "task", "transcribe"),
            "language": "et",  # From the test call
        }

        # Parameters that can be overridden (from transcribe_with_minimal_params logic)
        allowed_override_params = {"beam_size", "chunk_length", "no_speech_threshold", "condition_on_previous_text"}

        # Build expected kwargs from config
        expected_kwargs = expected_base.copy()
        for param in allowed_override_params:
            if hasattr(transcription_config, param):
                value = getattr(transcription_config, param)
                if value is not None:
                    expected_kwargs[param] = value

        print("  VERIFICATION RESULT:")  # noqa: T201
        print()  # noqa: T201

        all_correct = True
        # Verify all expected parameters are present and correct
        for param, expected_value in expected_kwargs.items():
            if param not in actual_kwargs:
                print(f"    ❌ {param}: MISSING (expected {expected_value})")  # noqa: T201
                all_correct = False
            elif actual_kwargs[param] != expected_value:
                actual_value = actual_kwargs[param]
                print(f"    ❌ {param}: {actual_value} (expected {expected_value})")  # noqa: T201
                all_correct = False
            else:
                print(f"    ✅ {param}: {actual_kwargs[param]} (matches config)")  # noqa: T201

        # Check for unexpected parameters (only allow expected ones)
        allowed_params = {"beam_size", "word_timestamps", "task", "language"}
        allowed_params.update(allowed_override_params)
        unexpected = set(actual_kwargs.keys()) - allowed_params
        if unexpected:
            print()  # noqa: T201
            print(f"    ⚠️  Unexpected parameters found: {unexpected}")  # noqa: T201
            print(f"       (These should not be present for variant {variant_number})")  # noqa: T201

        if all_correct and not unexpected:
            print()  # noqa: T201
            print("  ✅ Verification PASSED")  # noqa: T201
            return True
        else:
            print()  # noqa: T201
            print("  ❌ Verification FAILED")  # noqa: T201
            return False


def main() -> int:
    """Verify all enabled variants."""
    print("=" * 80)  # noqa: T201
    print("VERIFICATION: Transcription Parameters for All Enabled Variants")  # noqa: T201
    print("=" * 80)  # noqa: T201

    # Get enabled variants
    enabled_variants = get_builtin_variants()
    variant_numbers = sorted([v.number for v in enabled_variants])

    print(f"\nEnabled variants: {variant_numbers}")  # noqa: T201
    print()  # noqa: T201

    results = {}
    for variant_number in variant_numbers:
        results[variant_number] = verify_variant(variant_number)

    print()  # noqa: T201
    print("=" * 80)  # noqa: T201
    print("Summary:")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201
    for variant_number, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  Variant {variant_number}: {status}")  # noqa: T201

    all_passed = all(results.values())
    if all_passed:
        print()  # noqa: T201
        print("✅ All variants verified successfully!")  # noqa: T201
        return 0
    else:
        print()  # noqa: T201
        print("❌ Some variants failed verification!")  # noqa: T201
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
