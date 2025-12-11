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
    print(f"Variant {variant_number}: {variant.name} ({variant.description})")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print(f"  Preset: {variant.transcription_preset}")  # noqa: T201
    print(f"  Overrides: {variant.transcription_overrides}")  # noqa: T201
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
                    transcription_overrides=variant.transcription_overrides,
                )
            except Exception:  # noqa: B110
                # We expect it to fail since we're using fake audio, but we should have captured the call
                # The call should have been made before any error occurs
                if not recording_model.calls:
                    # If no call was captured, this is a real error
                    raise

        # Verify we captured a call
        if not recording_model.calls:
            print("  ❌ Error: model.transcribe() was never called!")  # noqa: T201
            return False

        # Get the actual kwargs that were passed
        actual_kwargs = recording_model.calls[0]["kwargs"]

        print("  ACTUAL parameters passed to model.transcribe():")  # noqa: T201
        for key, value in sorted(actual_kwargs.items()):
            print(f"    - {key}: {value}")  # noqa: T201

        # Verify expected base parameters
        expected_base = {
            "beam_size": 5,
            "word_timestamps": False,
            "task": "transcribe",
            "language": "et",
        }

        # Exclude override parameters from base check (they'll be checked separately)
        override_params = set(variant.transcription_overrides.keys()) if variant.transcription_overrides else set()

        all_correct = True
        for param, expected_value in expected_base.items():
            # Skip parameters that are overridden
            if param in override_params:
                continue
            if param not in actual_kwargs:
                print(f"    ❌ {param}: MISSING (expected {expected_value})")  # noqa: T201
                all_correct = False
            elif actual_kwargs[param] != expected_value:
                print(f"    ❌ {param}: {actual_kwargs[param]} (expected {expected_value})")  # noqa: T201
                all_correct = False

        # Verify override parameters
        if variant.transcription_overrides:
            print()  # noqa: T201
            print("  Override parameters:")  # noqa: T201
            for param, expected_value in variant.transcription_overrides.items():
                if param in actual_kwargs:
                    if actual_kwargs[param] == expected_value:
                        print(f"    ✅ {param}: {actual_kwargs[param]} (matches override)")  # noqa: T201
                    else:
                        print(f"    ❌ {param}: {actual_kwargs[param]} (expected {expected_value})")  # noqa: T201
                        all_correct = False
                else:
                    print(f"    ❌ {param}: MISSING (expected {expected_value} from override)")  # noqa: T201
                    all_correct = False

        # Check for unexpected parameters
        allowed_params = {"beam_size", "word_timestamps", "task", "language"}
        if variant.transcription_overrides:
            allowed_params.update(variant.transcription_overrides.keys())
        unexpected = set(actual_kwargs.keys()) - allowed_params
        if unexpected:
            print()  # noqa: T201
            print(f"    ⚠️  Unexpected parameters: {unexpected}")  # noqa: T201

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
