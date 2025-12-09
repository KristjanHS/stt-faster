#!/usr/bin/env python3
"""Verify variant 7 transcription parameters.

This script actually executes variant 7 (with mocked model) to verify
what parameters are passed to model.transcribe().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.orchestrator import PreprocessResult
from backend.variants.executor import transcribe_with_minimal_params
from backend.variants.registry import get_variant_by_number


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


def main() -> int:
    """Verify variant 7 configuration and parameters."""
    print("=" * 80)  # noqa: T201
    print("VARIANT 7 CONFIGURATION VERIFICATION")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201

    # Get variant 7
    variant_7 = get_variant_by_number(7)
    if variant_7 is None:
        print("❌ Variant 7 not found!")  # noqa: T201
        return 1

    print("Variant 7:")  # noqa: T201
    print(f"  Name: {variant_7.name}")  # noqa: T201
    print(f"  Description: {variant_7.description}")  # noqa: T201
    print(f"  Transcription preset: {variant_7.transcription_preset}")  # noqa: T201
    print(f"  Transcription overrides: {variant_7.transcription_overrides}")  # noqa: T201
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
            from backend.variants.executor import create_variant_transcribe_config

            transcription_config = create_variant_transcribe_config(variant_7)

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
            except Exception as e:
                # We expect it to fail since we're using fake audio, but we should have captured the call
                if not recording_model.calls:
                    print(f"❌ Error: Transcription failed before model.transcribe() was called: {e}")  # noqa: T201
                    return 1

        # Verify we captured a call
        if not recording_model.calls:
            print("❌ Error: model.transcribe() was never called!")  # noqa: T201
            return 1

        # Get the actual kwargs that were passed
        actual_kwargs = recording_model.calls[0]["kwargs"]
        actual_args = recording_model.calls[0]["args"]

        print("=" * 80)  # noqa: T201
        print("ACTUAL PARAMETERS PASSED TO model.transcribe():")  # noqa: T201
        print("=" * 80)  # noqa: T201
        print()  # noqa: T201
        print(f"  Args: {actual_args}")  # noqa: T201
        print("  Kwargs:")  # noqa: T201
        for key, value in sorted(actual_kwargs.items()):
            print(f"    - {key}: {value}")  # noqa: T201
        print()  # noqa: T201

        print("=" * 80)  # noqa: T201
        print("VERIFICATION RESULT:")  # noqa: T201
        print("=" * 80)  # noqa: T201
        print()  # noqa: T201

        # Verify expected parameters
        expected = {
            "beam_size": 5,
            "word_timestamps": False,
            "task": "transcribe",
            "language": "et",
        }

        all_correct = True
        for param, expected_value in expected.items():
            if param in actual_kwargs:
                actual_value = actual_kwargs[param]
                if actual_value == expected_value:
                    print(f"  ✅ {param}: {actual_value} (matches expected {expected_value})")  # noqa: T201
                else:
                    print(f"  ❌ {param}: {actual_value} (expected {expected_value}, got {actual_value})")  # noqa: T201
                    all_correct = False
            else:
                print(f"  ❌ {param}: MISSING (expected {expected_value})")  # noqa: T201
                all_correct = False

        # Verify no unexpected parameters (except language which is dynamic)
        allowed_params = {"beam_size", "word_timestamps", "task", "language"}
        unexpected = set(actual_kwargs.keys()) - allowed_params
        if unexpected:
            print()  # noqa: T201
            print(f"  ⚠️  Unexpected parameters found: {unexpected}")  # noqa: T201
            print("     (These should not be present for variant 7)")  # noqa: T201

        print()  # noqa: T201
        if all_correct and not unexpected:
            print("✅ All parameters match expectations!")  # noqa: T201
            print("   Variant 7 correctly uses only the base minimal parameters.")  # noqa: T201
        else:
            print("❌ Parameter verification failed!")  # noqa: T201
            return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
