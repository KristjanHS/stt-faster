from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.steps import loudness


@pytest.fixture
def mock_pyloudnorm():
    with patch("backend.preprocess.steps.loudness.pyln") as mock:
        yield mock


@pytest.fixture
def mock_soundfile():
    with patch("backend.preprocess.steps.loudness.sf") as mock:
        yield mock


def test_apply_loudnorm_uses_pyloudnorm_with_correct_settings(mock_pyloudnorm, mock_soundfile) -> None:
    # Setup
    input_path = Path("input.wav")
    output_path = Path("output.wav")
    sample_rate = 16000
    fake_samples = np.array([0.1, 0.2], dtype=np.float32)
    fake_loudness = -25.0
    normalized_audio = np.array([0.2, 0.4], dtype=np.float32)

    mock_soundfile.read.return_value = (fake_samples, sample_rate)

    meter_instance = MagicMock()
    meter_instance.integrated_loudness.return_value = fake_loudness
    mock_pyloudnorm.Meter.return_value = meter_instance

    mock_pyloudnorm.normalize.loudness.return_value = normalized_audio
    mock_pyloudnorm.normalize.peak.return_value = normalized_audio

    # Execute
    result = loudness.apply_loudnorm(input_path, output_path, sample_rate, preset="default")

    # Verify
    assert result.backend == "pyloudnorm"

    # Check soundfile read
    mock_soundfile.read.assert_called_once_with(input_path, always_2d=False)

    # Check Meter creation and usage
    # Note: _create_meter might use different args depending on pyln version logic,
    # but we catch that in specific tests. Here we assume successful creation.
    mock_pyloudnorm.Meter.assert_called()
    meter_instance.integrated_loudness.assert_called_once()

    # Check normalization call
    # Default preset I is -20.0
    mock_pyloudnorm.normalize.loudness.assert_called_once()
    call_args = mock_pyloudnorm.normalize.loudness.call_args
    # args: (waveform, input_loudness, target_loudness)
    assert np.array_equal(call_args[0][0], fake_samples)
    assert call_args[0][1] == fake_loudness
    assert call_args[0][2] == -20.0

    # Check peak normalization call (default preset TP is -2.0)
    mock_pyloudnorm.normalize.peak.assert_called_once()
    peak_call_args = mock_pyloudnorm.normalize.peak.call_args
    # args: (normalized_audio, target_peak)
    assert np.array_equal(peak_call_args[0][0], normalized_audio)
    assert peak_call_args[0][1] == -2.0

    # Check write
    mock_soundfile.write.assert_called_once_with(output_path, normalized_audio, sample_rate, subtype="PCM_16")


def test_apply_loudnorm_uses_boost_quiet_voices_preset(mock_pyloudnorm, mock_soundfile) -> None:
    input_path = Path("input.wav")
    output_path = Path("output.wav")
    sample_rate = 16000

    mock_soundfile.read.return_value = (np.array([0.1]), sample_rate)
    meter_instance = MagicMock()
    meter_instance.integrated_loudness.return_value = -30.0
    mock_pyloudnorm.Meter.return_value = meter_instance

    loudness.apply_loudnorm(input_path, output_path, sample_rate, preset="boost-quiet-voices")

    # Check target loudness validation
    # boost-quiet-voices I is -18.0
    mock_pyloudnorm.normalize.loudness.assert_called_once()
    assert mock_pyloudnorm.normalize.loudness.call_args[0][2] == -18.0


def test_apply_loudnorm_raises_on_sample_rate_mismatch(mock_soundfile) -> None:
    mock_soundfile.read.return_value = (np.array([0.1]), 44100)  # Mismatch

    with pytest.raises(StepExecutionError, match="unexpected sample rate"):
        loudness.apply_loudnorm(Path("in.wav"), Path("out.wav"), sample_rate=16000)


def test_apply_loudnorm_raises_on_pyloudnorm_failure(mock_pyloudnorm, mock_soundfile) -> None:
    mock_soundfile.read.return_value = (np.array([0.1]), 16000)
    mock_pyloudnorm.Meter.side_effect = RuntimeError("Original Pyln Error")

    with pytest.raises(StepExecutionError, match="pyloudnorm processing failed"):
        loudness.apply_loudnorm(Path("in.wav"), Path("out.wav"), sample_rate=16000)


def test_create_meter_supports_filter_true_keyword() -> None:
    class MeterWithFilter:
        def __init__(self, sample_rate: int, block_size: float = 0.4, filter_true: bool = False) -> None:
            self.kwargs = {
                "sample_rate": sample_rate,
                "block_size": block_size,
                "filter_true": filter_true,
            }

    pyln_mock = SimpleNamespace(Meter=MeterWithFilter)

    # We essentially mock how pyln is imported/used in _create_meter logic
    # But _create_meter uses the global 'pyln' or passed arg?
    # The current implementation of _create_meter imports pyloudnorm globally.
    # We can patch pyln.Meter globally.

    with patch("backend.preprocess.steps.loudness.pyln", pyln_mock):
        meter = loudness._create_meter(44100)
        assert meter.kwargs["filter_true"] is True


def test_create_meter_falls_back_when_filter_true_removed() -> None:
    class MeterWithoutFilter:
        def __init__(self, sample_rate: int, block_size: float = 0.4) -> None:
            self.kwargs = {
                "sample_rate": sample_rate,
                "block_size": block_size,
            }

    # Simulate TypeError when filter_true is passed
    def invalid_meter_init(*args, **kwargs):
        if "filter_true" in kwargs:
            raise TypeError("unexpected keyword argument 'filter_true'")
        return MeterWithoutFilter(*args, **kwargs)

    pyln_mock = MagicMock()
    pyln_mock.Meter.side_effect = invalid_meter_init

    with patch("backend.preprocess.steps.loudness.pyln", pyln_mock):
        loudness._create_meter(48000)

        # Should have called twice
        assert pyln_mock.Meter.call_count == 2
        # First call with filter_true
        assert pyln_mock.Meter.call_args_list[0][1]["filter_true"] is True
        # Second call without it
        assert "filter_true" not in pyln_mock.Meter.call_args_list[1][1]
