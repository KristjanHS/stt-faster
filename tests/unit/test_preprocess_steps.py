"""Unit tests for individual preprocess step functions in backend/variants/preprocess_steps.py.

These tests validate each preprocessing step function in isolation, ensuring:
- Correct ffmpeg filter construction
- Parameter passing
- Error handling
- Output metrics generation
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.variants.preprocess_steps import (
    aresampl_loudnorm_fixed,
    aresampl_loudnorm_fixed2,
    compressor_with_limiter,
    dynaudnorm_conservative,
    dynaudnorm_only,
    highlow_aform_loudnorm,
    highlow_nosampl_loudnorm,
    limiter_only,
    loudnorm_2pass_linear,
    loudnorm_only,
    loudnorm_with_highpass,
    peak_normalize_2pass,
    sox_peak_normalize,
    volume_with_limiter,
)


class MockFFmpegError(Exception):
    """Stand-in for ``ffmpeg.Error`` (carries ``stderr`` bytes like the real one)."""

    def __init__(self, cmd: str, stdout: bytes, stderr: bytes) -> None:
        self.cmd = cmd
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"ffmpeg error: {stderr.decode()}")


@pytest.fixture
def mock_ffmpeg() -> MagicMock:
    """MagicMock standing in for the ``ffmpeg`` module; each ffmpeg call returns a fresh stream mock."""
    module = MagicMock()
    module.input.return_value = MagicMock()
    module.filter.return_value = MagicMock()
    module.output.return_value = MagicMock()
    module.run.return_value = None
    return module


def _loader(module: MagicMock) -> Callable[[], Any]:
    """Build the ``ffmpeg_module`` loader the steps call instead of ``import ffmpeg``."""
    return lambda: module


class TestLoudnormOnly:
    """Tests for loudnorm_only step."""

    def test_loudnorm_only_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that loudnorm_only constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = loudnorm_only(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            loudnorm_preset="default",
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify ffmpeg.input was called
        mock_ffmpeg.input.assert_called_once_with(str(input_file))

        # Verify loudnorm filter was called with default preset params
        mock_ffmpeg.filter.assert_called_once()
        filter_call = mock_ffmpeg.filter.call_args
        assert filter_call[0][1] == "loudnorm"  # filter name
        # loudnorm params use uppercase keys (I, TP, LRA) or lowercase (i, tp, lra) depending on implementation
        filter_kwargs = filter_call[1]
        assert "I" in filter_kwargs or "i" in filter_kwargs  # integrated loudness
        assert "TP" in filter_kwargs or "tp" in filter_kwargs  # true peak
        assert "LRA" in filter_kwargs or "lra" in filter_kwargs  # loudness range

        # Verify output was called with correct parameters
        mock_ffmpeg.output.assert_called_once()
        output_call = mock_ffmpeg.output.call_args
        assert output_call[1]["ac"] == 1
        assert output_call[1]["ar"] == 16000
        assert output_call[1]["acodec"] == "pcm_s16le"

        # Verify metrics
        assert metrics.name.startswith("loudnorm_only")
        assert metrics.backend == "ffmpeg"
        assert metrics.duration >= 0

    def test_loudnorm_only_handles_ffmpeg_error(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that loudnorm_only handles ffmpeg errors correctly."""
        from backend.preprocess.errors import StepExecutionError

        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        mock_ffmpeg.Error = MockFFmpegError
        mock_ffmpeg.run.side_effect = MockFFmpegError("ffmpeg", b"", b"Error: Option not found")

        with pytest.raises(StepExecutionError) as excinfo:
            loudnorm_only(
                input_path=input_file,
                output_path=output_file,
                target_sample_rate=16000,
                target_channels=1,
                ffmpeg_module=_loader(mock_ffmpeg),
            )

        assert "ffmpeg failed" in str(excinfo.value).lower()


class TestVolumeWithLimiter:
    """Tests for volume_with_limiter step."""

    def test_volume_with_limiter_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that volume_with_limiter constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = volume_with_limiter(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            volume_db=3.0,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify volume filter was called
        filter_calls = mock_ffmpeg.filter.call_args_list
        volume_call = next((c for c in filter_calls if c[0][1] == "volume"), None)
        assert volume_call is not None

        # Verify limiter filter was called
        limiter_call = next((c for c in filter_calls if c[0][1] == "alimiter"), None)
        assert limiter_call is not None

        # Verify metrics
        assert metrics.name.startswith("volume_with_limiter")
        assert metrics.backend == "ffmpeg"


class TestPeakNormalize2Pass:
    """Tests for peak_normalize_2pass step."""

    def test_peak_normalize_2pass_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that peak_normalize_2pass constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        # First call returns volumedetect output, second call is for actual processing
        volumedetect_stderr = b"max_volume: -12.5 dB\nmean_volume: -20.0 dB"
        mock_ffmpeg.run.side_effect = [
            (b"", volumedetect_stderr),  # First pass: volumedetect
            None,  # Second pass: actual processing
        ]

        metrics = peak_normalize_2pass(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            target_db=-3.0,
            max_gain_db=3.0,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify two calls to ffmpeg.run (volumedetect + processing)
        assert mock_ffmpeg.run.call_count == 2

        # Verify first pass uses volumedetect
        first_output_call = mock_ffmpeg.output.call_args_list[0]
        assert first_output_call[1].get("af") == "volumedetect"

        # Verify second pass uses volume filter with calculated gain
        second_output_call = mock_ffmpeg.output.call_args_list[1]
        af_value = second_output_call[1].get("af", "")
        assert "volume=" in af_value or "alimiter" in af_value

        # Verify metrics
        assert metrics.name.startswith("peak_normalize_2pass")
        assert metrics.backend == "ffmpeg"


class TestLoudnormWithHighpass:
    """Tests for loudnorm_with_highpass step."""

    def test_loudnorm_with_highpass_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that loudnorm_with_highpass constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = loudnorm_with_highpass(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            loudnorm_preset="default",
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify highpass filter was called
        filter_calls = mock_ffmpeg.filter.call_args_list
        highpass_call = next((c for c in filter_calls if c[0][1] == "highpass"), None)
        assert highpass_call is not None

        # Verify loudnorm filter was called
        loudnorm_call = next((c for c in filter_calls if c[0][1] == "loudnorm"), None)
        assert loudnorm_call is not None

        # Verify metrics
        assert metrics.name.startswith("loudnorm_with_highpass")
        assert metrics.backend == "ffmpeg"


class TestDynaudnormOnly:
    """Tests for dynaudnorm_only step."""

    def test_dynaudnorm_only_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that dynaudnorm_only constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = dynaudnorm_only(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify dynaudnorm filter was called
        filter_call = mock_ffmpeg.filter.call_args
        assert filter_call[0][1] == "dynaudnorm"

        # Verify metrics
        assert metrics.name == "dynaudnorm_only"
        assert metrics.backend == "ffmpeg"


class TestHighlowAformLoudnorm:
    """Tests for highlow_aform_loudnorm step."""

    def test_highlow_aform_loudnorm_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that highlow_aform_loudnorm constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = highlow_aform_loudnorm(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify highpass filter was called
        filter_calls = mock_ffmpeg.filter.call_args_list
        highpass_call = next((c for c in filter_calls if c[0][1] == "highpass"), None)
        assert highpass_call is not None

        # Verify lowpass filter was called
        lowpass_call = next((c for c in filter_calls if c[0][1] == "lowpass"), None)
        assert lowpass_call is not None

        # Verify loudnorm filter was called
        loudnorm_call = next((c for c in filter_calls if c[0][1] == "loudnorm"), None)
        assert loudnorm_call is not None

        # Verify metrics
        assert metrics.name == "highlow_aform_loudnorm"
        assert metrics.backend == "ffmpeg"


class TestHighlowNosamplLoudnorm:
    """Tests for highlow_nosampl_loudnorm step."""

    def test_highlow_nosampl_loudnorm_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that highlow_nosampl_loudnorm constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = highlow_nosampl_loudnorm(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify highpass and lowpass filters were called
        filter_calls = mock_ffmpeg.filter.call_args_list
        highpass_call = next((c for c in filter_calls if c[0][1] == "highpass"), None)
        assert highpass_call is not None

        lowpass_call = next((c for c in filter_calls if c[0][1] == "lowpass"), None)
        assert lowpass_call is not None

        # Verify loudnorm filter was called
        loudnorm_call = next((c for c in filter_calls if c[0][1] == "loudnorm"), None)
        assert loudnorm_call is not None

        # Verify metrics
        assert metrics.name == "highlow_nosampl_loudnorm"
        assert metrics.backend == "ffmpeg"


class TestAresamplLoudnormFixed:
    """Tests for aresampl_loudnorm_fixed step."""

    def test_aresampl_loudnorm_fixed_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that aresampl_loudnorm_fixed constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = aresampl_loudnorm_fixed(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            target_i=-24.0,
            target_tp=-2.0,
            target_lra=15.0,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify aresample filter was called with osr (not ar)
        filter_calls = mock_ffmpeg.filter.call_args_list
        aresample_call = next((c for c in filter_calls if c[0][1] == "aresample"), None)
        assert aresample_call is not None
        assert "osr" in aresample_call[1] or aresample_call[1].get("osr") == 16000

        # Verify loudnorm filter was called
        loudnorm_call = next((c for c in filter_calls if c[0][1] == "loudnorm"), None)
        assert loudnorm_call is not None

        # Verify metrics
        assert metrics.name.startswith("aresampl_loudnorm_fixed")
        assert metrics.backend == "ffmpeg"


class TestAresamplLoudnormFixed2:
    """Tests for aresampl_loudnorm_fixed2 step."""

    def test_aresampl_loudnorm_fixed2_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that aresampl_loudnorm_fixed2 constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = aresampl_loudnorm_fixed2(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            target_i=-24.0,
            target_tp=-2.0,
            target_lra=15.0,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify aresample filter was called with osr and precision
        filter_calls = mock_ffmpeg.filter.call_args_list
        aresample_call = next((c for c in filter_calls if c[0][1] == "aresample"), None)
        assert aresample_call is not None
        assert aresample_call[1].get("precision") == "24"

        # Verify loudnorm filter was called
        loudnorm_call = next((c for c in filter_calls if c[0][1] == "loudnorm"), None)
        assert loudnorm_call is not None

        # Verify metrics
        assert metrics.name.startswith("aresampl_loudnorm_fixed2")
        assert metrics.backend == "ffmpeg"

    def test_aresampl_loudnorm_fixed2_handles_ffmpeg_error(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that aresampl_loudnorm_fixed2 handles ffmpeg errors correctly."""
        from backend.preprocess.errors import StepExecutionError

        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        mock_ffmpeg.Error = MockFFmpegError
        mock_ffmpeg.run.side_effect = MockFFmpegError("ffmpeg", b"", b"Error: Option not found")

        with pytest.raises(StepExecutionError) as excinfo:
            aresampl_loudnorm_fixed2(
                input_path=input_file,
                output_path=output_file,
                target_sample_rate=16000,
                target_channels=1,
                target_i=-24.0,
                target_tp=-2.0,
                target_lra=15.0,
                ffmpeg_module=_loader(mock_ffmpeg),
            )

        assert "ffmpeg failed" in str(excinfo.value).lower()


class TestLoudnorm2PassLinear:
    """Tests for loudnorm_2pass_linear step."""

    def test_loudnorm_2pass_linear_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that loudnorm_2pass_linear constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = loudnorm_2pass_linear(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify loudnorm filter was called with linear and dual_mono
        filter_call = mock_ffmpeg.filter.call_args
        assert filter_call[0][1] == "loudnorm"
        assert filter_call[1]["linear"] == "true"
        assert filter_call[1]["dual_mono"] == "true"

        # Verify metrics
        assert metrics.name == "loudnorm_2pass_linear"
        assert metrics.backend == "ffmpeg"


class TestLimiterOnly:
    """Tests for limiter_only step."""

    def test_limiter_only_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that limiter_only constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = limiter_only(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify alimiter filter was called
        filter_call = mock_ffmpeg.filter.call_args
        assert filter_call[0][1] == "alimiter"

        # Verify metrics
        assert metrics.name == "limiter_only"
        assert metrics.backend == "ffmpeg"


class TestSoxPeakNormalize:
    """Tests for sox_peak_normalize step."""

    def test_sox_peak_normalize_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that sox_peak_normalize constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = sox_peak_normalize(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            target_db=-3.0,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify volume filter was called with target dB
        filter_call = mock_ffmpeg.filter.call_args
        assert filter_call[0][1] == "volume"
        # Volume can be passed as string like "-3.0dB" or as a number
        volume_value = filter_call[1].get("volume", "")
        assert "-3.0" in str(volume_value) or volume_value == "-3.0dB"

        # Verify metrics
        assert metrics.name.startswith("sox_peak_normalize")
        assert metrics.backend == "ffmpeg"


class TestCompressorWithLimiter:
    """Tests for compressor_with_limiter step."""

    def test_compressor_with_limiter_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that compressor_with_limiter constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = compressor_with_limiter(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify compand filter was called
        filter_calls = mock_ffmpeg.filter.call_args_list
        compand_call = next((c for c in filter_calls if c[0][1] == "compand"), None)
        assert compand_call is not None

        # Verify alimiter filter was called
        limiter_call = next((c for c in filter_calls if c[0][1] == "alimiter"), None)
        assert limiter_call is not None

        # Verify metrics
        assert metrics.name == "compressor_with_limiter"
        assert metrics.backend == "ffmpeg"


class TestDynaudnormConservative:
    """Tests for dynaudnorm_conservative step."""

    def test_dynaudnorm_conservative_calls_ffmpeg_correctly(self, tmp_path: Path, mock_ffmpeg: MagicMock) -> None:
        """Test that dynaudnorm_conservative constructs correct ffmpeg pipeline."""
        input_file = tmp_path / "in.wav"
        output_file = tmp_path / "out.wav"
        input_file.write_bytes(b"\x00\x01")

        metrics = dynaudnorm_conservative(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            ffmpeg_module=_loader(mock_ffmpeg),
        )

        # Verify dynaudnorm filter was called with conservative parameters
        filter_call = mock_ffmpeg.filter.call_args
        assert filter_call[0][1] == "dynaudnorm"
        assert filter_call[1]["f"] == "150"
        assert filter_call[1]["g"] == "15"
        assert filter_call[1]["p"] == "0.95"

        # Verify metrics
        assert metrics.name == "dynaudnorm_conservative"
        assert metrics.backend == "ffmpeg"
