"""Unit tests for backend.transcribe module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from backend.exceptions import ModelLoadError, ModelNotFoundError
from backend.transcribe import (
    _get_estonian_model_path,
    _round_floats,
    _segment_to_payload,
    pick_model,
    transcribe,
    transcribe_to_json,
)

if TYPE_CHECKING:
    pass


class TestGetEstonianModelPath:
    """Tests for _get_estonian_model_path function."""

    @patch("backend.transcribe.snapshot_download")
    def test_success_with_ct2_folder(self, mock_download: Mock) -> None:
        """Test successful download when CT2 folder exists."""
        mock_download.return_value = "/tmp/model_cache"

        with patch("backend.transcribe.Path") as mock_path:
            mock_ct2_path = MagicMock()
            mock_ct2_path.exists.return_value = True
            mock_path.return_value.__truediv__.return_value = mock_ct2_path

            _get_estonian_model_path("TalTechNLP/whisper-test")

            assert mock_download.called
            assert mock_download.call_args[0][0] == "TalTechNLP/whisper-test"
            assert mock_download.call_args[1]["allow_patterns"] == ["ct2/*"]

    @patch("backend.transcribe.snapshot_download")
    @patch("backend.transcribe.Path")
    def test_raises_error_when_ct2_folder_missing(self, mock_path: Mock, mock_download: Mock) -> None:
        """Test that ModelNotFoundError is raised when CT2 folder doesn't exist."""
        mock_download.return_value = "/tmp/model_cache"

        mock_ct2_path = MagicMock()
        mock_ct2_path.exists.return_value = False
        mock_path.return_value.__truediv__.return_value = mock_ct2_path

        with pytest.raises(ModelNotFoundError, match="CT2 folder not found"):
            _get_estonian_model_path("TalTechNLP/whisper-test")


class TestPickModel:
    """Tests for pick_model function."""

    @patch("backend.transcribe._get_estonian_model_path")
    @patch("backend.transcribe.WhisperModel")
    def test_et_large_preset_success(self, mock_whisper: Mock, mock_get_path: Mock) -> None:
        """Test et-large preset with successful GPU initialization."""
        mock_get_path.return_value = "/tmp/model/ct2"
        mock_model = Mock()
        mock_whisper.return_value = mock_model

        result = pick_model("et-large")

        assert result == mock_model
        mock_get_path.assert_called_once_with("TalTechNLP/whisper-large-v3-turbo-et-verbatim")
        mock_whisper.assert_called_once_with("/tmp/model/ct2", device="cuda", compute_type="float16")

    @patch("backend.transcribe._get_estonian_model_path")
    @patch("backend.transcribe.WhisperModel")
    def test_et_large_fallback_to_cpu(
        self, mock_whisper: Mock, mock_get_path: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test et-large preset falls back to CPU when GPU fails."""
        mock_get_path.return_value = "/tmp/model/ct2"

        # First call (GPU) raises exception, second call (CPU) succeeds
        mock_cpu_model = Mock()
        mock_whisper.side_effect = [RuntimeError("CUDA not available"), mock_cpu_model]

        with caplog.at_level(logging.WARNING):
            result = pick_model("et-large")

        assert result == mock_cpu_model
        assert mock_whisper.call_count == 2
        assert mock_whisper.call_args_list[0][1] == {
            "device": "cuda",
            "compute_type": "float16",
        }
        assert mock_whisper.call_args_list[1][1] == {"device": "cpu", "compute_type": "int8"}
        assert "GPU initialization failed" in caplog.text

    @patch("backend.transcribe._get_estonian_model_path")
    @patch("backend.transcribe.WhisperModel")
    def test_et_large_fails_on_both_gpu_and_cpu(self, mock_whisper: Mock, mock_get_path: Mock) -> None:
        """Test that ModelLoadError is raised when both GPU and CPU fail."""
        mock_get_path.return_value = "/tmp/model/ct2"
        mock_whisper.side_effect = [
            RuntimeError("CUDA not available"),
            RuntimeError("CPU also failed"),
        ]

        with pytest.raises(ModelLoadError, match="Failed to load model .* on both GPU and CPU"):
            pick_model("et-large")

    @patch("backend.transcribe.snapshot_download")
    @patch("backend.transcribe.WhisperModel")
    def test_turbo_preset(self, mock_whisper: Mock, mock_snapshot: Mock) -> None:
        """Test turbo preset initialization."""
        mock_model = Mock()
        mock_whisper.return_value = mock_model
        mock_snapshot.return_value = "/tmp/model/turbo"

        result = pick_model("turbo")

        assert result == mock_model
        mock_snapshot.assert_called_once_with("Systran/faster-distil-whisper-large-v3")
        mock_whisper.assert_called_once_with("/tmp/model/turbo", device="cuda", compute_type="float16")

    @patch("backend.transcribe.snapshot_download")
    @patch("backend.transcribe.WhisperModel")
    def test_distil_preset(self, mock_whisper: Mock, mock_snapshot: Mock) -> None:
        """Test distil preset initialization."""
        mock_model = Mock()
        mock_whisper.return_value = mock_model
        mock_snapshot.return_value = "/tmp/model/distil"

        result = pick_model("distil")

        assert result == mock_model
        mock_snapshot.assert_called_once_with("Systran/faster-distil-whisper-large-v3")
        mock_whisper.assert_called_once_with("/tmp/model/distil", device="cuda", compute_type="float16")

    @patch("backend.transcribe.snapshot_download")
    @patch("backend.transcribe.WhisperModel")
    def test_large8gb_preset(self, mock_whisper: Mock, mock_snapshot: Mock) -> None:
        """Test large8gb preset initialization."""
        mock_model = Mock()
        mock_whisper.return_value = mock_model
        mock_snapshot.return_value = "/tmp/model/large-v3"

        result = pick_model("large8gb")

        assert result == mock_model
        mock_snapshot.assert_called_once_with("Systran/faster-whisper-large-v3")
        mock_whisper.assert_called_once_with("/tmp/model/large-v3", device="cuda", compute_type="int8_float16")

    @patch("backend.transcribe.WhisperModel")
    def test_fallback_preset(self, mock_whisper: Mock) -> None:
        """Test fallback to small CPU model for unknown presets."""
        mock_model = Mock()
        mock_whisper.return_value = mock_model

        result = pick_model("unknown")

        assert result == mock_model
        mock_whisper.assert_called_once_with("small", device="cpu", compute_type="int8")


class TestRoundFloats:
    """Tests for _round_floats function."""

    def test_round_single_float(self) -> None:
        """Test rounding a single float value."""
        result = _round_floats(3.14159265359, places=3)
        assert result == 3.142

    def test_round_float_in_dict(self) -> None:
        """Test rounding floats in a dictionary."""
        data = {"start": 1.23456, "end": 2.34567, "text": "hello"}
        result = _round_floats(data, places=3)
        assert result == {"start": 1.235, "end": 2.346, "text": "hello"}

    def test_round_floats_in_list(self) -> None:
        """Test rounding floats in a list."""
        data = [1.11111, 2.22222, 3.33333]
        result = _round_floats(data, places=2)
        assert result == [1.11, 2.22, 3.33]

    def test_round_nested_structure(self) -> None:
        """Test rounding floats in nested dict/list structure."""
        data = {
            "segments": [
                {"start": 1.23456, "end": 2.34567},
                {"start": 3.45678, "end": 4.56789},
            ],
            "duration": 10.123456,
        }
        result = _round_floats(data, places=3)
        assert result == {
            "segments": [
                {"start": 1.235, "end": 2.346},
                {"start": 3.457, "end": 4.568},
            ],
            "duration": 10.123,
        }

    def test_preserve_non_float_types(self) -> None:
        """Test that non-float types are preserved unchanged."""
        data = {
            "int": 42,
            "str": "text",
            "bool": True,
            "none": None,
            "float": 1.23456,
        }
        result = _round_floats(data, places=2)
        assert result == {
            "int": 42,
            "str": "text",
            "bool": True,
            "none": None,
            "float": 1.23,
        }

    def test_empty_structures(self) -> None:
        """Test handling of empty structures."""
        assert _round_floats({}) == {}
        assert _round_floats([]) == []
        assert _round_floats("") == ""


class TestSegmentToPayload:
    """Tests for _segment_to_payload function."""

    def test_basic_segment(self) -> None:
        """Test conversion of a basic segment."""
        segment = Mock()
        segment.id = 1
        segment.start = 0.0
        segment.end = 5.123456
        segment.text = "  Hello world  "
        segment.speaker = None

        result = _segment_to_payload(segment)

        assert result == {
            "id": 1,
            "start": 0.0,
            "end": 5.123,
            "text": "Hello world",
        }

    def test_segment_with_speaker(self) -> None:
        """Test segment with speaker information."""
        segment = Mock()
        segment.id = 2
        segment.start = 5.0
        segment.end = 10.5
        segment.text = "Speaker text"
        segment.speaker = "SPEAKER_01"

        result = _segment_to_payload(segment)

        assert result == {
            "id": 2,
            "start": 5.0,
            "end": 10.5,
            "text": "Speaker text",
            "speaker": "SPEAKER_01",
        }

    def test_segment_with_none_values(self) -> None:
        """Test that None values are excluded from payload."""
        segment = Mock()
        segment.id = None
        segment.start = 1.0
        segment.end = None
        segment.text = "text"
        segment.speaker = None

        result = _segment_to_payload(segment)

        assert result == {"start": 1.0, "text": "text"}
        assert "id" not in result
        assert "end" not in result
        assert "speaker" not in result

    def test_segment_text_stripping(self) -> None:
        """Test that segment text is properly stripped."""
        segment = Mock()
        segment.id = 1
        segment.start = 0.0
        segment.end = 1.0
        segment.text = "\n\t  Whitespace everywhere  \t\n"
        segment.speaker = None

        result = _segment_to_payload(segment)

        assert result["text"] == "Whitespace everywhere"


class TestTranscribe:
    """Tests for transcribe function."""

    @patch("backend.transcribe.pick_model")
    def test_transcribe_basic(self, mock_pick_model: Mock) -> None:
        """Test basic transcription."""
        mock_model = Mock()
        mock_pick_model.return_value = mock_model

        # Create mock segments
        mock_segment1 = Mock()
        mock_segment1.id = 0
        mock_segment1.start = 0.0
        mock_segment1.end = 5.0
        mock_segment1.text = "First segment"
        mock_segment1.speaker = None

        mock_segment2 = Mock()
        mock_segment2.id = 1
        mock_segment2.start = 5.0
        mock_segment2.end = 10.0
        mock_segment2.text = "Second segment"
        mock_segment2.speaker = None

        mock_info = Mock()
        mock_info.language = "et"
        mock_info.language_probability = 0.95
        mock_info.duration = 10.0

        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        result = transcribe("/path/to/audio.wav", preset="et-large")

        assert result["audio"] == "audio.wav"
        assert result["language"] == "et"
        assert result["language_probability"] == 0.95
        assert result["duration"] == 10.0
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "First segment"
        assert result["segments"][1]["text"] == "Second segment"

        # Verify transcribe was called with correct parameters
        mock_model.transcribe.assert_called_once_with(
            "/path/to/audio.wav",
            beam_size=5,
            word_timestamps=False,
            language="et",
            task="transcribe",
        )

    @patch("backend.transcribe.pick_model")
    def test_transcribe_non_estonian_model(self, mock_pick_model: Mock) -> None:
        """Test transcription with non-Estonian model (language=None)."""
        mock_model = Mock()
        mock_pick_model.return_value = mock_model

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 5.0

        mock_model.transcribe.return_value = ([], mock_info)

        transcribe("/path/to/audio.wav", preset="turbo")

        # Verify language parameter is None for non-Estonian models
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] is None

    @patch("backend.transcribe.pick_model")
    def test_transcribe_without_duration(self, mock_pick_model: Mock) -> None:
        """Test transcription when duration is None."""
        mock_model = Mock()
        mock_pick_model.return_value = mock_model

        mock_info = Mock()
        mock_info.language = "et"
        mock_info.language_probability = 0.95
        mock_info.duration = None

        mock_model.transcribe.return_value = ([], mock_info)

        result = transcribe("/path/to/audio.wav", preset="et-large")

        assert "duration" not in result
        assert "language" in result

    @patch("backend.transcribe.pick_model")
    def test_transcribe_filename_extraction(self, mock_pick_model: Mock) -> None:
        """Test that only filename is extracted from full path."""
        mock_model = Mock()
        mock_pick_model.return_value = mock_model

        mock_info = Mock()
        mock_info.language = "et"
        mock_info.language_probability = 0.95
        mock_info.duration = 10.0

        mock_model.transcribe.return_value = ([], mock_info)

        result = transcribe("/very/long/path/to/audio.wav", preset="et-large")

        assert result["audio"] == "audio.wav"


class TestTranscribeToJson:
    """Tests for transcribe_to_json function."""

    @patch("backend.transcribe.transcribe")
    @patch("builtins.open", new_callable=mock_open)
    @patch("backend.transcribe.json.dump")
    def test_transcribe_to_json_basic(self, mock_json_dump: Mock, mock_file: Mock, mock_transcribe: Mock) -> None:
        """Test basic JSON file writing."""
        mock_payload = {
            "audio": "test.wav",
            "language": "et",
            "segments": [],
        }
        mock_transcribe.return_value = mock_payload

        transcribe_to_json("test.wav", "test.json", preset="et-large")

        mock_transcribe.assert_called_once_with("test.wav", "et-large")
        mock_file.assert_called_once_with("test.json", "w", encoding="utf-8")
        mock_json_dump.assert_called_once()

        # Verify json.dump was called with correct parameters
        call_args = mock_json_dump.call_args
        assert call_args[0][0] == mock_payload
        assert call_args[1]["ensure_ascii"] is False
        assert call_args[1]["indent"] == 2

    @patch("backend.transcribe.transcribe")
    @patch("builtins.open", new_callable=mock_open)
    def test_transcribe_to_json_with_different_preset(self, mock_file: Mock, mock_transcribe: Mock) -> None:
        """Test JSON writing with different model preset."""
        mock_payload = {"audio": "test.wav", "language": "en"}
        mock_transcribe.return_value = mock_payload

        transcribe_to_json("test.wav", "test.json", preset="turbo")

        mock_transcribe.assert_called_once_with("test.wav", "turbo")
