"""Unit tests for backend.transcribe module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from backend.exceptions import ModelLoadError, ModelNotFoundError
from backend.model_loader import DeviceSelector, ModelLoader
from backend.preprocess.config import PreprocessConfig, TranscriptionConfig
from backend.preprocess.metrics import PreprocessMetrics
from backend.transcribe import (
    TranscriptionMetrics,
    _get_estonian_model_path,
    _round_floats,
    pick_model,
    segment_to_payload,
    transcribe,
    transcribe_and_save,
    transcribe_to_json,
    transcribe_to_text,
)


class RecordingDownloader:
    def __init__(self, return_path: str):
        self.return_path = return_path
        self.calls: list[tuple[str, Any]] = []

    def __call__(self, model_id: str, allow_patterns: Any = None, **kwargs: Any) -> str:  # noqa: ANN401
        self.calls.append((model_id, allow_patterns))
        return self.return_path


class RecordingResolver:
    def __init__(self, resolved_path: str):
        self.resolved_path = resolved_path
        self.calls: list[Any] = []

    def __call__(self, config) -> str:  # noqa: ANN001
        self.calls.append(config)
        return self.resolved_path


class FixedDeviceSelector(DeviceSelector):
    def __init__(self, device: str, compute_type: str):
        super().__init__()
        self.device = device
        self.compute_type = compute_type

    def select(self, config):  # noqa: ANN001
        return self.device, self.compute_type


class RecordingModelFactory:
    def __init__(self, side_effects: list[Any] | None = None, return_value: object | None = None):
        self.side_effects = side_effects or []
        self.return_value = return_value or object()
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, model_path: str, device: str, compute_type: str):
        self.calls.append((model_path, device, compute_type))
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return self.return_value


class FakeSegment:
    def __init__(
        self, seg_id: int | None, start: float | None, end: float | None, text: str, speaker: str | None = None
    ):
        self.id = seg_id
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker


class FakeTranscriptionInfo:
    def __init__(self, language: str | None, language_probability: float | None, duration: float | None):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class RecordingModel:
    def __init__(self, segments, info):  # noqa: ANN001
        self._segments = segments
        self._info = info
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._segments, self._info


class FakePreprocessResult:
    def __init__(
        self,
        output_path: Path,
        duration: float | None = None,
        metrics: PreprocessMetrics | None = None,
    ):
        self.output_path = output_path
        self.output_path = output_path
        self.input_info = (
            type("Info", (), {"duration": duration, "channels": 1, "sample_rate": 16000, "sample_format": "s16"})()
            if duration is not None
            else None
        )
        self.metrics = metrics or PreprocessMetrics(total_duration=0.0, steps=[])
        self.profile = "test"
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class TestGetEstonianModelPath:
    """Tests for _get_estonian_model_path function."""

    def test_success_with_ct2_folder(self, tmp_path: Path) -> None:
        """Test successful download when CT2 folder exists."""
        (tmp_path / "ct2").mkdir()
        downloader = RecordingDownloader(str(tmp_path))

        result = _get_estonian_model_path("TalTechNLP/whisper-test", downloader=downloader, path_cls=Path)

        assert result == str(tmp_path / "ct2")
        assert downloader.calls == [("TalTechNLP/whisper-test", ["ct2/*"])]

    def test_raises_error_when_ct2_folder_missing(self, tmp_path: Path) -> None:
        """Test that ModelNotFoundError is raised when CT2 folder doesn't exist."""
        downloader = RecordingDownloader(str(tmp_path))

        with pytest.raises(ModelNotFoundError, match="CT2 folder not found"):
            _get_estonian_model_path("TalTechNLP/whisper-test", downloader=downloader, path_cls=Path)


class TestPickModel:
    """Tests for pick_model function."""

    @pytest.mark.parametrize(
        "preset,expected_model_id,expected_cache_path",
        [
            ("et-large", "TalTechNLP/whisper-large-v3-turbo-et-verbatim", "/tmp/model/ct2"),
            ("turbo", "Systran/faster-distil-whisper-large-v3", "/tmp/model/turbo"),
            ("distil", "Systran/faster-distil-whisper-large-v3", "/tmp/model/distil"),
            ("large8gb", "Systran/faster-whisper-large-v3", "/tmp/model/large-v3"),
        ],
    )
    def test_preset_loads_correct_model(self, preset: str, expected_model_id: str, expected_cache_path: str) -> None:
        """Test that presets resolve to correct model IDs and load successfully (behavior test)."""
        resolver = RecordingResolver(expected_cache_path)
        factory = RecordingModelFactory(return_value="model")
        loader = ModelLoader(device_selector=FixedDeviceSelector("cuda", "float16"), model_factory=factory)

        # Test behavior: preset should resolve to correct model and load
        result = pick_model(preset, resolver=resolver, loader=loader)

        # Verify model was loaded (behavior contract)
        assert result == "model", f"Preset {preset} should load a model"
        # Verify correct model ID was resolved (behavior contract)
        assert resolver.calls, f"Preset {preset} should call resolver"
        assert resolver.calls[0].model_id == expected_model_id, f"Preset {preset} should resolve to {expected_model_id}"
        # Verify model factory was called with correct path (behavior contract)
        assert factory.calls == [(expected_cache_path, "cuda", "float16")], (
            f"Preset {preset} should load from {expected_cache_path}"
        )

    def test_et_large_fallback_to_cpu(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test et-large preset falls back to CPU when GPU fails (fallback behavior)."""
        resolver = RecordingResolver("/tmp/model/ct2")
        factory = RecordingModelFactory(side_effects=[RuntimeError("CUDA not available"), "cpu_model"])
        loader = ModelLoader(device_selector=FixedDeviceSelector("cuda", "float16"), model_factory=factory)

        with caplog.at_level(logging.WARNING):
            result = pick_model("et-large", resolver=resolver, loader=loader)

        # Test behavior: should fallback to CPU and succeed
        assert result == "cpu_model", "Should fallback to CPU when GPU fails"
        assert factory.calls[0] == ("/tmp/model/ct2", "cuda", "float16"), "Should try GPU first"
        assert factory.calls[1] == ("/tmp/model/ct2", "cpu", "int8"), "Should fallback to CPU"
        assert "GPU initialization failed" in caplog.text, "Should log GPU failure"

    def test_et_large_fails_on_both_gpu_and_cpu(self) -> None:
        """Test that ModelLoadError is raised when both GPU and CPU fail (error behavior)."""
        resolver = RecordingResolver("/tmp/model/ct2")
        factory = RecordingModelFactory(
            side_effects=[RuntimeError("CUDA not available"), RuntimeError("CPU also failed")]
        )
        loader = ModelLoader(device_selector=FixedDeviceSelector("cuda", "float16"), model_factory=factory)

        # Test behavior: should raise error when both devices fail
        with pytest.raises(ModelLoadError, match="Failed to load model .* on both GPU and CPU"):
            pick_model("et-large", resolver=resolver, loader=loader)

    def test_unknown_preset_raises_keyerror(self) -> None:
        """Test that unknown presets raise KeyError (error behavior)."""
        # Test behavior: invalid preset should raise clear error
        with pytest.raises(KeyError, match="Unknown preset 'unknown'"):
            pick_model("unknown")


class TestRoundFloats:
    """Tests for _round_floats function contract: floats rounded, structure preserved."""

    @pytest.mark.parametrize(
        "input_data,expected_output,places",
        [
            # Single float
            (3.14159265359, 3.142, 3),
            # Dictionary with floats
            ({"start": 1.23456, "end": 2.34567, "text": "hello"}, {"start": 1.235, "end": 2.346, "text": "hello"}, 3),
            # List with floats
            ([1.11111, 2.22222, 3.33333], [1.11, 2.22, 3.33], 2),
            # Nested structure (real-world use case)
            (
                {
                    "segments": [
                        {"start": 1.23456, "end": 2.34567},
                        {"start": 3.45678, "end": 4.56789},
                    ],
                    "duration": 10.123456,
                },
                {
                    "segments": [
                        {"start": 1.235, "end": 2.346},
                        {"start": 3.457, "end": 4.568},
                    ],
                    "duration": 10.123,
                },
                3,
            ),
        ],
    )
    def test_rounds_floats_preserves_structure(self, input_data: Any, expected_output: Any, places: int) -> None:
        """Test contract: floats are rounded, data structure is preserved."""
        result = _round_floats(input_data, places=places)
        assert result == expected_output, f"Structure should be preserved, floats rounded to {places} places"

    def test_preserves_non_float_types(self) -> None:
        """Test contract: non-float types are preserved unchanged."""
        data = {
            "int": 42,
            "str": "text",
            "bool": True,
            "none": None,
            "float": 1.23456,
        }
        result = _round_floats(data, places=2)
        # Test behavior: non-float types unchanged, float rounded
        assert result["int"] == 42
        assert result["str"] == "text"
        assert result["bool"] is True
        assert result["none"] is None
        assert result["float"] == 1.23

    def test_handles_empty_structures(self) -> None:
        """Test contract: empty structures are preserved."""
        assert _round_floats({}) == {}
        assert _round_floats([]) == []
        assert _round_floats("") == ""


class TestSegmentToPayload:
    """Tests for segment_to_payload function."""

    def test_basic_segment(self) -> None:
        """Test conversion of a basic segment."""
        segment = FakeSegment(1, 0.0, 5.123456, "  Hello world  ")

        result = segment_to_payload(segment)

        assert result == {
            "id": 1,
            "start": 0.0,
            "end": 5.123,
            "text": "Hello world",
        }

    def test_segment_with_speaker(self) -> None:
        """Test segment with speaker information."""
        segment = FakeSegment(2, 5.0, 10.5, "Speaker text", speaker="SPEAKER_01")

        result = segment_to_payload(segment)

        assert result == {
            "id": 2,
            "start": 5.0,
            "end": 10.5,
            "text": "Speaker text",
            "speaker": "SPEAKER_01",
        }

    def test_segment_with_none_values(self) -> None:
        """Test that None values are excluded from payload."""
        segment = FakeSegment(None, 1.0, None, "text")

        result = segment_to_payload(segment)

        assert result == {"start": 1.0, "text": "text"}
        assert "id" not in result
        assert "end" not in result
        assert "speaker" not in result

    def test_segment_text_stripping(self) -> None:
        """Test that segment text is properly stripped."""
        segment = FakeSegment(1, 0.0, 1.0, "\n\t  Whitespace everywhere  \t\n")

        result = segment_to_payload(segment)

        assert result["text"] == "Whitespace everywhere"


class TestTranscribe:
    """Tests for transcribe function."""

    def test_transcribe_basic(self, tmp_path: Path) -> None:
        """Test basic transcription."""
        processed_path = tmp_path / "processed.wav"
        processed_path.write_text("data")
        segments = [
            FakeSegment(0, 0.0, 5.0, "First segment"),
            FakeSegment(1, 5.0, 10.0, "Second segment"),
        ]
        info = FakeTranscriptionInfo(language="et", language_probability=0.95, duration=10.0)
        model = RecordingModel(segments, info)

        result = transcribe(
            "/path/to/audio.wav",
            preset="et-large",
            preprocess_config_provider=lambda: PreprocessConfig(enabled=False),
            preprocess_runner=lambda path, cfg: FakePreprocessResult(processed_path, duration=10.0),  # noqa: ARG005
            model_picker=lambda preset: model,  # noqa: ARG005
        )

        assert result["audio"] == "audio.wav"
        assert result["language"] == "et"
        assert result["language_probability"] == 0.95
        assert result["duration"] == 10.0
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "First segment"
        assert result["segments"][1]["text"] == "Second segment"

        # Verify transcribe was called with correct parameters
        assert model.calls[0]["args"][0] == str(processed_path)
        call_kwargs = model.calls[0]["kwargs"]
        # Use config defaults to ensure consistency
        default_config = TranscriptionConfig()
        assert call_kwargs["beam_size"] == default_config.beam_size
        assert call_kwargs["word_timestamps"] == default_config.word_timestamps
        assert call_kwargs["language"] == "et"
        assert call_kwargs["task"] == default_config.task

    def test_transcribe_non_estonian_model(self, tmp_path: Path) -> None:
        """Test transcription with non-Estonian model (language=None)."""
        processed_path = tmp_path / "processed.wav"
        processed_path.write_text("data")
        info = FakeTranscriptionInfo(language="en", language_probability=0.99, duration=5.0)
        model = RecordingModel([], info)

        transcribe(
            "/path/to/audio.wav",
            preset="turbo",
            preprocess_config_provider=lambda: PreprocessConfig(enabled=False),
            preprocess_runner=lambda path, cfg: FakePreprocessResult(processed_path),  # noqa: ARG005
            model_picker=lambda preset: model,  # noqa: ARG005
        )

        # Verify language parameter is None for non-Estonian models
        call_kwargs = model.calls[0]["kwargs"]
        assert call_kwargs["language"] is None

    def test_transcribe_without_duration(self, tmp_path: Path) -> None:
        """Test transcription when duration is None."""
        processed_path = tmp_path / "processed.wav"
        processed_path.write_text("data")
        info = FakeTranscriptionInfo(language="et", language_probability=0.95, duration=None)
        model = RecordingModel([], info)

        result = transcribe(
            "/path/to/audio.wav",
            preset="et-large",
            preprocess_config_provider=lambda: PreprocessConfig(enabled=False),
            preprocess_runner=lambda path, cfg: FakePreprocessResult(processed_path),  # noqa: ARG005
            model_picker=lambda preset: model,  # noqa: ARG005
        )

        assert "duration" not in result
        assert "language" in result

    def test_transcribe_filename_extraction(self, tmp_path: Path) -> None:
        """Test that only filename is extracted from full path."""
        processed_path = tmp_path / "processed.wav"
        processed_path.write_text("data")
        info = FakeTranscriptionInfo(language="et", language_probability=0.95, duration=10.0)
        model = RecordingModel([], info)

        result = transcribe(
            "/very/long/path/to/audio.wav",
            preset="et-large",
            preprocess_config_provider=lambda: PreprocessConfig(enabled=False),
            preprocess_runner=lambda path, cfg: FakePreprocessResult(processed_path),  # noqa: ARG005
            model_picker=lambda preset: model,  # noqa: ARG005
        )

        assert result["audio"] == "audio.wav"

    def test_transcribe_emits_metrics(self, tmp_path: Path) -> None:
        """Ensure metrics_collector receives structured data."""
        processed_path = tmp_path / "processed.wav"
        processed_path.write_text("data")
        info = FakeTranscriptionInfo(language="en", language_probability=0.9, duration=12.0)
        model = RecordingModel([], info)
        collected: list[TranscriptionMetrics] = []

        transcribe(
            "/path/to/audio.wav",
            preset="et-large",
            language="en",
            preprocess_config_provider=lambda: PreprocessConfig(enabled=False),
            preprocess_runner=lambda path, cfg: FakePreprocessResult(processed_path, duration=12.0),  # noqa: ARG005, ARG001
            model_picker=lambda preset: model,  # noqa: ARG005
            metrics_collector=collected.append,
        )

        assert collected, "metrics_collector should be invoked"
        metrics = collected[0]

        assert metrics.audio_path == "/path/to/audio.wav"
        assert metrics.preset == "et-large"
        assert metrics.requested_language == "en"
        assert metrics.applied_language == "en"
        assert metrics.detected_language == "en"
        assert metrics.language_probability == 0.9
        assert metrics.audio_duration == 12.0
        assert metrics.preprocess_enabled is False
        assert metrics.preprocess_profile == "test"
        assert metrics.target_sample_rate == 16000
        assert metrics.preprocess_steps == []
        assert metrics.speed_ratio is not None
        assert metrics.total_processing_time >= metrics.transcribe_duration


class TestTranscribeToJson:
    """Tests for transcribe_to_json function."""

    def test_transcribe_to_json_basic(self, tmp_path: Path) -> None:
        """Test basic JSON file writing."""
        payload = {"audio": "test.wav", "language": "et", "segments": []}
        calls: list[tuple[str, str, Any]] = []

        def fake_transcribe(
            audio_path: str, preset: str, language: str | None = None, **_kwargs: Any
        ) -> dict[str, Any]:
            calls.append((audio_path, preset, language))
            return payload

        json_path = tmp_path / "test.json"

        transcribe_to_json("test.wav", json_path, preset="et-large", transcribe_fn=fake_transcribe)

        assert calls == [("test.wav", "et-large", None)]
        assert json.loads(json_path.read_text()) == payload

    def test_transcribe_to_json_with_different_preset(self, tmp_path: Path) -> None:
        """Test JSON writing with different model preset."""
        payload = {"audio": "test.wav", "language": "en"}
        calls: list[tuple[str, str, Any]] = []

        def fake_transcribe(
            audio_path: str, preset: str, language: str | None = None, **_kwargs: Any
        ) -> dict[str, Any]:
            calls.append((audio_path, preset, language))
            return payload

        json_path = tmp_path / "test.json"

        transcribe_to_json("test.wav", json_path, preset="turbo", transcribe_fn=fake_transcribe)

        assert calls == [("test.wav", "turbo", None)]
        assert json.loads(json_path.read_text()) == payload


class TestTranscribeToText:
    """Tests for transcribe_to_text TXT format (range timestamps + optional speaker)."""

    def test_transcribe_to_text_with_speakers(self, tmp_path: Path) -> None:
        """Speaker-present segments emit `[start --> end] SPEAKER_NN: text` lines."""
        payload = {
            "segments": [
                {"start": 0.4, "end": 3.05, "text": " Hello", "speaker": "SPEAKER_00"},
                {"start": 3.1, "end": 5.7, "text": "World", "speaker": "SPEAKER_01"},
            ],
        }

        def fake_transcribe(
            audio_path: str, preset: str, language: str | None = None, **_kwargs: Any
        ) -> dict[str, Any]:  # noqa: ARG001
            return payload

        text_path = tmp_path / "out.txt"
        transcribe_to_text("a.wav", str(text_path), preset="et-large", transcribe_fn=fake_transcribe)

        assert text_path.read_text(encoding="utf-8") == (
            "[00:00:00.40 --> 00:00:03.05] SPEAKER_00: Hello\n[00:00:03.10 --> 00:00:05.70] SPEAKER_01: World\n"
        )

    def test_transcribe_to_text_without_speakers(self, tmp_path: Path) -> None:
        """Speaker-absent segments emit `[start --> end] text` lines (single space, no prefix)."""
        payload = {
            "segments": [
                {"start": 0.4, "end": 3.05, "text": " Hello"},
                {"start": 3.1, "end": 5.7, "text": "World"},
            ],
        }

        def fake_transcribe(
            audio_path: str, preset: str, language: str | None = None, **_kwargs: Any
        ) -> dict[str, Any]:  # noqa: ARG001
            return payload

        text_path = tmp_path / "out.txt"
        transcribe_to_text("a.wav", str(text_path), preset="et-large", transcribe_fn=fake_transcribe)

        assert text_path.read_text(encoding="utf-8") == (
            "[00:00:00.40 --> 00:00:03.05] Hello\n[00:00:03.10 --> 00:00:05.70] World\n"
        )

    def test_transcribe_to_text_handles_hour_boundary(self, tmp_path: Path) -> None:
        """Hours past 00 render in `hh:` slot (regression guard for mm overflow)."""
        payload = {"segments": [{"start": 3661.5, "end": 3662.0, "text": "x"}]}

        def fake_transcribe(
            audio_path: str, preset: str, language: str | None = None, **_kwargs: Any
        ) -> dict[str, Any]:  # noqa: ARG001
            return payload

        text_path = tmp_path / "out.txt"
        transcribe_to_text("a.wav", str(text_path), preset="et-large", transcribe_fn=fake_transcribe)

        line = text_path.read_text(encoding="utf-8")
        assert line.startswith("[01:01:01.50 --> 01:01:02.00]")

    def test_transcribe_to_text_clamps_pathological_timestamps(self, tmp_path: Path) -> None:
        """NaN / negative / inf segment times don't crash the writer."""
        payload = {
            "segments": [
                {"start": -0.01, "end": 0.5, "text": "neg"},
                {"start": float("nan"), "end": 1.0, "text": "nan"},
                {"start": 2.0, "end": float("inf"), "text": "inf"},
            ],
        }

        def fake_transcribe(
            audio_path: str, preset: str, language: str | None = None, **_kwargs: Any
        ) -> dict[str, Any]:  # noqa: ARG001
            return payload

        text_path = tmp_path / "out.txt"
        transcribe_to_text("a.wav", str(text_path), preset="et-large", transcribe_fn=fake_transcribe)

        contents = text_path.read_text(encoding="utf-8")
        assert "[00:00:00.00 --> 00:00:00.50] neg\n" in contents
        assert "[??:??:??.??" in contents
        assert " --> ??:??:??.??]" in contents


class TestTranscribeDiarize:
    """Tests for the diarize plumbing on transcribe()."""

    def test_transcribe_calls_annotate_when_diarize_true(self, tmp_path: Path) -> None:
        """Passing diarize=True invokes the diarize runner exactly once."""
        processed_path = tmp_path / "processed.wav"
        processed_path.write_text("data")
        info = FakeTranscriptionInfo(language="et", language_probability=0.95, duration=5.0)
        model = RecordingModel(
            [FakeSegment(0, 0.0, 5.0, "hello")],
            info,
        )

        runner_calls: list[tuple[str, dict[str, Any]]] = []

        def fake_runner(audio_path: str, **kwargs: Any) -> list[Any]:
            runner_calls.append((audio_path, kwargs))
            return []  # zero turns → annotate returns segments unchanged

        transcribe(
            "/path/to/audio.wav",
            preset="et-large",
            preprocess_config_provider=lambda: PreprocessConfig(enabled=False),
            preprocess_runner=lambda path, cfg: FakePreprocessResult(processed_path, duration=5.0),  # noqa: ARG005
            model_picker=lambda preset: model,  # noqa: ARG005
            diarize=True,
            num_speakers=3,
            diarize_runner=fake_runner,
        )

        assert len(runner_calls) == 1
        audio_path, runner_kwargs = runner_calls[0]
        assert audio_path == str(processed_path)
        assert runner_kwargs.get("num_speakers") == 3

    def test_transcribe_skips_annotate_when_diarize_false(self, tmp_path: Path) -> None:
        """Default diarize=False leaves the runner unused."""
        processed_path = tmp_path / "processed.wav"
        processed_path.write_text("data")
        info = FakeTranscriptionInfo(language="et", language_probability=0.95, duration=5.0)
        model = RecordingModel(
            [FakeSegment(0, 0.0, 5.0, "hello")],
            info,
        )

        def sentinel_runner(*args: Any, **kwargs: Any) -> list[Any]:  # noqa: ARG001
            raise AssertionError("diarize runner should not be called when diarize=False")

        transcribe(
            "/path/to/audio.wav",
            preset="et-large",
            preprocess_config_provider=lambda: PreprocessConfig(enabled=False),
            preprocess_runner=lambda path, cfg: FakePreprocessResult(processed_path, duration=5.0),  # noqa: ARG005
            model_picker=lambda preset: model,  # noqa: ARG005
            diarize=False,
            diarize_runner=sentinel_runner,
        )


class TestTranscribeAndSave:
    """Tests for diarize plumbing on transcribe_and_save (T3)."""

    def test_transcribe_and_save_forwards_diarize_kwargs(self, tmp_path: Path) -> None:
        """transcribe_and_save must forward diarize/num_speakers to transcribe_fn."""
        captured_calls: list[dict[str, Any]] = []

        def fake_transcribe(audio_path: str, preset: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            captured_calls.append(kwargs)
            return {"segments": [], "language": "et"}

        out_path = tmp_path / "out.json"
        transcribe_and_save(
            audio_path="test.wav",
            output_path=str(out_path),
            output_format="json",
            diarize=True,
            num_speakers=3,
            transcribe_fn=fake_transcribe,
        )

        assert len(captured_calls) == 1
        assert captured_calls[0].get("diarize") is True
        assert captured_calls[0].get("num_speakers") == 3

    def test_transcribe_and_save_both_format_forwards_to_both_paths(self, tmp_path: Path) -> None:
        """output_format='both' must forward diarize to both txt and json writers."""
        captured_calls: list[dict[str, Any]] = []

        def fake_transcribe(audio_path: str, preset: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            captured_calls.append(kwargs)
            return {"segments": [], "language": "et"}

        transcribe_and_save(
            audio_path="test.wav",
            output_path=str(tmp_path / "out"),
            output_format="both",
            diarize=True,
            num_speakers=4,
            transcribe_fn=fake_transcribe,
        )

        assert len(captured_calls) == 2
        for call in captured_calls:
            assert call.get("diarize") is True
            assert call.get("num_speakers") == 4
