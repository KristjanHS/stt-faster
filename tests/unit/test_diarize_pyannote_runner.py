"""Unit tests for backend.diarize.pyannote_runner — HF token / license error mapping."""

from __future__ import annotations

import logging
import struct
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from backend.diarize.errors import DiarizationConfigError


def _make_http_error(status_code: int) -> Exception:
    from huggingface_hub.errors import HfHubHTTPError

    # Bypass HfHubHTTPError.__init__ (which reads response.headers); set attrs directly.
    exc = HfHubHTTPError.__new__(HfHubHTTPError)
    Exception.__init__(exc, f"{status_code} error")
    response = type("R", (), {"status_code": status_code, "headers": {}})()
    exc.response = response  # type: ignore[assignment]
    exc.server_message = None
    return exc


def _install_pyannote_stub(monkeypatch: pytest.MonkeyPatch, from_pretrained_impl: Any) -> None:
    # Stub pyannote.audio in sys.modules so run_pyannote's lazy
    # `from pyannote.audio import Pipeline` resolves to a fake. The real import
    # costs ~5s (loads torch + sklearn + hf_hub); these error-mapping tests
    # never exercise pyannote internals, so paying it is pure overhead.
    stub_pipeline = type("Pipeline", (), {"from_pretrained": staticmethod(from_pretrained_impl)})
    stub_audio = types.ModuleType("pyannote.audio")
    stub_audio.Pipeline = stub_pipeline  # type: ignore[attr-defined]
    stub_root = types.ModuleType("pyannote")
    stub_root.audio = stub_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", stub_root)
    monkeypatch.setitem(sys.modules, "pyannote.audio", stub_audio)


class TestRunPyannoteErrors:
    def test_missing_hf_token_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.diarize.pyannote_runner import run_pyannote

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

        with pytest.raises(DiarizationConfigError) as exc_info:
            run_pyannote("fake.wav")
        assert "HF_TOKEN" in str(exc_info.value)
        assert "docs/diarization_setup.md" in str(exc_info.value)

    def test_huggingface_401_maps_to_config_error_with_token_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.diarize.pyannote_runner import run_pyannote

        monkeypatch.setenv("HF_TOKEN", "fake-token")

        def raise_401(*_: Any, **__: Any) -> Any:
            raise _make_http_error(401)

        _install_pyannote_stub(monkeypatch, raise_401)
        with pytest.raises(DiarizationConfigError) as exc_info:
            run_pyannote("fake.wav")
        msg = str(exc_info.value)
        assert "401" in msg
        assert "huggingface.co/settings/tokens" in msg
        assert "docs/diarization_setup.md" in msg

    def test_huggingface_403_maps_to_config_error_with_model_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.diarize.pyannote_runner import run_pyannote

        monkeypatch.setenv("HF_TOKEN", "fake-token")

        def raise_403(*_: Any, **__: Any) -> Any:
            raise _make_http_error(403)

        _install_pyannote_stub(monkeypatch, raise_403)
        with pytest.raises(DiarizationConfigError) as exc_info:
            run_pyannote("fake.wav")
        msg = str(exc_info.value)
        assert "403" in msg
        assert "pyannote/speaker-diarization-community-1" in msg
        assert "docs/diarization_setup.md" in msg


def _write_silent_wav(path: Path, *, sample_rate: int, channels: int, seconds: float) -> None:
    """Write a stdlib-only PCM-16 WAV (silence) for PyAV decode tests."""
    n_frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{n_frames * channels}h", *([0] * (n_frames * channels))))


class TestLoadAudioTensor:
    """`_load_audio_tensor` decodes via PyAV → mono float32 (channel, time)."""

    def test_mono_input_round_trips_shape_dtype_and_sample_rate(self, tmp_path: Path) -> None:
        import torch

        from backend.diarize.pyannote_runner import _load_audio_tensor

        path = tmp_path / "mono_16k_1s.wav"
        _write_silent_wav(path, sample_rate=16000, channels=1, seconds=1.0)

        waveform, sample_rate = _load_audio_tensor(str(path))

        assert sample_rate == 16000
        assert isinstance(waveform, torch.Tensor)
        assert str(waveform.dtype) == "torch.float32"
        assert waveform.ndim == 2
        # PyAV resample chunking can drop a sample or two at boundaries — allow a small window.
        assert waveform.shape[0] == 1
        assert 15990 <= waveform.shape[1] <= 16010

    def test_stereo_input_downmixes_to_mono(self, tmp_path: Path) -> None:
        from backend.diarize.pyannote_runner import _load_audio_tensor

        path = tmp_path / "stereo_22k.wav"
        _write_silent_wav(path, sample_rate=22050, channels=2, seconds=0.5)

        waveform, sample_rate = _load_audio_tensor(str(path))

        assert sample_rate == 22050
        # Stereo → mono via PyAV's resampler (layout="mono")
        assert waveform.shape[0] == 1
        assert waveform.shape[1] > 0


class TestDiarizeProgressHook:
    """Cadence + line-shape behaviour for ``_DiarizeProgressHook``.

    The hook's contract with pyannote is the gating piece of this module's
    progress logging — once shipped, regressions show up only in production
    log diffs. These tests pin the three behaviours that distinguish the
    hook from the deleted wall-clock heartbeat: step transitions always log,
    in-stage updates throttle to 60s, and the two log line shapes are
    selected by the presence of ``(completed, total)``.
    """

    def test_transition_logs_immediately_then_throttles_within_60s(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        clock = [1000.0]
        monkeypatch.setattr(pyannote_runner.time, "time", lambda: clock[0])
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None) as hook:
            hook("embeddings", None, total=None, completed=None)
            clock[0] += 30  # under throttle
            hook("embeddings", None, total=100, completed=10)
            clock[0] += 31  # cumulative 61s — throttle expires
            hook("embeddings", None, total=100, completed=20)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert len(msgs) == 2
        assert "embeddings" in msgs[0] and "elapsed 0.0 min" in msgs[0]
        assert "embeddings 20/100" in msgs[1] and "(20.0%)" in msgs[1]

    def test_step_change_always_logs_regardless_of_throttle(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        clock = [2000.0]
        monkeypatch.setattr(pyannote_runner.time, "time", lambda: clock[0])
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None) as hook:
            hook("speaker_segmentation", None, total=None, completed=None)
            clock[0] += 5  # well under throttle, but step changes
            hook("embeddings", None, total=None, completed=None)
            clock[0] += 5
            hook("speaker_counting", None, total=None, completed=None)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert len(msgs) == 3
        assert "speaker_segmentation" in msgs[0]
        assert "embeddings" in msgs[1]
        assert "speaker_counting" in msgs[2]

    def test_line_shape_with_and_without_progress_quantities(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        clock = [3000.0]
        monkeypatch.setattr(pyannote_runner.time, "time", lambda: clock[0])
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None) as hook:
            hook("embeddings", None, total=None, completed=None)
            clock[0] += 61
            hook("embeddings", None, total=4, completed=1)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert msgs[0] == "⌛ Diarization progress: embeddings, elapsed 0.0 min"
        assert msgs[1] == "⌛ Diarization progress: embeddings 1/4 (25.0%), elapsed 1.0 min"

    def test_entry_with_zero_progress_logs_no_quantities_and_skips_redundant_marker(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression: pyannote can emit ``(completed=0, total=N)`` at stage entry
        and follow up with ``(None, None)`` seconds later. Pre-fix we logged
        both — once as ``"segmentation 0/3509 (0.0%)"`` and once as a redundant
        ``"segmentation, elapsed 0.1 min"``. Now: one entry line, no quantities,
        and the follow-up marker is suppressed.
        """
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        clock = [4000.0]
        monkeypatch.setattr(pyannote_runner.time, "time", lambda: clock[0])
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None) as hook:
            hook("segmentation", None, total=3509, completed=0)
            clock[0] += 6
            hook("segmentation", None, total=None, completed=None)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert msgs == ["⌛ Diarization progress: segmentation, elapsed 0.0 min"]
