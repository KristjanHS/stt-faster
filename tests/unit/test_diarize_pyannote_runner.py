"""Unit tests for backend.diarize.pyannote_runner — HF token / license error mapping."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

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

        with patch("pyannote.audio.Pipeline.from_pretrained", side_effect=raise_401):
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

        with patch("pyannote.audio.Pipeline.from_pretrained", side_effect=raise_403):
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
