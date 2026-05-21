"""Unit tests for backend.diarize.pyannote_runner — HF token / license error mapping."""

from __future__ import annotations

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
        assert "pyannote/speaker-diarization-3.1" in msg
        assert "docs/diarization_setup.md" in msg
