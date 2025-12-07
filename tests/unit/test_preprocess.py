from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.metrics import StepMetrics
from backend.preprocess.orchestrator import PreprocessResult, preprocess_audio


class _SentinelTempDir:
    """TemporaryDirectory shim that tracks cleanup."""

    def __init__(self, base: Path):
        self.name = str(base)
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


def test_config_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STT_PREPROCESS_ENABLED", raising=False)
    monkeypatch.delenv("STT_PREPROCESS_TARGET_SR", raising=False)
    monkeypatch.delenv("STT_PREPROCESS_TARGET_CH", raising=False)
    monkeypatch.delenv("STT_PREPROCESS_TMP_DIR", raising=False)

    cfg = PreprocessConfig.from_env()

    assert cfg.enabled is False
    assert cfg.target_sample_rate == 16_000
    assert cfg.target_channels == 1
    assert cfg.temp_dir is None


def test_config_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STT_PREPROCESS_ENABLED", "1")
    monkeypatch.setenv("STT_PREPROCESS_TARGET_SR", "8000")
    monkeypatch.setenv("STT_PREPROCESS_TARGET_CH", "2")
    monkeypatch.setenv("STT_PREPROCESS_TMP_DIR", "/tmp/foo")

    cfg = PreprocessConfig.from_env()

    assert cfg.enabled is True
    assert cfg.target_sample_rate == 8000
    assert cfg.target_channels == 2
    assert cfg.temp_dir == "/tmp/foo"


def test_inspect_audio_parses_ffprobe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_audio = tmp_path / "sample.wav"
    fake_audio.write_bytes(b"data")

    payload: Dict[str, Any] = {
        "streams": [
            {
                "sample_rate": "44100",
                "channels": 2,
                "duration": "1.23",
                "sample_fmt": "s16",
            }
        ]
    }

    def fake_check_output(cmd: list[str], stderr: Any, text: bool) -> str:  # noqa: ARG001
        return json.dumps(payload)

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    info = inspect_audio(fake_audio)

    assert info == AudioInfo(channels=2, sample_rate=44100, duration=1.23, sample_format="s16")


def test_preprocess_audio_disabled_passthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_audio = tmp_path / "input.wav"
    fake_audio.write_bytes(b"data")

    dummy_info = AudioInfo(channels=1, sample_rate=16000, duration=0.5, sample_format="s16")
    monkeypatch.setattr("backend.preprocess.orchestrator.inspect_audio", lambda _: dummy_info)

    cfg = PreprocessConfig(enabled=False)
    result = preprocess_audio(fake_audio, cfg)

    assert result.output_path == fake_audio
    assert result.input_info == dummy_info
    assert result.metrics.total_duration == 0.0
    assert result.metrics.steps == []


def test_preprocess_audio_runs_downmix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_audio = tmp_path / "input.wav"
    fake_audio.write_bytes(b"data")

    dummy_info = AudioInfo(channels=2, sample_rate=44100, duration=0.5, sample_format="s16")
    monkeypatch.setattr("backend.preprocess.orchestrator.inspect_audio", lambda _: dummy_info)

    calls: dict[str, Any] = {}

    def fake_step(input_path: Path, output_path: Path, target_sample_rate: int, target_channels: int) -> StepMetrics:
        calls["input"] = input_path
        calls["output"] = output_path
        calls["sr"] = target_sample_rate
        calls["ch"] = target_channels
        output_path.write_bytes(b"processed")  # ensure path exists
        return StepMetrics(name="downmix_resample", backend="fake", duration=0.01)

    monkeypatch.setattr("backend.preprocess.orchestrator.downmix_and_resample", fake_step)

    temp_root = tmp_path / "custom_temp"
    temp_root.mkdir()
    temp_dir = _SentinelTempDir(base=temp_root)
    monkeypatch.setattr("tempfile.TemporaryDirectory", lambda prefix, dir=None: temp_dir)  # type: ignore[call-arg]

    cfg = PreprocessConfig(enabled=True, target_sample_rate=8000, target_channels=1, temp_dir=str(temp_root))
    result: PreprocessResult = preprocess_audio(fake_audio, cfg)

    assert result.output_path.parent == temp_root
    assert result.output_path.exists()
    assert result.input_info == dummy_info
    assert result.metrics.steps and result.metrics.steps[0].name == "downmix_resample"
    assert calls["input"] == fake_audio
    assert calls["sr"] == 8000
    assert calls["ch"] == 1

    result.cleanup()
    assert temp_dir.cleaned is True
