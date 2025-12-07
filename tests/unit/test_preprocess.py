from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Dict

import pytest

import backend.preprocess.orchestrator as orchestrator
from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import PreprocessError
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.orchestrator import PreprocessResult, preprocess_audio


def test_config_from_env_defaults() -> None:
    cfg = PreprocessConfig.from_env(env={})

    assert cfg.enabled is False
    assert cfg.target_sample_rate == 16_000
    assert cfg.target_channels == 1
    assert cfg.temp_dir is None
    assert cfg.profile == "auto"


def test_config_from_env_overrides() -> None:
    cfg = PreprocessConfig.from_env(
        env={
            "STT_PREPROCESS_ENABLED": "1",
            "STT_PREPROCESS_TARGET_SR": "8000",
            "STT_PREPROCESS_TARGET_CH": "2",
            "STT_PREPROCESS_TMP_DIR": "/tmp/foo",
            "STT_PREPROCESS_PROFILE": "cpu",
        }
    )

    assert cfg.enabled is True
    assert cfg.target_sample_rate == 8000
    assert cfg.target_channels == 2
    assert cfg.temp_dir == "/tmp/foo"
    assert cfg.profile == "cpu"


def test_inspect_audio_parses_ffprobe(tmp_path: Path) -> None:
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

    info = inspect_audio(fake_audio, run_command=fake_check_output)

    assert info == AudioInfo(channels=2, sample_rate=44100, duration=1.23, sample_format="s16")


def test_inspect_audio_formats_ffprobe_errors(tmp_path: Path) -> None:
    fake_audio = tmp_path / "broken.wav"
    fake_audio.write_bytes(b"data")

    def fake_check_output(cmd: list[str], stderr: Any, text: bool) -> str:  # noqa: ARG001
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="[mov,mp4,m4a] moov atom not found\n/tmp/foo: Invalid data found when processing input\n{\n}\n",
        )

    with pytest.raises(PreprocessError) as excinfo:
        inspect_audio(fake_audio, run_command=fake_check_output)

    message = str(excinfo.value)
    assert "ffprobe failed" in message
    assert "moov atom not found" in message
    assert "Invalid data" in message
    assert "\n" not in message


def test_preprocess_audio_disabled_passthrough(
    dummy_audio_info_mono,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    fake_audio = tmp_path / "input.wav"
    fake_audio.write_bytes(b"data")

    cfg = PreprocessConfig(enabled=False)
    result = preprocess_audio(fake_audio, cfg, inspector=lambda _: dummy_audio_info_mono)

    assert result.output_path == fake_audio
    assert result.input_info == dummy_audio_info_mono
    assert result.metrics.total_duration == 0.0
    assert result.metrics.steps == []
    assert result.metrics.snr_delta is None
    assert result.profile == "disabled"


def test_preprocess_audio_runs_downmix(preprocess_pipeline_stubs, tmp_path: Path) -> None:  # noqa: ANN001
    fake_audio = tmp_path / "input.wav"
    fake_audio.write_bytes(b"data")

    calls: dict[str, Any] = preprocess_pipeline_stubs["calls"]
    temp_dir = preprocess_pipeline_stubs["temp_dir"]
    temp_root = preprocess_pipeline_stubs["temp_root"]

    cfg = PreprocessConfig(enabled=True, target_sample_rate=8000, target_channels=1, temp_dir=str(temp_root))
    result: PreprocessResult = preprocess_audio(
        fake_audio,
        cfg,
        inspector=preprocess_pipeline_stubs["inspector"],
        downmix_fn=preprocess_pipeline_stubs["downmix_fn"],
        loudnorm_fn=preprocess_pipeline_stubs["loudnorm_fn"],
        denoise_fn=preprocess_pipeline_stubs["denoise_fn"],
        snr_estimator=preprocess_pipeline_stubs["snr_estimator"],
        temp_dir_factory=preprocess_pipeline_stubs["temp_dir_factory"],
        gpu_check=preprocess_pipeline_stubs["gpu_check"],
    )

    assert result.output_path.parent == temp_root
    assert result.output_path.exists()
    assert result.input_info is not None
    assert result.metrics.steps and result.metrics.steps[0].name == "downmix_resample"
    assert result.metrics.steps[1].name == "loudnorm"
    assert result.metrics.steps[2].name == "denoise_light"
    assert calls["input"] == fake_audio
    assert calls["loudnorm_input"] == temp_root / "downmixed.wav"
    assert calls["loudnorm_sr"] == 8000
    assert calls["denoise_input"] == temp_root / "loudnorm.wav"
    assert calls["sr"] == 8000
    assert calls["ch"] == 1
    assert calls["denoise_sr"] == 8000
    assert result.metrics.snr_before == 1.0
    assert result.metrics.snr_after == 3.0
    assert result.metrics.snr_delta == 2.0
    assert result.profile == "gpu"

    result.cleanup()
    assert temp_dir.cleaned is True


def test_resolve_profile_prefers_gpu() -> None:
    profile, note = orchestrator._resolve_profile("auto", gpu_check=lambda: True)

    assert profile == "gpu"
    assert note is None


def test_resolve_profile_falls_back_to_cpu() -> None:
    profile, note = orchestrator._resolve_profile("gpu", gpu_check=lambda: False)

    assert profile == "cpu"
    assert note is not None
    assert "falling back to cpu" in note.lower()
