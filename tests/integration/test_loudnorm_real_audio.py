from __future__ import annotations

from pathlib import Path

import pytest

from backend.preprocess import PreprocessConfig, preprocess_audio
from backend.preprocess.errors import PreprocessError


def test_loudnorm_runs_on_real_audio(tmp_path: Path, real_audio_sample: Path, ffmpeg_tooling: tuple[str, str]) -> None:
    ffmpeg_path, ffprobe_path = ffmpeg_tooling
    assert ffmpeg_path and ffprobe_path

    cfg = PreprocessConfig(enabled=True, temp_dir=str(tmp_path))

    try:
        result = preprocess_audio(real_audio_sample, cfg)
    except PreprocessError as exc:  # pragma: no cover - integration guard
        pytest.fail(f"Preprocess failed in environment: {exc}")

    assert result.output_path.exists()
    assert any(metric.name == "loudnorm" for metric in result.metrics.steps)
    assert result.metrics.snr_before is not None
    assert result.metrics.snr_after is not None
    assert result.metrics.snr_delta is not None

    result.cleanup()
