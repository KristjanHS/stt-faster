from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest

from backend.preprocess import PreprocessConfig, preprocess_audio
from backend.preprocess.errors import PreprocessError


def test_preprocess_pipeline_runs_full_path(
    tmp_path: Path,
    real_audio_sample: Path,
    ffmpeg_tooling: Tuple[str, str],
) -> None:
    ffmpeg_path, ffprobe_path = ffmpeg_tooling
    assert ffmpeg_path and ffprobe_path

    cfg = PreprocessConfig(enabled=True, temp_dir=str(tmp_path))

    try:
        result = preprocess_audio(real_audio_sample, cfg)
    except PreprocessError as exc:  # pragma: no cover - integration guard
        pytest.fail(f"Preprocess failed in environment: {exc}")

    assert result.output_path.exists()
    step_names = [metric.name for metric in result.metrics.steps]
    assert "downmix_resample" in step_names
    assert "loudnorm" in step_names
    assert "denoise_light" in step_names

    result.cleanup()
