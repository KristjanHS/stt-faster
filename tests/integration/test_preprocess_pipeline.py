from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from backend.preprocess import PreprocessConfig, preprocess_audio
from backend.preprocess.errors import PreprocessError


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        pytest.skip(f"{name} not available; skipping preprocess integration")


def test_preprocess_pipeline_runs_full_path(tmp_path: Path) -> None:
    source = Path("tests/test.mp3")
    if not source.exists():
        pytest.skip("tests/test.mp3 not available; preprocess integration requires real audio sample")

    _require_tool("ffmpeg")
    _require_tool("ffprobe")

    cfg = PreprocessConfig(enabled=True, temp_dir=str(tmp_path))

    try:
        result = preprocess_audio(source, cfg)
    except PreprocessError as exc:  # pragma: no cover - integration guard
        pytest.fail(f"Preprocess failed in environment: {exc}")

    assert result.output_path.exists()
    step_names = [metric.name for metric in result.metrics.steps]
    assert "downmix_resample" in step_names
    assert "loudnorm" in step_names
    assert "denoise_light" in step_names

    result.cleanup()
