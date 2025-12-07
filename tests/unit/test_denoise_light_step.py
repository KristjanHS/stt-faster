from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.steps import denoise_light


def test_apply_light_denoise_invokes_spectral_gate(tmp_path: Path) -> None:  # noqa: ANN001
    input_path = tmp_path / "input.wav"
    samples = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    sf.write(input_path, samples, 16_000, subtype="PCM_16")

    calls: dict[str, object] = {}

    def fake_reducer(y, sr, **kwargs):  # noqa: ANN001
        calls["audio_dtype"] = getattr(y, "dtype", None)
        calls["rate"] = sr
        calls["kwargs"] = kwargs
        return y

    output_path = tmp_path / "output.wav"
    metric = denoise_light.apply_light_denoise(
        input_path,
        output_path,
        sample_rate=16_000,
        noise_reducer=fake_reducer,
    )

    assert metric.name == "denoise_light"
    assert output_path.exists()
    assert calls["rate"] == 16_000
    assert calls["audio_dtype"] == np.float32
    assert calls["kwargs"]["stationary"] is True
    assert calls["kwargs"]["use_tqdm"] is False


def test_apply_light_denoise_rejects_rate_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    samples = np.zeros(800, dtype=np.float32)
    sf.write(input_path, samples, 8_000, subtype="PCM_16")

    with pytest.raises(StepExecutionError) as excinfo:
        denoise_light.apply_light_denoise(input_path, tmp_path / "out.wav", sample_rate=16_000)

    assert "unexpected sample rate" in str(excinfo.value).lower()


def test_apply_light_denoise_wraps_processing_error(tmp_path: Path) -> None:  # noqa: ANN001
    input_path = tmp_path / "input.wav"
    samples = np.ones(1600, dtype=np.float32)
    sf.write(input_path, samples, 16_000, subtype="PCM_16")

    def failing_reducer(*args, **kwargs):  # noqa: ANN001,ARG001
        raise RuntimeError("spectral gate failed")

    with pytest.raises(StepExecutionError) as excinfo:
        denoise_light.apply_light_denoise(
            input_path,
            tmp_path / "out.wav",
            sample_rate=16_000,
            noise_reducer=failing_reducer,
        )

    assert "noisereduce failed" in str(excinfo.value).lower()


def test_apply_light_denoise_wraps_write_error(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    samples = np.ones(1600, dtype=np.float32)
    sf.write(input_path, samples, 16_000, subtype="PCM_16")

    def _boom(*args, **kwargs):  # noqa: ANN001,ARG001
        raise OSError("disk full")

    def passthrough(y, sr, **kwargs):  # noqa: ANN001,ARG001
        return y

    with pytest.raises(StepExecutionError) as excinfo:
        denoise_light.apply_light_denoise(
            input_path,
            tmp_path / "out.wav",
            sample_rate=16_000,
            noise_reducer=passthrough,
            writer=_boom,
        )

    assert "failed to write audio" in str(excinfo.value).lower()
