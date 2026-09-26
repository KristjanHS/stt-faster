"""Unit tests for backend.diarize.pyannote_runner — model loading + audio decode."""

from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

from backend.diarize.errors import DiarizationConfigError, DiarizationRuntimeError
from backend.diarize.model import NOT_INSTALLED


def _stub_pipeline_import(from_pretrained_impl: Any) -> Any:
    # Fake for run_pyannote's `import_pipeline` seam, standing in for the lazy
    # `from pyannote.audio import Pipeline`. The real import costs ~5s (loads
    # torch + sklearn + hf_hub); these model-load tests never exercise
    # pyannote internals, so paying it is pure overhead.
    stub_pipeline = type("Pipeline", (), {"from_pretrained": staticmethod(from_pretrained_impl)})
    return lambda: stub_pipeline


def _missing_model(_env: Any) -> Path:
    raise DiarizationConfigError(NOT_INSTALLED)


class TestRunPyannoteModelLoad:
    def test_missing_model_raises_config_error_before_importing_pyannote(self) -> None:
        from backend.diarize.pyannote_runner import run_pyannote

        def no_import() -> Any:
            raise AssertionError("pyannote must not be imported when the model is missing")

        with pytest.raises(DiarizationConfigError, match="Speaker model not installed"):
            run_pyannote("fake.wav", env={}, import_pipeline=no_import, resolve_model=_missing_model)

    def test_loads_resolved_dir_without_token(self, tmp_path: Path) -> None:
        from backend.diarize.pyannote_runner import run_pyannote

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        with pytest.raises(DiarizationConfigError, match="returned None"):
            run_pyannote(
                "fake.wav",
                env={"HF_TOKEN": "must-not-be-sent"},
                import_pipeline=_stub_pipeline_import(record),
                resolve_model=lambda _env: tmp_path,
            )
        assert calls == [((str(tmp_path),), {})]


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


class _FakeClock:
    """Settable stand-in for ``time.time`` fed to the hook's ``clock`` seam."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fake_clock() -> _FakeClock:
    return _FakeClock()


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
        self, fake_clock: _FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        fake_clock.now = 1000.0
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None, clock=fake_clock) as hook:
            hook("embeddings", None, total=None, completed=None)
            fake_clock.now += 30  # under throttle
            hook("embeddings", None, total=100, completed=10)
            fake_clock.now += 31  # cumulative 61s — throttle expires
            hook("embeddings", None, total=100, completed=20)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert len(msgs) == 2
        assert "embeddings" in msgs[0] and "elapsed 0.0 min" in msgs[0]
        assert "embeddings 20/100" in msgs[1] and "(20.0%)" in msgs[1]

    def test_step_change_always_logs_regardless_of_throttle(
        self, fake_clock: _FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        fake_clock.now = 2000.0
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None, clock=fake_clock) as hook:
            hook("speaker_segmentation", None, total=None, completed=None)
            fake_clock.now += 5  # well under throttle, but step changes
            hook("embeddings", None, total=None, completed=None)
            fake_clock.now += 5
            hook("speaker_counting", None, total=None, completed=None)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert len(msgs) == 3
        assert "speaker_segmentation" in msgs[0]
        assert "embeddings" in msgs[1]
        assert "speaker_counting" in msgs[2]

    def test_line_shape_with_and_without_progress_quantities(
        self, fake_clock: _FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        fake_clock.now = 3000.0
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None, clock=fake_clock) as hook:
            hook("embeddings", None, total=None, completed=None)
            fake_clock.now += 61
            hook("embeddings", None, total=4, completed=1)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert msgs[0] == "⌛ Diarization progress: embeddings, elapsed 0.0 min"
        assert msgs[1] == "⌛ Diarization progress: embeddings 1/4 (25.0%), elapsed 1.0 min"

    def test_on_progress_gets_every_entry_and_quantity_unthrottled(self, fake_clock: _FakeClock) -> None:
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        calls: list[tuple[str, int | None, int | None]] = []
        with _DiarizeProgressHook(
            audio_duration=None, clock=fake_clock, on_progress=lambda *a: calls.append(a)
        ) as hook:
            hook("segmentation", None, total=None, completed=None)
            hook("segmentation", None, total=4, completed=1)  # under the log throttle, still forwarded
            hook("segmentation", None, total=None, completed=None)  # no quantities: nothing new
            hook("embeddings", None, total=4, completed=0)  # entry wins over its 0/N

        assert calls == [("segmentation", None, None), ("segmentation", 1, 4), ("embeddings", None, None)]

    def test_a_failing_on_progress_does_not_abort_the_run(self, fake_clock: _FakeClock) -> None:
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        def broken(*_args: object) -> None:
            raise OSError("stdout closed")

        with _DiarizeProgressHook(audio_duration=None, clock=fake_clock, on_progress=broken) as hook:
            hook("segmentation", None, total=None, completed=None)
            hook("segmentation", None, total=4, completed=1)

    def test_entry_with_zero_progress_logs_no_quantities_and_skips_redundant_marker(
        self, fake_clock: _FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression: pyannote can emit ``(completed=0, total=N)`` at stage entry
        and follow up with ``(None, None)`` seconds later. Pre-fix we logged
        both — once as ``"segmentation 0/3509 (0.0%)"`` and once as a redundant
        ``"segmentation, elapsed 0.1 min"``. Now: one entry line, no quantities,
        and the follow-up marker is suppressed.
        """
        from backend.diarize import pyannote_runner
        from backend.diarize.pyannote_runner import _DiarizeProgressHook

        fake_clock.now = 4000.0
        caplog.set_level(logging.INFO, logger=pyannote_runner.LOGGER.name)

        with _DiarizeProgressHook(audio_duration=None, clock=fake_clock) as hook:
            hook("segmentation", None, total=3509, completed=0)
            fake_clock.now += 6
            hook("segmentation", None, total=None, completed=None)

        msgs = [r.getMessage() for r in caplog.records if r.name == pyannote_runner.LOGGER.name]
        assert msgs == ["⌛ Diarization progress: segmentation, elapsed 0.0 min"]


def test_import_opts_out_of_pyannote_metrics() -> None:
    # Fresh interpreter with an explicit env lacking the var: the module-level
    # opt-out must set it on import (no reload / os.environ mutation here).
    env = {k: v for k, v in os.environ.items() if k != "PYANNOTE_METRICS_ENABLED"}
    probe = (
        "import os, sys\n"
        "import backend.diarize.pyannote_runner\n"
        "sys.stdout.write(os.environ.get('PYANNOTE_METRICS_ENABLED', '<unset>'))\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout == "0"


class _Segment:
    start = 0.0
    end = 1.0


class _Annotation:
    def itertracks(self, *, yield_label: bool) -> list[tuple[Any, None, str]]:
        return [(_Segment(), None, "SPEAKER_00")]


class _FakePipeline:
    """Records each device it is moved to; the first ``fail_on`` call on that device raises."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.devices: list[str] = ["cpu"]
        self.fail_on = fail_on

    def to(self, device: Any) -> None:
        self.devices.append(str(device))

    def __call__(self, _audio: Any, **_kwargs: Any) -> Any:
        if self.devices[-1] == self.fail_on:
            raise RuntimeError("cuDNN version incompatibility")
        return type("DiarizeOutput", (), {"speaker_diarization": _Annotation()})()


class TestRunPyannoteDevice:
    def _run(self, tmp_path: Path, pipeline: _FakePipeline, env: dict[str, str], cuda: bool) -> list[Any]:
        from backend.diarize.pyannote_runner import run_pyannote

        wav = tmp_path / "a.wav"
        _write_silent_wav(wav, sample_rate=16000, channels=1, seconds=0.1)
        return run_pyannote(
            str(wav),
            env=env,
            import_pipeline=_stub_pipeline_import(lambda _dir: pipeline),
            resolve_model=lambda _env: tmp_path,
            cuda_available=lambda: cuda,
        )

    def test_gpu_when_cuda_available(self, tmp_path: Path) -> None:
        pipeline = _FakePipeline()
        assert len(self._run(tmp_path, pipeline, {}, cuda=True)) == 1
        assert pipeline.devices == ["cpu", "cuda"]

    def test_stt_device_cpu_keeps_it_off_the_gpu(self, tmp_path: Path) -> None:
        pipeline = _FakePipeline()
        self._run(tmp_path, pipeline, {"STT_DEVICE": "cpu"}, cuda=True)
        assert pipeline.devices == ["cpu"]

    def test_gpu_failure_retries_on_cpu(self, tmp_path: Path) -> None:
        pipeline = _FakePipeline(fail_on="cuda")
        assert len(self._run(tmp_path, pipeline, {}, cuda=True)) == 1
        assert pipeline.devices == ["cpu", "cuda", "cpu"]

    def test_cpu_failure_is_a_runtime_error(self, tmp_path: Path) -> None:
        with pytest.raises(DiarizationRuntimeError, match="cuDNN"):
            self._run(tmp_path, _FakePipeline(fail_on="cpu"), {}, cuda=False)
