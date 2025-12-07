from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from backend.preprocess.io import AudioInfo
from backend.preprocess.metrics import StepMetrics

from backend.database import TranscriptionDatabase


@pytest.fixture(scope="session", autouse=True)
def _disable_network_for_unit_tests() -> None:
    """Block real sockets for unit tests; allow Unix sockets for pytest internals."""
    from pytest_socket import disable_socket

    disable_socket(allow_unix_socket=True)


@pytest.fixture
def temp_db() -> Iterator[TranscriptionDatabase]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db = TranscriptionDatabase(db_path)
    yield db
    db.close()

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_folder() -> Iterator[Path]:
    """Create a temporary folder for testing."""
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class _SentinelTempDir:
    """TemporaryDirectory shim that tracks cleanup."""

    def __init__(self, base: Path):
        self.name = str(base)
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


@pytest.fixture
def dummy_audio_info_mono() -> AudioInfo:
    return AudioInfo(channels=1, sample_rate=16000, duration=0.5, sample_format="s16")


@pytest.fixture
def dummy_audio_info_stereo() -> AudioInfo:
    return AudioInfo(channels=2, sample_rate=44100, duration=0.5, sample_format="s16")


@pytest.fixture
def preprocess_pipeline_stubs(
    tmp_path: Path,
    dummy_audio_info_stereo: AudioInfo,
):
    """Stub preprocess pipeline steps and temp dir handling."""
    calls: dict[str, object] = {}

    def fake_inspect(_: Path) -> AudioInfo:
        return dummy_audio_info_stereo

    def fake_downmix(input_path: Path, output_path: Path, target_sample_rate: int, target_channels: int) -> StepMetrics:
        calls["input"] = input_path
        calls["output"] = output_path
        calls["sr"] = target_sample_rate
        calls["ch"] = target_channels
        output_path.write_bytes(b"processed")  # ensure path exists
        return StepMetrics(name="downmix_resample", backend="fake", duration=0.01)

    def fake_loudnorm(input_path: Path, output_path: Path, sample_rate: int) -> StepMetrics:
        calls["loudnorm_input"] = input_path
        calls["loudnorm_sr"] = sample_rate
        output_path.write_bytes(b"normalized")
        return StepMetrics(name="loudnorm", backend=str(input_path), duration=0.02)

    def fake_denoise(input_path: Path, output_path: Path, sample_rate: int) -> StepMetrics:
        calls["denoise_input"] = input_path
        calls["denoise_sr"] = sample_rate
        output_path.write_bytes(b"denoised")
        return StepMetrics(name="denoise_light", backend=str(input_path), duration=0.01)

    def fake_snr(path: Path, sample_rate: int, channels: int) -> float:  # noqa: ARG001
        if "input" in calls and path == calls["input"]:
            return 1.0
        return 3.0

    temp_root = tmp_path / "custom_temp"
    temp_root.mkdir()
    temp_dir = _SentinelTempDir(base=temp_root)

    def temp_dir_factory(*, prefix: str, dir: str | None = None):  # noqa: ARG001
        return temp_dir

    return {
        "calls": calls,
        "temp_dir": temp_dir,
        "temp_root": temp_root,
        "inspector": fake_inspect,
        "downmix_fn": fake_downmix,
        "loudnorm_fn": fake_loudnorm,
        "denoise_fn": fake_denoise,
        "snr_estimator": fake_snr,
        "temp_dir_factory": temp_dir_factory,
        "gpu_check": lambda: True,
    }
