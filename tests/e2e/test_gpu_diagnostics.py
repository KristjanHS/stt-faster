"""GPU diagnostics E2E tests (hard requirements, no graceful skips)."""

from __future__ import annotations

import ctypes
import logging
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.network,
]

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def require_gpu_environment() -> dict[str, str]:
    """Validate GPU prerequisites once; fail fast on missing pieces.

    Logs diagnostic information for debugging GPU issues.
    """
    info: dict[str, str] = {}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        info["nvidia_smi"] = result.stdout.strip()
        LOGGER.info("GPU Diagnostics - NVIDIA Driver / GPUs:\n%s", info["nvidia_smi"])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.fail(f"nvidia-smi unavailable or failing: {exc}")

    try:
        ctypes.CDLL("libcudart.so")
        info["cuda_runtime"] = "available"
        LOGGER.info("GPU Diagnostics - CUDA Runtime: available")
    except OSError as exc:
        pytest.fail(f"CUDA runtime (libcudart.so) not available: {exc}")

    try:
        ctypes.CDLL("libcudnn.so")
        info["cudnn"] = "available"
        LOGGER.info("GPU Diagnostics - cuDNN: available")
    except OSError as exc:
        pytest.fail(f"cuDNN (libcudnn.so) not available: {exc}")

    try:
        import ctranslate2  # type: ignore

        info["ctranslate2_version"] = getattr(ctranslate2, "__version__", "unknown")
        device_count = ctranslate2.get_cuda_device_count()
        if device_count <= 0:
            pytest.fail(f"ctranslate2 reports no CUDA devices (count={device_count})")
        info["ctranslate2_cuda_devices"] = str(device_count)
        LOGGER.info("GPU Diagnostics - ctranslate2: %s (CUDA devices: %s)", info["ctranslate2_version"], device_count)
    except ImportError as exc:
        pytest.fail(f"ctranslate2 not installed: {exc}")
    except (AttributeError, RuntimeError) as exc:
        pytest.fail(f"ctranslate2 cannot enumerate CUDA devices: {exc}")

    try:
        import faster_whisper  # type: ignore

        info["faster_whisper_version"] = getattr(faster_whisper, "__version__", "unknown")
        LOGGER.info("GPU Diagnostics - faster-whisper: %s", info["faster_whisper_version"])
    except ImportError as exc:
        pytest.fail(f"faster-whisper not installed: {exc}")

    return info


@pytest.fixture(scope="session")
def tiny_gpu_model(require_gpu_environment: dict[str, str]) -> "WhisperModel":
    """Load the tiny Whisper model on GPU (fails hard if unavailable)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - import guard
        pytest.fail(f"faster-whisper import failed: {exc}")

    try:
        LOGGER.info("Loading tiny Whisper model on CUDA for diagnostics...")
        return WhisperModel("tiny", device="cuda", compute_type="int8")
    except Exception as exc:  # pragma: no cover - diagnostic guard
        pytest.fail(f"Failed to load tiny model on GPU: {exc}")


def test_nvidia_smi_reports_gpus(require_gpu_environment: dict[str, str]) -> None:
    """Ensure nvidia-smi reports at least one GPU with details."""
    output = require_gpu_environment["nvidia_smi"]
    assert output, "nvidia-smi returned empty output"
    lines = [line for line in output.split("\n") if line.strip()]
    assert lines, "nvidia-smi did not list any GPUs"
    assert all("," in line for line in lines), "nvidia-smi output missing expected columns"


def test_cuda_and_cudnn_present(require_gpu_environment: dict[str, str]) -> None:
    """Confirm CUDA runtime and cuDNN libraries are present."""
    assert require_gpu_environment["cuda_runtime"] == "available"
    assert require_gpu_environment["cudnn"] == "available"


def test_ctranslate2_sees_cuda_devices(require_gpu_environment: dict[str, str]) -> None:
    """Validate ctranslate2 detects CUDA devices."""
    device_count = int(require_gpu_environment["ctranslate2_cuda_devices"])
    assert device_count > 0


def test_faster_whisper_on_gpu(tiny_gpu_model: "WhisperModel") -> None:
    """Test that faster-whisper performs actual transcription on GPU (behavior test)."""
    # Test behavior: model must load and be usable
    assert tiny_gpu_model is not None, "Model failed to load"

    # Test behavior: verify GPU is actually used
    # Check that model has device information indicating GPU usage
    model_device = getattr(tiny_gpu_model.model, "device", None)
    if model_device:
        assert "cuda" in str(model_device).lower() or "gpu" in str(model_device).lower(), (
            f"Model not using GPU device, got: {model_device}"
        )

    # Test behavior: perform actual transcription to verify GPU functionality
    import numpy as np

    # Create a minimal test audio (1 second of silence at 16kHz)
    sample_rate = 16000
    duration = 1.0
    audio_samples = np.zeros(int(sample_rate * duration), dtype=np.float32)

    # Transcribe using GPU model
    segments, info = tiny_gpu_model.transcribe(audio_samples, beam_size=1)

    # Verify transcription produces results (even if empty for silence)
    assert info is not None, "Transcription info should be returned"
    assert info.language is not None, "Language detection should work"

    # Verify segments are iterable (behavior contract)
    segment_list = list(segments)
    # For silence, we might get empty segments, which is valid behavior
    LOGGER.info("GPU transcription test completed: language=%s, segments=%d", info.language, len(segment_list))
