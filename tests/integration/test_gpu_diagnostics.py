"""Environment validation tests for GPU/CUDA setup.

IMPORTANT: These tests verify system-level GPU/CUDA installation, NOT actual
transcription functionality or performance.

Purpose:
- Deployment validation: Verify GPU drivers/libraries are installed correctly
- Developer setup: Help developers confirm their GPU environment is configured
- Troubleshooting: Identify specific missing components when GPU isn't working

These are NOT integration tests of business logic. They check if:
- NVIDIA drivers are present (nvidia-smi works)
- CUDA/cuDNN libraries can be loaded
- ctranslate2 recognizes GPU devices
- faster-whisper can load models on GPU

For actual GPU transcription testing, see test_cli_commands.py with real audio files.

Tests are automatically skipped in CPU-only environments.
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


# Mark all tests in this module as optional GPU tests
pytestmark = pytest.mark.gpu


class TestNvidiaDriver:
    """Tests for NVIDIA driver availability."""

    def test_nvidia_smi_available(self) -> None:
        """Test if nvidia-smi command is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            assert result.returncode == 0, "nvidia-smi command failed"
            assert "NVIDIA" in result.stdout, "nvidia-smi output doesn't contain NVIDIA"
        except FileNotFoundError:
            pytest.skip("nvidia-smi not found - NVIDIA driver not installed")

    def test_nvidia_driver_shows_gpu(self) -> None:
        """Test if nvidia-smi shows at least one GPU."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            gpu_names = result.stdout.strip().split("\n")
            assert len(gpu_names) > 0, "No GPUs detected by nvidia-smi"
            assert all(name.strip() for name in gpu_names), "Empty GPU names detected"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Cannot query GPU information")


class TestCudaLibraries:
    """Tests for CUDA library availability."""

    def test_cuda_runtime_library_available(self) -> None:
        """Test if CUDA runtime library can be loaded."""
        import ctypes

        try:
            ctypes.CDLL("libcudart.so")
        except OSError as e:
            pytest.fail(f"CUDA runtime library (libcudart.so) not found: {e}")

    def test_cudnn_library_available(self) -> None:
        """Test if cuDNN library can be loaded."""
        import ctypes

        try:
            ctypes.CDLL("libcudnn.so")
        except OSError as e:
            pytest.fail(f"cuDNN library (libcudnn.so) not found: {e}")


class TestCTranslate2:
    """Tests for ctranslate2 CUDA support."""

    def test_ctranslate2_installed(self) -> None:
        """Test if ctranslate2 is installed."""
        try:
            import ctranslate2

            assert hasattr(ctranslate2, "__version__")
        except ImportError:
            pytest.fail("ctranslate2 not installed")

    def test_ctranslate2_cuda_device_count(self) -> None:
        """Test if ctranslate2 detects at least one CUDA device.

        This is the most reliable way to check ctranslate2 CUDA support
        across different API versions.
        """
        try:
            import ctranslate2
        except ImportError:
            pytest.skip("ctranslate2 not installed")

        try:
            device_count = ctranslate2.get_cuda_device_count()
            assert device_count > 0, f"Expected > 0 CUDA devices, got {device_count}"
        except (AttributeError, RuntimeError) as e:
            pytest.fail(f"ctranslate2 cannot detect CUDA devices: {e}")


class TestFasterWhisper:
    """Tests for faster-whisper GPU support."""

    def test_faster_whisper_installed(self) -> None:
        """Test if faster-whisper is installed."""
        try:
            import faster_whisper

            assert hasattr(faster_whisper, "__version__")
        except ImportError:
            pytest.fail("faster-whisper not installed")

    @pytest.mark.slow
    def test_load_tiny_model_on_gpu(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test loading a tiny Whisper model on GPU.

        This verifies that faster-whisper can actually instantiate a model
        on GPU, not just that libraries are present. This is the most
        complete environment validation test.

        Marked 'slow' because it downloads and loads a model (~75MB).
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            pytest.skip("faster-whisper not installed")

        with caplog.at_level(logging.INFO):
            try:
                model = WhisperModel("tiny", device="cuda", compute_type="int8")
            except Exception as e:
                pytest.fail(f"Failed to load tiny model on GPU: {e}")

            # Assert outside try/except to avoid catching assertion failures
            assert model is not None, "Model should not be None after loading"


class TestGPUDiagnosticsSummary:
    """Summary test that provides diagnostic information."""

    def test_print_gpu_environment_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Print comprehensive GPU environment information for debugging."""
        info_lines = [
            "\n" + "=" * 60,
            "GPU ENVIRONMENT DIAGNOSTIC INFORMATION",
            "=" * 60,
        ]

        # NVIDIA Driver
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            info_lines.append("\n✓ NVIDIA Driver:")
            info_lines.extend(f"  {line}" for line in result.stdout.strip().split("\n"))
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            info_lines.append(f"\n✗ NVIDIA Driver: Not available ({e})")

        # CUDA Libraries
        import ctypes

        try:
            ctypes.CDLL("libcudart.so")
            info_lines.append("\n✓ CUDA Runtime: Available")
        except OSError:
            info_lines.append("\n✗ CUDA Runtime: Not available")

        try:
            ctypes.CDLL("libcudnn.so")
            info_lines.append("✓ cuDNN: Available")
        except OSError:
            info_lines.append("✗ cuDNN: Not available")

        # ctranslate2
        try:
            import ctranslate2

            info_lines.append(f"\n✓ ctranslate2: {ctranslate2.__version__}")
            try:
                if ctranslate2.cuda.is_cuda_available():
                    device_count = ctranslate2.get_cuda_device_count()
                    info_lines.append(f"  CUDA support: Available ({device_count} devices)")
                else:
                    info_lines.append("  CUDA support: Not available")
            except AttributeError:
                info_lines.append("  CUDA support: Cannot check")
        except ImportError:
            info_lines.append("\n✗ ctranslate2: Not installed")

        # faster-whisper
        try:
            import faster_whisper

            info_lines.append(f"\n✓ faster-whisper: {faster_whisper.__version__}")
        except ImportError:
            info_lines.append("\n✗ faster-whisper: Not installed")

        info_lines.append("=" * 60 + "\n")

        # Output info (captured by capsys for display)
        output = "\n".join(info_lines)

        # Use capsys to output without triggering T201 (print) rule
        import sys

        sys.stdout.write(output + "\n")

        # This test is informational - it prints GPU diagnostic info for debugging
        # No assertions needed as it's purely for information display


@pytest.fixture(scope="module", autouse=True)
def check_gpu_availability(gpu_available: bool) -> None:
    """Auto-run fixture to check if GPU tests should run.

    This will cause all GPU tests to be skipped if no GPU is available.
    Uses the shared gpu_available fixture from integration/conftest.py.
    """
    if not gpu_available:
        pytest.skip("No GPU available - skipping all GPU tests", allow_module_level=True)
