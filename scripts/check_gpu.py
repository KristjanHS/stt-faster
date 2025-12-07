#!/usr/bin/env python3
"""Diagnostic script to check CUDA/cuDNN installation and GPU availability."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_nvidia_driver() -> bool:
    """Check if NVIDIA driver is installed."""
    try:
        import subprocess  # nosec B404

        result = subprocess.run(  # nosec B603 B607
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("✓ NVIDIA driver is installed")
            logger.info(f"\n{result.stdout}")
            return True
        logger.error("✗ nvidia-smi command failed")
        return False
    except FileNotFoundError:
        logger.error("✗ nvidia-smi not found - NVIDIA driver may not be installed")
        return False


def check_cuda_libraries() -> bool:
    """Check if CUDA libraries are available."""
    try:
        import ctypes

        # Try to load CUDA runtime library
        try:
            ctypes.CDLL("libcudart.so")
            logger.info("✓ CUDA runtime library (libcudart.so) found")
            return True
        except OSError:
            logger.error("✗ CUDA runtime library (libcudart.so) not found")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking CUDA libraries: {e}")
        return False


def check_cudnn_libraries() -> bool:
    """Check if cuDNN libraries are available."""
    try:
        import ctypes

        # Try to load cuDNN library
        try:
            ctypes.CDLL("libcudnn.so")
            logger.info("✓ cuDNN library (libcudnn.so) found")
            return True
        except OSError:
            logger.error("✗ cuDNN library (libcudnn.so) not found")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking cuDNN libraries: {e}")
        return False


def check_ctranslate2() -> bool:
    """Check if ctranslate2 is installed and can detect CUDA."""
    try:
        import ctranslate2

        logger.info(f"✓ ctranslate2 version: {ctranslate2.__version__}")

        # Check for CUDA support
        try:
            cuda_available = ctranslate2.cuda.is_cuda_available()
            cuda_device_count = ctranslate2.get_cuda_device_count()

            if cuda_available:
                logger.info(f"✓ ctranslate2 CUDA support: Available ({cuda_device_count} devices)")
                return True
            logger.warning("✗ ctranslate2 CUDA support: Not available")
            return False
        except AttributeError:
            logger.warning("⚠ Cannot check CUDA availability in ctranslate2")
            return False

    except ImportError:
        logger.error("✗ ctranslate2 not installed")
        return False


def check_faster_whisper() -> bool:
    """Check if faster-whisper is installed."""
    try:
        import faster_whisper

        logger.info(f"✓ faster-whisper version: {faster_whisper.__version__}")
        return True
    except ImportError:
        logger.error("✗ faster-whisper not installed")
        return False


def test_model_load() -> bool:
    """Try to load a small Whisper model on GPU."""
    try:
        from faster_whisper import WhisperModel

        logger.info("\n--- Testing Model Load ---")
        logger.info("Attempting to load 'tiny' model on GPU...")

        try:
            WhisperModel("tiny", device="cuda", compute_type="int8")
            logger.info("✓ Successfully loaded model on GPU")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load model on GPU: {e}")
            logger.info("\nAttempting to load 'tiny' model on CPU as fallback...")
            try:
                WhisperModel("tiny", device="cpu", compute_type="int8")
                logger.info("✓ Successfully loaded model on CPU")
                logger.warning("⚠ GPU loading failed, but CPU works")
                return False
            except Exception as cpu_error:
                logger.error(f"✗ Failed to load model on CPU: {cpu_error}")
                return False

    except ImportError:
        logger.error("✗ Cannot test model loading - faster-whisper not installed")
        return False


def print_recommendations(results: dict[str, bool]) -> None:
    """Print recommendations based on diagnostic results."""
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)

    if all(results.values()):
        logger.info("✓ All checks passed! GPU acceleration should work.")
        return

    if not results.get("nvidia_driver"):
        logger.info("\n1. Install NVIDIA driver:")
        logger.info("   Ubuntu/Debian:")
        logger.info("   sudo apt-get update")
        logger.info("   sudo ubuntu-drivers autoinstall")
        logger.info("   sudo reboot")

    if not results.get("cuda_libs"):
        logger.info("\n2. Install CUDA Toolkit:")
        logger.info("   Ubuntu/Debian:")
        logger.info("   sudo apt-get update")
        logger.info("   sudo apt-get install -y cuda-toolkit-12-6")

    if not results.get("cudnn_libs"):
        logger.info("\n3. Install cuDNN:")
        logger.info("   Ubuntu/Debian:")
        logger.info("   sudo apt-get update")
        logger.info("   sudo apt-get install -y cudnn9-cuda-12")

    if not results.get("ctranslate2"):
        logger.info("\n4. Reinstall ctranslate2 with CUDA support:")
        logger.info("   pip install --force-reinstall ctranslate2==4.4.0")

    if not results.get("model_load"):
        logger.info("\n5. If all libraries are installed but model loading fails:")
        logger.info("   - Verify LD_LIBRARY_PATH includes CUDA/cuDNN paths")
        logger.info("   - Check GPU memory (nvidia-smi)")
        logger.info("   - Try reinstalling faster-whisper:")
        logger.info("     pip install --force-reinstall faster-whisper")

    logger.info("\n6. Environment variables (add to ~/.bashrc):")
    logger.info('   export HF_HOME="$HOME/.cache/hf"')
    logger.info('   export HF_HUB_CACHE="$HF_HOME/hub"')
    logger.info("   export HF_XET_HIGH_PERFORMANCE=1")

    logger.info("\n7. WSL2 Users (if running on Windows):")
    logger.info("   - Ensure you have WSL2 with GPU support enabled")
    logger.info("   - Install NVIDIA drivers on Windows (not in WSL)")
    logger.info("   - Follow WSL2 CUDA setup guide:")
    logger.info("     https://docs.nvidia.com/cuda/wsl-user-guide/")


def main() -> int:
    """Run all diagnostic checks."""
    logger.info("=" * 60)
    logger.info("GPU/CUDA DIAGNOSTIC TOOL")
    logger.info("=" * 60)
    logger.info("")

    results = {
        "nvidia_driver": check_nvidia_driver(),
        "cuda_libs": check_cuda_libraries(),
        "cudnn_libs": check_cudnn_libraries(),
        "ctranslate2": check_ctranslate2(),
        "faster_whisper": check_faster_whisper(),
        "model_load": test_model_load(),
    }

    print_recommendations(results)

    # Return 0 if all checks passed, 1 otherwise
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
