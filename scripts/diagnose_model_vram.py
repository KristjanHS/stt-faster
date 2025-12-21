#!/usr/bin/env python3
"""Diagnose model VRAM usage and compute type.

This script loads the Estonian model and reports:
- Actual compute type being used
- VRAM usage
- Model configuration
"""

import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import torch (optional)
try:
    import torch  # type: ignore[reportMissingImports]

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

from backend.model_config import get_preset
from backend.transcribe import pick_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_gpu_memory_nvidia_smi() -> tuple[float | None, float | None]:
    """Get GPU memory usage using nvidia-smi.

    Returns:
        Tuple of (used_memory_mb, total_memory_mb) or (None, None) if unavailable
    """
    try:
        result = subprocess.run(  # nosec B603, B607
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse output: "1234, 5678" -> used, total
        parts = result.stdout.strip().split(", ")
        if len(parts) == 2:
            used_mb = float(parts[0])
            total_mb = float(parts[1])
            return used_mb, total_mb
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return None, None


def get_gpu_memory_torch() -> float | None:
    """Get current GPU memory usage in MB using PyTorch."""
    if not TORCH_AVAILABLE or torch is None:
        return None
    if not torch.cuda.is_available():  # type: ignore[union-attr]
        return None
    return torch.cuda.memory_allocated() / (1024**2)  # type: ignore[union-attr]


def format_memory(mb: float | None) -> str:
    """Format memory in MB to human-readable format."""
    if mb is None:
        return "N/A"
    if mb < 1024:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.2f} GB"


def main() -> int:
    """Diagnose model VRAM usage."""
    logger.info("=" * 60)
    logger.info("Estonian Model VRAM Diagnostic")
    logger.info("=" * 60)

    # Check GPU availability
    gpu_name = None

    # Try PyTorch first
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():  # type: ignore[union-attr]
        gpu_name = torch.cuda.get_device_name(0)  # type: ignore[union-attr]
        logger.info(f"GPU (via PyTorch): {gpu_name}")
        logger.info(f"CUDA Version: {torch.version.cuda}")  # type: ignore[union-attr]
        logger.info(f"PyTorch Version: {torch.version.__version__}")  # type: ignore[union-attr]
    else:
        # Try nvidia-smi
        used_mem, total_mem = get_gpu_memory_nvidia_smi()
        if used_mem is not None:
            logger.info("GPU detected via nvidia-smi")
            logger.info(f"  Total VRAM: {format_memory(total_mem)}")
        else:
            logger.warning("⚠️  Cannot detect GPU. CUDA may not be available.")
            logger.warning("  Continuing with model load diagnostics...")

    # Get model config
    config = get_preset("et-large")
    logger.info("\n" + "=" * 60)
    logger.info("Model Configuration:")
    logger.info(f"  Model ID: {config.model_id}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Compute Type: {config.compute_type}")
    logger.info(f"  Is Estonian: {config.is_estonian}")

    # Check environment variables
    stt_device = os.getenv("STT_DEVICE")
    if stt_device:
        logger.info(f"\nSTT_DEVICE env var: {stt_device}")
        logger.info("  (This may override config settings)")

    # Get initial GPU memory
    initial_memory = None
    initial_total = None

    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():  # type: ignore[union-attr]
        torch.cuda.empty_cache()  # type: ignore[union-attr]
        initial_memory = get_gpu_memory_torch()
    else:
        # Use nvidia-smi
        used_mem, total_mem = get_gpu_memory_nvidia_smi()
        if used_mem is not None:
            initial_memory = used_mem
            initial_total = total_mem

    logger.info(f"\nInitial GPU Memory: {format_memory(initial_memory)}")
    if initial_total:
        logger.info(f"Total GPU Memory: {format_memory(initial_total)}")

    # Load model
    logger.info("\n" + "=" * 60)
    logger.info("Loading model...")
    try:
        model = pick_model("et-large")
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}", exc_info=True)
        return 1

    # Check memory after loading
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():  # type: ignore[union-attr]
        torch.cuda.synchronize()  # type: ignore[union-attr]
        after_load_memory = get_gpu_memory_torch()
    else:
        # Use nvidia-smi
        used_mem, _ = get_gpu_memory_nvidia_smi()
        after_load_memory = used_mem

    memory_used = (after_load_memory - initial_memory) if (initial_memory and after_load_memory) else None

    logger.info("\n" + "=" * 60)
    logger.info("Memory Usage:")
    logger.info(f"  Initial: {format_memory(initial_memory)}")
    logger.info(f"  After Load: {format_memory(after_load_memory)}")
    if memory_used is not None:
        logger.info(f"  Model VRAM: {format_memory(memory_used)}")
    else:
        logger.warning("  Model VRAM: Could not calculate (need both initial and after-load measurements)")

    # Check model attributes (if available)
    logger.info("\n" + "=" * 60)
    logger.info("Model Attributes:")
    if hasattr(model, "model"):
        model_attr = getattr(model, "model", None)
        logger.info(f"  Model type: {type(model_attr)}")
        if model_attr is not None and hasattr(model_attr, "device"):
            logger.info(f"  Device: {getattr(model_attr, 'device', 'N/A')}")  # type: ignore[arg-type]
        if model_attr is not None and hasattr(model_attr, "compute_type"):
            logger.info(f"  Compute type: {getattr(model_attr, 'compute_type', 'N/A')}")  # type: ignore[arg-type]

    # Expected memory ranges
    logger.info("\n" + "=" * 60)
    logger.info("Expected VRAM Usage (for large-v3):")
    logger.info("  float32: ~6-7 GB")
    logger.info("  float16: ~3-4 GB")
    logger.info("  int8_float16: ~2.5-3 GB")
    logger.info("  int8: ~2-3 GB")
    logger.info(f"\nYour model is using: {format_memory(memory_used)}")

    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("Analysis:")
    if memory_used and memory_used < 3000:
        logger.warning("⚠️  VRAM usage is lower than expected for float16 (~3-4 GB).")
        logger.warning("  This suggests the model might be using:")
        logger.warning("    - int8_float16 (mixed precision)")
        logger.warning("    - int8 quantization")
        logger.warning("    - Or the model is optimized/smaller than expected")
        logger.info("\n  To use pure float16, ensure:")
        logger.info("    1. compute_type='float16' in config")
        logger.info("    2. STT_DEVICE env var is not overriding it")
        logger.info("    3. Model files support float16")
    elif memory_used and 3000 <= memory_used < 5000:
        logger.info("✓ VRAM usage is in expected range for float16")
    elif memory_used and memory_used >= 5000:
        logger.info("✓ VRAM usage suggests float32 (highest quality)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
