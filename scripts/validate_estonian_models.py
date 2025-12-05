#!/usr/bin/env python3
"""Validate Estonian Whisper models from TalTechNLP.

This script checks if the Estonian models can be accessed and are compatible
with faster-whisper before implementing them in the main transcription system.

Usage:
    .venv/bin/python scripts/validate_estonian_models.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def check_huggingface_access() -> bool:
    """Check if we can access Hugging Face Hub.

    Returns:
        True if access is successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        LOGGER.info("✓ Hugging Face Hub access: OK")
        return True
    except ImportError:
        LOGGER.error("✗ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as error:
        LOGGER.error("✗ Failed to access Hugging Face Hub: %s", error)
        return False


def check_model_exists(model_id: str) -> bool:
    """Check if a model exists on Hugging Face.

    Args:
        model_id: The model ID (e.g., 'TalTechNLP/whisper-medium-et')

    Returns:
        True if model exists, False otherwise
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        LOGGER.info("✓ Model '%s' exists on Hugging Face", model_id)
        LOGGER.info("  - Model ID: %s", info.id)
        LOGGER.info("  - Last modified: %s", info.lastModified)
        if hasattr(info, "pipeline_tag"):
            LOGGER.info("  - Pipeline: %s", info.pipeline_tag)
        return True
    except Exception as error:
        LOGGER.error("✗ Model '%s' not found or inaccessible: %s", model_id, error)
        return False


def check_faster_whisper_compatibility(model_id: str, use_cpu: bool = True) -> bool:
    """Check if model can be loaded with faster-whisper.

    Args:
        model_id: The model ID to test
        use_cpu: If True, test with CPU (safer for validation). If False, try CUDA.

    Returns:
        True if model loads successfully, False otherwise
    """
    try:
        from faster_whisper import WhisperModel

        device = "cpu" if use_cpu else "cuda"
        compute_type = "int8" if use_cpu else "int8_float16"

        LOGGER.info("Attempting to load model '%s' with faster-whisper...", model_id)
        LOGGER.info("  - Device: %s", device)
        LOGGER.info("  - Compute type: %s", compute_type)

        # This will download the model if not cached
        model = WhisperModel(model_id, device=device, compute_type=compute_type)

        LOGGER.info("✓ Model '%s' loaded successfully with faster-whisper", model_id)
        return True

    except ImportError:
        LOGGER.error("✗ faster-whisper not installed. Install with: pip install faster-whisper")
        return False
    except Exception as error:
        LOGGER.error("✗ Failed to load model '%s': %s", model_id, error, exc_info=True)
        return False


def main() -> int:
    """Run all validation checks.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    LOGGER.info("=" * 70)
    LOGGER.info("Estonian Whisper Model Validation")
    LOGGER.info("=" * 70)

    models_to_check = [
        "TalTechNLP/whisper-medium-et",
        "TalTechNLP/whisper-large-et",
    ]

    # Step 1: Check Hugging Face access
    LOGGER.info("\n[1/3] Checking Hugging Face access...")
    if not check_huggingface_access():
        LOGGER.error("Cannot proceed without Hugging Face access")
        return 1

    # Step 2: Check model existence
    LOGGER.info("\n[2/3] Checking model existence on Hugging Face...")
    all_exist = True
    for model_id in models_to_check:
        if not check_model_exists(model_id):
            all_exist = False

    if not all_exist:
        LOGGER.error("Not all models exist on Hugging Face")
        return 1

    # Step 3: Test loading with faster-whisper (CPU only for safety)
    LOGGER.info("\n[3/3] Testing model loading with faster-whisper...")
    LOGGER.info("Note: Using CPU for validation (safer, but slower)")
    LOGGER.info("First load will download models - this may take a few minutes...")

    all_compatible = True
    for model_id in models_to_check:
        LOGGER.info("")
        if not check_faster_whisper_compatibility(model_id, use_cpu=True):
            all_compatible = False

    # Summary
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("Validation Summary")
    LOGGER.info("=" * 70)

    if all_exist and all_compatible:
        LOGGER.info("✓ All checks passed!")
        LOGGER.info("✓ Estonian models are ready to be integrated")
        return 0
    else:
        LOGGER.error("✗ Some checks failed")
        LOGGER.error("✗ Estonian models may not be compatible")
        return 1


if __name__ == "__main__":
    sys.exit(main())
