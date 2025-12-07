"""Model loading with device selection and fallback logic.

This module handles the actual loading of Whisper models with automatic
GPU->CPU fallback and environment-based device configuration.
"""

import logging
import os
import time
from typing import Callable, Optional, cast

from faster_whisper import WhisperModel

from backend.exceptions import ModelLoadError
from backend.model_config import ComputeType, DeviceType, ModelConfig

LOGGER = logging.getLogger(__name__)


def _default_model_factory(model_path: str, device: DeviceType, compute_type: ComputeType) -> WhisperModel:
    return WhisperModel(model_path, device=device, compute_type=compute_type)


_ALLOWED_COMPUTE_TYPES = {"int8", "float16", "int8_float16", "float32"}


class DeviceSelector:
    """Selects the appropriate device (CPU/GPU) for model loading.

    Device selection priority:
    1. STT_DEVICE environment variable (accepts device[/compute_type], e.g., cuda/float16)
    2. Preferred device from configuration
    3. Auto-fallback to CPU if GPU fails (handled by ModelLoader)
    """

    def select(self, config: ModelConfig) -> tuple[DeviceType, ComputeType]:
        """Select device and compute type based on environment and preferences.

        Args:
            config: Model configuration with preferred device and compute type

        Returns:
            Tuple of (device, compute_type) to use
        """
        raw_env = os.getenv("STT_DEVICE", "").strip()
        if raw_env:
            normalized = raw_env.lower()
            parts = normalized.split("/", 1)
            device = parts[0]
            compute_override_input = None if len(parts) == 1 else parts[1]
            compute_override: ComputeType | None = None
            if device in ("cpu", "cuda"):
                if compute_override_input and compute_override_input not in _ALLOWED_COMPUTE_TYPES:
                    LOGGER.warning(
                        "Ignoring unsupported compute type from STT_DEVICE (%s); falling back to defaults",
                        compute_override_input,
                    )
                elif compute_override_input:
                    compute_override = cast(ComputeType, compute_override_input)
                compute_type: ComputeType = "int8" if device == "cpu" else (compute_override or config.compute_type)
                if device == "cpu" and compute_override and compute_override != "int8":
                    LOGGER.warning(
                        "STT_DEVICE requested CPU compute (%s) which is forced to int8 for compatibility.",
                        compute_override,
                    )
                LOGGER.info("Using device from STT_DEVICE env: %s (compute=%s)", device, compute_type)
                return device, compute_type
            LOGGER.warning("Ignoring STT_DEVICE=%s because '%s' is not a supported device", raw_env, device)

        # Use preferred device and compute type from config
        return config.device, config.compute_type


class ModelLoader:
    """Loads Whisper models with automatic fallback handling.

    Handles GPU->CPU fallback, performance timing, and error recovery.
    """

    def __init__(
        self,
        device_selector: Optional[DeviceSelector] = None,
        model_factory: Callable[[str, DeviceType, ComputeType], WhisperModel] | None = None,
    ):
        """Initialize ModelLoader.

        Args:
            device_selector: Device selection strategy (defaults to DeviceSelector)
            model_factory: Factory used to construct WhisperModel (defaults to WhisperModel)
        """
        self._device_selector = device_selector or DeviceSelector()
        self._model_factory: Callable[[str, DeviceType, ComputeType], WhisperModel] = (
            model_factory or _default_model_factory
        )

    def load(self, model_path: str, config: ModelConfig) -> WhisperModel:
        """Load a Whisper model with automatic GPU->CPU fallback.

        Args:
            model_path: Path to the model (local or cached)
            config: Model configuration with preferred device/compute type

        Returns:
            Loaded WhisperModel instance

        Raises:
            ModelLoadError: If model fails to load on both GPU and CPU
        """
        device, compute_type = self._device_selector.select(config)

        # If device is CPU, use it directly (no fallback needed)
        if device == "cpu":
            LOGGER.info("Using CPU device for model: %s", model_path)
            return self._load_on_device(model_path, "cpu", "int8")

        # Try GPU first, fall back to CPU if it fails
        try:
            LOGGER.info("Attempting to load model on GPU: %s", model_path)
            return self._load_on_device(model_path, device, compute_type)
        except Exception as error:
            LOGGER.warning(
                (
                    "⚠️  GPU initialization failed for %s (requested %s/%s): %s. "
                    "Falling back to CPU int8; expect slower transcription."
                ),
                model_path,
                device,
                compute_type,
                error,
            )
            try:
                return self._load_on_device(model_path, "cpu", "int8")
            except Exception as cpu_error:
                raise ModelLoadError(
                    f"Failed to load model {model_path} on both GPU and CPU: {cpu_error}"
                ) from cpu_error

    def _load_on_device(
        self,
        model_path: str,
        device: DeviceType,
        compute_type: ComputeType,
    ) -> WhisperModel:
        """Load model on specific device with timing.

        Args:
            model_path: Path to the model
            device: Device to use (cpu/cuda)
            compute_type: Quantization/precision type

        Returns:
            Loaded WhisperModel instance
        """
        start_time = time.time()
        model = self._model_factory(model_path, device, compute_type)
        load_time = time.time() - start_time

        device_name = "GPU" if device == "cuda" else "CPU"
        LOGGER.info("⏱️  Model loaded on %s in %.2f seconds", device_name, load_time)

        return model
