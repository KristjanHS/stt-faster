"""Model loading with device selection and fallback logic.

This module handles the actual loading of Whisper models with automatic
GPU->CPU fallback and environment-based device configuration.
"""

import logging
import os
import time
from typing import Optional, Callable

from faster_whisper import WhisperModel

from backend.exceptions import ModelLoadError
from backend.model_config import ComputeType, DeviceType, ModelConfig

LOGGER = logging.getLogger(__name__)


def _default_model_factory(model_path: str, device: DeviceType, compute_type: ComputeType) -> WhisperModel:
    return WhisperModel(model_path, device=device, compute_type=compute_type)


class DeviceSelector:
    """Selects the appropriate device (CPU/GPU) for model loading.

    Device selection priority:
    1. STT_DEVICE environment variable (cpu/cuda)
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
        env_device = os.getenv("STT_DEVICE", "").lower()

        if env_device in ("cpu", "cuda"):
            device = env_device  # type: ignore[assignment]
            compute_type: ComputeType = "int8" if device == "cpu" else config.compute_type
            LOGGER.info("Using device from STT_DEVICE env: %s", device)
            return device, compute_type

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
                "GPU initialization failed for %s: %s. Falling back to CPU int8; expect slower transcription.",
                model_path,
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
