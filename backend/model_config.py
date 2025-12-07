"""Model configuration and presets.

This module defines available model presets and their configurations,
separating configuration from model loading logic.
"""

from dataclasses import dataclass
from typing import Literal

DeviceType = Literal["cpu", "cuda"]
ComputeType = Literal["int8", "float16", "int8_float16", "float32"]


@dataclass
class ModelConfig:
    """Configuration for a Whisper model.

    Attributes:
        model_id: HuggingFace model ID or local path identifier
        device: Preferred device (cpu/cuda) - may fallback to CPU
        compute_type: Quantization/precision type
        is_estonian: Whether this is an Estonian-specific model (requires special path handling)
    """

    model_id: str
    device: DeviceType = "cuda"
    compute_type: ComputeType = "float16"
    is_estonian: bool = False


# Model presets - easy to extend with new models
PRESETS: dict[str, ModelConfig] = {
    # Estonian models (TalTech fine-tuned)
    "et-large": ModelConfig(
        model_id="TalTechNLP/whisper-large-v3-turbo-et-verbatim",
        device="cuda",
        compute_type="float16",
        is_estonian=True,
    ),
    # English/multilingual models (Systran faster-whisper)
    "turbo": ModelConfig(
        model_id="Systran/faster-distil-whisper-large-v3",
        device="cuda",
        compute_type="float16",
    ),
    "distil": ModelConfig(
        model_id="Systran/faster-distil-whisper-large-v3",
        device="cuda",
        compute_type="float16",
    ),
    "large8gb": ModelConfig(
        model_id="Systran/faster-whisper-large-v3",
        device="cuda",
        compute_type="int8_float16",  # Try original large-v3 with mixed INT8/FP16 to fit in 8 GB
    ),
    # Fallback preset (portable, CPU-friendly)
    "small": ModelConfig(
        model_id="small",
        device="cpu",
        compute_type="int8",
    ),
}


def get_preset(name: str) -> ModelConfig:
    """Get model configuration by preset name.

    Args:
        name: Preset name (e.g., 'et-large', 'turbo')

    Returns:
        ModelConfig for the requested preset

    Raises:
        KeyError: If preset name is not recognized
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]
