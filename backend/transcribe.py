"""
MINIMALISTIC solution to convert wav to a structured JSON transcript:

SYSTRAN/faster-whisper — a faster Whisper reimplementation (≈18.4k ⭐ as of Oct 5, 2025).
Why it's popular: up to ~4× faster and lower memory via CTranslate2; easy Python API; great on CPU or modest GPUs.

# Run these commands TO INITIALIZE:
source .venv/bin/activate

# to Install:
pip install -U hf_transfer huggingface_hub faster-whisper
pip install --force-reinstall ctranslate2==4.4.0
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12

# For faster HF transfer and cache:
# keep in your ~/.bashrc
export HF_HOME="$HOME/.cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1

# To run:
python transcribe.py
"""

import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, cast

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

from backend.exceptions import ModelNotFoundError
from backend.model_config import ModelConfig, get_preset
from backend.model_loader import ModelLoader

if TYPE_CHECKING:
    from faster_whisper import Segment, TranscriptionInfo


LOGGER = logging.getLogger(__name__)

# Default transcription parameters
DEFAULT_BEAM_SIZE = 5
DEFAULT_WORD_TIMESTAMPS = False
DEFAULT_TASK = "transcribe"

# Float rounding precision for JSON output
FLOAT_PRECISION = 3


def _get_estonian_model_path(model_id: str) -> str:
    """Download Estonian model and return path to CT2 subfolder.

    Estonian models from TalTechNLP store CT2 files in a 'ct2' subdirectory.
    This helper downloads only the CT2 files, skipping unnecessary formats.

    Args:
        model_id: HuggingFace model ID (e.g., 'TalTechNLP/whisper-large-v3-turbo-et-verbatim')

    Returns:
        Path to the CT2 model directory

    Raises:
        ModelNotFoundError: If CT2 folder doesn't exist in model
    """
    # Only download ct2 folder, skip transformers/native formats
    model_path = snapshot_download(model_id, allow_patterns=["ct2/*"])  # nosec B615
    ct2_path = Path(model_path) / "ct2"
    if not ct2_path.exists():
        raise ModelNotFoundError(f"CT2 folder not found in {model_id}. Expected at: {ct2_path}")
    return str(ct2_path)


def _get_cached_model_path(model_id: str) -> str:
    """Download/retrieve cached Whisper model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., 'Systran/faster-whisper-large-v3')

    Returns:
        Path to the cached model directory
    """
    return snapshot_download(model_id)  # nosec B615


def _resolve_model_path(config: ModelConfig) -> str:
    """Resolve the model path based on configuration.

    Args:
        config: Model configuration

    Returns:
        Path to the model (local or cached from HuggingFace)
    """
    if config.is_estonian:
        return _get_estonian_model_path(config.model_id)
    # For non-Estonian models, check if it's a simple preset name or HF model ID
    if "/" in config.model_id:
        # HuggingFace model ID (e.g., "Systran/faster-whisper-large-v3")
        return _get_cached_model_path(config.model_id)
    # Simple model name (e.g., "small") - faster-whisper will download it
    return config.model_id


def pick_model(preset: str = "et-large") -> WhisperModel:
    """Load a Whisper model by preset name.

    This is the main entry point for loading models. It uses configuration-driven
    model selection with automatic device selection and GPU->CPU fallback.

    Available presets: et-large, turbo, distil, large8gb, small

    Args:
        preset: Model preset name (see PRESETS in model_config.py)

    Returns:
        Loaded WhisperModel instance

    Raises:
        KeyError: If preset is not recognized
        ModelLoadError: If model fails to load
        ModelNotFoundError: If Estonian model CT2 folder not found

    Example:
        >>> model = pick_model("et-large")  # Estonian model on GPU (or CPU fallback)
        >>> model = pick_model("small")     # Small model on CPU
    """
    # Get configuration for the preset
    config = get_preset(preset)

    # Resolve the model path (download if needed)
    model_path = _resolve_model_path(config)

    # Load the model with automatic fallback
    loader = ModelLoader()
    return loader.load(model_path, config)


def _round_floats(value: Any, places: int = FLOAT_PRECISION) -> Any:
    if isinstance(value, dict):
        return {key: _round_floats(item, places) for key, item in cast(dict[Any, Any], value).items()}
    if isinstance(value, list):
        return [_round_floats(item, places) for item in cast(list[Any], value)]
    if isinstance(value, float):
        return round(value, places)
    return value


def _segment_to_payload(segment: "Segment") -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "id": getattr(segment, "id", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": segment.text.strip(),
    }

    speaker = getattr(segment, "speaker", None)
    if speaker is not None:
        data["speaker"] = speaker

    cleaned = {key: value for key, value in data.items() if value is not None}
    return _round_floats(cleaned)


def transcribe(path: str, preset: str = "et-large") -> Dict[str, Any]:
    LOGGER.info("🎤 Starting transcription of: %s", os.path.basename(path))
    overall_start = time.time()

    model = pick_model(preset)
    segments: Iterable["Segment"]
    info: "TranscriptionInfo"

    # Force Estonian language for Estonian models
    language = "et" if preset.startswith("et-") else None

    LOGGER.info("🔄 Transcribing audio...")
    transcribe_start = time.time()
    segments, info = model.transcribe(
        path,
        beam_size=DEFAULT_BEAM_SIZE,
        word_timestamps=DEFAULT_WORD_TIMESTAMPS,
        language=language,
        task=DEFAULT_TASK,
    )

    segment_payloads = [_segment_to_payload(segment) for segment in segments]
    transcribe_time = time.time() - transcribe_start

    payload: Dict[str, Any] = {
        "audio": os.path.basename(path),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": segment_payloads,
    }

    duration = getattr(info, "duration", None)
    if duration is not None:
        payload["duration"] = duration

    overall_time = time.time() - overall_start

    # Log timing summary
    LOGGER.info("⏱️  Transcription completed in %.2f seconds", transcribe_time)
    if duration:
        speed_ratio = duration / transcribe_time if transcribe_time > 0 else 0
        LOGGER.info("⚡ Speed: %.2fx realtime (%.1fs audio in %.1fs)", speed_ratio, duration, transcribe_time)
    LOGGER.info("✅ Total processing time: %.2f seconds", overall_time)

    cleaned = {key: value for key, value in payload.items() if value is not None}
    return _round_floats(cleaned)


def transcribe_to_json(audio_path: str, json_path: str, preset: str = "et-large") -> None:
    payload = transcribe(audio_path, preset)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    input_audio_path = project_root / "UserData" / "RelMan.wav"
    output_json_path = input_audio_path.with_suffix(".json")

    transcribe_to_json(str(input_audio_path), str(output_json_path), preset="et-large")
    LOGGER.info("Wrote JSON transcript to %s", output_json_path)
