"""
MINIMALISTIC solution to convert wav to a structured JSON transcript:

SYSTRAN/faster-whisper — a faster Whisper reimplementation (≈18.4k ⭐ as of Oct 5, 2025).
Why it’s popular: up to ~4× faster and lower memory via CTranslate2; easy Python API; great on CPU or modest GPUs.

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
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from faster_whisper import Segment, TranscriptionInfo


LOGGER = logging.getLogger(__name__)


def pick_model(preset: str = "turbo") -> WhisperModel:
    def _load(model_name: str, *, default_device: str, default_type: str) -> WhisperModel:
        try:
            return WhisperModel(model_name, device=default_device, compute_type=default_type)
        except Exception as error:  # fall back to CPU when GPU libs are missing
            LOGGER.error(
                "GPU initialisation failed for %s (%s). Falling back to CPU int8; expect slower transcription. "
                "Verify CUDA/cuDNN installation if GPU acceleration is desired.",
                model_name,
                error,
                exc_info=True,
            )
            return WhisperModel(model_name, device="cpu", compute_type="int8")

    if preset == "turbo":
        # Best default for 8 GB
        return _load("large-v3-turbo", default_device="cuda", default_type="float16")
    if preset == "distil":
        # English-only; very fast; near large-v3 accuracy
        return _load("distil-large-v3", default_device="cuda", default_type="float16")
    if preset == "large8gb":
        # Try original large-v3 with mixed INT8/FP16 to fit in 8 GB
        return _load("large-v3", default_device="cuda", default_type="int8_float16")
    # Fallback (portable): CPU int8
    return WhisperModel("small", device="cpu", compute_type="int8")


def _segment_to_payload(segment: "Segment") -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "id": getattr(segment, "id", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": segment.text.strip(),
    }

    optional_attrs = (
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
    )
    for attr in optional_attrs:
        value = getattr(segment, attr, None)
        if value is not None:
            data[attr] = value

    speaker = getattr(segment, "speaker", None)
    if speaker is not None:
        data["speaker"] = speaker

    words_payload: List[Dict[str, Any]] = []
    words = getattr(segment, "words", None)
    if words:
        for word in words:
            word_payload: Dict[str, Any] = {
                "start": getattr(word, "start", None),
                "end": getattr(word, "end", None),
                "text": getattr(word, "word", "").strip(),
            }
            probability = getattr(word, "probability", None)
            if probability is not None:
                word_payload["probability"] = probability
            # Drop keys with None values to keep payload compact.
            words_payload.append({key: value for key, value in word_payload.items() if value is not None})
    if words_payload:
        data["words"] = words_payload

    return {key: value for key, value in data.items() if value is not None}


def transcribe(path: str, preset: str = "distil") -> Dict[str, Any]:
    model = pick_model(preset)
    segments: Iterable["Segment"]
    info: "TranscriptionInfo"
    segments, info = model.transcribe(path, beam_size=5, word_timestamps=True)

    segment_payloads = [_segment_to_payload(segment) for segment in segments]
    payload: Dict[str, Any] = {
        "audio": os.path.basename(path),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": segment_payloads,
    }

    duration = getattr(info, "duration", None)
    if duration is not None:
        payload["duration"] = duration

    return {key: value for key, value in payload.items() if value is not None}


def transcribe_to_json(audio_path: str, json_path: str, preset: str = "distil") -> None:
    payload = transcribe(audio_path, preset)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    input_audio_path = project_root / "UserData" / "RelMan.wav"
    output_json_path = input_audio_path.with_suffix(".json")

    transcribe_to_json(str(input_audio_path), str(output_json_path), preset="distil")
    LOGGER.info("Wrote JSON transcript to %s", output_json_path)
