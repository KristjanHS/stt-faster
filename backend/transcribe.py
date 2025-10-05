"""
MINIMALISTIC solution to convert wav to text:

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

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from faster_whisper import Segment, TranscriptionInfo


def pick_model(preset: str = "turbo") -> WhisperModel:
    if preset == "turbo":
        # Best default for 8 GB
        return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    if preset == "distil":
        # English-only; very fast; near large-v3 accuracy
        return WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
    if preset == "large8gb":
        # Try original large-v3 with mixed INT8/FP16 to fit in 8 GB
        return WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    # Fallback (portable): CPU int8
    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(path: str, preset: str = "distil") -> str:
    model = pick_model(preset)
    segments: Iterable["Segment"]
    _info: "TranscriptionInfo"
    segments, _info = model.transcribe(path, beam_size=5)
    return " ".join(s.text.strip() for s in segments)


def transcribe_to_md(audio_path: str, md_path: str, preset: str = "distil") -> None:
    model = pick_model(preset)
    segments: Iterable["Segment"]
    info: "TranscriptionInfo"
    segments, info = model.transcribe(audio_path, beam_size=5)

    lines: list[str] = []
    lines.append(f"# Transcript of `{os.path.basename(audio_path)}`\n")
    lines.append(f"**Detected language**: {info.language}  \n")
    lines.append(f"**Language probability**: {info.language_probability:.4f}\n")
    lines.append("---\n")

    for seg in segments:
        start = getattr(seg, "start", None)
        end = getattr(seg, "end", None)
        if start is not None and end is not None:
            lines.append(f"**{start:.1f}s -> {end:.1f}s**  \n")
        lines.append(seg.text.strip() + "\n\n")

    md_text = "\n".join(lines)

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(md_text)


if __name__ == "__main__":
    input_audio = "RelMan.wav"
    output_md = os.path.splitext(input_audio)[0] + ".md"
    transcribe_to_md(input_audio, output_md, preset="distil")
    print(f"Wrote markdown transcript to {output_md}")
