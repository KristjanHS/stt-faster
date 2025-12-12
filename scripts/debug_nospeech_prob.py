"""Debug script for analyzing no_speech_prob values from faster-whisper CT2 models.

This script helps diagnose whether no_speech_threshold can actually filter segments
based on no_speech_prob values from the model.

Interpretation:
    If max(no_speech_prob) == 0.0 (or ~0.000000 for all segments), that's strong
    evidence that no_speech_threshold can't ever fire on this model in CT2/faster-whisper.
    This matches reported behavior for v3/v3-turbo CT2 models where the attribute
    exists but is always zero.

    If max(no_speech_prob) > 0.0 (especially > 0.5), then no_speech_threshold
    CAN fire and the filtering mechanism is functional for this model.

Example results for Estonian model (TalTechNLP/whisper-large-v3-turbo-et-verbatim):
    - max(no_speech_prob) = 0.999512 indicates the threshold mechanism works
    - This model does NOT exhibit the v3/v3-turbo CT2 bug where values are always zero
"""

from __future__ import annotations

import argparse
from statistics import mean

from faster_whisper import WhisperModel


def run(model_dir: str, audio_path: str) -> None:
    model = WhisperModel(model_dir, device="cuda", compute_type="float16")

    # IMPORTANT: disable vad_filter, otherwise VAD may hide silence-related behavior
    segments, _info = model.transcribe(
        audio_path,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
        beam_size=1,
        chunk_length=30,
    )
    segs = list(segments)

    if not segs:
        print("No segments produced. Use an audio sample that contains *some speech*.")  # noqa: T201
        return

    nos = [float(s.no_speech_prob) for s in segs]  # type: ignore[attr-defined]
    avl = [float(s.avg_logprob) for s in segs]  # type: ignore[attr-defined]

    print(f"segments: {len(segs)}")  # noqa: T201
    print(f"no_speech_prob: min={min(nos):.6f} mean={mean(nos):.6f} max={max(nos):.6f}")  # noqa: T201
    print(f"avg_logprob:    min={min(avl):.3f} mean={mean(avl):.3f} max={max(avl):.3f}")  # noqa: T201
    print("\nFirst 15 segments:")  # noqa: T201
    for s in segs[:15]:
        txt = (s.text or "").replace("\n", " ")
        no_sp = s.no_speech_prob  # type: ignore[attr-defined]
        avg_lp = s.avg_logprob  # type: ignore[attr-defined]
        print(f"{s.start:7.2f}-{s.end:7.2f}  no_speech={no_sp:.6f}  avg_lp={avg_lp:.3f}  {txt[:120]}")  # noqa: T201


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Path to CT2 model directory")
    ap.add_argument("--audio", required=True, help="Path to audio file")
    args = ap.parse_args()
    run(args.model_dir, args.audio)


# Example usage (Estonian model - replace model-dir with your CT2 model path):
# python scripts/debug_nospeech_prob.py \
#   --model-dir ~/.cache/hf/hub/models--TalTechNLP--whisper-large-v3-turbo-et-verbatim/snapshots/XXXXX/ct2 \
#   --audio /path/to/audio.wav
