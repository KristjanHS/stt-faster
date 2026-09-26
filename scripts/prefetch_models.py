#!/usr/bin/env python3
"""Pre-fetch the Whisper models and the pinned speaker-diarization model.

Downloads CT2 weights for the Estonian and English presets so the first
transcription run on a fresh machine doesn't pay for ~3 GB of HuggingFace
downloads. Pure download via the existing model resolvers - no
``WhisperModel(...)`` construction, so this never probes CUDA on a CPU-only box.

Usage::

    .venv/Scripts/python scripts/prefetch_models.py    # Windows
    .venv/bin/python    scripts/prefetch_models.py    # Linux/WSL
    .venv/bin/python    scripts/prefetch_models.py --diarization-only
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.diarize.model import DIARIZATION_REPO, fetch_model
from backend.transcribe import _get_cached_model_path, _get_estonian_model_path

LOGGER = logging.getLogger("stt-faster.prefetch")

ESTONIAN_MODEL = "TalTechNLP/whisper-large-v3-turbo-et-verbatim"
ENGLISH_MODEL = "Systran/faster-distil-whisper-large-v3"


def prefetch_diarization() -> None:
    LOGGER.info("Prefetching speaker-diarization model: %s", DIARIZATION_REPO)
    LOGGER.info("  -> %s", fetch_model())


def prefetch_all() -> None:
    LOGGER.info("Prefetching Estonian model (ct2/* only): %s", ESTONIAN_MODEL)
    et_path = _get_estonian_model_path(ESTONIAN_MODEL)
    LOGGER.info("  -> %s", et_path)

    LOGGER.info("Prefetching English model (full snapshot): %s", ENGLISH_MODEL)
    en_path = _get_cached_model_path(ENGLISH_MODEL)
    LOGGER.info("  -> %s", en_path)

    prefetch_diarization()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--diarization-only", action="store_true", help="fetch only the speaker model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        prefetch_diarization() if args.diarization_only else prefetch_all()
    except Exception:
        LOGGER.exception("Model prefetch failed")
        return 1
    LOGGER.info("Prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
